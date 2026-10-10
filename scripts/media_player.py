#!/usr/bin/env python3
"""
Python Media Player — PyQt6 + libVLC.

Architettura:
  * libVLC notifica gli eventi su un proprio thread interno. Ogni evento viene
    trasferito sul thread GUI tramite segnali Qt in QueuedConnection: nessuna
    chiamata libVLC e nessun accesso ai widget avviene mai dal thread di VLC.
  * `Playlist` è l'unica sorgente di verità per percorsi e ordine di
    riproduzione (permutazione Fisher–Yates per lo shuffle, con cronologia).
  * La scansione delle cartelle gira su un QThreadPool dedicato e cancellabile.
  * Ogni parametro vive in `PlayerConfig` (Pydantic, immutabile, validato).

Dipendenze: python-vlc, PyQt6, pydantic>=2
"""
from __future__ import annotations

import ctypes
import functools
import itertools
import logging
import os
import random
import re
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import vlc
from pydantic import (
    BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator,
)
from PyQt6.QtCore import (
    QByteArray, QElapsedTimer, QKeyCombination, QMargins, QObject, QPoint, QRect,
    QRunnable, QThreadPool, QTimer, Qt, pyqtSignal,
)
from PyQt6.QtGui import (
    QCloseEvent, QColor, QDragEnterEvent, QDropEvent, QGuiApplication,
    QKeySequence, QMouseEvent, QPalette, QShortcut, QShowEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QSizePolicy, QSlider, QSplitter, QStyle, QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger("mediaplayer")

# Conversioni di unità e limiti di tipo (non soglie arbitrarie).
_MS_PER_S = 1000
_S_PER_MIN = 60
_S_PER_H = 3600
_QT_INT_MAX = 2**31 - 1  # QSlider usa int32 con segno: ~24.8 giorni in ms

_LABEL_PLAY = "▶ Play"
_LABEL_PAUSE = "⏸ Pausa"
_LABEL_NO_TIME = "--:--"


# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #
class PlayerConfig(BaseModel):
    """Configurazione immutabile e validata: nessun numero magico nel codice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_size: tuple[PositiveInt, PositiveInt] = (1050, 620)
    splitter_ratio: tuple[PositiveInt, PositiveInt] = (7, 3)

    # Le etichette temporali hanno risoluzione di 1 s: campionare a >= 2 Hz
    # garantisce che nessun secondo venga saltato a video (Nyquist).
    ui_refresh_hz: float = Field(default=5.0, ge=2.0, le=60.0)

    initial_volume: int = Field(default=70, ge=0)
    max_volume: int = Field(default=100, ge=1, le=200)  # libVLC: fino a 200 (amplificazione)
    volume_step: PositiveInt = 5
    seek_step_ms: PositiveInt = 5_000
    # "Precedente" oltre questa posizione riavvia il brano (convenzione standard).
    prev_restart_threshold_ms: int = Field(default=3_000, ge=0)
    status_message_ms: int = Field(default=5_000, ge=0)
    # PiP: frazione massima dello schermo disponibile occupata per lato; il
    # riquadro rispetta l'aspect ratio del video entro entrambi i limiti.
    pip_screen_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    pip_corner: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-right"
    shutdown_timeout_ms: int = Field(default=2_000, ge=0)

    extensions: frozenset[str] = Field(
        default=frozenset({
            ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm",
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a",
        }),
        min_length=1,
    )
    vlc_args: tuple[str, ...] = (
        "--no-video-title-show",
        "--no-mouse-events",
        "--no-keyboard-events",
    )

    @field_validator("extensions")
    @classmethod
    def _normalise_extensions(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in value
        )

    @model_validator(mode="after")
    def _check_consistency(self) -> PlayerConfig:
        if self.initial_volume > self.max_volume:
            raise ValueError("initial_volume deve essere <= max_volume")
        if self.volume_step > self.max_volume:
            raise ValueError("volume_step deve essere <= max_volume")
        return self

    @property
    def ui_refresh_interval_ms(self) -> int:
        return round(_MS_PER_S / self.ui_refresh_hz)

    @property
    def file_dialog_filter(self) -> str:
        pattern = " ".join(f"*{ext}" for ext in sorted(self.extensions))
        return f"File multimediali ({pattern});;Tutti i file (*)"


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> list[int | str]:
    """Ordinamento naturale: 'traccia2' < 'traccia10'.

    re.split con gruppo di cattura alterna testo (indici pari) e numeri
    (indici dispari): i tipi sono quindi sempre allineati nel confronto.
    """
    return [
        int(tok) if i % 2 else tok.casefold()
        for i, tok in enumerate(_NUM_RE.split(text))
    ]


def format_time(ms: int) -> str:
    if ms < 0:
        return _LABEL_NO_TIME
    total_s = ms // _MS_PER_S
    hours, rem = divmod(total_s, _S_PER_H)
    minutes, seconds = divmod(rem, _S_PER_MIN)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _enum_value(value: Any) -> int:
    """Valore intero di un enum python-vlc letto da una struct ctypes."""
    return int(getattr(value, "value", value))


# Orientamenti libVLC con assi scambiati (left_top, left_bottom, right_top, right_bottom).
_TRANSPOSED_ORIENTATIONS = frozenset(
    _enum_value(o) for o in (vlc.VideoOrient.left_top, vlc.VideoOrient.left_bottom,
                             vlc.VideoOrient.right_top, vlc.VideoOrient.right_bottom))


# --- Lettura delle tracce: binding ctypes propri --------------------------
# python-vlc 3.0.x ha due difetti su libvlc_media_tracks_get:
#   * MediaTrack dichiara audio/video/subtitle come tre puntatori consecutivi,
#     ma nell'ABI di libVLC 3 sono una UNION: leggere .video dereferenzia
#     memoria sbagliata (segfault verificato);
#   * Media.tracks_get() non chiama mai libvlc_media_tracks_release (leak).
# Le struct sotto rispecchiano libvlc_media.h di VLC 3; si leggono solo i campi
# necessari, sempre tramite puntatore (la dimensione totale è irrilevante).
class _LibvlcVideoTrack(ctypes.Structure):
    _fields_ = (
        ("height", ctypes.c_uint), ("width", ctypes.c_uint),
        ("sar_num", ctypes.c_uint), ("sar_den", ctypes.c_uint),
        ("frame_rate_num", ctypes.c_uint), ("frame_rate_den", ctypes.c_uint),
        ("orientation", ctypes.c_int),
    )


class _LibvlcTrackUnion(ctypes.Union):
    _fields_ = (
        ("audio", ctypes.c_void_p),
        ("video", ctypes.POINTER(_LibvlcVideoTrack)),
        ("subtitle", ctypes.c_void_p),
    )


class _LibvlcMediaTrack(ctypes.Structure):
    _fields_ = (
        ("codec", ctypes.c_uint32), ("original_fourcc", ctypes.c_uint32),
        ("id", ctypes.c_int), ("type", ctypes.c_int),
        ("profile", ctypes.c_int), ("level", ctypes.c_int),
        ("u", _LibvlcTrackUnion),
        ("bitrate", ctypes.c_uint),
        ("language", ctypes.c_char_p), ("description", ctypes.c_char_p),
    )


_libvlc_tracks_get = vlc.dll.libvlc_media_tracks_get
_libvlc_tracks_get.restype = ctypes.c_uint  # argtypes assenti: Media passa _as_parameter_
_libvlc_tracks_release = vlc.dll.libvlc_media_tracks_release
_libvlc_tracks_release.restype = None


_F = TypeVar("_F", bound=Callable[..., Any])


def _guard_alive(method: _F) -> _F:
    """Scarta eventi accodati che arrivano dopo il rilascio delle risorse libVLC."""
    @functools.wraps(method)
    def wrapper(self: MediaPlayer, *args: Any) -> Any:
        if self._closing:
            return None
        return method(self, *args)
    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Ponte eventi libVLC -> thread GUI
# --------------------------------------------------------------------------- #
class VlcEventBridge(QObject):
    """
    libVLC invoca le callback sul proprio thread interno. Da lì è vietato:
      1) toccare widget Qt (non thread-safe);
      2) richiamare libVLC sullo stesso player (es. set_media()/play() dentro
         EndReached): deadlock documentato di libVLC.
    Le callback quindi emettono soltanto un segnale; i receiver sono collegati
    in QueuedConnection e girano sul thread GUI, in ordine FIFO.
    """

    playing = pyqtSignal()
    paused = pyqtSignal()
    stopped = pyqtSignal()
    end_reached = pyqtSignal()
    error = pyqtSignal()
    # Vout creato/distrutto o nuovo elementary stream: le dimensioni del
    # video sorgente possono essere cambiate.
    video_geometry_hint = pyqtSignal()

    def __init__(self, player: vlc.MediaPlayer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._manager = player.event_manager()
        et = vlc.EventType
        # Riferimenti forti alle callback: ctypes non deve poterle raccogliere.
        self._bindings: tuple[tuple[Any, Callable[[Any], None]], ...] = (
            (et.MediaPlayerPlaying, lambda _e: self.playing.emit()),
            (et.MediaPlayerPaused, lambda _e: self.paused.emit()),
            (et.MediaPlayerStopped, lambda _e: self.stopped.emit()),
            (et.MediaPlayerEndReached, lambda _e: self.end_reached.emit()),
            (et.MediaPlayerEncounteredError, lambda _e: self.error.emit()),
            (et.MediaPlayerVout, lambda _e: self.video_geometry_hint.emit()),
            (et.MediaPlayerESAdded, lambda _e: self.video_geometry_hint.emit()),
        )
        for event_type, callback in self._bindings:
            self._manager.event_attach(event_type, callback)

    def detach(self) -> None:
        for event_type, _ in self._bindings:
            self._manager.event_detach(event_type)


# --------------------------------------------------------------------------- #
# Modello playlist
# --------------------------------------------------------------------------- #
class Playlist:
    """
    Sorgente di verità unica per percorsi e ordine di riproduzione.

    `_order` è una permutazione degli indici di `_paths`; `_cursor` è la
    posizione corrente dentro `_order`. In modalità sequenziale `_order` è
    l'identità; in shuffle è una permutazione Fisher–Yates: ogni brano viene
    riprodotto esattamente una volta per ciclo e "Precedente" ripercorre la
    cronologia reale. Mutata solo dal thread GUI (nessun lock necessario).
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._paths: list[str] = []
        self._order: list[int] = []
        self._cursor = -1
        self._shuffle = False

    def __len__(self) -> int:
        return len(self._paths)

    def path(self, index: int) -> str:
        return self._paths[index]

    @property
    def current_index(self) -> int:
        if 0 <= self._cursor < len(self._order):
            return self._order[self._cursor]
        return -1

    def extend(self, paths: Iterable[str]) -> None:
        start = len(self._paths)
        self._paths.extend(paths)
        new = list(range(start, len(self._paths)))
        if self._shuffle:
            # I nuovi brani entrano nella parte ancora da riprodurre del ciclo.
            future = self._order[self._cursor + 1:] + new
            self._rng.shuffle(future)
            self._order = self._order[: self._cursor + 1] + future
        else:
            self._order.extend(new)

    def remove(self, index: int) -> bool:
        """Rimuove un brano; ritorna True se era quello corrente."""
        was_current = index == self.current_index
        pos = self._order.index(index)
        self._order.pop(pos)
        self._order = [i - 1 if i > index else i for i in self._order]
        del self._paths[index]
        if pos <= self._cursor:
            self._cursor -= 1
        return was_current

    def clear(self) -> None:
        self._paths.clear()
        self._order.clear()
        self._cursor = -1

    def set_shuffle(self, enabled: bool) -> None:
        if enabled == self._shuffle:
            return
        self._shuffle = enabled
        current = self.current_index
        n = len(self._paths)
        if enabled:
            rest = [i for i in range(n) if i != current]
            self._rng.shuffle(rest)
            self._order = ([current] if current >= 0 else []) + rest
            self._cursor = 0 if current >= 0 else -1
        else:
            self._order = list(range(n))
            self._cursor = current

    def jump_to(self, index: int) -> None:
        """Selezione esplicita: in shuffle il brano viene inserito subito dopo
        la cronologia, preservando l'insieme dei brani non ancora riprodotti."""
        if not self._shuffle:
            self._cursor = index
            return
        pos = self._order.index(index)
        self._order.pop(pos)
        if pos <= self._cursor:
            self._cursor -= 1
        self._order.insert(self._cursor + 1, index)
        self._cursor += 1

    def advance(self, step: int, *, wrap: bool) -> int | None:
        n = len(self._order)
        if n == 0:
            return None
        if self._cursor < 0:
            target = 0 if step > 0 else n - 1
        else:
            target = self._cursor + step

        if 0 <= target < n:
            self._cursor = target
        elif not wrap:
            return None
        elif self._shuffle and target >= n:
            # Nuovo ciclo: nuova permutazione, senza ripetere subito l'ultimo brano.
            last = self.current_index
            self._rng.shuffle(self._order)
            if n > 1 and self._order[0] == last:
                j = self._rng.randrange(1, n)
                self._order[0], self._order[j] = self._order[j], self._order[0]
            self._cursor = 0
        else:
            self._cursor = target % n
        return self.current_index


# --------------------------------------------------------------------------- #
# Scansione cartelle in background
# --------------------------------------------------------------------------- #
class _ScanSignals(QObject):
    finished = pyqtSignal(int, str, list)  # scan_id, root, files


class FolderScanTask(QRunnable):
    def __init__(
        self, scan_id: int, root: str, extensions: frozenset[str],
        cancel: threading.Event,
    ) -> None:
        super().__init__()
        self.signals = _ScanSignals()
        self._scan_id = scan_id
        self._root = root
        self._extensions = extensions
        self._cancel = cancel

    def run(self) -> None:
        matched: list[str] = []

        def _on_error(err: OSError) -> None:
            log.warning("Scansione: %s", err)

        for dirpath, dirnames, filenames in os.walk(self._root, onerror=_on_error):
            if self._cancel.is_set():
                return
            dirnames.sort(key=natural_key)  # attraversamento deterministico
            matched.extend(
                os.path.join(dirpath, f)
                for f in sorted(filenames, key=natural_key)
                if os.path.splitext(f)[1].lower() in self._extensions
            )
        self.signals.finished.emit(self._scan_id, self._root, matched)


# --------------------------------------------------------------------------- #
# Widget
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PipSavedState:
    """Stato della finestra da ripristinare all'uscita dal PiP."""
    geometry: QByteArray          # include lo stato massimizzato
    flags: Qt.WindowType
    margins: QMargins


class VideoFrame(QFrame):
    doubleClicked = pyqtSignal()
    # Emesso quando un trascinamento supera la soglia di drag del sistema.
    moveRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # libVLC necessita di una finestra nativa propria; senza forzare gli
        # antenati a diventare nativi (evita flicker e overhead di compositing).
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(Qt.GlobalColor.black))
        self.setPalette(palette)
        self._press_pos: QPoint | None = None

    # Lo spostamento parte solo dopo QApplication.startDragDistance(): avviarlo
    # sulla pressione entrerebbe nel loop di move nativo e il sistema non
    # riconoscerebbe più il doppio click (che serve a uscire dal PiP).
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            travelled = (event.position().toPoint() - self._press_pos).manhattanLength()
            if travelled >= QApplication.startDragDistance():
                self._press_pos = None
                self.moveRequested.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class SeekSlider(QSlider):
    """Slider orizzontale: il click sul binario porta la maniglia sotto il
    cursore (e avvia il trascinamento) invece di avanzare di pageStep."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            style = self.style()
            cc = QStyle.ComplexControl.CC_Slider
            handle = style.subControlRect(cc, opt, QStyle.SubControl.SC_SliderHandle, self)
            click = event.position().toPoint()
            if not handle.contains(click):
                groove = style.subControlRect(cc, opt, QStyle.SubControl.SC_SliderGroove, self)
                span = groove.width() - handle.width()
                offset = click.x() - groove.x() - handle.width() // 2
                self.setValue(QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), offset, span, opt.upsideDown))
        super().mousePressEvent(event)


def _vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


# --------------------------------------------------------------------------- #
# Finestra principale
# --------------------------------------------------------------------------- #
class MediaPlayer(QMainWindow):
    def __init__(self, config: PlayerConfig) -> None:
        super().__init__()
        self._cfg = config
        self._closing = False
        self._video_bound = False
        self._length_ms = 0
        self._consecutive_errors = 0
        # Seek da tastiera: libVLC applica set_time() in modo asincrono, quindi
        # get_time() subito dopo restituisce ancora la posizione VECCHIA. Le
        # pressioni ravvicinate si accumulano sull'ultimo target richiesto.
        self._seek_target_ms = -1
        self._seek_clock = QElapsedTimer()
        self._pre_fullscreen_state = Qt.WindowState.WindowNoState
        self._pip_saved: _PipSavedState | None = None  # not None <=> PiP attivo

        instance = vlc.Instance(list(config.vlc_args))
        if instance is None:
            raise RuntimeError("Inizializzazione libVLC fallita: verificare VLC e vlc_args.")
        self._instance: vlc.Instance = instance
        self._player: vlc.MediaPlayer = self._instance.media_player_new()
        # Input ownership belongs to Qt. Set it on the player object itself
        # (documented libVLC contract) instead of relying on the instance-level
        # --no-mouse-events being inherited by the platform video output.
        # Must be called before the first play(): the vout reads it on creation.
        self._player.video_set_mouse_input(False)
        self._player.video_set_key_input(False)
        self._playlist = Playlist(random.Random())

        # Un solo worker: le cartelle vengono accodate in ordine FIFO di aggiunta.
        self._scan_pool = QThreadPool(self)
        self._scan_pool.setMaxThreadCount(1)
        self._scan_cancel = threading.Event()
        self._scan_ids = itertools.count()
        self._active_scans: dict[int, _ScanSignals] = {}

        self.setWindowTitle("Python Media Player")
        self.resize(*config.window_size)
        self.setAcceptDrops(True)

        self._init_ui()
        self._init_shortcuts()
        self._refresh_playlist_counter()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(config.ui_refresh_interval_ms)
        self._ui_timer.timeout.connect(self._on_ui_tick)

        self._events = VlcEventBridge(self._player, self)
        queued = Qt.ConnectionType.QueuedConnection
        self._events.playing.connect(self._on_vlc_playing, queued)
        self._events.paused.connect(self._on_vlc_paused, queued)
        self._events.stopped.connect(self._on_vlc_stopped, queued)
        self._events.end_reached.connect(self._on_vlc_end_reached, queued)
        self._events.error.connect(self._on_vlc_error, queued)
        self._events.video_geometry_hint.connect(self._on_video_geometry_hint, queued)

        self._player.audio_set_volume(config.initial_volume)
        self._reset_transport_ui()

    # ------------------------------------------------------------------ UI --
    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        self._main_layout = main_layout

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.video_frame = VideoFrame()
        self.video_frame.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        self.video_frame.doubleClicked.connect(self._on_video_double_click)
        self.video_frame.moveRequested.connect(self._on_video_move_requested)
        self.splitter.addWidget(self.video_frame)

        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Playlist:</b>"))
        header.addStretch(1)
        self.lbl_playlist_counter = QLabel()
        self.lbl_playlist_counter.setToolTip("Brano corrente / totale")
        header.addWidget(self.lbl_playlist_counter)
        right_layout.addLayout(header)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setUniformItemSizes(True)  # layout O(1) su liste grandi
        self.list_widget.itemDoubleClicked.connect(self._on_playlist_double_click)
        # Contatore: si aggancia ai segnali del modello, così qualunque
        # mutazione (aggiunta, rimozione, svuotamento) lo aggiorna senza
        # chiamate sparse nel codice. Gli aggiornamenti sono coalescenti:
        # 50.000 inserimenti producono un solo refresh al giro di event loop.
        self._counter_refresh = QTimer(self)
        self._counter_refresh.setSingleShot(True)
        self._counter_refresh.setInterval(0)
        self._counter_refresh.timeout.connect(self._refresh_playlist_counter)
        model = self.list_widget.model()
        model.rowsInserted.connect(self._counter_refresh.start)
        model.rowsRemoved.connect(self._counter_refresh.start)
        model.modelReset.connect(self._counter_refresh.start)
        right_layout.addWidget(self.list_widget)

        pl_btn_layout = QHBoxLayout()
        for text, slot in (
            ("Aggiungi File", self.add_files),
            ("Aggiungi Cartella", self.add_folder),
            ("Svuota", self.clear_playlist),
        ):
            btn = QPushButton(text)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(slot)
            pl_btn_layout.addWidget(btn)
        right_layout.addLayout(pl_btn_layout)

        mode_layout = QHBoxLayout()
        self.chk_shuffle = QCheckBox("Casuale")
        self.chk_shuffle.toggled.connect(self._playlist.set_shuffle)
        self.chk_repeat = QCheckBox("Ripeti playlist")
        self.chk_repeat.setChecked(True)  # comportamento dell'originale
        for chk in (self.chk_shuffle, self.chk_repeat):
            chk.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            mode_layout.addWidget(chk)
        right_layout.addLayout(mode_layout)

        self.splitter.addWidget(self.right_widget)
        for i, factor in enumerate(self._cfg.splitter_ratio):
            self.splitter.setStretchFactor(i, factor)
        # Stretch 1 vs 0: all surplus vertical space goes to the video area.
        main_layout.addWidget(self.splitter, stretch=1)

        # Controlli inferiori
        self.bottom_controls_widget = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_controls_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        progress_layout = QHBoxLayout()
        self.lbl_current_time = QLabel()
        self.progress_slider = SeekSlider()
        self.progress_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.progress_slider.sliderMoved.connect(self._on_slider_moved)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)
        self.lbl_total_time = QLabel()
        progress_layout.addWidget(self.lbl_current_time)
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.lbl_total_time)
        bottom_layout.addLayout(progress_layout)

        controls = QHBoxLayout()
        self.btn_prev = QPushButton("⏮ Prec")
        self.btn_prev.setToolTip("Precedente (B: sempre il file precedente)")
        self.btn_play = QPushButton(_LABEL_PLAY)
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setToolTip("Stop (S)")
        self.btn_next = QPushButton("Succ ⏭")
        self.btn_next.setToolTip("Successivo (N)")
        for btn, slot in (
            (self.btn_prev, self.play_previous),
            (self.btn_play, self.play_pause),
            (self.btn_stop, self.stop),
            (self.btn_next, self.play_next),
        ):
            # NoFocus: altrimenti Space attiva sia il bottone a fuoco sia lo shortcut.
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        controls.addWidget(_vsep())
        controls.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.volume_slider.setRange(0, self._cfg.max_volume)
        self.volume_slider.setValue(self._cfg.initial_volume)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        controls.addWidget(self.volume_slider)
        self.lbl_volume = QLabel(f"{self._cfg.initial_volume}%")
        controls.addWidget(self.lbl_volume)

        controls.addWidget(_vsep())
        self.btn_fullscreen = QPushButton("⛶ Schermo Intero")
        self.btn_fullscreen.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        controls.addWidget(self.btn_fullscreen)

        self.btn_pip = QPushButton("⧉ PiP")
        self.btn_pip.setToolTip("Picture-in-Picture (P)")
        self.btn_pip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_pip.clicked.connect(self.toggle_pip)
        controls.addWidget(self.btn_pip)

        bottom_layout.addLayout(controls)
        # Maximum: the controls never grow beyond their sizeHint().
        self.bottom_controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                                                  QSizePolicy.Policy.Maximum)
        main_layout.addWidget(self.bottom_controls_widget, stretch=0)

        # Barra di stato: a sinistra messaggi temporanei (eventi), a destra
        # un indicatore permanente del brano in riproduzione.
        self.lbl_now_playing = QLabel()
        self.statusBar().addPermanentWidget(self.lbl_now_playing)

    def _init_shortcuts(self) -> None:
        k = Qt.Key
        window_ctx = Qt.ShortcutContext.WindowShortcut
        step = self._cfg.seek_step_ms
        bindings: tuple[tuple[tuple[QKeyCombination, ...], Callable[[], None]], ...] = (
            ((QKeyCombination(k.Key_Space), QKeyCombination(k.Key_MediaTogglePlayPause),
              QKeyCombination(k.Key_MediaPlay), QKeyCombination(k.Key_MediaPause)),
             self.play_pause),
            ((QKeyCombination(k.Key_S), QKeyCombination(k.Key_MediaStop)), self.stop),
            ((QKeyCombination(k.Key_R),), self.restart_current),
            ((QKeyCombination(k.Key_MediaNext), QKeyCombination(k.Key_N)), self.play_next),
            ((QKeyCombination(k.Key_B),), self.play_previous_file),
            ((QKeyCombination(k.Key_MediaPrevious),), self.play_previous),
            ((QKeyCombination(k.Key_F), QKeyCombination(k.Key_F11)), self.toggle_fullscreen),
            ((QKeyCombination(k.Key_Escape),), self._on_escape),
            ((QKeyCombination(k.Key_P),), self.toggle_pip),
            ((QKeyCombination(k.Key_Right),), lambda: self._seek_relative(+step)),
            ((QKeyCombination(k.Key_Left),), lambda: self._seek_relative(-step)),
            ((QKeyCombination(Qt.KeyboardModifier.ControlModifier, k.Key_Up),),
             lambda: self._step_volume(+1)),
            ((QKeyCombination(Qt.KeyboardModifier.ControlModifier, k.Key_Down),),
             lambda: self._step_volume(-1)),
        )
        for combos, slot in bindings:
            for combo in combos:
                self._add_shortcut(combo, slot, self, window_ctx)
        self._add_shortcut(QKeyCombination(k.Key_Delete), self._remove_selected,
                           self.list_widget, Qt.ShortcutContext.WidgetShortcut)

    @staticmethod
    def _add_shortcut(combo: QKeyCombination, slot: Callable[[], None],
                      parent: QWidget, context: Qt.ShortcutContext) -> None:
        shortcut = QShortcut(QKeySequence(combo), parent)
        shortcut.setContext(context)
        shortcut.activated.connect(slot)

    def _status(self, message: str) -> None:
        self.statusBar().showMessage(message, self._cfg.status_message_ms)

    # ---------------------------------------------------- output video ----
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._video_bound:
            self._bind_video_output()
            self._video_bound = True

    def _bind_video_output(self) -> None:
        """Associazione unica dell'handle nativo, dopo che la finestra esiste."""
        win_id = int(self.video_frame.winId())
        if sys.platform.startswith("linux"):
            if QGuiApplication.platformName() != "xcb":
                log.error("libVLC richiede X11/xcb per l'embedding (piattaforma Qt: %s)",
                          QGuiApplication.platformName())
            self._player.set_xwindow(win_id)
        elif sys.platform == "win32":
            self._player.set_hwnd(win_id)
        elif sys.platform == "darwin":
            self._player.set_nsobject(win_id)

    # ------------------------------------------------------ fullscreen ----
    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self._exit_fullscreen()
            return
        self._exit_pip()  # modalità mutuamente esclusive
        self._pre_fullscreen_state = self.windowState()  # preserva "massimizzato"
        self.right_widget.hide()
        self.bottom_controls_widget.hide()
        self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        self.setWindowState(self._pre_fullscreen_state)
        self.right_widget.show()
        self.bottom_controls_widget.show()

    def _on_escape(self) -> None:
        if self._pip_saved is not None:
            self._exit_pip()
        else:
            self._exit_fullscreen()

    def _on_video_double_click(self) -> None:
        if self._pip_saved is not None:
            self._exit_pip()
        else:
            self.toggle_fullscreen()

    def _on_video_move_requested(self) -> None:
        handle = self.windowHandle()
        if self._pip_saved is not None and handle is not None:
            handle.startSystemMove()  # move nativo del window manager

    # ------------------------------------------------------------- PiP ----
    # Perché QWindow.setFlags() e non QWidget.setWindowFlags(): quest'ultimo
    # chiama setParent() e RICREA la finestra nativa, distruggendo l'HWND/XID
    # figlio in cui libVLC sta renderizzando (schermo nero o crash). QWindow
    # aggiorna gli stili della finestra esistente sul posto: il vout di VLC
    # non si accorge di nulla e la riproduzione prosegue senza interruzioni.
    def toggle_pip(self) -> None:
        if self._pip_saved is not None:
            self._exit_pip()
        else:
            self._enter_pip()

    def _pip_chrome(self) -> tuple[QWidget, ...]:
        return (self.right_widget, self.bottom_controls_widget, self.statusBar())

    def _enter_pip(self) -> None:
        handle = self.windowHandle()
        if handle is None or self._pip_saved is not None:
            return
        self._exit_fullscreen()
        self._pip_saved = _PipSavedState(
            geometry=self.saveGeometry(),
            flags=handle.flags(),
            margins=self._main_layout.contentsMargins(),
        )
        if self.isMaximized():
            self.showNormal()
        for widget in self._pip_chrome():
            widget.hide()
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        handle.setFlags(handle.flags()
                        | Qt.WindowType.FramelessWindowHint
                        | Qt.WindowType.WindowStaysOnTopHint)
        self._defer_after_frame_change(self._apply_pip_geometry)

    def _exit_pip(self) -> None:
        handle = self.windowHandle()
        saved = self._pip_saved
        if handle is None or saved is None:
            return
        self._pip_saved = None
        handle.setFlags(saved.flags)
        self._main_layout.setContentsMargins(saved.margins)
        for widget in self._pip_chrome():
            widget.show()
        self._defer_after_frame_change(
            functools.partial(self._apply_restored_geometry, saved.geometry))

    # Cambiare la cornice (FramelessWindowHint) su Windows fa sì che Qt accodi
    # una notifica ASINCRONA di cambio geometria, con il rettangolo valido in
    # quel momento. Una setGeometry() immediata viene poi sovrascritta da quella
    # notifica obsoleta: è il riquadro sbagliato che vedevi. La geometria va
    # quindi applicata dopo che gli eventi del window system sono stati
    # consegnati. Il dispatcher Qt consegna sempre gli eventi del window system
    # prima degli eventi accodati, quindi una chiamata accodata a 0 ms arriva
    # deterministicamente dopo la notifica obsoleta (nessuna attesa temporale).
    def _defer_after_frame_change(self, apply: Callable[[], None]) -> None:
        QTimer.singleShot(0, apply)

    @_guard_alive
    def _apply_pip_geometry(self) -> None:
        if self._pip_saved is not None:  # l'utente potrebbe essere già uscito
            self.setGeometry(self._pip_geometry(anchor=None))

    @_guard_alive
    def _apply_restored_geometry(self, geometry: QByteArray) -> None:
        if self._pip_saved is None:  # l'utente potrebbe essere già rientrato
            self.restoreGeometry(geometry)

    @_guard_alive
    def _on_video_geometry_hint(self) -> None:
        """Il video sorgente è (di nuovo) noto: in PiP il riquadro si adatta
        mantenendo fermo l'angolo di ancoraggio della posizione attuale."""
        if self._pip_saved is not None:
            self.setGeometry(self._pip_geometry(anchor=self.geometry()))

    def _pip_geometry(self, anchor: QRect | None) -> QRect:
        """Riquadro PiP con l'aspect del video. `anchor=None` lo posiziona
        nell'angolo configurato dello schermo; altrimenti mantiene fermo il
        corrispondente angolo del rettangolo dato (finestra già spostata)."""
        area = (self.screen().availableGeometry() if anchor is None
                else (QGuiApplication.screenAt(anchor.center()) or self.screen()).availableGeometry())
        aspect = self._display_aspect(area)
        f = self._cfg.pip_screen_fraction
        # Massimo riquadro con quell'aspect che sta in (f·W) x (f·H):
        # gestisce sia video orizzontali sia verticali.
        width = min(area.width() * f, area.height() * f * aspect)
        w, h = round(width), round(width / aspect)
        corner = self._cfg.pip_corner
        if anchor is None:
            margin = self.style().pixelMetric(QStyle.PixelMetric.PM_LayoutRightMargin)
            left, top = area.left() + margin, area.top() + margin
            right, bottom = area.right() + 1 - margin, area.bottom() + 1 - margin
        else:
            left, top = anchor.left(), anchor.top()
            right, bottom = anchor.right() + 1, anchor.bottom() + 1
        x = left if corner.endswith("left") else right - w
        y = top if corner.startswith("top") else bottom - h
        return QRect(x, y, w, h)

    def _display_aspect(self, fallback_area: QRect) -> float:
        """Aspect ratio di visualizzazione, in ordine di affidabilità:
        1) tracce del media (larghezza·SAR / altezza, rotazione inclusa);
        2) dimensioni del vout (pixel quadrati assunti);
        3) frame corrente; 4) schermo."""
        source = self._source_display_aspect()
        if source is not None:
            return source
        try:
            vw, vh = self._player.video_get_size(0)
        except vlc.VLCException:
            vw = vh = 0
        for w, h in ((vw, vh),
                     (self.video_frame.width(), self.video_frame.height()),
                     (fallback_area.width(), fallback_area.height())):
            if w > 0 and h > 0:
                return w / h
        return 1.0  # irraggiungibile: uno schermo ha sempre area positiva

    def _source_display_aspect(self) -> float | None:
        """DAR dalla prima traccia video del media corrente (vedi _LibvlcMediaTrack)."""
        media = self._player.get_media()
        if media is None:
            return None
        tracks = ctypes.POINTER(ctypes.POINTER(_LibvlcMediaTrack))()
        count = _libvlc_tracks_get(media, ctypes.byref(tracks))
        try:
            for i in range(count):
                track = tracks[i].contents
                if track.type != _enum_value(vlc.TrackType.video) or not track.u.video:
                    continue
                video = track.u.video.contents
                if video.width == 0 or video.height == 0:
                    continue
                sar = (video.sar_num / video.sar_den
                       if video.sar_num > 0 and video.sar_den > 0 else 1.0)
                dar = video.width * sar / video.height
                # Orientamenti trasposti (rotazioni di 90°, es. video da smartphone).
                if _enum_value(video.orientation) in _TRANSPOSED_ORIENTATIONS:
                    dar = 1.0 / dar
                return dar
            return None
        finally:
            if count:
                _libvlc_tracks_release(tracks, count)
            media.release()  # get_media() restituisce un riferimento trattenuto

    # -------------------------------------------------------- playlist ----
    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleziona file multimediali", "", self._cfg.file_dialog_filter)
        if files:
            self._append_to_playlist(files)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella multimediale")
        if folder:
            self._start_scan(folder)

    def _start_scan(self, root: str) -> None:
        scan_id = next(self._scan_ids)
        task = FolderScanTask(scan_id, root, self._cfg.extensions, self._scan_cancel)
        # Riferimento forte ai segnali finché il risultato non è consegnato.
        self._active_scans[scan_id] = task.signals
        task.signals.finished.connect(self._on_scan_finished,
                                      Qt.ConnectionType.QueuedConnection)
        self.statusBar().showMessage(f"Scansione di {root}…")
        self._scan_pool.start(task)

    @_guard_alive
    def _on_scan_finished(self, scan_id: int, root: str, files: list[str]) -> None:
        self._active_scans.pop(scan_id, None)
        if files:
            self._append_to_playlist(files)
        self._status(f"{len(files)} file trovati in {root}")

    def _append_to_playlist(self, paths: Sequence[str]) -> None:
        # Normalizzazione al confine d'ingresso: QFileDialog e QUrl restituiscono
        # separatori '/' anche su Windows, mentre vlc_path2uri() esige il
        # separatore nativo dopo la lettera di unità ("C:\\...") e altrimenti
        # fallisce con ENOTSUP -> libvlc_media_new_path() ritorna NULL.
        paths = [os.path.abspath(p) for p in paths]
        was_idle = self._playlist.current_index < 0 and not self._is_active()
        self._playlist.extend(paths)
        self.list_widget.setUpdatesEnabled(False)
        try:
            for path in paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setToolTip(path)
                self.list_widget.addItem(item)
        finally:
            self.list_widget.setUpdatesEnabled(True)
        if was_idle:
            self._consecutive_errors = 0
            self._advance(+1, wrap=True)

    def _refresh_playlist_counter(self) -> None:
        total = len(self._playlist)
        current = self._playlist.current_index
        position = str(current + 1) if current >= 0 else "–"
        self.lbl_playlist_counter.setText(f"{position} / {total}")

    def clear_playlist(self) -> None:
        self.stop()
        self._playlist.clear()
        self.list_widget.clear()

    def _remove_selected(self) -> None:
        rows = sorted((self.list_widget.row(i) for i in self.list_widget.selectedItems()),
                      reverse=True)
        removed_current = False
        for row in rows:  # ordine decrescente: gli indici residui restano validi
            removed_current |= self._playlist.remove(row)
            self.list_widget.takeItem(row)
        if removed_current and self._is_active():
            self._advance(+1, wrap=self.chk_repeat.isChecked())

    def _on_playlist_double_click(self, item: QListWidgetItem) -> None:
        row = self.list_widget.row(item)
        if 0 <= row < len(self._playlist):
            self._consecutive_errors = 0
            self._playlist.jump_to(row)
            self._play_index(row)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files: list[str] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.isdir(path):
                self._start_scan(path)
            elif os.path.splitext(path)[1].lower() in self._cfg.extensions:
                files.append(path)
        if files:
            self._append_to_playlist(files)
        event.acceptProposedAction()

    # --------------------------------------------------- riproduzione ----
    def _is_active(self) -> bool:
        return self._player.get_state() in (
            vlc.State.Opening, vlc.State.Buffering, vlc.State.Playing, vlc.State.Paused)

    def _play_index(self, index: int) -> None:
        # media_new_path: il percorso è trattato come file, non come MRL
        # (nomi con '#', '?' o ':' non vengono interpretati come URI).
        path = self._playlist.path(index)
        media = self._instance.media_new_path(path)
        if media is None:
            # libVLC ha rifiutato il percorso: stesso trattamento di un errore di
            # riproduzione, differito all'event loop (niente ricorsione).
            log.error("libVLC non accetta il percorso: %s", path)
            self.list_widget.setCurrentRow(index)
            self._counter_refresh.start()
            QTimer.singleShot(0, self._on_vlc_error)
            return
        self._player.set_media(media)
        media.release()  # il player trattiene il proprio riferimento
        self._length_ms = 0
        self.list_widget.setCurrentRow(index)
        self._counter_refresh.start()
        if self._player.play() == -1:
            # Differito al giro successivo dell'event loop: niente ricorsione
            # su playlist lunghe composte da file illeggibili.
            QTimer.singleShot(0, self._on_vlc_error)

    def _advance(self, step: int, *, wrap: bool) -> None:
        index = self._playlist.advance(step, wrap=wrap)
        if index is None:
            self._player.stop()
            self._reset_transport_ui()
            return
        self._play_index(index)

    def play_pause(self) -> None:
        state = self._player.get_state()
        if state in (vlc.State.Playing, vlc.State.Buffering):
            self._player.set_pause(1)
        elif state == vlc.State.Paused:
            self._player.set_pause(0)
        elif state == vlc.State.Opening:
            return  # transizione in corso: un secondo play() sarebbe ridondante
        else:
            self._consecutive_errors = 0
            index = self._playlist.current_index
            if index >= 0:
                self._play_index(index)
            else:
                self._advance(+1, wrap=True)

    def stop(self) -> None:
        self._player.stop()
        self._reset_transport_ui()

    def restart_current(self) -> None:
        """Riporta il brano corrente all'inizio.

        In riproduzione o pausa basta un seek (nessuna riapertura del file, lo
        stato di pausa viene mantenuto). Da Stop/Ended il vout non esiste più:
        il file va riaperto.
        """
        index = self._playlist.current_index
        if index < 0:
            return
        if self._is_active() and self._player.is_seekable():
            self._player.set_time(0)
            self._seek_target_ms = -1  # annulla un eventuale accumulo di ←/→
            self.progress_slider.setValue(0)
            self.lbl_current_time.setText(format_time(0))
            return
        self._consecutive_errors = 0
        self._play_index(index)

    def play_next(self) -> None:
        self._consecutive_errors = 0
        self._advance(+1, wrap=True)

    def play_previous(self) -> None:
        if self._is_active() and self._player.get_time() > self._cfg.prev_restart_threshold_ms:
            self._player.set_time(0)
            return
        self.play_previous_file()

    def play_previous_file(self) -> None:
        """Sempre il file precedente, senza la regola di riavvio di ⏮."""
        self._consecutive_errors = 0
        self._advance(-1, wrap=True)

    # ---------------------------------------------------- seek / volume ----
    def _on_slider_moved(self, value: int) -> None:
        # Durante il drag solo anteprima: un seek per pixel saturerebbe demuxer e decoder.
        self.lbl_current_time.setText(format_time(value))

    def _on_slider_released(self) -> None:
        if self._player.is_seekable():
            self._player.set_time(self.progress_slider.value())

    def _seek_in_flight(self) -> bool:
        """True finché le pressioni rientrano nell'intervallo che il sistema
        considera 'digitazione consecutiva' (QStyleHints, non una soglia nostra)."""
        return (self._seek_target_ms >= 0 and self._seek_clock.isValid()
                and self._seek_clock.elapsed()
                < QGuiApplication.styleHints().keyboardInputInterval())

    def _seek_relative(self, delta_ms: int) -> None:
        if self._length_ms <= 0 or not self._player.is_seekable():
            return
        base = self._seek_target_ms if self._seek_in_flight() else self._player.get_time()
        if base < 0:
            return
        target = min(max(0, base + delta_ms), self._length_ms)
        self._player.set_time(target)
        self._seek_target_ms = target
        self._seek_clock.restart()
        # Feedback immediato: la UI mostra il target, non la posizione ritardata.
        self.progress_slider.setValue(min(target, self.progress_slider.maximum()))
        self.lbl_current_time.setText(format_time(target))

    def _on_volume_changed(self, value: int) -> None:
        self._player.audio_set_volume(value)
        self.lbl_volume.setText(f"{value}%")

    def _step_volume(self, direction: int) -> None:
        self.volume_slider.setValue(self.volume_slider.value()
                                    + direction * self._cfg.volume_step)

    # ------------------------------------------------- eventi libVLC ----
    @_guard_alive
    def _on_vlc_playing(self) -> None:
        self._consecutive_errors = 0
        # L'uscita audio esiste solo ora: un volume impostato prima può andare perso.
        self._player.audio_set_volume(self.volume_slider.value())
        self.btn_play.setText(_LABEL_PAUSE)
        self.lbl_now_playing.setText(f"▶ {self._current_name()}")
        if not self._ui_timer.isActive():
            self._ui_timer.start()

    @_guard_alive
    def _on_vlc_paused(self) -> None:
        self.btn_play.setText(_LABEL_PLAY)
        self.lbl_now_playing.setText(f"⏸ {self._current_name()}")
        self._ui_timer.stop()

    def _current_name(self) -> str:
        index = self._playlist.current_index
        return os.path.basename(self._playlist.path(index)) if index >= 0 else ""

    @_guard_alive
    def _on_vlc_stopped(self) -> None:
        self._reset_transport_ui()

    @_guard_alive
    def _on_vlc_end_reached(self) -> None:
        self._advance(+1, wrap=self.chk_repeat.isChecked())

    @_guard_alive
    def _on_vlc_error(self) -> None:
        self._consecutive_errors += 1
        index = self._playlist.current_index
        name = os.path.basename(self._playlist.path(index)) if index >= 0 else "?"
        log.warning("Errore di riproduzione: %s", name)
        # Limite naturale: un intero giro di playlist senza alcun file valido.
        if self._consecutive_errors >= len(self._playlist):
            self._consecutive_errors = 0
            self._status("Nessun file riproducibile: riproduzione interrotta.")
            self.stop()
            return
        self._status(f"Impossibile riprodurre {name}: passo al successivo.")
        self._advance(+1, wrap=True)

    @_guard_alive
    def _on_ui_tick(self) -> None:
        if self._length_ms <= 0:
            length = self._player.get_length()
            if length > 0:
                self._length_ms = length
                self.progress_slider.setRange(0, min(length, _QT_INT_MAX))
                self.lbl_total_time.setText(format_time(length))
        self.progress_slider.setEnabled(self._length_ms > 0 and self._player.is_seekable())

        if self.progress_slider.isSliderDown() or self._seek_in_flight():
            return
        now = self._player.get_time()
        if now >= 0:
            self.progress_slider.setValue(min(now, self.progress_slider.maximum()))
            self.lbl_current_time.setText(format_time(now))

    def _reset_transport_ui(self) -> None:
        self._ui_timer.stop()
        self._length_ms = 0
        self.btn_play.setText(_LABEL_PLAY)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setEnabled(False)
        self.lbl_current_time.setText(format_time(0))
        self.lbl_total_time.setText(_LABEL_NO_TIME)
        self._seek_target_ms = -1
        self.lbl_now_playing.clear()

    # -------------------------------------------------------- shutdown ----
    def closeEvent(self, event: QCloseEvent) -> None:
        """Ordine obbligato: libVLC va fermato e rilasciato PRIMA che Qt
        distrugga la finestra nativa in cui sta renderizzando."""
        self._closing = True
        self._ui_timer.stop()
        self._scan_cancel.set()
        self._scan_pool.clear()
        if not self._scan_pool.waitForDone(self._cfg.shutdown_timeout_ms):
            log.warning("Scansione ancora attiva allo shutdown (filesystem lento?)")
        self._events.detach()
        self._player.stop()
        self._player.release()
        self._instance.release()
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
        # set_xwindow() richiede un XID: sotto Wayland si usa XWayland (xcb).
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    app = QApplication(sys.argv)
    window = MediaPlayer(PlayerConfig())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
