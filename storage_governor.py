"""
storage_governor.py

Graded, hysteretic write admission for the data drive (janitor.monitor_path).

WHY: the janitor frees space hourly, but data is produced continuously. Before
this, the only protection was a one-shot latch: one failed cleanup set
emergency_stop_active, which inhibits ALL capture and is never cleared -- a
remote station stayed dead after a single bad hour, even once SFTP uploads or
the next session made old data deletable. It also shared that latch with the
watchdog's thread-crash emergency, so it could not simply be cleared.

LEVELS (decided from the live filesystem, statvfs, on every admission check):
    OK    normal.
    SHED  usage >= janitor.threshold: suspend low-value writes (STORAGE_SHED_DIRS:
          debug, rejected, daylight snapshots) and wake the janitor NOW. Held
          until usage < janitor.target -- the janitor's own hysteresis band.
    HALT  free space < janitor.storage_reserve_mb: refuse every artifact write
          and stop acquisition (the scheduler watches `halt_event`). Released
          only once free >= reserve AND usage < threshold, so it cannot flap
          at the reserve boundary.

A write is also refused if it would itself push free space below the reserve
(expected_bytes known up front: an image's raw size, a video's input frames),
so the last free megabytes stay available for logs, SQLite/journald and the OS.

Recovery is automatic: whenever uploads or the janitor free space, the next
check drops the level; the scheduler restarts acquisition on its own.
"""
import logging
import os
import shutil
import threading
from enum import IntEnum


class StorageLevel(IntEnum):
    OK = 0
    SHED = 1
    HALT = 2


def next_level(prev, used_pct, free_bytes, threshold_pct, target_pct, reserve_bytes):
    """Pure transition function (unit-testable). See module docstring for the rules."""
    if free_bytes < reserve_bytes:
        return StorageLevel.HALT
    if prev == StorageLevel.HALT and used_pct >= threshold_pct:
        return StorageLevel.HALT
    if used_pct >= threshold_pct:
        return StorageLevel.SHED
    if prev >= StorageLevel.SHED and used_pct >= target_pct:
        return StorageLevel.SHED
    return StorageLevel.OK


class StorageGovernor:
    def __init__(self, monitor_path, janitor_cfg_getter, shed_dirs, on_transition=None):
        """
        janitor_cfg_getter: callable returning the CURRENT janitor config dict (so
        a config reload is honoured without rebuilding the governor).
        shed_dirs: absolute directories whose writes are suspended at SHED.
        on_transition(old, new, used_pct, free_bytes): called outside the lock.
        """
        self._path = monitor_path
        self._cfg = janitor_cfg_getter
        self._shed_dirs = [os.path.realpath(d) for d in shed_dirs if d]
        self._on_transition = on_transition
        self._lock = threading.Lock()
        self._level = StorageLevel.OK
        self.halt_event = threading.Event()   # set while HALT (the scheduler stops acquisition)
        self.released_event = threading.Event()  # set while NOT halted (the scheduler waits on it)
        self.released_event.set()
        self.wake_janitor = threading.Event()  # set on entering SHED/HALT from a write-time check
        self._refused = {}                     # dir -> count since the last summary

    @property
    def level(self):
        return self._level

    def refresh(self):
        """Re-evaluate the level from the live filesystem; returns (level, used_pct, free_bytes)."""
        du = shutil.disk_usage(self._path)
        # Same definition as psutil.disk_usage().percent, which the janitor's
        # threshold/target are compared against: used / (used + available), i.e.
        # excluding the filesystem's root-reserved blocks. free = f_bavail, the
        # space an unprivileged writer can actually use.
        denom = du.used + du.free
        used_pct = 100.0 * du.used / denom if denom else 100.0
        cfg = self._cfg()
        reserve = cfg["storage_reserve_mb"] * 1024 * 1024
        with self._lock:
            old = self._level
            new = next_level(old, used_pct, du.free, cfg["threshold"], cfg["target"], reserve)
            self._level = new
        if new != old:
            if new == StorageLevel.HALT:
                self.released_event.clear()
                self.halt_event.set()
            else:
                self.halt_event.clear()
                self.released_event.set()
            if new > old:
                self.wake_janitor.set()
            if self._on_transition:
                self._on_transition(old, new, used_pct, du.free)
        return new, used_pct, du.free

    def interrupt(self):
        """Wake every thread sleeping on governor events (shutdown)."""
        self.wake_janitor.set()
        self.released_event.set()

    def _is_shed(self, dest_path):
        d = os.path.realpath(os.path.dirname(dest_path) or ".")
        return any(d == s or d.startswith(s + os.sep) for s in self._shed_dirs)

    def admits(self, dest_path, expected_bytes=0):
        """True if writing ~expected_bytes to dest_path is allowed right now."""
        try:
            level, _, free = self.refresh()
        except OSError as e:
            # Cannot even stat the drive: the write itself would fail the same way.
            logging.error(f"Storage: cannot stat {self._path} ({e}); refusing write to {dest_path}.")
            return False
        reserve = self._cfg()["storage_reserve_mb"] * 1024 * 1024
        allowed = (
            level != StorageLevel.HALT
            and not (level == StorageLevel.SHED and self._is_shed(dest_path))
            and free - int(expected_bytes) >= reserve
        )
        if not allowed:
            key = os.path.dirname(dest_path)
            with self._lock:
                self._refused[key] = self._refused.get(key, 0) + 1
        return allowed

    def pop_refusal_summary(self):
        """{dir: refused_count} since the last call (reported by the janitor, not per write)."""
        with self._lock:
            out, self._refused = self._refused, {}
        return out
