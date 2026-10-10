#!/usr/bin/env python3
"""
RYUSEI PIPELINE - Autonomous All-Sky Observatory
Code name Meteora
-------------------------------------------------
A multi-threaded computer vision pipeline designed for 
continuous meteor detection, astronomical tracking, and time-lapse generation. 
Highly optimized for the Raspberry Pi 5.

Key Capabilities:
- Smart Motion Detection: A lightning-fast 8-bit adaptive OpenCV engine that 
  dynamically absorbs the Earth's rotation and weather to eliminate false positives.
- Thermal & Optical Calibration: Real-time Dark Frame scaling via DS18B20 
  temperature sensors, and automated synthetic Flat-Field map generation.
- AI Gatekeepers (TensorFlow Lite): Dual AI models. One classifies events 
  (Meteor vs. Aircraft/Insect) to maintain a pure dataset, while the other 
  monitors lens health (Rain/Fog/Dirt) to protect the system during bad weather.
- Hardware Resilience: Self-healing architecture featuring a master Watchdog, 
  RAM Out-Of-Memory (OOM) protection, GPIO power monitoring, and a disk Janitor.
- Rich Telemetry: Outputs H.264 event videos, Maximum Intensity Projection (MIP) 
  stacks, continuous Keograms, and detailed JSON/CSV scientific telemetry.
- Live Web Dashboard: A secure, interactive Flask server for real-time monitoring, 
  OTA updates, config reloading, and viewing the live Activity Timeline.

 !!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!
 Always take one good dark frame at ~25°C and save it as dark_fallback.png. It is the "Insurance Policy" for your observatory!

 pip3 install ai-edge-litert --break-system-packages
 python3 -c 'import ai_edge_litert.interpreter as tflite; print("AI Engine is Ready!")'

 It should print "AI Engine is Ready!" perfectly. Your AI engine is installed and ready to go!
 You can now enable ai_classification in your config.json and start classifying meteors.

 TODO:
 camera sensor long disconnections : check power supply
 
 
 run in virtual enviroment if needed
 # Delete the old venv
 rm -rf /home/pi/venv

 # Create a new one with system hardware packages linked!
 python3 -m venv /home/pi/venv --system-site-packages

 # Activate it
 source /home/pi/venv/bin/activate
 
 # Deactivate it
 deactivate
 
 export SFTP_PASS=password
 export RYUSEI_DASH_TOKEN=your_long_secret_token
 ./ryusei_V201_17.py --config config.json --force_archive

 to get the correct red_gain and blue_gain:
 python3 -c "from picamera2 import Picamera2; import time; p=Picamera2(); p.configure(p.create_still_configuration()); p.start(); time.sleep(4); p.capture_array(); m=p.capture_metadata(); g=m.get('ColourGains'); print(f'\n--- RECOMMENDED GAINS ---\nRed: {g[0]:.2f}\nBlue: {g[1]:.2f}\n-------------------------') if g else print('\nMetadata empty. Try running again.'); p.stop()"
"""

import argparse
import json
import logging
import os
import sys

# HARDWARE POWER-DRAW THROTTLE
if "--half_power" in sys.argv:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
import csv
import faulthandler
import gc
import glob
import hmac
import html
import io
import math
import queue
import random
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import cv2
try:
    import gpiod
except ImportError:
    gpiod = None
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import ntplib
import numpy as np
import math
np.math = math   # FIX: NumPy 2.0 Monkey Patch for Tetra3 compatibility
import paramiko
import psutil
import requests
import zmq  # for the debug viewer
import tetra3
import py_compile
import glob

try:
    import termios
    import tty
    import select
except ImportError:
    print("Keyboard listener disabled (termios not available on this platform).")

from collections import deque
from datetime import datetime
from datetime import time as dtime
from enum import Enum, auto
from zipfile import ZipFile
from typing import List, Optional, Tuple
from pydantic import (BaseModel, Field, ValidationError, confloat, conint, constr, field_validator, model_validator)
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from config_models import (
    DEFAULTS, SKY_SCORE_VALUES, JANITOR_CLEANUP_PRIORITY, JANITOR_PROTECTED_FILES, STORAGE_SHED_DIRS, EVENT_LOG_FIELDS,
    RESOLUTION_TUNING_PROFILES,
    CaptureConfig, DetectionConfig, SolveReadinessConfig,
    AstrometryConfig, TimelapseConfig, TimelapseVideoConfig, EventsConfig,
    MonitorConfig, PowerMonitorConfig, LedStatusConfig, DaylightConfig,
    JanitorConfig, DashboardConfig, HeartbeatConfig, EventLogConfig,
    HealthMonitorConfig, SftpConfig, NtpConfig, AIClassificationConfig,
    AIObservatoryConfig, GeneralConfig, MainConfig,
)

from enums import SkyConditionStatus, SKY_STATE_MACHINE, DEBUG_LEVELS
from astrometry import AstrometryClass
from concurrency import ConcurrencyClass
from sky_monitor import SkyMonitorClass
from stacking import StackingClass
from event import EventClass
from timelapse import TimelapseClass
from janitor import JanitorClass
from thermal import ThermalClass
from system_services import SystemServicesClass
from dashboard import DashboardClass
from file_utils import FileUtilsClass
from detection import DetectionClass
from capture import CaptureClass
from camera_capture import CameraCaptureClass
from simulator import FrameSimulatorClass
from sftp import SFTPUploaderClass
from thermal_dark import ThermalDarkManagerClass
from neural import NeuralObserverClass
from flat import DynamicFlatEngineClass
from event_footprints import EventFootprintRegistry
from storage_governor import StorageGovernor
from session_archive import archive_name
from satellites import SatelliteCatalog
from locations import LocationStore

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    tflite = None

def init_gpio():
    if gpiod is None:
        raise ImportError("gpiod library is not installed")
    for chip_path in ["/dev/gpiochip4", "/dev/gpiochip0"]:
        if os.path.exists(chip_path):
            return gpiod.Chip(chip_path), chip_path
    raise FileNotFoundError("No known Raspberry Pi GPIO chip found")


try:
    GPIO_CHIP, chip_path = init_gpio()
    IS_GPIO_AVAILABLE = True
    logging.info(f"GPIO enabled on {chip_path}")
except (ImportError, FileNotFoundError, OSError) as e:
    IS_GPIO_AVAILABLE = False
    GPIO_CHIP = None
    logging.warning(
        f"GPIO support is DISABLED (Reason: {e}). Power monitor and LED will not run."
    )

# --------------------------- Logging ---------------------------

class ThreadColorLogFormatter(logging.Formatter):
    """
    A custom logging formatter that assigns a unique color to each thread
    """

    RESET = "\033[0m"

    # A palette of high-contrast colors for thread names
    THREAD_COLORS = [
        # Standard
        "\033[36m",  # Cyan
        "\033[33m",  # Yellow
        "\033[35m",  # Magenta
        "\033[32m",  # Green
        "\033[34m",  # Blue
        "\033[31m",  # Red
        # Bright
        "\033[96m",  # Bright Cyan
        "\033[93m",  # Bright Yellow
        "\033[95m",  # Bright Magenta
        "\033[92m",  # Bright Green
        "\033[94m",  # Bright Blue
        "\033[91m",  # Bright Red
        # Bold
        "\033[36;1m",  # Bold Cyan
        "\033[33;1m",  # Bold Yellow
        "\033[35;1m",  # Bold Magenta
        "\033[32;1m",  # Bold Green
        "\033[34;1m",  # Bold Blue
    ]

    LEVEL_COLORS = {
        logging.DEBUG: "\033[90m",  # Grey
        logging.INFO: "\033[37m",  # White
        logging.WARNING: "\033[93;1m",  # Bright Yellow Bold
        logging.ERROR: "\033[91;1m",  # Bright Red Bold
        logging.CRITICAL: "\033[97;41m",  # White on Red Background
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_color_map = {}
        self.lock = threading.Lock()

    def _get_thread_color(self, thread_name):
        # Assigns a persistent color to a thread name if not already assigned.
        with self.lock:
            if thread_name not in self.thread_color_map:
                new_color = self.THREAD_COLORS[
                    len(self.thread_color_map) % len(self.THREAD_COLORS)
                ]
                self.thread_color_map[thread_name] = new_color
            return self.thread_color_map[thread_name]

    def format(self, record):
        # Formats the log record with level and thread colors.
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        thread_color = self._get_thread_color(record.threadName)

        # Create the formatted string with embedded ANSI color codes
        log_line = (
            f"\033[90m{self.formatTime(record, self.datefmt)}{self.RESET} "
            f"{level_color}[{record.levelname}]{self.RESET} "
            f"{thread_color}[{record.threadName}]{self.RESET} "
            f"{record.getMessage()}"
        )

        return log_line

class Pipeline(AstrometryClass, ConcurrencyClass, SkyMonitorClass, StackingClass, EventClass, TimelapseClass, JanitorClass, ThermalClass, SystemServicesClass, DashboardClass, FileUtilsClass, DetectionClass, CaptureClass):

    def __init__(self, cfg, config_path=None, force_archive=False, half_power=False, simulate=False):

        self.version = "201_21"
        
        self.cfg = cfg

        self.config_path = config_path or "config.json"  # config path

        # DS18B20 Temperature Sensor Initialization
        self._init_thermal_sensors()

        # Get the base path from the janitor config. Default to the script's directory.
        base_path = self.cfg["janitor"].get("monitor_path")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set the paths and files.
        self.events_out_dir = os.path.join(base_path, "output/events")
        self.timelapse_out_dir = os.path.join(base_path, "output/timelapse")
        self.daylight_out_dir = os.path.join(base_path, "output/daylight")
        self.calibration_out_dir = os.path.join(base_path, "output/calibration")
        self.rejected_out_dir = os.path.join(base_path, "output/rejected")
        self.debug_out_dir = os.path.join(base_path, "output/debug")
        self.general_log_dir = os.path.join(base_path, self.cfg["general"]["log_dir"])
        self.monitor_out_file = os.path.join(self.general_log_dir, "system_monitor.csv")
        self.event_log_out_file = os.path.join(self.general_log_dir, "events.csv")
        self.logs_archive = os.path.join(self.general_log_dir, "archive")
        self.crash_file = os.path.join(script_dir, "crash_count.txt")
        self.mip_path = os.path.join(self.general_log_dir, "masterpiece_live.png")
        self.startrail_path = os.path.join(self.general_log_dir, "startrail.png")
        self.health_log_out_file = os.path.join(self.general_log_dir, "health_stats.csv")
        self.backlog_file_path = os.path.join(self.general_log_dir, "sftp_backlog.txt")
        self.camera_k_file = os.path.join(self.general_log_dir, "camera_k.csv")
        self.ai_filtered_out_dir = os.path.join(base_path, "output/ai_filtered")

        self.assets_dir = os.path.join(script_dir, "assets")
        self.flat_field_dir = os.path.join(self.assets_dir, "flat_field")
        self.dynamic_flat_path = os.path.join(self.flat_field_dir, "master_dynamic_flat.png")

        dark_dir_path = os.path.join(self.assets_dir, "darks")
        self.dark_manager = ThermalDarkManagerClass(
            dark_dir=dark_dir_path, ref_exp_us=self.cfg["detection"]["dark_reference_exposure_us"]
        )

        # Guarantee the log directory exists so rotation doesn't fail
        os.makedirs(self.general_log_dir, exist_ok=True)
        
        self._validate_output_directories()
        
        if force_archive:
            self._force_archive_all_logs()

        self._setup_logging()
        self._enable_fault_forensics()

        self._enforce_exposure_safety_limits()

        self.half_power = half_power
        # Apply OpenCV thread limit if half-power is active
        if self.half_power:
            # "Half power" = half the cores, derived from the machine rather than assuming 4.
            cv2.setNumThreads(max(1, (os.cpu_count() or 2) // 2))
            logging.info("HALF-POWER MODE ACTIVE: CPU threads limited.")

        # self._validate_output_directories()

        self._apply_resolution_aware_tuning()

        # Global stop signal for the entire application
        self.running = threading.Event()
        self.running.set()

        # Stop signal for the capture workers
        self.capture_running = threading.Event()
        self.capture_running.set()

        # Lists to manage different types of threads
        self.worker_threads = []
        self.control_threads = []

        self.acq_q = queue.Queue(maxsize=self.effective_max_q)  # Queue for the acquisition frame
        self.event_q = queue.Queue(maxsize=self.effective_max_event_q)  # Queue for the event frame
        self.timelapse_q = queue.Queue(maxsize=self.effective_max_q)  # Queue for timelapse frame
        self.event_stack_q = queue.Queue(maxsize=self.effective_max_event_q)  # Queue for event stacks
        rec_q = self.cfg["general"]["record_queue_maxsize"]
        self.event_log_q = queue.Queue(maxsize=rec_q)  # Queue for the event logger
        self.health_q = queue.Queue(maxsize=rec_q)  # Queue for health statistics
        # Holds full frames: sized to the largest stack a test shot can request (the
        # nighttime stack), not a fixed 100 frames (~280 MB at 1280x720 if it ever filled).
        self.calibration_q = queue.Queue(maxsize=self.cfg["timelapse"]["nighttime_stack_N"])
        self.sftp_dispatch_q = queue.Queue(maxsize=rec_q)  # Queue for SFTP dispatcher

        self.last_calibration_image_path = None
        self.is_calibrating = threading.Event()
        self.is_calibrating.clear()
        self.event_counter = 0

        self.memory_critical = threading.Event()  # RAM Protection Flag
        self.memory_critical.clear()

        if simulate:
            # Use the fake camera
            self.camera = FrameSimulatorClass(self.cfg["capture"])
        else:
            # Use the real camera
            self.camera = CameraCaptureClass(self.cfg["capture"])
        self._reconcile_capture_limits_with_sensor()

        self.background = None
        self.camera_fatal_error = threading.Event()  # Set by capture_loop if fatal
        self.consecutive_camera_failures = 0
        self.consecutive_restart_failures = 0
        self.config_reloaded_ack = threading.Event()  # Triggers for config reload

        # --- Initialize the SFTP ---
        self.sftp_uploader = None
        if self.cfg["sftp"].get("enabled"):
            try:
                self.sftp_uploader = SFTPUploaderClass(self.cfg["sftp"], self)
            except ImportError:
                logging.error("SFTP is enabled, but 'paramiko' library is not installed. Please run 'pip3 install paramiko'.")
            except Exception as e:
                logging.error(f"Failed to initialize SFTP uploader: {e}")

        # Initialize status LED
        self.led_line = None
        if IS_GPIO_AVAILABLE and self.cfg["led_status"].get("enabled"):
            try:
                self.led_pin = self.cfg["led_status"]["pin"]
                # Create a LineSettings object to define the line's configuration.
                settings = gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT, output_value=gpiod.line.Value.ACTIVE)  # Set initial value to ON (booting)
                # Request the line with the specified settings.
                self.led_line = GPIO_CHIP.request_lines(consumer="ryusei-pipeline-led", config={self.led_pin: settings})
                logging.info(f"Status LED enabled on GPIO pin {self.led_pin}. LED is ON for startup sequence.")

            except Exception as e:
                logging.error(f"Failed to initialize status LED on pin {self.led_pin}: {e}", exc_info=True)
                if self.led_line:
                    self.led_line.release()  # Ensure release on failure
                self.led_line = None
        else:
            logging.info("Status LED is disabled.")

        self._load_calibration_frames()

        ctx = zmq.Context()
        self.debug_pub = ctx.socket(zmq.PUB)
        try:
            self.debug_pub.bind("tcp://*:5555")
        except zmq.ZMQError:
            logging.info("ZMQ port 5555 already in use. Debug publisher disabled.")

        self.last_night_journal = ("Initial sequence started. The first chronicle will be written at sunrise.")
        self.session_stats = self._reset_session_stats()
        self.night_chronicle = self._load_chronicle()

        self.current_sensor_temp_cam = None
        self.current_sensor_temp_ext = None

        if self.ds18b20_cam_path:
            # We only care about the CAMERA sensor for dynamic dark frames!
            pass
        else:
            logging.warning("Hardware: Camera Thermal Sensor NOT found. Dynamic Darks will be disabled.")
            self.use_dynamic_darks = False

        self.sky_score = 0.0
        self.daylight_stack_n = 1  # During the daylight the stack_n lowered to avoid burn images. Uncontrolled. NOT CHANGE!
        self.effective_stack_n = self.cfg["timelapse"]["nighttime_stack_N"]

        self.sky_monitor_in_progress = threading.Event()
        self.sky_monitor_in_progress.clear()  # when sky_monitor triggers the scheduler must be set in wait status
        self.bright_timelapse_count = 0

        self.daylight_mode_active = threading.Event()
        self.daylight_mode_active.clear()  # Default to night mode (detection on)
        self.timelapse_interval_sec = self.cfg["timelapse"].get("daymode_interval_sec")
        self.base_timelapse_interval = self.timelapse_interval_sec
        self.timelapse_frames_queued = 0
        self.weather_hold_active = threading.Event()
        self.weather_hold_active.clear()  # Start in good weather status
        self.event_in_progress = threading.Event()
        self.event_in_progress.clear()  # Start in the "no event" state
        self.reload_config_signal = threading.Event()
        self.request_sky_analysis_trigger = threading.Event()
        self.status_lock = threading.Lock()  # A dedicated lock for status variables
        self.thread_heartbeats = {}  # name -> (Thread, time.monotonic() of last ping); see ping_watchdog
        # Detector -> timelapse channel for sigma-clip protection of real transients.
        self.event_footprints = EventFootprintRegistry()
        # Graded write admission for the data drive (see storage_governor.py).
        self.storage = StorageGovernor(
            base_path,
            lambda: self.cfg["janitor"],
            [getattr(self, a) for a in STORAGE_SHED_DIRS],
            on_transition=self._on_storage_transition,
        )
        # Observing sites database (assets/locations.json, see locations.py). The site in
        # use stays in config.json; make sure it is always among the selectable ones.
        self.location_store = LocationStore(os.path.join(self.assets_dir, "locations.json"))
        try:
            added = self.location_store.ensure_contains(self.cfg["general"])
            if added:
                logging.info(f"Locations: current site '{added['location']}' added to the locations database.")
        except (OSError, ValueError) as e:
            logging.error(f"Locations: database unavailable ({e}); the configured site is still used.")

        # Local satellite catalogue + matcher (satellites.py), kept with the other
        # reference data in assets/satellites. Refreshed in the background
        # (_maybe_refresh_satellites).
        sat_dir = os.path.join(self.assets_dir, "satellites")
        self._migrate_satellite_cache(os.path.join(base_path, "satellites"), sat_dir)
        self.satellite_catalog = SatelliteCatalog(
            sat_dir,
            lambda: self.cfg["satellites"],
            requests.get,
            # Identify the client, as CelesTrak asks of automated software.
            user_agent=f"Ryusei/{self.version} (all-sky meteor station; GP CSV, 1 download/group/day)",
            # No storage-governor check: the governor watches the DATA drive, and assets/
            # sits with the program. The catalogue is a few MB, replaced in place daily.
            admits=None,
            on_failure=lambda group, rec: self.log_health_event(
                "WARNING", "SATELLITE_DOWNLOAD_FAILED",
                f"GROUP={group} {rec['status']}: {rec['message'][:200]} (failure {rec['failures']} in a row)",
            ),
        )
        # Tracks abandoned _run_bounded() worker threads: name -> (thread_obj, registered_at).
        # watchdog_loop checks these for genuine zombies (still alive well past
        # their own timeout) and cleans up entries for ones that finished late
        # on their own.
        self.fire_and_forget_threads = {}
        self.fire_and_forget_lock = threading.Lock()
        self.astrometry_lock = threading.Lock()
        self.wcs_cache = None
        self.last_illuminance = "N/A"
        self.last_sky_conditions = {"stddev": "N/A", "stars": "N/A"}
        self.last_moon_status = {
            "impact": 0.0,
            "in_fov": False,
            "alt": 0.0,
            "dist": 180.0,
        }
        self.power_status = "DISABLED"  # Can be OK, LOST, SHUTTING_DOWN, or DISABLED
        self.last_heartbeat_status = "N/A"
        self.noise_baseline_stddev = None
        self.last_effective_threshold = "N/A"
        self.effective_threshold = None
        self.threshold_min = self.cfg["detection"].get("min_threshold")
        self.threshold_max = self.cfg["detection"].get("max_threshold")
#        self.last_moon_info = None
        self.last_calibration_error = None
        self.last_event_files = {"image": None, "video": None}
        self.timelapse_next_capture_time = float("-inf")  # monotonic deadline for the next timelapse frame
        self.daylight_next_capture_time = float("-inf")  # monotonic deadline for the next daylight frame
        self.capture_stable = threading.Event()
        self.session_start_time = None
        self.camera_lock = threading.Lock()
        self.current_exposure_us = self.cfg["capture"].get("exposure_us")
        self.current_gain = self.cfg["capture"].get("gain")
        self.in_maintenance_mode = threading.Event()
        self.in_maintenance_mode.clear()  # Start in normal operating mode
        self.watchdog_pause = threading.Event()
        self.emergency_stop_active = threading.Event()
        self.emergency_stop_active.clear()  # Set in the "not stopped" state
        self.initial_sftp_sweep_complete = threading.Event()
        self.initial_sftp_sweep_complete.clear()  # Start in the "not complete" state
        self.initial_janitor_check_complete = threading.Event()
        self.timelapse_complete_event = threading.Event()
        self.timelapse_complete_event.set()  # Set initially to allow first run
        self.timelapse_flush_done = threading.Event()  # set once a FLUSHed partial stack is on disk
        self.timelapse_flush_done.set()
        self.initial_janitor_check_complete.clear()  # Start in the "not complete" state
        self.system_ready = threading.Event()  # Master "All Systems Go" signal
        self.maintenance_timeout_lock = threading.Lock()
        self.maintenance_timeout_until = (0)   # time.monotonic() deadline when the timeout expires (0 = no timeout active)
        self.maintenance_timeout_duration_sec = self.cfg["general"].get("maintenance_timeout")
        self.last_sky_status = None
        self.last_daylight_mode = None
        self.background_reset_time = float("-inf")  # monotonic; -inf = never reset (no cooldown at startup)
        self.calibration_k = 0.0
        self.auto_exposure_converged = False
        self.ae_reference = None  # {"product": exposure*gain, "target": brightness it was aimed at, "t": epoch} of the last trusted auto-exposure (see _ae_make_reference)
        self.ae_candidate = None  # this check's result, pending the sky status (see _ae_finalize_reference)
        self.ae_last_run_clamped = False
        self.ae_recheck_due_at = None  # time.monotonic() when the lightweight exposure re-check is due (None = none pending)
        self.last_raw_frame_info = None  # last captured frame + the exposure/gain it was really taken with
        self.use_ram_for_check = True
        self.last_light_check_source = "acquired"
        self.snapshot_frame = None
        self.last_reject_save_time = float("-inf")  # monotonic; -inf = never saved
        # A list of editable checkbox in the dashboard
        self.editable_booleans = [
            "auto_exposure_tuning",
            "auto_sensitivity_tuning",
            "stack_align",
            "auto_dark_enabled",
            "auto_flat_enabled",
        ]

        self.last_dark_update_time = float("-inf")  # monotonic; -inf so the first frame always gets a dark
        self.ai = NeuralObserverClass(self.cfg["ai_classification"], self.cfg["ai_observatory"])
        self.flat_engine = DynamicFlatEngineClass(self)
        if tflite is None:
            self.cfg["ai_classification"]["enabled"] = False
            logging.warning(
                "Tensorflow lite NOT found. The AI capability will be disabled."
            )
        self.disk_used_mb = 0
        self.disk_percent = 0
        self._initial_cleanup()

    # -----------------
    def _setup_logging(self):
        """Configures the root logger, ensuring all timestamps are in local time."""
        log_dir = self.general_log_dir
        max_mb = self.cfg.get("janitor", {}).get("log_rotation_mb")
#        backup_count = self.cfg.get("janitor", {}).get("log_backup_count")

        # os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "pipeline.log")

        # Formatter.converter defaults to time.localtime already; left unset
        # (rather than explicitly assigned) so it stays local time reference.
        local_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
        )

        color_local_formatter = ThreadColorLogFormatter(
            "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
        )

        file_handler = logging.FileHandler(log_path)

        # logging.FileHandler(log_path)

        file_handler.setFormatter(local_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_local_formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # set the log to info level
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.info("=" * 60)
        logging.info(
            f"--- Ryusei Pipeline v{self.version} Starting ---"
        )
        logging.info("=" * 60)
        logging.info("Logging started. All timestamps are in local time.")

    # -----------------
    def _initial_cleanup(self):
        # Scans and removes orphaned temporary directories.
        logging.info("Performing initial cleanup check...")
        cleanup_count = 0
        if os.path.isdir(self.events_out_dir):
            try:
                with os.scandir(self.events_out_dir) as it:
                    for entry in it:
                        # Temporary directories are our specific target
                        if entry.is_dir() and entry.name.endswith("_frames"):
                            logging.warning(
                                "Found orphaned temp directory from previous run: %s. Deleting.",
                                entry.name,
                            )
                            try:
                                shutil.rmtree(entry.path)
                                cleanup_count += 1
                            except Exception:
                                logging.exception(
                                    "Error deleting orphaned directory %s", entry.path
                                )
            except Exception:
                logging.exception("Error during initial cleanup scan.")
        if cleanup_count > 0:
            logging.info(
                "Initial cleanup removed %d orphaned directories.", cleanup_count
            )

        # Clean up legacy nested debug directories to free disk space
        for legacy_dir in ["debug_dumps", "debug_detection"]:
            legacy_path = os.path.join(self.general_log_dir, legacy_dir)
            if os.path.isdir(legacy_path):
                try:
                    shutil.rmtree(legacy_path)
                    logging.info(
                        "Initial cleanup removed legacy debug directory: %s", legacy_dir
                    )
                except Exception:
                    pass

    # -----------------
    def _reset_session_stats(self):
        return {
            "start_time": datetime.now(),
            "last_check_time": time.monotonic(),
            "total_events": 0,
            "rejected_contours": 0,
            "accepted_events": 0,
            "clear_sky_minutes": 0,
            "cloudy_sky_minutes": 0,
            "error_mins": 0,
            "avg_sky_score": [],
            "max_moon_impact": 0.0,
            "moon_visible_minutes": 0,
            "data_written_mb": 0.0,
            "peak_cpu_temp": 0.0,
            "avg_lux": [],
            "sky_history": [],
            "star_count_history": [],
            "stddev_history": [],
            "moon_penalty_history": [],
            "bias_history": [],
            "obs_condition": "N/A (Disabled)",  # Default value
            "obs_confidence": 0.0,  # Default value
            "ai_interceptions": 0,  # Good to reset this too
            "loop_time_ms": 0.0,
            "recent_events": []
        }

    # -----------------
    def _load_chronicle(self):
        path = os.path.join(self.general_log_dir, "night_chronicle.json")
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                logging.info("Night Chronicle loaded: %d nights on record.", len(data))
                return data
        except Exception as e:
            logging.warning("Could not load night chronicle: %s", e)
        return []

    # -----------------
    def _update_chronicle(self, hours, sky_avg, clarity, sky_history):
        record = {
            "date": self.session_stats["start_time"].strftime("%Y-%m-%d"),
            "hours_watched": round(hours, 2),
            "events_accepted": self.session_stats["accepted_events"],
            "clarity_pct": round(clarity, 1),
            "sky_score_avg": round(sky_avg, 3),
            "peak_temp_c": round(self.session_stats["peak_cpu_temp"], 1),
            "max_moon_impact": round(self.session_stats["max_moon_impact"], 3),
            "data_written_mb": round(self.session_stats["data_written_mb"], 1),
            "sky_trend": self._generate_atmosphere_fingerprint(sky_history).get(
                "trend"
            ),
        }
        # Avoid duplicate dates
        self.night_chronicle = [
            n for n in self.night_chronicle if n["date"] != record["date"]
        ]
        self.night_chronicle.append(record)
        # Keep last 365 nights
        self.night_chronicle = self.night_chronicle[-365:]
        path = os.path.join(self.general_log_dir, "night_chronicle.json")
        self.save_json_atomic(path, self.night_chronicle)
        logging.info(
            "Night Chronicle updated. Total nights on record: %d",
            len(self.night_chronicle),
        )

    # -----------------
    def _get_chronicle_context(self):
        """Returns percentile ranks and historical comparisons for tonight."""
        c = self.night_chronicle
        if len(c) < 2:
            return {}
        all_events = [n["events_accepted"] for n in c]
        all_clarity = [n["clarity_pct"] for n in c]
        all_scores = [n["sky_score_avg"] for n in c]
        tonight_events = self.session_stats["accepted_events"]
        tonight_clarity = (
            self.session_stats["clear_sky_minutes"]
            / max(
                1,
                self.session_stats["clear_sky_minutes"]
                + self.session_stats["cloudy_sky_minutes"],
            )
            * 100
        )
        tonight_score = (
            float(np.mean(self.session_stats["avg_sky_score"]))
            if self.session_stats["avg_sky_score"]
            else 0.0
        )

        def percentile_rank(val, lst):
            return round(sum(1 for x in lst if x <= val) / len(lst) * 100)

        return {
            "nights_total": len(c),
            "events_avg": round(np.mean(all_events), 1),
            "events_tonight": tonight_events,
            "events_pct": percentile_rank(tonight_events, all_events),
            "events_record": max(all_events),
            "clarity_avg": round(np.mean(all_clarity), 1),
            "clarity_tonight": round(tonight_clarity, 1),
            "clarity_pct": percentile_rank(tonight_clarity, all_clarity),
            "score_avg": round(np.mean(all_scores), 3),
            "score_tonight": round(tonight_score, 3),
            "score_pct": percentile_rank(tonight_score, all_scores),
        }

    # -----------------
    def _generate_atmosphere_fingerprint(self, sky_history):
        """Analyses the intra-night sky quality trend from sky_history snapshots."""
        if len(sky_history) < 2:
            return {
                "trend": "stable",
                "summary": "Insufficient sky checks for trend analysis.",
                "sparkline": [],
            }
        scores = [h["score"] for h in sky_history]
        times = [h["t"] for h in sky_history]
        # Linear slope via least-squares
        n = len(scores)
        t0 = times[0]
        xs = [(t - t0) / 3600.0 for t in times]  # hours from session start
        mx = sum(xs) / n
        my = sum(scores) / n
        num = sum((xs[i] - mx) * (scores[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
        # Classify
        if slope > 0.08:
            trend = "improving"
            verb = "steadily improved"
        elif slope < -0.08:
            trend = "degrading"
            verb = "gradually degraded"
        else:
            trend = "stable"
            verb = "remained stable"
        best = max(sky_history, key=lambda h: h["score"])
        worst = min(sky_history, key=lambda h: h["score"])
        best_t = datetime.fromtimestamp(best["t"]).strftime("%H:%M")
        worst_t = datetime.fromtimestamp(worst["t"]).strftime("%H:%M")
        summary = (
            f"Atmospheric quality {verb} through the session "
            f"(slope {slope:+.2f}/hr). "
            f"Peak transparency: score {best['score']:.2f} at {best_t} "
            f"({best['stars']} stars, stddev {best['stddev']:.1f}). "
            f"Worst moment: score {worst['score']:.2f} at {worst_t}."
        )
        return {
            "trend": trend,
            "slope": round(slope, 3),
            "summary": summary,
            "sparkline": [round(s, 2) for s in scores],
            "best": best,
            "worst": worst,
        }

    # -----------------
    def _apply_resolution_aware_tuning(self):
        """Automatically optimizes logic variables based on the total pixel count."""
        w = self.cfg["capture"]["width"]
        h = self.cfg["capture"]["height"]
        mp = (w * h) / 1_000_000.0  # Megapixels

        # Find the profile that matches the resolution closest
        available_profiles = sorted(RESOLUTION_TUNING_PROFILES.keys())
        # Default to the lowest profile
        best_match_mp = available_profiles[0]
        for p_mp in available_profiles:
            if mp >= p_mp:
                best_match_mp = p_mp

        profile = RESOLUTION_TUNING_PROFILES[best_match_mp]

        logging.warning(f"Applying '{best_match_mp}MP' Tuning Profile.")

        self.description = profile["description"]
        self.min_area = profile["min_area"]
        self.blur_size = profile["blur"]
        self.morph_size = profile["morph"]
        self.min_changes_required = profile["min_changes"]
        self.star_mask_radius = profile["radius"]
        self.alignment_contrast_floor = profile["std_floor"]
        self.effective_max_q = profile["max_q"]
        self.effective_max_event_frames = profile["max_event_frames"]
        self.effective_max_event_q = profile["max_event_q"]
        self.min_mask_px = profile["min_mask_px"]
        self.ecc_gauss_size = profile["ecc_gauss"]
        self.max_warp_factor = profile["warp_f"]
        self.top_hat_factor = profile["th_f"]
        self.pre_event_frames = profile["pre_f"]
        self.event_cooldown_frames = profile["cool_f"]

        logging.info(f"Tuning profile select: {self.description}")
        logging.info(
            f"Parameters set: Area>{self.min_area}, Motion>{self.min_changes_required}, MaskRadius={self.star_mask_radius}"
        )

    # -----------------
    def run(self):
        """
        The main entry point to start and run the entire pipeline application.
        """
        shutdown_event = threading.Event()

        def handle_signal(sig, frame):
            if not shutdown_event.is_set():
                logging.warning(
                    "Shutdown signal received! Initiating graceful shutdown..."
                )
                self.stop()
                shutdown_event.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        self.log_health_event("WARNING", "START", "The system is starting.")

        self.start()
        time.sleep(0.5)

        logging.info("Verifying thread health before engaging system...")
        all_threads = self.worker_threads + self.control_threads
        startup_failed = False

        for t in all_threads:
            if not t.is_alive():
                logging.critical(
                    f"STARTUP ERROR: Thread '{t.name}' failed to start or died immediately."
                )
                startup_failed = True

        if startup_failed:
            logging.critical(
                "Startup Aborted: One or more critical threads failed to start. Exiting."
            )
            self.stop()
            sys.exit(1)

        logging.info(
            f"Health Check Passed: All {len(all_threads)} threads are running."
        )

        # self._led_startup_signal()

        logging.info("All threads launched. Setting 'System Ready' signal.")
        self.system_ready.set()

        try:
            logging.info("Running pipeline indefinitely (Ctrl+C to stop).")
            shutdown_event.wait()
        finally:
            if not shutdown_event.is_set():
                logging.info("Main loop finished, initiating shutdown...")
                self.stop()

        logging.info("Pipeline has been stopped. Main thread exiting.")
        # stop() has already joined every worker/control thread and handlers flush on
        # every emit, so nothing remains to wait for (a fixed 2 s sleep was removed).
        self._execute_post_shutdown_script()

    # -----------------
    def start(self):
        logging.info("Starting all pipeline services...")

        if self.cfg.get("detection", {}).get("auto_sensitivity_tuning"):
            gen_cfg = self.cfg.get("general")
            logging.info("Auto-Sensitivity Tuning is ENABLED. Lunar awareness active.")
            logging.info("Site location: %s", gen_cfg.get("location") or "Unknown")
            logging.info(
                "  -> Coordinates : Lat %.4f, Lon %.4f",
                gen_cfg.get("latitude"),
                gen_cfg.get("longitude"),
            )
            logging.info(
                "  -> Camera Point: Azimuth %.1f°, Altitude %.1f°",
                gen_cfg.get("camera_azimuth"),
                gen_cfg.get("camera_altitude"),
            )
        else:
            logging.info(
                "Auto-Sensitivity Tuning is DISABLED. Using static thresholds."
            )

        # power_pin = self.cfg["general"].get("power_monitor_pin")

        self.worker_threads = [
            threading.Thread(target=self.event_writer_loop, name="EventWriterThread", daemon=True),
            threading.Thread(target=self.timelapse_loop, name="TimelapseThread", daemon=True),
            threading.Thread(target=self.event_stacker_loop, name="EventStackerThread", daemon=True),
            threading.Thread(target=self.calibration_worker_loop, name="CalibrationThread", daemon=True),
        ]

        if self.cfg["event_log"].get("enabled"):
            self.worker_threads.append(threading.Thread(target=self.event_logger_loop, name="EventLoggerThread", daemon=True))
        else:
            logging.info("Event log is disabled in configuration.")

        for t in self.worker_threads:
            t.start()
        logging.info("All worker threads started.")

        # 1. Add mandatory control threads
        self.control_threads = [
            threading.Thread(target=self.watchdog_loop, name="WatchdogThread", daemon=True),
            threading.Thread(target=self.system_monitor_loop, name="MonitorThread", daemon=True),
            threading.Thread(target=self.janitor_loop, name="JanitorThread", daemon=True),
            threading.Thread(target=self.scheduler_loop, name="SchedulerThread", daemon=True),
            threading.Thread(target=self.health_monitor_loop, name="HealthMonitorThread", daemon=True),
            threading.Thread(target=self.config_reloader_loop, name="ConfigReloaderThread", daemon=True),
            threading.Thread(target=self.keyboard_listener_loop, name="KeyboardListenerThread", daemon=True),
        ]

        # if self.ds18b20_path:
        if self.ds18b20_cam_path or self.ds18b20_ext_path:
            self.control_threads.append(threading.Thread(target=self.thermal_sensor_loop, name="ThermalSensorThread", daemon=True))

        # 2. Add optional control threads based on configuration
        if self.cfg["power_monitor"].get("enabled") and IS_GPIO_AVAILABLE:
            self.control_threads.append(threading.Thread(target=self.power_monitor_loop, name="PowerMonitorThread", daemon=True))
        else:
            if not IS_GPIO_AVAILABLE and self.cfg["power_monitor"].get("enabled"):
                logging.warning("Power monitor is enabled but GPIO is not available. Thread will not be started.")
            else:
                logging.info("Power monitor is disabled in configuration.")
            self.power_status = "DISABLED"

        if self.cfg["heartbeat"].get("enabled") and self.cfg["heartbeat"].get("url"):
            self.control_threads.append(threading.Thread(target=self.heartbeat_loop, name="HeartbeatThread", daemon=True))
        else:
            if self.cfg["heartbeat"].get("enabled") and not self.cfg["heartbeat"].get("url"):
                logging.warning("Heartbeat is enabled but no URL is configured. Thread will not be started.")
            else:
                logging.info("Heartbeat is disabled in configuration.")

        if self.cfg["ntp"].get("enabled"):
            self.control_threads.append(threading.Thread(target=self.ntp_sync_loop, name="NTPThread", daemon=True))
        else:
            logging.info("NTP is disabled in configuration.")

        if self.cfg["daylight"].get("enabled"):
            self.control_threads.append(threading.Thread(target=self.sky_monitor_loop, name="SkyMonitorThread", daemon=True))
        else:
            logging.info("Sky monitor is disabled in configuration.")

        if self.cfg["dashboard"].get("enabled"):
            self.control_threads.append(threading.Thread(target=self.dashboard_loop, name="DashboardThread", daemon=True))
            # The maintenance watchdog is tied to the dashboard's pause feature
            self.control_threads.append(threading.Thread(target=self.maintenance_watchdog_loop, name="MaintWatchdogThread", daemon=True))
        else:
            logging.info("Dashboard (and Maintenance Watchdog) is disabled in configuration.")

        if self.led_line:
            self.control_threads.append(threading.Thread(target=self.led_status_loop, name="LedStatusThread", daemon=True))
        else:
            logging.info("Status LED disabled in configuration.")

        for t in self.control_threads:
            t.start()
        logging.info("All control threads started.")

        # 3. Add SFTP threads if enabled
        if self.sftp_uploader:
            sftp_thread = threading.Thread(target=self.sftp_uploader.worker_loop, name="SftpThread", daemon=True)
            sftp_thread.start()
            self.control_threads.append(sftp_thread)  # Add to control threads for clean shutdown
            sftp_dispatcher_thread = threading.Thread(target=self.sftp_dispatcher_loop, name="SFTPDispatchThread", daemon=True)
            sftp_dispatcher_thread.start()
            self.control_threads.append(sftp_dispatcher_thread)
            logging.info("SFTP Uploader thread started.")

            # --- Perform the initial sweep AFTER threads are running but BEFORE capture starts ---
            try:
                logging.info("Performing mandatory startup sweep for un-uploaded files...")
                self.sweep_and_enqueue_unuploaded()
                logging.info("Startup SFTP sweep complete.")
            except Exception as e:
                logging.exception(f"An error occurred during startup SFTP sweep: {e}")
            finally:
                # CRITICAL: Always set the event, even if the sweep fails, to prevent a deadlock.
                self.initial_sftp_sweep_complete.set()
        else:
            # If SFTP is not enabled, we don't need to sweep, so we can proceed immediately.
            self.initial_sftp_sweep_complete.set()

    # -----------------
    def stop(self):
        """
        Performs a robust, multi-stage shutdown of all pipeline services.
        """
        if not self.running.is_set():
            logging.warning("Shutdown already in progress.")
            return

        logging.info("=" * 60)
        logging.info("--- INITIATING GRACEFUL SHUTDOWN ---")
        logging.info("=" * 60)

        logging.info("Writing journal.")
        with self.status_lock:
            self.last_night_journal = self.generate_morning_journal()

        # PAUSE WATCHDOG DURING SHUTDOWN
        logging.warning("Pausing watchdog for final shutdown procedure.")
        self.watchdog_pause.set()

        # Stop all new data production
        logging.info("Setting global stop signal for all threads.")
        self.running.clear()  # Global stop signal
        self.storage.interrupt()  # janitor and scheduler sleep on governor events; let them see running is clear

        # Specifically stop the data production chain first
        self.stop_producer_thread()  # This now handles both Capture and Detection

        # Signal all consumer and control threads to terminate
        logging.info(
            "Posting final sentinels to all remaining worker and control queues..."
        )
        queues_to_signal = [
            self.timelapse_q,
            self.event_q,
            self.event_stack_q,
            self.event_log_q,
            self.health_q,
            self.calibration_q,
        ]
        if self.sftp_uploader:
            self.sftp_uploader.stop()  # This handles the SFTP queues

        for q in queues_to_signal:
            try:
                q.put_nowait(None)
            except (queue.Full, AttributeError):
                pass

        # Wait for all remaining threads to finish cleanly
        logging.info("Waiting for all threads to complete their final tasks...")
        all_threads = self.worker_threads + self.control_threads
        shutdown_timeout = self.cfg["general"]["thread_shutdown_timeout_sec"]

        for t in all_threads:
            if (
                t == threading.current_thread()
                or t.name == "DashboardThread"
                or not t.is_alive()
            ):
                continue

            t.join(timeout=shutdown_timeout)
            if t.is_alive():
                logging.error(
                    f"Thread '{t.name}' failed to join and is still alive. Shutdown may be incomplete."
                )

        # Release hardware resources
        if self.camera:
            try:
                self.camera.release()
                self.camera = None
            except Exception:
                pass

        if self.led_line:
            try:
                logging.info("Turning off status LED.")
                self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
                self.led_line.release()
            except Exception as e:
                logging.error(f"Error releasing LED GPIO pin: {e}")

        logging.info("=" * 60)
        logging.info("--- PIPELINE SHUTDOWN COMPLETE ---")
        logging.info("=" * 60)

    # -----------------
    # Highest selectable debug level. Cycling wraps 4 -> 0.
    MAX_DEBUG_LEVEL = 4

    def keyboard_listener_loop(self):
        """
        Interactive hotkey listener.

        Press Ctrl+D to cycle the runtime debug verbosity level:
        0 -> 1 -> 2 -> 3 -> 4 -> 0 (each press increments by one, wrapping
        back to 0 after MAX_DEBUG_LEVEL). The new value is written straight
        into self.cfg["general"]["debug_level"], which every debug hook reads
        live, so the change takes effect immediately. It is NOT persisted to
        config.json - a restart returns to the configured level.

        Only runs when the pipeline is attached to an interactive terminal
        (TTY). When launched headless (e.g. as a systemd service) or on a
        platform without termios, the thread logs why and then idles for the
        lifetime of the process (it must stay alive so the startup thread
        health-check does not treat it as a crashed thread).
        """
        def _idle_until_shutdown():
            while self.running.is_set():
                time.sleep(1.0)

        stdin = sys.stdin
        try:
            is_tty = stdin is not None and stdin.isatty()
        except Exception:
            is_tty = False
        if not is_tty:
            logging.info("Keyboard listener disabled (stdin is not an interactive TTY).")
            _idle_until_shutdown()
            return

        # try:
            # import termios
            # import tty
            # import select
        # except ImportError:
            # logging.info("Keyboard listener disabled (termios not available on this platform).")
            # _idle_until_shutdown()
            # return

        fd = stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except termios.error as e:
            logging.info(f"Keyboard listener disabled (cannot access terminal: {e}).")
            _idle_until_shutdown()
            return

        CYCLE_KEY = b"\x04"  # Ctrl+D

        logging.info(
            "Keyboard listener active. Press Ctrl+D to cycle debug level "
            f"(0..{self.MAX_DEBUG_LEVEL}, then back to 0)."
        )

        try:
            # cbreak leaves Ctrl+C (SIGINT) working for graceful shutdown while
            # delivering every other key immediately and unbuffered.
            tty.setcbreak(fd)
            while self.running.is_set():
                # Poll with a timeout so shutdown is noticed promptly.
                ready, _, _ = select.select([fd], [], [], 0.5)
                if not ready:
                    continue
                ch = os.read(fd, 1)
                if ch != CYCLE_KEY:
                    continue
                current = self.cfg["general"].get("debug_level") or 0
                new_level = (current + 1) % (self.MAX_DEBUG_LEVEL + 1)
                self.cfg["general"]["debug_level"] = new_level
                state = "OFF" if new_level == 0 else f"Level {new_level} active"
                logging.warning(f"HOTKEY: Debug level {current} -> {new_level} ({state}).")
                
                # if new_level == 1: logging.info(f"Debug level set to Stats")
                # elif new_level == 2: logging.info(f"Debug level set to Images")
                # elif new_level == 3: logging.info(f"Debug level set to Full/Stream")
                # elif new_level == 4: logging.info(f"Debug level set to Variables dump")
                if new_level in DEBUG_LEVELS:
                    logging.info(f"Debug level set to {DEBUG_LEVELS[new_level]}")
    
        except Exception as e:
            logging.error(f"Keyboard listener thread crashed: {e}", exc_info=True)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
            logging.info("Keyboard listener stopped.")

    # -----------------
    def scheduler_loop(self):
        """
        A dedicated control thread that acts as the master timer. It starts and
        stops the producer thread based on the main schedule.
        """
        max_restart_attempts = self.cfg["general"].get("max_restart_failures")
        logging.info("Master scheduler started. Will check every 60 seconds.")

        # --- Wait for the initial SFTP sweep to complete before doing anything. ---
        logging.info("Scheduler is waiting for initial SFTP sweep to complete...")
        self.initial_sftp_sweep_complete.wait()
        logging.info("Scheduler: SFTP sweep complete. Waiting for initial Janitor check...")
        self.initial_janitor_check_complete.wait()
        logging.info("Scheduler: All startup checks passed. Cleared to start normal operations.")

        logging.info("Scheduler: Waiting for 'System Ready' signal.")
        self.system_ready.wait()
        logging.info("Scheduler: Cleared to start normal operations.")

        try:
            while self.running.is_set():
                # If the schedule trigger when sky_monitoris in prograss, can lead an uncontrolled status
                # Pause scheduler if Sky Monitor is busy doing hardware math
                if self.sky_monitor_in_progress.is_set():
                    logging.info("Scheduler: Sky monitor is in progress, pausing schedule evaluation.")
                    # Sky monitor can legitimately run for minutes (waiting for the
                    # current timelapse stack's safe window, then a multi-attempt
                    # auto-exposure hunt) -- this loop has no bound of its own, so
                    # without a heartbeat here a long-but-legitimate sky monitor
                    # pass gets SchedulerThread flagged as a hung/zombie thread and
                    # the whole process force-restarted. Same class of bug already
                    # fixed for the AE loop and the stacking loop.
                    while (
                        self.sky_monitor_in_progress.is_set() and self.running.is_set()
                    ):
                        self.ping_watchdog()
                        time.sleep(1.0)

                now_str = datetime.now().strftime("%H:%M")
                shutdown_time = self.cfg["general"].get("shutdown_time")

                # Scheduled System Shutdown Check
                # logging.info(f"skap pre shutdown - shutdown_time:{shutdown_time}, now_str:{now_str}")
                if shutdown_time and now_str == shutdown_time:
                    # logging.info(f"skap enter shutdown")
                    logging.warning(f"Scheduler: Reached scheduled shutdown time ({shutdown_time}).")
                    # Wait for the video thread to finish
                    video_thread = next(
                        (
                            t
                            for t in threading.enumerate()
                            if t.name == "TimelapseVideoThread"
                        ),
                        None,
                    )

                    if video_thread and video_thread.is_alive():
                        logging.info("Scheduler: Waiting for End-of-Session video creation to complete before shutdown...")

                        # Bounded by the video thread's OWN budget (the same value its zombie
                        # detection uses): waiting any shorter abandons a legitimate encode,
                        # any longer waits on a thread the watchdog already considers dead.
                        if self.join_with_heartbeat(
                            video_thread, self.cfg["general"]["timelapse_video_thread_timeout_sec"]
                        ):
                            logging.info("Scheduler: Video creation thread has finished.")
                        else:
                            logging.error("Scheduler: Video creation timed out! Forcing shutdown anyway.")

                    # Perform Shutdown
                    logging.info("Scheduler: Proceeding with system shutdown sequence.")
                    # (A former 2 s "log flushing" sleep was removed: FileHandler/StreamHandler
                    # flush on every emit -- see perform_system_action.)
                    self.perform_system_action("shutdown", reason="Scheduled Daily Shutdown")
                    # logging.info(f"skap exit shutdown")
                    break

                # check the status of the schedule (true -> in time) and the producer (true -> acquire)
                is_active_schedule = self.within_schedule()
                is_capturing = self.producer_thread_active()

                if self.storage.halt_event.is_set() and not self.emergency_stop_active.is_set():
                    # Recoverable: unlike EMERGENCY STOP this clears by itself as soon as
                    # the governor sees enough free space again; acquisition then resumes
                    # through the normal schedule logic below on a later pass.
                    if is_capturing:
                        self.stop_producer_thread()
                    logging.warning("Scheduler: storage HALT (data drive below its reserve). Capture paused until space is recovered.")
                    self.storage.refresh()  # re-evaluate now: uploads/cleanup may already have freed space
                    # Sleep until the governor releases HALT (or shutdown interrupts it), at
                    # most one pressure-mode janitor period -- the cadence at which space is
                    # actively reclaimed while under pressure.
                    self.wait_event_with_heartbeat(
                        self.storage.released_event,
                        self.cfg["janitor"]["pressure_check_interval_min"] * 60,
                    )
                    continue

                if self.emergency_stop_active.is_set():
                    if is_capturing:
                        self.stop_producer_thread()
                    logging.warning("Scheduler: System is in EMERGENCY STOP due to an error. All capture is inhibited.")
                    self.log_health_event("CRITICAL", "EMERGENCY_STOP", "Critical state.")
                    for _ in range(self.cfg["general"]["scheduler_interval_sec"]):
                        self.ping_watchdog()
                        if not self.running.is_set():
                            break
                        time.sleep(1)
                    continue  # Skip the rest of the scheduler logic

                if (self.camera_fatal_error.is_set()):  # the camera reach max_camera_failures
                    logging.error(
                        "Scheduler detected fatal camera error. Attempting restart (%d/%d)...",
                        self.consecutive_restart_failures + 1,
                        max_restart_attempts,
                    )
                    self.log_health_event(
                        "ERROR", "CAMERA_DETECT_FAIL", "Fatal camera error."
                    )
                    self.stop_producer_thread()  # Ensure it's fully stopped
                    time.sleep(0.1)
                    self.camera_fatal_error.clear()  # Reset for new attempt
                    self.start_producer_thread()  # Attempt restart

                    # Give the restarted producer time to fail again or stabilise.
                    def _settle(slice_sec):
                        time.sleep(slice_sec)
                        return not self.running.is_set()  # stop waiting early on shutdown
                    self.wait_with_heartbeat(_settle, self.cfg["general"]["camera_restart_settle_sec"])

                    if not self.producer_thread_active():
                        self.consecutive_restart_failures += 1
                        logging.critical(
                            "Producer thread failed to restart. (%d/%d)",
                            self.consecutive_restart_failures,
                            max_restart_attempts,
                        )
                        if self.consecutive_restart_failures >= max_restart_attempts:
                            msg = f"!!! FAILED TO RECOVER AFTER {max_restart_attempts} ATTEMPTS. INITIATING SYSTEM REBOOT. !!!"

                            self.perform_system_action("reboot", reason=msg)
                            break
                    else:
                        logging.info("Producer thread restarted successfully after fatal error.")
                        self.consecutive_restart_failures = 0  # Reset on success

                # START/STOP LOGIC
                # logging.info(f"skap pre schedule check")
                if is_active_schedule and not is_capturing:
                    if self.in_maintenance_mode.is_set():
                        # If we are in schedule but not capturing BECAUSE the user paused it,
                        # DO NOTHING. Log a message and let the maintenance timeout or user handle it.
                        logging.info("Scheduler: System is in scheduled active time but is currently in user-initiated Maintenance Mode.")
                    elif self.is_calibrating.is_set():
                        # Also, do nothing if a calibration is running outside the schedule time
                        logging.info("Scheduler: System is in scheduled active time but is currently performing a calibration.")
                    else:
                        logging.info("Scheduler: Active schedule window has begun. Starting producer.")
                        # BACKUP PREVIOUS MASTERPIECE
                        # Archive names are dated by the night the image belongs to (its last
                        # write time, see session_archive.night_of). The session start time
                        # used before is the PROCESS start time after a restart: a station
                        # rebooted during the day filed last night under today's date.
                        if os.path.exists(self.mip_path):
                            try:
                                archive_path = archive_name("masterpiece", self.mip_path, self.logs_archive)
                                shutil.move(self.mip_path, archive_path)
                                logging.info(f"Backed up previous night's masterpiece to {archive_path}")

                                # CREATE FRESH CANVAS FOR NEW SESSION.
                                self.nightly_masterpiece = None
                                dummy = np.zeros((self.cfg["capture"]["height"], self.cfg["capture"]["width"], 3),dtype=np.uint8)
                                self._safe_save_image(self.mip_path, dummy)

                            except Exception as e:
                                logging.error(f"Failed to backup masterpiece: {e}")

                        # Backup previous star trail (same dating; collision-safe -- the old code
                        # silently overwrote an existing star trail of the same date)
                        if os.path.exists(self.startrail_path):
                            try:
                                archive_path = archive_name("startrail", self.startrail_path, self.logs_archive)
                                shutil.move(self.startrail_path, archive_path)
                                logging.info(f"Backed up previous night's starttrail to {archive_path}")
                            except Exception as e:
                                logging.error(f"Failed to backup startrail: {e}")

                        self.session_stats = {
                            k: (
                                None
                                if k == "start_time"
                                else [] if isinstance(v, list) else 0
                            )
                            for k, v in self.session_stats.items()
                        }
                        # last_check_time is a clock reading, not a counter: zeroing it
                        # made the session's first sky check add (now - 0) -- i.e. the
                        # whole clock value -- to clear/cloudy/moon/error minutes.
                        self.session_stats["last_check_time"] = time.monotonic()
                        self.consecutive_restart_failures = 0
                        # Record session start time
                        self.session_start_time = datetime.now()

                        self.capture_stable.clear()
                        self.session_stats = self._reset_session_stats()

                        try:
                            self.watchdog_pause.set()
                            self.start_producer_thread()
                        finally:
                            self.watchdog_pause.clear()

                        logging.info("Scheduler: Requesting an immediate sky condition check.")
                        self.request_sky_analysis_trigger.set()

                elif (
                    not is_active_schedule
                    and is_capturing
                    and not self.is_calibrating.is_set()
                ):
                    logging.info("Scheduler: Active schedule window has ended. Stopping producer.")
                    self.consecutive_restart_failures = 0
                    with self.status_lock:
                        self.last_night_journal = self.generate_morning_journal()
                    self.stop_producer_thread()
                    # Trigger end-of-session video creation
                    self.start_timelapse_video_creation()

                elif not is_active_schedule and not is_capturing:
                    self.consecutive_restart_failures = 0
                    if (
                        not self.in_maintenance_mode.is_set()
                        and not self.is_calibrating.is_set()
                    ):
                        logging.info("Scheduler: Outside the schedule time.")

                for _ in range(self.cfg["general"]["scheduler_interval_sec"]):
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break
                    time.sleep(1)
        except Exception as e:
            logging.critical("CRITICAL FATAL ERROR IN SCHEDULER THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "SCHEDULER_CRASH", str(e))

    # -----------------
    def _enable_fault_forensics(self):
        """
        Native-crash and hang forensics, independent of the logging module.

        faulthandler writes Python stacks with raw write(2) calls from a
        signal-safe C handler: no Python locks, no buffers. That makes it the
        only reliable record for (a) native crashes inside libcamera, OpenCV or
        NumPy (SIGSEGV/SIGBUS/SIGABRT/SIGFPE/SIGILL), which otherwise kill the
        process with zero Python trace, and (b) watchdog-forced exits (see
        _dump_exit_forensics).

        Also registers SIGUSR1 as an on-demand dump: `kill -USR1 <pid>` prints
        every thread's stack into the file WITHOUT stopping the pipeline --
        a live hang can be diagnosed remotely without py-spy. (Unregistered,
        SIGUSR1's default action would terminate the process.)

        The file must stay open for the process lifetime (faulthandler keeps
        only its fd), so it cannot be rotated by the janitor while running;
        it is size-checked and rotated here, at startup, before it is opened.
        """
        self.fault_log_path = os.path.join(self.general_log_dir, "faulthandler.log")
        self._fault_file = None
        try:
            max_bytes = self.cfg["janitor"]["log_rotation_mb"] * 1024 * 1024
            if (
                os.path.exists(self.fault_log_path)
                and os.path.getsize(self.fault_log_path) > max_bytes
            ):
                os.makedirs(self.logs_archive, exist_ok=True)
                tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.replace(
                    self.fault_log_path,
                    os.path.join(self.logs_archive, f"faulthandler_{tstamp}.log"),
                )
            # Unbuffered: every byte goes straight to the fd, same path faulthandler uses.
            self._fault_file = open(self.fault_log_path, "ab", buffering=0)
            faulthandler.enable(file=self._fault_file, all_threads=True)
            faulthandler.register(
                signal.SIGUSR1, file=self._fault_file, all_threads=True, chain=False
            )
            logging.info(
                f"Fault forensics armed: native crashes and forced exits dump all "
                f"thread stacks to {self.fault_log_path} (on demand: kill -USR1 {os.getpid()})."
            )
        except OSError as e:
            # Forensics are diagnostic, never a reason to refuse to run.
            logging.error(f"Fault forensics unavailable ({e}); continuing without them.")
            self._fault_file = None

    # -----------------
    def _dump_exit_forensics(self, reason, also_run=()):
        """
        All-thread stack dump right before a watchdog-forced os._exit: the one
        artifact that says where the hung thread was actually stuck.

        Written to two independent sinks, stderr first (journald: a pipe, not
        the USB drive) then faulthandler.log (on the log drive), so a hang
        caused by the log drive itself still leaves a trace. Uses os.write and
        faulthandler.dump_traceback on raw fds -- no logging locks, no Python
        buffers. One daemon thread PER SINK, all joined against a single
        deadline of general.exit_forensics_timeout_sec: a sink that blocks
        (dead drive, stalled journald) can neither starve the other sink nor
        delay the exit beyond that bound. Callables in also_run (best-effort
        work that MAY block, e.g. logging calls) run under the same deadline.
        """
        # faulthandler identifies threads only by ident; the watchdog reports them
        # by name. Print the mapping so the dump can be read against the reason.
        thread_map = "".join(
            f"  0x{t.ident:016x}  {t.name}\n"
            for t in threading.enumerate()
            if t.ident is not None
        )
        header = (
            f"\n=== {datetime.now().isoformat()} FORCED EXIT pid={os.getpid()} "
            f"reason: {reason} ===\nThreads:\n{thread_map}"
        ).encode("utf-8", "replace")

        sinks = []
        try:
            if sys.stderr is not None:
                sinks.append(sys.stderr.fileno())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            pass
        fault_file = getattr(self, "_fault_file", None)
        if fault_file is not None:
            sinks.append(fault_file.fileno())
        def _dump(fd):
            try:
                os.write(fd, header)
                faulthandler.dump_traceback(file=fd, all_threads=True)
            except OSError:
                pass  # this sink is dead; the other one is independent

        workers = [
            threading.Thread(target=_dump, args=(fd,), name=f"ExitForensics-{fd}", daemon=True)
            for fd in sinks
        ] + [
            threading.Thread(target=fn, name="ExitForensics-extra", daemon=True)
            for fn in also_run
        ]
        for w in workers:
            w.start()
        deadline = time.monotonic() + self.cfg["general"]["exit_forensics_timeout_sec"]
        for w in workers:
            w.join(max(0.0, deadline - time.monotonic()))

    # -----------------
    @staticmethod
    def _migrate_satellite_cache(old_dir, new_dir):
        """
        One-time move of the catalogue from its first location (<monitor_path>/satellites)
        to assets/satellites. state.json moves WITH the data: it records when each group
        was last downloaded, and losing it would trigger an immediate re-download that
        CelesTrak refuses (HTTP 403) within 2 hours of the previous one.
        """
        if os.path.abspath(old_dir) == os.path.abspath(new_dir) or not os.path.isdir(old_dir):
            return
        try:
            os.makedirs(new_dir, exist_ok=True)
            moved = []
            for name in os.listdir(old_dir):
                src, dst = os.path.join(old_dir, name), os.path.join(new_dir, name)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
                    moved.append(name)
            if not os.listdir(old_dir):
                os.rmdir(old_dir)
            if moved:
                logging.info(f"Satellites: catalogue moved to {new_dir} ({', '.join(sorted(moved))}).")
        except OSError as e:
            logging.error(f"Satellites: could not move the catalogue from {old_dir} to {new_dir}: {e}")

    # -----------------
    def perform_system_action(self, action="exit", reason="Unknown reason"):
        """
        Unified handler for system-level state changes.
        Actions:
        - "exit": Exits script with status 1 (triggers systemd service restart).
        - "reboot": Gracefully stops pipeline, then reboots operating system.
        - "shutdown": Gracefully stops pipeline, then halts operating system.
        """
        banner = [
            "=" * 60,
            f"!!! SYSTEM ACTION TRIGGERED: {action.upper()} !!!",
            f"!!! Reason: {reason} !!!",
            "=" * 60,
        ]

        if action == "exit":
            # The forced-exit path must never block on the logging module. Every
            # logging call takes the handler's lock, and the thread the watchdog
            # just declared hung may be stuck INSIDE a handler write (dead USB
            # drive) holding that lock: an unconditional logging.critical() here
            # would deadlock the watchdog itself and the process would never
            # exit. So the banner is logged best-effort in a bounded thread,
            # concurrently with the lock-free stack dump, then we exit no matter
            # what. No logging.shutdown(): FileHandler/StreamHandler flush on
            # every emit already, and shutdown() takes the same locks.
            def _log_exit_banner():
                for line in banner:
                    logging.critical(line)
                logging.info("Exiting script with status 1 for service manager restart.")

            self._dump_exit_forensics(reason, also_run=(_log_exit_banner,))
            os._exit(1)

        for line in banner:
            logging.critical(line)

        # For reboot/shutdown, we MUST try to stop gracefully first to save data.
        try:
            self.stop()
        except Exception as e:
            logging.error(f"Error during graceful stop before {action}: {e}")

        self._execute_post_shutdown_script()

        # Execute OS-level commands
        if action == "reboot":
            logging.info("Executing OS reboot...")
            subprocess.run(["sudo", "reboot"], check=False)
        elif action == "shutdown":
            logging.info("Executing OS halt...")
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    # -----------------
    def perform_ota_update(self):
        """Executes an Enterprise Git Pull, verifies syntax across all files, and restarts."""
        logging.warning("="*60)
        logging.warning("OTA UPDATE INITIATED (Git Architecture)")
        logging.warning("="*60)

        script_dir = os.path.dirname(os.path.abspath(__file__))

        try:

            # 1. Discard any accidental local code edits (but ignore config.json)
            subprocess.run(["git", "reset", "--hard"], cwd=script_dir, capture_output=True)

            # 2. Execute Git Pull
            logging.info("Fetching latest code from Git repository...")
            result = subprocess.run(["git", "pull"], capture_output=True, text=True, cwd=script_dir)
            
            if result.returncode != 0:
                logging.error(f"OTA ABORTED: Git pull failed:\n{result.stderr}")
                return False
                
            if "Already up to date." in result.stdout:
                logging.info("Repository is already up to date. No restart needed.")
                return True 

            logging.info(f"Git pull successful:\n{result.stdout.strip()}")

            # 3. Syntax Check (First Line of Defense for MULTIPLE files)
            logging.info("Verifying Python syntax for all .py modules...")
            py_files = glob.glob(os.path.join(script_dir, "*.py"))
            
            for py_file in py_files:
                py_compile.compile(py_file, doraise=True)
                
            logging.info("All modules passed syntax verification.")

            # 4. Trigger Restart
            logging.warning("OTA Update applied successfully! Restarting pipeline...")
            def delayed_restart():
                time.sleep(2)
                self.perform_system_action("exit", reason="OTA Update Applied (Git)")
            
            threading.Thread(target=delayed_restart).start()
            return True

        except py_compile.PyCompileError as e:
            logging.error(f"OTA ABORTED: Syntax Error detected in downloaded files!\n{e}")
            logging.warning("Executing Git Rollback to previous working state...")
            # HEAD@{1} tells git to jump back in time to the state immediately before the pull
            subprocess.run(["git", "reset", "--hard", "HEAD@{1}"], cwd=script_dir)
            return False
            
        except Exception as e:
            logging.error(f"OTA ABORTED: Unexpected error: {e}")
            subprocess.run(["git", "reset", "--hard", "HEAD@{1}"], cwd=script_dir)
            return False

    # -----------------
    def generate_morning_journal(self):
        duration = datetime.now() - self.session_stats["start_time"]
        hours = duration.total_seconds() / 3600.0
        mean_lux = (
            np.mean(self.session_stats["avg_lux"])
            if self.session_stats["avg_lux"]
            else 0.0
        )

        clarity = (
            self.session_stats["clear_sky_minutes"]
            / max(
                1,
                self.session_stats["clear_sky_minutes"]
                + self.session_stats["cloudy_sky_minutes"],
            )
            * 100
        )
        sky_avg = (
            np.mean(self.session_stats["avg_sky_score"])
            if self.session_stats["avg_sky_score"]
            else 0.0
        )

        fp = self._generate_atmosphere_fingerprint(
            self.session_stats.get("sky_history", [])
        )
        ctx = self._get_chronicle_context()
        self._update_chronicle(
            hours, sky_avg, clarity, self.session_stats.get("sky_history", [])
        )

        eff = 100.0
        if self.session_stats["total_events"] > 0:
            eff = 100 - (
                self.session_stats["rejected_contours"]
                / self.session_stats["total_events"]
                * 100
            )

        lines = [
            f"THE WATCHMAN'S JOURNAL | Night of {self.session_stats['start_time'].strftime('%B %d, %Y')} Session Ending"
        ]
        lines.append("-" * 60)
        lines.append(f"• SESSION: Vigilance kept for {hours:.1f} hours.")
        lines.append(f"• ENVIRONMENT: Average sky illuminance: {mean_lux:.4f} Lux.")
        lines.append(
            f"• THE SKY: Observed {clarity:.1f}% clarity with an average transparency of {sky_avg:.2f}."
        )

        if self.session_stats["max_moon_impact"] > 0.1:
            lines.append(
                f"• LUNAR INFLUENCE: The Moon was in FOV for {self.session_stats['moon_visible_minutes']:.2f}m (Max Impact: {int(self.session_stats['max_moon_impact']*100)}%)."
            )

        lines.append(
            f"• EFFICIENCY: Analyzed {self.session_stats["total_events"]:,} potential triggers; ignored {self.session_stats["rejected_contours"]:,} noise artifacts ({eff:.1f}% accuracy)."
        )
        lines.append(
            f"• DETECTIONS: Isolated {self.session_stats["accepted_events"]} candidate celestial events."
        )
        lines.append(
            f"• STORAGE: Recorded {self.session_stats["data_written_mb"]:.1f} MB of new astronomical data to disk."
        )
        lines.append(
            f"• VITALS: Peak operating temperature: {self.session_stats['peak_cpu_temp']:.1f}°C."
        )
        lines.append(f"• ATMOSPHERE: {fp['summary']}")
        if ctx:
            lines.append(
                f"• CHRONICLE ({ctx['nights_total']} nights): "
                f"Tonight's {ctx['events_tonight']} events ranks in the "
                f"{ctx['events_pct']}th percentile (avg {ctx['events_avg']:.1f}, "
                f"record {ctx['events_record']}). "
                f"Sky quality {ctx['score_pct']}th percentile."
            )

        # GRAPH GENERATION
        pdf_supported = True
        plot_path = None
        try:
            matplotlib.use("Agg")  # Safe for headless Pi

            # Use dark background
            plt.style.use("dark_background")

            if (
                self.session_stats.get("sky_history")
                and len(self.session_stats["sky_history"]) > 1
            ):
                fig, ax1 = plt.subplots(figsize=(10, 4), facecolor="#1e1e1e")
                ax1.set_facecolor("#1e1e1e")

                # Define Labels and Line Colors
                C_SKY = "#61afef"
                C_MOON = "#e5c07b"
                C_STARS = "#98c379"
                C_NOISE = "#c678dd"
                C_DIVIDER = "#777777"

                # PRIMARY AXIS (Left): Sky Score & Moon Penalty
                t_sky = [
                    datetime.fromtimestamp(h["t"])
                    for h in self.session_stats["sky_history"]
                ]

                # 1. Sky Score (Cyan)
                scores = [h["score"] for h in self.session_stats["sky_history"]]
                (line1,) = ax1.plot(
                    t_sky,
                    scores,
                    color=C_SKY,
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label="Sky Score",
                    zorder=5,
                )

                ax1.axhline(
                    y=SKY_SCORE_VALUES["excellent"],
                    color="#555555",
                    linestyle="--",
                    alpha=0.8,
                    label="Excellent Threshold",
                )
                ax1.axhline(
                    y=SKY_SCORE_VALUES["good"],
                    color="#555555",
                    linestyle="--",
                    alpha=0.8,
                    label="Good Threshold",
                )
                ax1.axhline(
                    y=SKY_SCORE_VALUES["fair"],
                    color="#555555",
                    linestyle="--",
                    alpha=0.8,
                    label="Cloudy Threshold",
                )

                # 2. Moon Penalty (Orange/Amber)
                if self.session_stats.get("moon_penalty_history"):
                    t_moon = [
                        datetime.fromtimestamp(h["t"])
                        for h in self.session_stats["moon_penalty_history"]
                    ]
                    penalties = [
                        h["val"] for h in self.session_stats["moon_penalty_history"]
                    ]
                    (line2,) = ax1.plot(
                        t_moon,
                        penalties,
                        color=C_MOON,
                        linestyle="--",
                        alpha=0.6,
                        label="Moon Penalty",
                        zorder=4,
                    )

                # ax1.set_ylabel('Score / Penalty', color='#61afef', fontweight='bold')
                ax1.tick_params(axis="y", colors=C_SKY)
                ax1.grid(color="#333333", linestyle=":", alpha=0.5)

                # MULTI-COLOR LEFT LABEL
                # Place label "Score / Penalty" the same color of own graph
                ax1.annotate(
                    "Score",
                    xy=(-0.07, 0.65),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    color=C_SKY,
                    fontweight="bold",
                    fontsize=11,
                )
                ax1.annotate(
                    "/",
                    xy=(-0.07, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    color=C_DIVIDER,
                    fontweight="bold",
                    fontsize=11,
                )
                ax1.annotate(
                    "Penalty",
                    xy=(-0.07, 0.32),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    color=C_MOON,
                    fontweight="bold",
                    fontsize=11,
                )

                # SECONDARY AXIS (Right): Stars & StdDev
                ax2 = ax1.twinx()

                # 3. Star Count (Green)
                if self.session_stats.get("star_count_history"):
                    t_stars = [
                        datetime.fromtimestamp(h["t"])
                        for h in self.session_stats["star_count_history"]
                    ]
                    stars = [h["val"] for h in self.session_stats["star_count_history"]]
                    (line3,) = ax2.plot(
                        t_stars,
                        stars,
                        color=C_STARS,
                        marker="x",
                        markersize=4,
                        linestyle="-",
                        alpha=0.8,
                        label="Stars",
                        zorder=3,
                    )

                # 4. StdDev (Purple/Magenta)
                if self.session_stats.get("stddev_history"):
                    t_std = [
                        datetime.fromtimestamp(h["t"])
                        for h in self.session_stats["stddev_history"]
                    ]
                    stddevs = [h["val"] for h in self.session_stats["stddev_history"]]
                    (line4,) = ax2.plot(
                        t_std,
                        stddevs,
                        color=C_NOISE,
                        linestyle=":",
                        alpha=0.7,
                        label="Noise (StdDev)",
                        zorder=2,
                    )

                # ax2.set_ylabel('Count / Noise', color='#98c379', fontweight='bold')
                ax2.tick_params(axis="y", colors="#98c379")

                # MULTI-COLOR RIGHT LABEL
                # Place label "Count / Noise" the same color of own graph
                ax2.annotate(
                    "Count",
                    xy=(1.07, 0.65),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="center",
                    va="center",
                    color=C_STARS,
                    fontweight="bold",
                    fontsize=11,
                )
                ax2.annotate(
                    "/",
                    xy=(1.07, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="center",
                    va="center",
                    color=C_DIVIDER,
                    fontweight="bold",
                    fontsize=11,
                )
                ax2.annotate(
                    "Noise",
                    xy=(1.07, 0.35),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="center",
                    va="center",
                    color=C_NOISE,
                    fontweight="bold",
                    fontsize=11,
                )

                # TERTIARY AXIS (Overlay): Bias Mode
                if self.session_stats.get("bias_history"):
                    t_bias = [
                        datetime.fromtimestamp(h["t"])
                        for h in self.session_stats["bias_history"]
                    ]
                    bias_vals = [
                        1 if h["bias"] else 0
                        for h in self.session_stats["bias_history"]
                    ]
                    ax3 = ax1.twinx()
                    # Offset the axis so it doesn't overlap labels
                    ax3.spines.right.set_position(("axes", 1.15))
                    ax3.fill_between(
                        t_bias, 0, bias_vals, color="#e06c75", alpha=0.2, step="post"
                    )
                    ax3.set_yticks([])  # Hide Y ticks for the background fill
                    ax3.set_ylim(0, 1)

                # Legends and Formatting
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                plt.title(
                    f"Celestial Metrics & Atmospheric Stability",
                    color="#ffffff",
                    pad=25,
                    fontsize=14,
                )

                # Combine legends from both axes
                plot_lines = [line1]
                if self.session_stats.get("moon_penalty_history"):
                    plot_lines.append(line2)
                if self.session_stats.get("star_count_history"):
                    plot_lines.append(line3)
                if self.session_stats.get("stddev_history"):
                    plot_lines.append(line4)

                labels = [l.get_label() for l in plot_lines]
                ax1.legend(
                    plot_lines,
                    labels,
                    loc="upper left",
                    fontsize="small",
                    frameon=True,
                    facecolor="#1e1e1e",
                    edgecolor="#444444",
                )

                fig.tight_layout()
                plot_path = os.path.join(
                    self.general_log_dir, f"temp_plot_{int(time.time())}.png"
                )
                plt.savefig(plot_path, dpi=200, facecolor=fig.get_facecolor())
                plt.close(fig)

        except Exception as e:
            pdf_supported = False
            logging.error(f"Failed to generate complex journal graph: {e}")

        # 2. DOCUMENT GENERATION (Elegant PDF)
        rpt_name_base = self.session_stats["start_time"].strftime("journal_%Y%m%d")

        if pdf_supported:
            try:
                pdf_path = os.path.join(self.general_log_dir, rpt_name_base + ".pdf")
                pdf = FPDF()
                pdf.add_page()

                # PDF Styling
                C_DARK = (40, 44, 52)  # Deep grey/blue for headers
                C_ACCENT = (198, 120, 221)  # Ryusei Purple
                C_TEXT = (80, 80, 80)  # Soft grey for text

                # Header (Logo Injection)
                logo_path = os.path.join(self.assets_dir, "logo_pdf.jpg")

                if os.path.exists(logo_path):
                    # Center a 65mm wide logo on A4 paper (210mm wide) -> x = 72.5
                    pdf.image(logo_path, x=72.5, w=65)
                    pdf.ln(2)
                else:
                    pdf.set_font("Helvetica", "B", 18)
                    pdf.set_text_color(*C_DARK)
                    pdf.cell(
                        0,
                        8,
                        text="RYUSEI OBSERVATORY",
                        new_x=XPos.LMARGIN,
                        new_y=YPos.NEXT,
                        align="C",
                    )

                pdf.set_font("Helvetica", "I", 11)
                pdf.set_text_color(*C_ACCENT)
                pdf.cell(
                    0,
                    5,
                    text=f"The Watchman's Journal - {self.session_stats['start_time'].strftime('%B %d, %Y')}",
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                    align="C",
                )
                pdf.ln(3)

                # Helper Function for Clean Grids
                def add_section(title, data_dict):
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(*C_ACCENT)
                    pdf.cell(
                        0, 6, text=title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT
                    )

                    # Draw subtle underline
                    pdf.set_draw_color(*C_ACCENT)
                    pdf.set_line_width(0.3)
                    y = pdf.get_y()
                    pdf.line(10, y, 200, y)
                    pdf.ln(2)

                    for k, v in data_dict.items():
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.set_text_color(*C_DARK)
                        pdf.cell(45, 5, text=k, new_x=XPos.RIGHT)

                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(*C_TEXT)
                        # Sanitize strings for FPDF
                        safe_v = (
                            str(v)
                            .replace("•", "*")
                            .replace("°", " deg")
                            .replace("—", "-")
                        )
                        safe_v = safe_v.encode("latin-1", "ignore").decode("latin-1")
                        pdf.multi_cell(
                            0, 5, text=safe_v, new_x=XPos.LMARGIN, new_y=YPos.NEXT
                        )
                    pdf.ln(2)

                # Build Sections
                add_section(
                    "Mission Parameters",
                    {
                        "Vigilance Kept:": f"{hours:.1f} hours",
                        "Average Illuminance:": f"{mean_lux:.4f} Lux",
                        "Peak Temperature:": f"{self.session_stats['peak_cpu_temp']:.1f} °C",
                        "Data Recorded:": f"{self.session_stats['data_written_mb']:.1f} MB",
                    },
                )

                add_section(
                    "Detection Analytics",
                    {
                        "Sky Clarity:": f"{clarity:.1f}% (Avg Score: {sky_avg:.2f})",
                        "Celestial Events:": f"{self.session_stats["accepted_events"]} isolated",
                        "Detector Efficiency:": f"{eff:.1f}% ({self.session_stats["rejected_contours"]:,} noise artifacts rejected)",
                    },
                )

                if ctx:
                    add_section(
                        "Historical Chronicle",
                        {
                            "Database Size:": f"{ctx['nights_total']} recorded nights",
                            "Event Ranking:": f"{ctx['events_pct']}th percentile (Tonight: {ctx['events_tonight']}, Avg: {ctx['events_avg']:.1f})",
                            "Sky Quality:": f"{ctx['score_pct']}th percentile",
                        },
                    )

                add_section(
                    "Atmospheric Telemetry",
                    {
                        "Night Trend:": fp["summary"],
                        "Lunar Impact:": (
                            f"In FOV for {self.session_stats['moon_visible_minutes']:.1f}m (Max Flare: {int(self.session_stats['max_moon_impact']*100)}%)"
                            if self.session_stats["max_moon_impact"] > 0.1
                            else "None"
                        ),
                    },
                )

                # Insert Graph
                if plot_path and os.path.exists(plot_path):
                    pdf.ln(1)
                    pdf.image(plot_path, x=15, w=180)  # Fits perfectly on A4
                    os.remove(plot_path)  # Cleanup

                # Footer Quote
                pdf.ln(2)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(
                    0,
                    4,
                    text='"The stars are the streetlights of eternity." - Ryusei Intelligence',
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                    align="C",
                )

                pdf.output(pdf_path)
                logging.info(f"Morning Journal PDF successfully generated: {pdf_path}")
            except Exception as e:
                logging.error(f"Failed to compile PDF: {e}")

        if not pdf_supported or not os.path.exists(pdf_path):
            # Fallback to standard Text File if PDF failed
            txt_path = os.path.join(self.general_log_dir, rpt_name_base + ".txt")
            try:
                with open(txt_path, "w") as f:
                    f.write("\n".join(lines))
            except Exception as e:
                logging.warning(f"Could not write txt journal file: {e}")

        return "\n".join(lines)

    # -----------------
    def start_producer_thread(self):
        """Starts ONLY the camera capture thread."""
        if any(t.name == "CaptureThread" for t in self.worker_threads if t.is_alive()):
            logging.warning("Capture thread appears to be already running.")
            return

        # The sensor's lifecycle follows the producer's: stop_producer_thread() stops it
        # (hardware stop to interrupt long exposures), so every start must make sure it
        # is streaming again. Without this, any start after a stop -- a test shot after
        # Pause, the maintenance auto-resume, the scheduler's restart -- ran the capture
        # loop against a stopped sensor: no frame ever arrived and the CaptureThread
        # blocked forever inside capture_request(), unstoppable (-> THREAD_JOIN_TIMEOUT
        # -> reboot).
        ensure = getattr(self.camera, "ensure_streaming", None)
        if ensure is not None:
            with self.camera_lock:
                try:
                    ensure()
                except Exception as e:
                    logging.error(f"Cannot start the camera sensor: {e}. Producer not started.")
                    self.log_health_event("ERROR", "CAMERA_START_FAIL", str(e))
                    return

        # Ensure the capture running flag is set before starting
        self.capture_running.set()

        capture_thread = threading.Thread(
            target=self.capture_loop, name="CaptureThread", daemon=True
        )
        detection_thread = threading.Thread(
            target=self.detection_loop, name="DetectionThread", daemon=True
        )

        capture_thread.start()
        detection_thread.start()

        self.worker_threads.append(capture_thread)
        self.worker_threads.append(detection_thread)

        logging.info("Data production pipeline started successfully.")

        # Detection only processes frames once the SkyMonitor has validated the sky
        # (capture_stable). Any start without a valid check must therefore REQUEST
        # one, or detection waits forever -- the maintenance auto-resume did exactly
        # that after a test shot had cleared capture_stable. Centralised here so no
        # start path can forget it. Not during a test shot: a sky check would change
        # camera settings in the middle of it.
        if not self.capture_stable.is_set() and not self.is_calibrating.is_set():
            logging.info("Capture is not validated yet: requesting an immediate sky check.")
            self.request_sky_analysis_trigger.set()

    # -----------------
    def stop_producer_thread(self):
        """
        Atomically and safely stops all data producer threads.
        Optimized V188: Handles long-exposures by breaking hardware locks and using dynamic timeouts.
        """
        producer_names_to_stop = ["CaptureThread", "DetectionThread"]
        threads_to_stop = [
            t
            for t in self.worker_threads
            if t.name in producer_names_to_stop and t.is_alive()
        ]

        if not threads_to_stop:
            logging.warning(
                "stop_producer_thread was called, but no producer threads were found running."
            )

            return

        # Pause Watchdog
        try:
            logging.warning("Pausing watchdog for producer thread shutdown procedure.")
            self.watchdog_pause.set()
            self.update_worker_list(threads_to_stop)
        finally:
            logging.info("Resuming watchdog. Producer threads will now be joined.")
            self.watchdog_pause.clear()

        # Signal Shutdown
        self.capture_running.clear()

        # Force "Poison Pill" into Queue (Critical for DetectionThread)
        # If queue is full, drain one item to make room for the sentinel
        try:
            self.acq_q.put_nowait(None)
        except queue.Full:
            try:
                self.acq_q.get_nowait()  # Discard old frame
                self.acq_q.put_nowait(None)  # Insert stop signal
            except Exception:
                pass

        logging.info("Resetting background model for next session.")
        self.background = None

        # Stop budget, derived: a producer blocked in capture_request() or in a camera
        # sync (flush) returns within the in-flight pipeline depth x the frame period
        # (see _frame_wait_budget_sec / camera_capture.py) -- measured with the CURRENT
        # exposure. The old formula used the CONFIGURED initial exposure_us: with AE
        # running at a longer exposure the budget was too short and a normal stop was
        # escalated to THREAD_JOIN_TIMEOUT -> reboot.
        safe_timeout = self._frame_wait_budget_sec() + self.cfg["general"]["shutdown_join_slack_sec"]
        shutdown_deadline = time.monotonic() + safe_timeout

        logging.info(
            f"Waiting for producer threads to terminate (Timeout: {safe_timeout:.1f}s)..."
        )

        # Signal SkyMonitor to abort immediately
        self.capture_running.clear()

        # Give the SkyMonitor a moment to see the 'clear' signal and exit its lock
        time.sleep(0.5)

        # If we are in long exposure, read() blocks. We force the camera to stop NOW.
        if self.camera and hasattr(self.camera, "picam2"):
            try:
                # This breaks the blocking capture_array() call in the other thread
                logging.info("Sending hardware stop signal to interrupt exposure...")
                self.camera.picam2.stop()
            except Exception:
                pass  # Camera might already be stopped

        # Join Threads
        for thread in threads_to_stop:
            while thread.is_alive():
                if time.monotonic() > shutdown_deadline:
                    logging.critical(
                        f"!!! FAILED TO STOP THREAD '{thread.name}' within the timeout. !!!"
                    )
                    msg = f"THREAD_JOIN_TIMEOUT: Thread {thread.name} refused to stop."
                    self.log_health_event("CRITICAL", "THREAD_JOIN_TIMEOUT", msg)
                    # Record WHERE it is stuck before the reboot erases the evidence
                    # (bounded, lock-free; see _dump_exit_forensics).
                    self._dump_exit_forensics(msg)
                    self.perform_system_action("reboot", reason=msg)
                    break

                thread.join(timeout=0.5)

            if not thread.is_alive():
                logging.info(f"Thread '{thread.name}' has successfully terminated.")
                # Drop its heartbeat too, not just from worker_threads. The hang
                # detector in watchdog_loop tracks threads by name in
                # thread_heartbeats independently of worker_threads; a stopped
                # thread never pings again, so its last heartbeat would just sit
                # there aging until it crosses watchdog_hung_thread_sec and the
                # watchdog force-exits over a thread that was deliberately
                # stopped, not hung. Done only here (confirmed dead), not
                # earlier in update_worker_list -- while the thread is still
                # alive it could still ping and silently recreate the stale
                # entry before it actually stops.
                self.thread_heartbeats.pop(thread.name, None)

        # Tell the Timelapse Thread to process its partial buffer. timelapse_flush_done
        # is cleared first and set by the timelapse thread once the partial stack is
        # on disk, so end-of-session media can wait for it instead of sleeping.
        self.timelapse_flush_done.clear()
        try:
            self.timelapse_q.put_nowait("FLUSH")
        except queue.Full:
            logging.warning("Timelapse queue full: FLUSH not delivered; the partial stack will not be written.")
            self.timelapse_flush_done.set()

        logging.info("Data production chain has been cleanly stopped.")

    # -----------------
    def update_worker_list(self, threads_to_stop):

        if not threads_to_stop:
            logging.warning(
                "stop_producer_thread called but no producer threads were found running."
            )
            return

        # Immediately remove these threads from the main worker list.
        # This prevents the watchdog race condition.
        self.worker_threads = [
            t for t in self.worker_threads if t not in threads_to_stop
        ]
        logging.info(
            f"Removed {[t.name for t in threads_to_stop]} from watchdog monitoring list."
        )

    # -----------------
    def producer_thread_active(self):
        """Checks if the single CaptureThread is alive."""
        return any(
            t.name == "CaptureThread" and t.is_alive() for t in self.worker_threads
        )

    # -----------------
    def _tally_data_write(self, file_path):
        """Adds the size of a newly written file to the session statistics."""
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                size_mb = size_bytes / (1024 * 1024)
                with self.status_lock:
                    self.session_stats["data_written_mb"] += size_mb
        except Exception as e:
            logging.warning(f"Failed to tally file size for {file_path}: {e}")

    # -----------------
    def get_metadata(
        self,
        n_frames=1,
        mean_brightness=None,
        effective_threshold=None,
        override_exposure=None,
        override_gain=None,
        alignment_features=None,
        stddev=None,
        bias_reasons=None,
        stack_info=None,
        image_filename=None
    ):
        # Overrides if provided (for snapshots), otherwise use live system state
        exposure_us = (
            override_exposure
            if override_exposure is not None
            else self.current_exposure_us
        )
        gain = override_gain if override_gain is not None else self.current_gain

        actual_stacked = n_frames
        ai_label = None
        ai_confidence = None
        is_verified_meteor = None
        event_score = None
        ra_deg = None
        dec_deg = None
        
        if stack_info:
            stddev = stddev if stddev is not None else stack_info.get("stddev")
            alignment_features = (
                alignment_features
                if alignment_features is not None
                else stack_info.get("anchors")
            )
            bias_reasons = (
                bias_reasons
                if bias_reasons is not None
                else stack_info.get("bias_reasons")
            )
            actual_stacked = stack_info.get("actual_frames_stacked", n_frames)
            ai_label = stack_info.get("ai_label")
            ai_confidence = stack_info.get("ai_confidence")
            is_verified_meteor = stack_info.get("is_verified_meteor")
            event_score = stack_info.get("event_score")
            ra_deg = stack_info.get("ra_deg")
            dec_deg = stack_info.get("dec_deg")
            
        if mean_brightness:
            lux_val = self.calculate_lux(mean_brightness, exposure_us, gain)
        else:
            lux_val = 0.0

        # Mathematically adjust the total integration time if frames were rejected
        equivalent_exposure_s = (exposure_us * actual_stacked) / 1e6

        with self.status_lock:
            moon_impact = self.last_moon_status.get("impact", 0.0)
            moon_in_fov = self.last_moon_status.get("in_fov", False)
            moon_alt = self.last_moon_status.get("alt", 0.0)
            moon_dist = self.last_moon_status.get("dist", 180.0)
            obs_cond = self.session_stats.get("obs_condition")
            obs_conf = round(float(self.session_stats.get("obs_confidence", 0.0)), 2)

        meta = {
            "hostname": self.cfg["general"].get("hostname"),
            "ver": self.version,
            "location": self.cfg["general"].get("location") or "Unknown",
            "latitude": self.cfg["general"].get("latitude"),
            "longitude": self.cfg["general"].get("longitude"),
            "azimuth": self.cfg["general"].get("camera_azimuth"),
            "altitude": self.cfg["general"].get("camera_altitude"),
            "timestamp": datetime.now().isoformat(),
            "obs_condition": obs_cond,
            "obs_confidence": obs_conf,
            "moon_impact": round(moon_impact, 3),
            "moon_in_fov": bool(moon_in_fov),
            "moon_alt": round(moon_alt, 2),
            "moon_dist": round(moon_dist, 2),
        }

        if image_filename:
            meta["image_filename"] = image_filename

        meta.update(
            {
                "auto_exposure_tuning": self.cfg["capture"].get("auto_exposure_tuning"),
                "exposure_us": exposure_us,
                "frames_target": n_frames,  # How many it TRIED to stack
                "frames_stacked": actual_stacked,  # How many it ACTUALLY stacked
                "equivalent_exposure_s": equivalent_exposure_s,
                "gain": round(gain, 2),
                "red_gain": self.cfg["capture"].get("red_gain"),
                "blue_gain": self.cfg["capture"].get("blue_gain"),
                "lux": round(float(lux_val), 5),
                "lux_source": getattr(self, "last_light_check_source", "acquired"),
                "sensor_temp_cam_c": (
                    self.current_sensor_temp_cam
                    if self.current_sensor_temp_cam is not None
                    else "N/A"
                ),
                "sensor_temp_ext_c": (
                    self.current_sensor_temp_ext
                    if self.current_sensor_temp_ext is not None
                    else "N/A"
                ),
                # "sensor_temp_c": self.current_sensor_temp if self.current_sensor_temp is not None else "N/A",
                "dark_frame_used": getattr(self, "active_dark_frame_name", "None"),
            }
        )

        if ai_label is not None:
            meta["ai_label"] = ai_label
            meta["ai_confidence"] = ai_confidence
            meta["is_verified_meteor"] = is_verified_meteor
            if event_score is not None:
                meta["event_score"] = event_score

        if mean_brightness is not None:
            meta["mean_brightness"] = round(mean_brightness, 2)
            
        # Add Astrometry to JSON Output
        if ra_deg is not None:
            meta["ra_deg"] = ra_deg
            meta["dec_deg"] = dec_deg

        if effective_threshold is not None:
            meta["effective_threshold"] = round(effective_threshold, 2)
        if alignment_features is not None:
            meta["alignment_features"] = alignment_features
        if stddev is not None:
            meta["sky_stddev"] = round(stddev, 3)
        if bias_reasons:
            meta["bias_reasons"] = bias_reasons

        return meta

    # -----------------
    def event_logger_loop(self):
        """
        A dedicated thread that waits for event summary dictionaries and appends
        them as a new line in a structured CSV file.
        """
        logging.info(
            "Event logger started. Will append summaries to %s", self.event_log_out_file
        )

        # Write the CSV header if the file is new. Without this the file stays
        # headerless until the janitor first rotates it, which makes every
        # csv.DictReader consumer (dashboard event stats / detailed events)
        # misread the first event row as the header.
        header = ",".join(EVENT_LOG_FIELDS) + "\n"
        try:
            if not os.path.exists(self.event_log_out_file):
                with open(self.event_log_out_file, "w") as f:
                    f.write(header)
            else:
                # Schema migration: columns are only ever APPENDED (EVENT_LOG_FIELDS), so
                # an older file just needs its header line upgraded; its rows simply lack
                # the trailing fields. Atomic rewrite.
                with open(self.event_log_out_file, "r", encoding="utf-8", newline="") as f:
                    content = f.read()
                first, _, rest = content.partition("\n")
                if first.startswith("filename,") and first.strip() != header.strip() \
                        and header.startswith(first.strip() + ","):
                    tmp = self.event_log_out_file + ".tmp"
                    with open(tmp, "w", encoding="utf-8", newline="") as f:
                        f.write(header + rest)
                    os.replace(tmp, self.event_log_out_file)
                    logging.info("events.csv: header upgraded with the new columns.")
        except Exception as e:
            logging.error(f"Could not initialise events.csv header: {e}")

        try:
            while True:
                try:
                    # This is a blocking call, so the thread uses zero CPU while waiting
                    log_data = self.event_log_q.get()

                    if log_data == "STATUS":
                        self.ping_watchdog()
                        continue
    
                    # Check for the sentinel to exit
                    if log_data is None:
                        logging.info("Event logger received sentinel. Exiting.")
                        break

                    is_verified = log_data.get("is_verified_meteor")
                    status_text = "KEPT" if is_verified else "QUARANTINED"

                    # Format the data into a CSV line
                    line = (
                        f"{log_data['filename']},{log_data['timestamp_local']},{log_data['start_time_local']},"
                        f"{log_data['duration_sec']:.2f},{log_data['num_frames_video']},"
                        f"{log_data['num_frames_stack']},{log_data['event_score']:.3f},{log_data['ai_label']},{status_text},{log_data.get('ra_deg', '')},{log_data.get('dec_deg', '')},"
                        # Satellite names may contain commas/quotes ("ISS (ZARYA)", etc.): CSV-quote.
                        f"\"{str(log_data.get('satellite', '')).replace(chr(34), chr(34) * 2)}\"\n"
                    )

                    with open(self.event_log_out_file, "a") as f:
                        f.write(line)

                    # Increment and display the event counter
                    self.event_counter += 1
                    logging.info(
                        "Event #%d logged successfully. Total events captured: %d",
                        self.event_counter,
                        self.event_counter,
                    )

                except Exception as e:
                    logging.exception("Error in event_logger_loop: %s", e)
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN EVENT LOGGER THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "EVENT_LOGGER_CRASH", str(e))

    # -----------------
    def health_monitor_loop(self):
        """
        A stateful worker that maintains a persistent counter of health events.
        It reads the stats file into memory, updates counts based on events from
        the health_q, and rewrites the file.
        """

        # In-memory state of the health statistics
        # Format: { "EVENT_TYPE": {"count": N, "level": "LEVEL", "last_message": "..."} }
        health_stats = {}

        # Read the existing stats file on startup
        try:
            if os.path.exists(self.health_log_out_file):
                with open(
                    self.health_log_out_file, "r", encoding="utf-8", newline=""
                ) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert count back to an integer
                        health_stats[row["event_type"]] = {
                            "count": int(row["count"]),
                            "level": row["level"],
                            "last_message": row["last_message"],
                        }
                logging.info(
                    f"Health monitor loaded {len(health_stats)} existing event types."
                )
        except Exception as e:
            logging.error(f"Could not load existing health stats file: {e}")

        try:
            while self.running.is_set():
                try:
                    # Wait for a new health event to arrive
                    health_event = self.health_q.get(timeout=1.0)

                    if health_event == "STATUS":
                        self.ping_watchdog()
                        continue
    
                    if health_event is None:
                        break  # Sentinel for shutdown

                    event_type = health_event["event_type"]

                    if event_type in health_stats:
                        # If event type exists, increment the counter
                        health_stats[event_type]["count"] += 1
                        # Always update with the latest message and level
                        health_stats[event_type]["last_message"] = health_event[
                            "message"
                        ]
                        health_stats[event_type]["level"] = health_event["level"]
                    else:
                        # If it's a new event type, create a new entry
                        health_stats[event_type] = {
                            "count": 1,
                            "level": health_event["level"],
                            "last_message": health_event["message"],
                        }

                    try:
                        # Use a temporary file for an atomic write to prevent corruption
                        temp_path = self.health_log_out_file + ".tmp"
                        with open(temp_path, "w", encoding="utf-8", newline="") as f:
                            # Define the fieldnames in the order you want them
                            fieldnames = [
                                "event_type",
                                "count",
                                "level",
                                "last_message",
                            ]
                            writer = csv.DictWriter(f, fieldnames=fieldnames)

                            writer.writeheader()
                            # Sort by count so the file is always nicely ordered
                            sorted_events = sorted(
                                health_stats.items(),
                                key=lambda item: item[1]["count"],
                                reverse=True,
                            )
                            for event_type, data in sorted_events:
                                writer.writerow(
                                    {
                                        "event_type": event_type,
                                        "count": data["count"],
                                        "level": data["level"],
                                        "last_message": data["last_message"],
                                    }
                                )
                        # Atomically replace the old file with the new one
                        os.rename(temp_path, self.health_log_out_file)
                    except Exception as e:
                        logging.error(f"Failed to write updated health stats file: {e}")

                except queue.Empty:
                    continue  # This is normal, just loop again
                except Exception as e:
                    logging.exception("Error in health_monitor_loop: %s", e)
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN HEALTH MONITOR THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "HEALTH_MONITOR_CRASH", str(e))

    # -----------------
    def log_health_event(self, level, event_type, message):
        """Creates a structured health event and puts it on the health queue."""
        if not self.cfg.get("health_monitor", {}).get("enabled"):
            return

        event = {
            "level": level,
            "event_type": event_type,
            "message": message.replace(",", ";"),  # Sanitize commas for CSV
        }
        try:
            self.health_q.put_nowait(event)
        except queue.Full:
            logging.warning("Health event queue is full. A statistic was dropped.")

    # -----------------
    def get_health_statistics(self, filepath):
        """Reads the pre-aggregated health stats CSV file and returns it as a list of dicts."""
        if not os.path.exists(filepath):
            return []

        try:
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                # The file is already sorted by count, so we just return the rows
                stats = list(reader)
            return stats
        except Exception as e:
            # Return an error that can be displayed on the dashboard
            return [
                {
                    "event_type": "ERROR_READING_STATS",
                    "count": 1,
                    "last_message": str(e),
                    "level": "ERROR",
                }
            ]

    # -----------------
    def maintenance_watchdog_loop(self):
        """
        A dedicated thread that monitors the maintenance mode timeout.
        If the timeout is reached, it automatically resumes the pipeline.
        """
        logging.info("Maintenance Watchdog started.")
        try:
            while self.running.is_set():
                timeout_target = 0
                with self.maintenance_timeout_lock:
                    if self.maintenance_timeout_until > 0:
                        timeout_target = self.maintenance_timeout_until

                if timeout_target > 0:
                    # A timeout is active, check if it has expired
                    if time.monotonic() >= timeout_target:
                        logging.warning(
                            "Maintenance mode timeout reached. Automatically resuming pipeline..."
                        )
                        self.in_maintenance_mode.clear()
                        # Reset the timer
                        with self.maintenance_timeout_lock:
                            self.maintenance_timeout_until = 0

                        # Only resume if we are within the active schedule
                        if self.within_schedule():
                            self.start_producer_thread()
                        else:
                            logging.info(
                                "Maintenance timeout expired, but system is outside of active schedule. Staying idle."
                            )

                # Sleep for the configured resolution before checking again
                for _ in range(self.cfg["general"]["maintenance_poll_sec"]):
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break
                    time.sleep(1)
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN MAINTENANCE WATCHDOG THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "MAINTENANCE_WATCHDOG_CRASH", str(e))

    # -----------------
    def system_monitor_loop(self):
        # Write the header if the file is new. Otherwise the file stays
        # headerless until the janitor first rotates it, which breaks anything
        # that parses it with csv.DictReader.
        try:
            if not os.path.exists(self.monitor_out_file):
                with open(self.monitor_out_file, "w") as f:
                    f.write(
                        "timestamp,cpu_temp,sensor_temp_cam,sensor_temp_ext,cpu_percent,mem_used_mb,mem_percent,disk_used_mb,disk_percent,gpu_percent,load1,load5,load15,uptime_sec,core_voltage_v\n"
                    )
        except Exception as e:
            logging.error(f"Could not initialise system_monitor.csv header: {e}")

        try:
            while self.running.is_set():
                self.ping_watchdog()
                interval = self.cfg["monitor"].get("interval_sec")
                ts = datetime.now().isoformat()

                # CPU/SoC temperature
                temps = psutil.sensors_temperatures()
                cpu_temp = (
                    temps.get("cpu_thermal", [None])[0].current
                    if "cpu_thermal" in temps
                    else 0
                )
                self.session_stats["peak_cpu_temp"] = max(
                    self.session_stats["peak_cpu_temp"], cpu_temp
                )

                # CPU use
                cpu_percent = psutil.cpu_percent(interval=None)

                # memoriy
                mem = psutil.virtual_memory()
                mem_used_mb = mem.used / (1024 * 1024)

                # ACTIVE OOM RAM PROTECTION
                if mem.percent >= self.cfg["monitor"].get("ram_limit_percent"):
                    if not self.memory_critical.is_set():
                        logging.critical(
                            f"SYSTEM RAM CRITICAL ({mem.percent:.1f}%) - Suspending event recording to prevent OS OOM Kill!"
                        )
                        self.log_health_event(
                            "CRITICAL",
                            "OOM_WARNING",
                            f"RAM at {mem.percent:.1f}%. Event capture suspended.",
                        )
                        self.memory_critical.set()
                elif mem.percent <= self.cfg["monitor"].get("ram_recovery_percent"):
                    if self.memory_critical.is_set():
                        logging.info(
                            f"SYSTEM RAM RECOVERED ({mem.percent:.1f}%) - Resuming normal event recording."
                        )
                        self.memory_critical.clear()

                # disk
                disk = psutil.disk_usage(self.cfg["janitor"].get("monitor_path"))
                self.disk_percent = disk.percent
                self.disk_used_mb = disk.used / (1024 * 1024)

                # GPU usage
                gpu_percent = 0
                try:
                    gpu_stat = subprocess.check_output(
                        ["vcgencmd", "measure_utilisation"], text=True, timeout=self.cfg["general"]["hw_query_timeout_sec"]
                    )
                    parts = dict(p.split("=") for p in gpu_stat.strip().split())
                    if "gpu" in parts:
                        gpu_percent = parts["gpu"].replace("%", "")
                except subprocess.TimeoutExpired:
                    logging.warning(
                        "MonitorThread: vcgencmd did not respond within 3s -- "
                        "skipping GPU stat this cycle rather than blocking the "
                        "thread that also handles OOM protection."
                    )
                except Exception:
                    gpu_percent = 0

                # load average
                load1, load5, load15 = os.getloadavg()

                # system uptime
                uptime_sec = time.time() - psutil.boot_time()

                uv_now, uv_past, throttled_now, throttle_hex = (
                    self.check_throttling_and_voltage()
                )

                # Under-Voltage Logic
                if not getattr(self, "uv_history_logged", False) and uv_past:
                    logging.warning(
                        f"Hardware Power: The system has experienced an Under-Voltage event since booting! (Code: {throttle_hex})"
                    )
                    self.log_health_event(
                        "WARNING",
                        "POWER_UNDER_VOLTAGE",
                        "Historical under-voltage detected.",
                    )
                    self.uv_history_logged = True

                if uv_now:
                    logging.critical(
                        f"HARDWARE ALERT: ACTIVE UNDER-VOLTAGE DETECTED! Power supply is failing! (Code: {throttle_hex})"
                    )
                    self.log_health_event(
                        "CRITICAL",
                        "POWER_UNDER_VOLTAGE_ACTIVE",
                        "Active under-voltage! System may crash!",
                    )

                # Throttling Logic
                was_throttled = getattr(self, "is_currently_throttled", False)

                if throttled_now and not was_throttled:
                    logging.warning(
                        f"Hardware Performance: CPU is NOW THROTTLING! Speed reduced due to heat or power limits. (Code: {throttle_hex})"
                    )
                    self.log_health_event(
                        "WARNING",
                        "CPU_THROTTLED",
                        "System is actively throttling speed.",
                    )
                    self.is_currently_throttled = True

                elif not throttled_now and was_throttled:
                    logging.info(
                        "Hardware Performance: CPU Throttling has ended. Normal speed restored."
                    )
                    self.is_currently_throttled = False

                # Get Sensor Temp for CSV
                # s_temp = self.current_sensor_temp
                # s_temp_str = f"{s_temp:.1f}" if s_temp is not None else ""
                s_temp_cam = self.current_sensor_temp_cam
                s_temp_ext = self.current_sensor_temp_ext
                s_t_cam_str = f"{s_temp_cam:.1f}" if s_temp_cam is not None else ""
                s_t_ext_str = f"{s_temp_ext:.1f}" if s_temp_ext is not None else ""

                core_voltage_v = self.get_core_voltage()

                line = (
                    f"{ts},{cpu_temp},{s_t_cam_str},{s_t_ext_str},{cpu_percent},{mem_used_mb:.1f},{mem.percent},"
                    f"{self.disk_used_mb:.1f},{self.disk_percent},{gpu_percent},{load1:.2f},"
                    f"{load5:.2f},{load15:.2f},{int(uptime_sec)},{core_voltage_v:.4f}\n"
                )
                with open(self.monitor_out_file, "a") as f:
                    f.write(line)

                # Trigger the full variable dump
                if self.cfg["general"].get("debug_level") == 4:
                    self.publish_state_telemetry()

                time.sleep(interval)

        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN SYSTEM MONITOR THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "SYSTEM_MONITOR_CRASH", str(e))

    # -----------------
    def power_monitor_loop(self):
        """
        Monitor a GPIO pin for power loss using the gpiod library.
        If a sustained outage is detected, it initiates
        a graceful shutdown and system halt.
        """

        if not IS_GPIO_AVAILABLE:
            logging.warning("gpiod library not found. Power monitor is disabled.")
            return

        power_pin = self.cfg["power_monitor"].get("pin")
        shutdown_delay_sec = self.cfg["power_monitor"].get("shutdown_delay_sec")
        check_interval = self.cfg["power_monitor"]["check_interval_sec"]

        line_request = None
        try:
            settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.INPUT, bias=gpiod.line.Bias.PULL_UP
            )

            line_request = GPIO_CHIP.request_lines(
                consumer="ryusei-pipeline-power", config={power_pin: settings}
            )
            logging.info(
                f"Power monitor started on GPIO pin {power_pin} (shutdown delay: {shutdown_delay_sec}s)."
            )
        except Exception:
            logging.exception(
                "Failed to initialize GPIO pin %d for power monitor. The thread will now exit.",
                power_pin,
            )
            if line_request:
                line_request.release()
            return

        power_lost = False
        power_loss_time = 0

        try:
            while self.running.is_set():
                self.ping_watchdog()
                try:
                    # Assumes power is OK when the pin is HIGH (1) and LOST when LOW (0).
                    is_power_ok = (
                        line_request.get_value(power_pin) == gpiod.line.Value.ACTIVE
                    )

                    if not is_power_ok and not power_lost:
                        # Power outage
                        power_lost = True
                        power_loss_time = time.monotonic()
                        with self.status_lock:
                            self.power_status = "LOST"
                        logging.warning(
                            "Initial power loss detected. Starting %d second confirmation timer.",
                            shutdown_delay_sec,
                        )
                        self.log_health_event(
                            "WARNING", "POWER_ERROR", "Initial power loss detected"
                        )

                    elif is_power_ok and power_lost:
                        # Power was restored.
                        power_lost = False
                        with self.status_lock:
                            self.power_status = "OK"
                        logging.info("Power was restored. Cancelling shutdown timer.")

                    elif not is_power_ok and power_lost:
                        if time.monotonic() - power_loss_time > shutdown_delay_sec:
                            msg = f"!!! CONFIRMED POWER OUTAGE (lost for >{shutdown_delay_sec} sec) !!!"

                            with self.status_lock:
                                self.power_status = "SHUTTING DOWN"
                            self.perform_system_action("shutdown", reason=msg)
                            break

                    time.sleep(check_interval)
                    
                except Exception:
                    logging.exception(
                        "An error occurred in the power monitor loop. The thread will exit."
                    )
                    break  # Exit the loop on error
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN POWER MONITOR THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "POWER_MONITOR_CRASH", str(e))

        if line_request:
            line_request.release()

    # -----------------
    def led_status_loop(self):
        """
        A dedicated thread that handles the startup signal AND the status indication.
        Consolidating logic here prevents race conditions on the GPIO line.
        """
        if not self.led_line:
            return

        logging.info("LED Status thread started. Waiting for System Ready...")

        # 1. Wait for the main thread to signal that initialization is done
        if not self.system_ready.wait(timeout=30):
            logging.warning("LED: Timed out waiting for System Ready signal.")

        # 2. Perform Startup Signal (Triple Blink)
        # This is safe here because this thread has exclusive control of the line
        try:
            self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
            time.sleep(0.5)
            for _ in range(3):
                self.led_line.set_value(self.led_pin, gpiod.line.Value.ACTIVE)
                time.sleep(0.15)
                self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
                time.sleep(0.15)
            logging.info("LED startup signal complete.")
        except Exception as e:
            logging.error(f"LED startup signal failed: {e}")

        logging.info("LED entering continuous status loop.")

        # 3. Enter the main status indication loop
        try:
            while self.running.is_set():
                self.ping_watchdog()
                # Default: A slow "breathing" pulse for idle
                on_time, off_time = 0.1, 2.5

                try:
                    # Determine pattern based on system state priority
                    if (
                        self.emergency_stop_active.is_set()
                        or self.camera_fatal_error.is_set()
                        or self.storage.halt_event.is_set()
                    ):
                        on_time, off_time = 0.1, 0.1  # Rapid flicker for ERROR

                    elif self.event_in_progress.is_set():
                        on_time, off_time = 0.25, 0.25  # Fast blink for EVENT

                    elif self.producer_thread_active():
                        on_time, off_time = 1.0, 1.0  # Slow pulse for CAPTURING

                    # Execute the pattern
                    self.led_line.set_value(self.led_pin, gpiod.line.Value.ACTIVE)

                    # Check running flag frequently during sleep to allow fast shutdown
                    expiry = time.monotonic() + on_time
                    while self.running.is_set() and time.monotonic() < expiry:
                        time.sleep(0.1)

                    if not self.running.is_set():
                        break

                    self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
                    expiry = time.monotonic() + off_time
                    while self.running.is_set() and time.monotonic() < expiry:
                        time.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error in LED status loop: {e}")
                    time.sleep(5)
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN LED STATUS THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "LED_STATUS_CRASH", str(e))

    # -----------------
    def _execute_post_shutdown_script(self):
        """
        Executes a user-defined external python script (with arguments) after the pipeline shuts down.
        Blocks the OS from halting until the parser finishes.
        """
        cmd_string = self.cfg.get("general", {}).get("post_shutdown_script")

        if cmd_string:
            try:
                # Split the string into a list: ['/path/to/script.py', '--arg1', 'val1']
                cmd_parts = shlex.split(cmd_string)
                if not cmd_parts:
                    return

                # The first item is always the physical script file
                script_path = cmd_parts[0]

                if os.path.exists(script_path):
                    logging.info("=" * 60)
                    logging.info(
                        f"Triggering external Post-Shutdown script: {cmd_string}"
                    )
                    try:
                        # Build the command: [python3, script.py, arg1, arg2...]
                        full_cmd = [sys.executable] + cmd_parts
                        subprocess.run(full_cmd, check=False)
                        logging.info(
                            "External script execution completed successfully."
                        )
                    except Exception as e:
                        logging.error(f"Failed to execute post-shutdown script: {e}")
                    logging.info("=" * 60)
                else:
                    logging.warning(
                        f"Post-shutdown script not found at '{script_path}'. Skipping execution."
                    )
            except Exception as e:
                logging.error(f"Error parsing post_shutdown_script command string: {e}")

    # -----------------
    def _validate_output_directories(self):
        """Verifies that all required output directories are writable."""
        dirs_to_check = [
            self.events_out_dir,
            self.timelapse_out_dir,
            self.daylight_out_dir,
            self.calibration_out_dir,
            self.general_log_dir,
            self.rejected_out_dir,
            self.debug_out_dir,
            self.ai_filtered_out_dir,
            self.logs_archive,
            self.flat_field_dir,
        ]

        for d in dirs_to_check:
            try:
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)

                if not os.access(d, os.W_OK):
                    print(f"Directory {d} is NOT writable! Check permissions.")
                else:
                    print(f"Validated directory: {d} is writable.")
            except Exception as e:
                print(f"Error validating directory {d}: {e}")

    # -----------------
    def within_schedule(self):
        start_str = self.cfg["general"].get("start_time")
        end_str = self.cfg["general"].get("end_time")

        # If either time is cleared/None, the system runs 24/7 without stopping
        if not start_str or not end_str:
            return True

        start_h, start_m = map(int, start_str.split(":"))
        end_h, end_m = map(int, end_str.split(":"))
        start = dtime(start_h, start_m)
        end = dtime(end_h, end_m)
        current = datetime.now().time()

        if start <= end:
            return start <= current <= end
        else:
            return current >= start or current <= end

    # -----------------
    @classmethod
    def load_and_validate_config(cls, path=None, merge=True):
        """
        Loads a config from a file path, checks for typos/unknown keys,
        merges with defaults (unless merge=False), and validates using Pydantic.
        """
        # If the caller only wants the raw, unmerged file, use the legacy loader.
        if not merge:
            return cls.legacy_load_config(path, merge=False)

        user_cfg = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    user_cfg = json.load(f)
            except json.JSONDecodeError as e:
                print(
                    f"FATAL: Error decoding JSON from config file '{path}': {e}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Check for Typos / Unknown Keys
            unknown_keys = cls._validate_keys_exist(user_cfg, DEFAULTS)
            if unknown_keys:
                print("=" * 60, file=sys.stderr)
                print(
                    "FATAL: CONFIGURATION ERROR - UNKNOWN KEYS FOUND", file=sys.stderr
                )
                print(
                    "The following keys in your config file are not recognized:",
                    file=sys.stderr,
                )
                for k in unknown_keys:
                    print(f"  - {k}", file=sys.stderr)
                print(
                    "\nPlease check for spelling errors or obsolete settings.",
                    file=sys.stderr,
                )
                print("=" * 60, file=sys.stderr)
                sys.exit(1)
            # -------------------------------------------

        try:
            # Pydantic's model_validate will automatically use the defaults defined
            # in the schema for any keys missing from user_cfg. This is the merge step.
            validated_model = MainConfig.model_validate(user_cfg)
            validated_dict = validated_model.model_dump()

            # Only print this message on initial startup
            if path:
                print(
                    "Configuration loaded, structure verified, and validated successfully."
                )
            return validated_dict
        except ValidationError as e:  # Catches Pydantic's ValidationError
            print("=" * 60, file=sys.stderr)
            print("FATAL: CONFIGURATION VALIDATION FAILED", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print(f"Error details:\n{e}", file=sys.stderr)
            print("\nPlease correct your config file and try again.", file=sys.stderr)
            sys.exit(1)

    # -----------------
    @classmethod
    def _validate_keys_exist(cls, user_cfg, default_cfg, parent_path=""):
        """
        Recursively checks if keys in user_cfg exist in default_cfg.
        Returns a list of unknown key paths.
        """
        unknown_keys = []
        for key, value in user_cfg.items():
            current_path = f"{parent_path}.{key}" if parent_path else key

            if key not in default_cfg:
                unknown_keys.append(current_path)
            elif isinstance(value, dict) and isinstance(default_cfg.get(key), dict):
                # Recurse into nested dictionaries
                unknown_keys.extend(
                    cls._validate_keys_exist(value, default_cfg[key], current_path)
                )

        return unknown_keys

    # -----------------
    @classmethod
    def legacy_load_config(cls, path=None, merge=True):
        # This is the old load_config, kept as a fallback.
        if merge:
            cfg = json.loads(json.dumps(DEFAULTS))  # copy.deepcopy(DEFAULTS)
        else:
            cfg = {}
        if path and os.path.exists(path):
            with open(path, "r") as f:
                user_cfg = json.load(f)
            if merge:
                for k, v in user_cfg.items():
                    if isinstance(v, dict) and k in cfg:
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
            else:
                cfg = user_cfg.copy()
        return cfg

# --------------------
# MAIN EXECUTION BLOCK
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ryusei Pipeline Pi5")
    # parser.add_argument("--config", type=str, help="Path to config JSON file", default="config.json")
    parser.add_argument(
        "--config", type=str, help="Path to config JSON file", default=None
    )
    parser.add_argument(
        "--create_config",
        action="store_true",
        help="Create a default config.json file and exit.",
    )
    parser.add_argument(
        "--dark_frame", action="store_true", help="Create the dark frame and exit."
    )
    parser.add_argument(
        "--bias_frame",
        action="store_true",
        help="Capture 50 frames at min exposure for Bias.",
    )
    parser.add_argument(
        "--flat_frame",
        action="store_true",
        help="Capture 50 frames (auto-exposed) for Flat.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode with a fake camera feed.",
    )
    parser.add_argument(
        "--force_archive",
        action="store_true",
        help="Force archiving of all log and CSV files on startup.",
    )
    parser.add_argument(
        "--half_power",
        action="store_true",
        help="Throttle CPU usage to 50% to prevent transient power spike.",
    )
    args = parser.parse_args()

    # Helper to capture calibration stacks
    def capture_calib_stack(pipeline, count, desc, out_name):
        print(f"Capturing {count} {desc} frames...")
        frames = []

        # The camera constructor already drained the frames in flight with the old
        # settings (flush); every frame read from here on carries the calibration settings.
        for i in range(count):
            # read() returns (main, raw). The old code treated the tuple as the image
            # and crashed on .astype() at the first frame.
            frame, _raw = pipeline.camera.read()
            if frame is None:
                print(f"\n  WARNING: Frame {i+1} failed", file=sys.stderr)
                continue
            frames.append(frame)
            print(f"  Progress: {len(frames)}/{count}", end="\r")

        if frames:
            # Robust combine (per-pixel sigma-clipped mean, RAM-tiled) instead of a plain
            # mean: one cosmic ray or satellite in a single frame otherwise becomes a
            # permanent false hot pixel in the master frame.
            h, w = frames[0].shape[:2]
            n = len(frames)
            stacked = pipeline._combine_mean_stack(frames, [1.0] * n, [None] * n, h, w)
            avg = np.clip(stacked, 0, 255).astype(np.uint8)
            cv2.imwrite(out_name, avg, [int(cv2.IMWRITE_PNG_COMPRESSION), pipeline.cfg["general"]["png_compression"]])
            print(f"\nSUCCESS: Saved {desc} frame to {out_name}")
        else:
            print(f"\nERROR: Failed to acquire any frames for {desc}.")

    # Calibration Logic
    if args.dark_frame or args.bias_frame or args.flat_frame:
        config = Pipeline.load_and_validate_config(args.config)

        if args.bias_frame:
            print("--- Master Bias Creation ---")
            print("Cover the lens completely (Pitch black).")
            input("Press Enter to start...")
            # Bias = the sensor's MINIMUM exposure and gain. Requesting 1 us / gain 0
            # lets CameraCaptureClass.achievable() clamp to the sensor's reported
            # minimums, whatever the sensor (a fixed 100 us is not the minimum of most).
            config["capture"]["exposure_us"] = 1
            config["capture"]["gain"] = 0.0
            desc, out_name = "Bias", "master_bias.png"

        elif args.dark_frame:
            print("--- Master Dark Creation ---")
            print("Cover the lens completely (Pitch black).")
            input("Press Enter to start...")
            # Darks use the current production settings for exposure/gain
            desc, out_name = "Dark", "master_dark.png"

        elif args.flat_frame:
            print("--- Master Flat Creation ---")
            print("Point camera at an evenly illuminated surface (white wall/t-shirt).")
            input("Press Enter to start...")
            desc, out_name = "Flat", "master_flat.png"

        app = Pipeline(cfg=config, simulate=args.simulate)
        try:
            capture_calib_stack(app, config["detection"]["calibration_library_frames"], desc, out_name)
        finally:
            if app.camera:
                app.camera.release()
        sys.exit(0)

    # Handle --create-config argument
    if args.create_config:
        config_path = "config.json"  # Define a default name
        if os.path.exists(config_path):
            print(
                f"Configuration file '{config_path}' already exists. No action taken."
            )
        else:
            try:
                # Use the static method to ensure consistency
                Pipeline.save_json_atomic(config_path, DEFAULTS)
                print(
                    f"Successfully created default configuration file at '{config_path}'."
                )
                print("Please review and edit this file before running the pipeline.")
            except IOError as e:
                print(
                    f"ERROR: Could not write configuration file to '{config_path}': {e}",
                    file=sys.stderr,
                )
        sys.exit(0)  # Exit after handling config creation

    if args.config:
        if not os.path.exists(args.config):
            print(
                f"ERROR: Configuration file not found at '{args.config}'.",
                file=sys.stderr,
            )
            print(
                "Please create one or run with --create-config to generate a default file.",
                file=sys.stderr,
            )
            sys.exit(1)
        config_to_load = args.config
        target_config_path = args.config
    else:
        config_to_load = None
        # If the user later saves via Dashboard, it creates this file.
        target_config_path = "config.json"

    # 1. Load the configuration from the file.
    config = Pipeline.load_and_validate_config(config_to_load)

    # 2. Create an instance of the main application class.
    app = Pipeline(
        cfg=config,
        config_path=target_config_path,
        force_archive=args.force_archive,
        half_power=args.half_power,
        simulate=args.simulate,
    )

    # 3. Run the application.
    app.run()
