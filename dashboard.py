"""
dashboard.py

Flask web dashboard core: DashboardClass.dashboard_loop() (builds the Flask app,
the token-auth decorator and the directories/params every page needs, then
hands the app to each page module to register its routes), and the
debug/telemetry publishing methods. Composed into Pipeline via multiple
inheritance, same as every other *Class mixin.

The routes themselves live alongside this file, split by page/concern so no
single file is a 2000-line wall of nested Flask routes:
  dashboard_theme.py     THEME_ROOT / BASE_CSS / render_page (leaf module, no cycle)
  dashboard_control.py   system-time, pause/resume, test-shot, restart/reboot/OTA
  dashboard_settings.py  /settings (view + save)
  dashboard_files.py     static file serving + the Files manager page
  dashboard_sessions.py  /sessions, /night_archive (archived-night browsing)
  dashboard_stats.py     /stats + its JSON APIs
  dashboard_locations.py /locations (observing-site list + switch)
  dashboard_main.py      "/" -- the main status/control page

Split out of a single dashboard_mixin.py by a scripted, line-range extraction
(see old_ver/pre_dashboard_split/ for the pre-split original) -- every moved
route's body is byte-for-byte unchanged, only its physical location moved
and its indentation reduced by one level (nested in a register_*_routes()
function instead of in dashboard_loop() directly).

Imports here are exactly what THIS file actually uses -- verified by grep
after the split. Flask itself is imported locally inside dashboard_loop (a
deferred, optional-dependency style import), same as the page modules'
own flask imports, which are reached only after that succeeds.
"""
import os
import json
import logging
import threading
import hmac
import socket
import queue
import cv2
import numpy as np
import zmq
from enum import Enum
from datetime import datetime

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class DashboardClass:
    """Flask dashboard and telemetry/debug publishing, composed into Pipeline.

    The routes themselves live in dashboard_control.py, dashboard_settings.py,
    dashboard_files.py, dashboard_sessions.py, dashboard_stats.py,
    dashboard_locations.py and dashboard_main.py -- dashboard_loop() just
    builds the shared Flask app, the auth decorator and the directories/params
    every page needs, then hands the app to each module's register_*_routes()
    to add its routes.
    """

    def dashboard_loop(self):
        """A dedicated thread to run a Flask web server for status and control."""
        dash_cfg = self.cfg.get("dashboard")

        try:
            from functools import wraps

            from flask import Flask, abort, request

            # Deferred like the Flask import above: only imported once Flask
            # itself is confirmed present, so these modules' own `from flask
            # import ...` lines never run (and can't raise) when Flask isn't
            # installed -- the except below still catches it either way,
            # whichever import actually fails.
            from dashboard_control import register_control_routes
            from dashboard_settings import register_settings_routes
            from dashboard_files import register_file_routes
            from dashboard_sessions import register_session_routes
            from dashboard_stats import register_stats_routes
            from dashboard_locations import register_location_routes
            from dashboard_main import register_main_routes
        except ImportError:
            logging.error(
                "Flask library not found. 'pip3 install Flask'. Dashboard disabled."
            )
            return

        # Read the token from the environment.
        DASH_TOKEN = os.environ.get("RYUSEI_DASH_TOKEN")

        # If the token is not set or is empty, refuse to start.
        if not DASH_TOKEN:
            logging.critical("=" * 60)
            logging.critical("!!! SECURITY ALERT: RYUSEI_DASH_TOKEN is NOT SET. !!!")
            logging.critical(
                "!!! All access to the dashboard will be DENIED as a security precaution. !!!"
            )
            logging.critical(
                "!!! To enable the dashboard, set this environment variable to a long, random string. !!!"
            )
            logging.critical("=" * 60)
        else:
            logging.info("Dashboard authentication is ENABLED. Token has been loaded.")

        def require_token(f):
            @wraps(f)
            def inner(*args, **kwargs):
                # Rule 1: If the DASH_TOKEN was never configured on the server, deny all access.
                if not DASH_TOKEN:
                    abort(401)  # Access Denied

                # Rule 2: If the token IS configured, the user must provide the correct one.
                token_header = request.headers.get("X-Auth-Token")
                token_query = request.args.get("token")

                if (token_query and hmac.compare_digest(token_query, DASH_TOKEN)) or (
                    token_header and hmac.compare_digest(token_header, DASH_TOKEN)
                ):
                    return f(*args, **kwargs)  # Access Granted

                # If we reach here, the token was wrong or missing.
                logging.warning(
                    f"Dashboard: Unauthorized access attempt from {request.remote_addr}."
                )
                abort(401)  # Access Denied

            return inner

        app = Flask(__name__, static_folder=None)

        # PARAMETER
        EDITABLE_PARAMS = {
            "capture": [
                "exposure_us",
                "gain",
                "red_gain",
                "blue_gain",
                "moon_normal_penalty",
                "moon_max_penalty",
            ],
            "detection": ["target_adaptation_time"],
            "timelapse": ["nighttime_stack_N", "cloudy_interval_sec"],
        }
        TIMELAPSE_DIR = os.path.abspath(self.timelapse_out_dir)
        EVENTS_DIR = os.path.abspath(self.events_out_dir)
        ALLOWED_DIRS = [TIMELAPSE_DIR, EVENTS_DIR]

        register_control_routes(app, self, DASH_TOKEN, require_token, EDITABLE_PARAMS)
        register_settings_routes(app, self, DASH_TOKEN, require_token)
        register_file_routes(app, self, DASH_TOKEN, require_token, ALLOWED_DIRS, TIMELAPSE_DIR, EVENTS_DIR)
        register_session_routes(app, self, DASH_TOKEN, require_token)
        register_stats_routes(app, self, DASH_TOKEN, require_token)
        register_location_routes(app, self, DASH_TOKEN, require_token)
        register_main_routes(app, self, DASH_TOKEN, require_token, EDITABLE_PARAMS, TIMELAPSE_DIR)

        host = dash_cfg.get("host", "0.0.0.0")
        port = dash_cfg["port"]
        logging.info(
            "Starting status dashboard. Access at http://%s:%d or http://<IP_ADDRESS>:%d",
            socket.gethostname(),
            port,
            port,
        )
        app.logger.disabled = True
        logging.getLogger("werkzeug").disabled = True

        try:
            app.run(host=host, port=port)
        except OSError as e:
            logging.critical("CRITICAL FATAL ERROR IN DASHBOARD THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "DASHBOARD_CRASH", str(e))



    def publish_debug(self, msg_type, payload, image=None):
        """
        Standardized ZMQ publisher.
        msg_type: 'stats', 'event', 'alignment'
        payload: dict
        image: optional numpy array
        """
        # Level 0 = Off
        if self.cfg["general"]["debug_level"] < 1:
            return

        # Level 1 = Stats only (no images over ZMQ unless level 3)
        if image is not None and self.cfg["general"]["debug_level"] < 3:
            image = None

        try:
            # Add common metadata
            payload["type"] = msg_type
            payload["ts_local"] = datetime.now().isoformat()

            # Serialize JSON
            json_header = json.dumps(payload, default=str).encode("utf-8")

            if image is not None:
                ret, jpg = cv2.imencode(
                    ".jpg",
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.cfg["general"]["jpg_quality_zmq"]],
                )
                if ret:
                    # Send multipart: [Topic, JSON, ImageBytes]
                    self.debug_pub.send_multipart(
                        [b"debug", json_header, jpg.tobytes()], flags=zmq.NOBLOCK
                    )
            else:
                self.debug_pub.send_multipart(
                    [b"debug", json_header], flags=zmq.NOBLOCK
                )
        except Exception as e:
            # Never crash the pipeline due to a debug error
            logging.error(f"Debug Publish Error: {e}")

    # -----------------
    def publish_state_telemetry(self):
        """Sends the full internal variable state via ZMQ."""
        # Only run if debug is enabled to save CPU
        if self.cfg["general"]["debug_level"] < 1:
            return

        try:
            data = self.get_telemetry_data()
            payload = {
                "type": "state_telemetry",
                "ts_local": datetime.now().isoformat(),
                "variables": data,
            }
            # Serialize and send (No image part)
            json_header = json.dumps(payload, default=str).encode("utf-8")
            self.debug_pub.send_multipart([b"debug", json_header], flags=zmq.NOBLOCK)
        except Exception as e:
            logging.error(f"Telemetry Publish Error: {e}")

    # -----------------
    def get_telemetry_data(self):
        """Filters self.__dict__ for logic variables, excluding all images and hardware objects."""
        telemetry = {}

        # 1. Define objects that should be ignored or summarized
        # We explicitly exclude frames/arrays to keep the packet small
        exclude_types = (
            np.ndarray,
            cv2.VideoCapture,
            zmq.Socket,
            threading.Lock,
            Picamera2,
        )
        exclude_keys = [
            "background",
            "master_dark",
            "flat_multiplier",
            "camera",
            "debug_pub",
        ]

        for key, value in self.__dict__.items():
            if key in exclude_keys or isinstance(value, exclude_types):
                continue

            # Convert Complex Objects to Serializables

            # Thread Status
            if isinstance(value, threading.Thread):
                telemetry[key] = f"Thread(Alive={value.is_alive()})"

            # Event Status (Booleans)
            elif isinstance(value, threading.Event):
                telemetry[key] = value.is_set()

            # Queue Status (Current Depth)
            elif isinstance(value, queue.Queue):
                telemetry[key] = f"Queue(Size={value.qsize()})"

            # Enums (SkyConditionStatus)
            elif isinstance(value, Enum):
                telemetry[key] = value.name

            # Standard Primitives
            elif isinstance(value, (int, float, str, bool, type(None), list, dict)):
                telemetry[key] = value

            # Everything else to string
            else:
                telemetry[key] = str(value)

        return telemetry

    # -----------------
    # 10. UTILITIES & STATIC METHODS
    # -----------------
