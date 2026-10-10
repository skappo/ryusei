"""
dashboard_control.py

Capture pause/resume, system-time, restart/reboot/OTA and the test-shot
routes for the Flask dashboard. Split out of dashboard.py -- method bodies
are unchanged, only their physical location (same "pure move" approach used
for the earlier mixin extractions). Registered into the shared Flask app
by dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
this is a plain function), kept under that name so every `self.xxx` call
copied from the original nested route(s) needed no rewriting.
"""
import logging
import subprocess
import threading
import time
import html
from datetime import datetime

from flask import redirect, request, url_for

from config_models import MainConfig
from dashboard_theme import render_page


def register_control_routes(app, self, DASH_TOKEN, require_token, EDITABLE_PARAMS):
    """System-time, pause/resume, test-shot, restart/reboot/OTA and satellite-catalogue routes."""

    # START: SYSTEM TIME PAGE
    @app.route("/system_time", methods=["GET", "POST"])
    @require_token
    def system_time():
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""

        if request.method == "POST":
            try:
                new_date = request.form.get("new_date")
                new_time = request.form.get("new_time")
                if not new_date or not new_time:
                    raise ValueError("Date and time fields cannot be empty.")

                # Validate the format using datetime.strptime
                # This raises a ValueError if the string is malformed or not a real date.
                try:
                    # Format with seconds: "YYYY-MM-DD HH:MM:SS"
                    parsed_dt = datetime.strptime(
                        f"{new_date.strip()} {new_time.strip()}",
                        "%Y-%m-%d %H:%M:%S",
                    )
                except ValueError:
                    # Fallback for browsers that omit seconds (e.g. "YYYY-MM-DD HH:MM")
                    parsed_dt = datetime.strptime(
                        f"{new_date.strip()} {new_time.strip()}", "%Y-%m-%d %H:%M"
                    )

                # Rebuild the string directly from the validated Python object
                datetime_str = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")

                logging.warning(
                    f"Dashboard user is attempting to set system time to: {datetime_str}"
                )

                # Securely execute the command
                cmd = ["sudo", "date", "-s", datetime_str]
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, timeout=10
                )

                logging.info(
                    f"System time successfully set. Command output: {result.stdout.strip()}"
                )
                return redirect(
                    url_for("system_time", token=DASH_TOKEN, success="true")
                )

            except (
                ValueError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as e:
                error_message = str(e)
                if hasattr(e, "stderr") and getattr(e, "stderr"):
                    error_message = e.stderr.strip()
                logging.error(f"Failed to set system time: {error_message}")
                # URL-encode the error message for safe transport
                from urllib.parse import quote

                return redirect(
                    url_for(
                        "system_time", token=DASH_TOKEN, error=quote(error_message)
                    )
                )

        # Handle GET request (display the page)
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        loc = self.cfg["general"].get("location") or ""
        feedback_html = ""
        if request.args.get("success"):
            feedback_html = '<div class="banner ok"><b>System time updated.</b></div>'
        elif request.args.get("error"):
            error_msg = html.escape(request.args.get("error"))
            feedback_html = (f'<div class="banner err"><b>Could not set the time:</b> {error_msg}<br>'
                             f'<small>The service user needs sudo rights for /bin/date.</small></div>')

        body = f"""
        <h1 class="page-title">System time</h1>
        <p class="page-intro">Correct the clock when the station booted without network time (NTP).</p>
        {feedback_html}
        <section class="panel">
          <h2>Current system time</h2>
          <p style="font-size:26px;margin:0 0 4px;font-weight:600">{current_time}</p>
          <p class="muted" style="margin:0">{current_date}, local time</p>
        </section>
        <section class="panel">
          <h2>Set a new time</h2>
          <form method="post" action="/system_time{token_param}"
                onsubmit="return confirm('Change the system time? This affects log and file timestamps.');">
            <div class="field"><label for="new_date">Date</label>
              <input type="date" id="new_date" name="new_date" value="{current_date}"></div>
            <div class="field"><label for="new_time">Time (local)</label>
              <input type="time" id="new_time" name="new_time" value="{current_time}" step="1"></div>
            <button type="submit" class="btn primary">Set time</button>
          </form>
          <p class="note">Changing the clock mid-session shifts file and log timestamps; prefer doing it while idle.</p>
        </section>"""
        return render_page("System time", body, token_param=token_param, current="/system_time",
                           location=loc, narrow=True)

    # END: SYSTEM TIME PAGE

    # Pause the acquisition.
    @app.route("/api/pause")
    @require_token
    def pause_capture():
        logging.warning(
            "Pause command received from dashboard. Stopping worker threads."
        )
        self.stop_producer_thread()
        self.in_maintenance_mode.set()
        with self.maintenance_timeout_lock:
            self.maintenance_timeout_until = (
                time.monotonic() + self.maintenance_timeout_duration_sec
            )
        return redirect(url_for("dashboard", token=DASH_TOKEN))

    # Resume the acquisition.
    @app.route("/api/resume")
    @require_token
    def resume_capture():
        logging.warning(
            "Resume command received from dashboard. Restarting worker threads."
        )
        #            self.start_producer_thread()
        self.in_maintenance_mode.clear()
        with self.maintenance_timeout_lock:
            self.maintenance_timeout_until = 0

        if self.within_schedule():
            logging.warning(
                "Resume command received from dashboard. Restarting worker threads as we are within schedule."
            )
            with self.camera_lock:
                try:
                    ensure = getattr(self.camera, "ensure_streaming", None)
                    if ensure is not None:
                        ensure()  # idempotent: starts only a stopped sensor
                    logging.info("Hardware Start: Success.")
                    test_frame, _ = self.camera.read()
                    if test_frame is None:
                        raise RuntimeError(
                            "Camera started but returned empty frame during validation."
                        )
                    logging.info(
                        "Hardware Validation: Success (Sensor responsive)."
                    )
                except Exception as e:
                    # HARDWARE FAILURE PATH
                    error_msg = (
                        f"Hardware Failure: Could not restart sensor. Reason: {e}"
                    )
                    logging.critical(error_msg)

                    # Log to the health system so it appears on the dashboard
                    self.log_health_event(
                        "CRITICAL", "HARDWARE_RESUME_FAIL", str(e)
                    )

                    # Force Maintenance Mode back ON to prevent thread loops from spinning
                    self.in_maintenance_mode.set()
                    self.camera_fatal_error.set()  # Triggers the scheduler logic to attempt a full restart later

                    # Tell the user exactly what happened
                    return redirect(
                        url_for(
                            "dashboard",
                            token=DASH_TOKEN,
                            resume_status="fail",
                            error=str(e),
                        )
                    )
            self.capture_stable.clear()  # Force SkyMonitor to run a fresh calibration
            self.request_sky_analysis_trigger.set()
            self.start_producer_thread()
            return redirect(
                url_for("dashboard", token=DASH_TOKEN, resume_status="success")
            )
        else:
            logging.warning(
                "Resume command received, but system is outside of active schedule. "
                "Staying idle and letting the scheduler take over at the next active window."
            )
            # Redirect with an informational message
            return redirect(
                url_for("dashboard", token=DASH_TOKEN, resume_status="ignored")
            )

    # Show EDITABLE_PARAMS, save and test for image test
    @app.route("/api/save_and_test", methods=["POST"])
    @require_token
    def save_and_test():
        logging.info("Dashboard: Received /api/save_and_test request.")
        if self.producer_thread_active():
            logging.warning(
                "Dashboard: Capture threads ARE active, skipping test shot trigger."
            )
            return redirect(url_for("dashboard", token=DASH_TOKEN))

        try:
            # Stage 1: Clear old status and save new config
            with self.status_lock:
                self.last_calibration_error = None
                self.last_calibration_image_path = None

            with self.maintenance_timeout_lock:
                self.maintenance_timeout_until = (
                    time.monotonic() + self.maintenance_timeout_duration_sec
                )

            logging.info("Dashboard: Saving new configuration from dashboard.")

            changes_from_form = {}
            for section, params in EDITABLE_PARAMS.items():
                for param in params:
                    form_key = f"{section}_{param}"
                    if form_key in request.form:
                        if section not in changes_from_form:
                            changes_from_form[section] = {}
                        changes_from_form[section][param] = request.form[form_key]

            try:
                validated_changes = MainConfig.model_validate(
                    changes_from_form
                ).model_dump(exclude_unset=True)
            except Exception as e:
                logging.error(
                    f"Dashboard: New configuration failed validation: {e}"
                )
                return redirect(url_for("dashboard", token=DASH_TOKEN))

            config_on_disk = self.legacy_load_config(self.config_path, merge=False)
            if not isinstance(config_on_disk, dict):
                config_on_disk = {}

            def recursive_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = recursive_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            config_to_save = recursive_update(config_on_disk, validated_changes)
            self.save_json_atomic(self.config_path, config_to_save)

            # Stage 2: Trigger and WAIT for the reloader to finish
            logging.info(
                "Dashboard: Triggering config reloader and waiting for completion..."
            )
            self.config_reloaded_ack.clear()  # Ensure ack is cleared before signaling
            self.reload_config_signal.set()

            # Wait for a longer, more realistic timeout.
            if not self.config_reloaded_ack.wait(timeout=20.0):
                logging.error(
                    "Dashboard: Timed out waiting for config reloader ack."
                )
                with self.status_lock:
                    self.last_calibration_error = (
                        "Timed out waiting for config reload."
                    )
                return redirect(url_for("dashboard", token=DASH_TOKEN))

            self.config_reloaded_ack.clear()
            logging.info("Dashboard: Config reloader acknowledged completion.")

            # Stage 3: Start calibration process ONLY after reload is complete
            if self.is_calibrating.is_set():
                logging.warning("Dashboard: Calibration is already in progress.")
            else:
                logging.info(
                    "Dashboard: Setting calibration mode and starting producer thread..."
                )
                self.is_calibrating.set()
                self.start_producer_thread()

        except Exception as e:
            logging.exception(
                "A critical error occurred while saving the configuration: %s", e
            )
            with self.status_lock:
                self.last_calibration_error = f"An unexpected error occurred: {e}"

        return redirect(url_for("dashboard", token=DASH_TOKEN))

    @app.route("/api/restart_service", methods=["POST"])
    @require_token
    def restart_service():
        logging.warning("User-initiated restart command received from dashboard.")

        # Use a separate thread to give Flask time to return a response to the user's browser
        # before the main process exits.
        def delayed_restart():
            time.sleep(1)
            self.perform_system_action(
                "exit", reason="User-initiated restart from dashboard"
            )

        threading.Thread(target=delayed_restart).start()

        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
        refresh_url = url_for("dashboard", _external=True) + token_param
        return render_page(
            "Restarting", f"""<section class="panel status-card">
              <h1 class="warn">Restarting the program</h1>
              <p>The pipeline is restarting; this takes a few seconds.</p>
              <p class="muted">You will be taken back to the dashboard automatically.
                <a href="{refresh_url}">Go now</a></p></section>""",
            token_param=token_param, location=self.cfg["general"].get("location") or "",
            head_extra=f'<meta http-equiv="refresh" content="5;url={refresh_url}">')

    @app.route("/api/reboot_system", methods=["POST"])
    @require_token
    def reboot_system():
        logging.warning(
            "User-initiated SYSTEM REBOOT command received from dashboard."
        )

        def delayed_reboot():
            time.sleep(2)
            # This calls the existing system logic for OS reboot
            self.perform_system_action(
                "reboot", reason="User-initiated reboot from dashboard"
            )

        threading.Thread(target=delayed_reboot).start()

        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
        return render_page(
            "Rebooting", """<section class="panel status-card">
              <h1 class="err">Rebooting the system</h1>
              <p>The Raspberry Pi is restarting. The dashboard is back in about a minute.</p>
              <p class="muted">Reload this page after that.</p></section>""",
            token_param=token_param, location=self.cfg["general"].get("location") or "")
    

    # 3. PDF Archive Manager
