"""
system_services_mixin.py

    sftp_dispatcher_loop
    sweep_and_enqueue_unuploaded
    heartbeat_loop
    ntp_sync_loop
    config_reloader_loop
    
Heartbeat, NTP sync, config hot-reload, and SFTP dispatch/backlog-sweep
methods for the Ryusei pipeline, extracted from the main Pipeline class
as a pure move -- method bodies are unchanged, only their physical
location. Composed into Pipeline via multiple inheritance.

Two non-contiguous chunks in the original file: sftp_dispatcher_loop +
sweep_and_enqueue_unuploaded (contiguous with each other), and separately
heartbeat_loop + ntp_sync_loop + config_reloader_loop (contiguous with
each other). Confirmed via an explicit interleaving check before
extraction -- maintenance_watchdog_loop, system_monitor_loop, and
power_monitor_loop sit between the two groups in the original file and
are deliberately NOT included here.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls
self.ping_watchdog (ConcurrencyClass), self.load_and_validate_config (a
classmethod not yet extracted), self._apply_resolution_aware_tuning,
self._enforce_exposure_safety_limits, self.log_health_event,
self.perform_system_action, self.stop_producer_thread -- all still
directly in Pipeline's own body. Must be provided somewhere in the
composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import json
import subprocess
import queue
import logging
import time
import ntplib
import requests
from datetime import datetime

from capture import CaptureClass
from camera_capture import CameraCaptureClass


class SystemServicesClass:
    """Heartbeat, NTP sync, config reload, and SFTP dispatch, composed into Pipeline."""

    def sftp_dispatcher_loop(self):
        """
        A smart dispatcher that acts as a persistent, disk-backed queue for the SFTP uploader.
        Implements the logic to use a disk file as a backlog when the memory queue is full.
        """
        if not self.sftp_uploader:
            logging.info("SFTP Dispatcher: Uploader is disabled. Thread will exit.")
            return

        logging.info(
            f"SFTP Dispatcher started. Using backlog file: {self.backlog_file_path}"
        )

        try:
            while self.running.is_set():
                try:
                    # Check if we can flush from the disk backlog first.
                    upload_q = self.sftp_uploader.upload_q

                    if upload_q.qsize() < upload_q.maxsize and os.path.exists(
                        self.backlog_file_path
                    ):
                        logging.info(
                            "SFTP Dispatcher: Upload queue has space. Flushing from disk backlog."
                        )
                        temp_backlog = []
                        with open(self.backlog_file_path, "r") as f:
                            temp_backlog = [line.strip() for line in f if line.strip()]

                        files_flushed = 0
                        while temp_backlog and not upload_q.full():
                            file_path = temp_backlog.pop(0)
                            self.sftp_uploader.upload_q.put(file_path)
                            files_flushed += 1

                        # Rewrite the backlog file with the remaining items
                        with open(self.backlog_file_path, "w") as f:
                            for item in temp_backlog:
                                f.write(item + "\n")

                        if not temp_backlog:
                            os.remove(
                                self.backlog_file_path
                            )  # Clean up the file if it's empty

                        logging.info(
                            f"SFTP Dispatcher: Flushed {files_flushed} items from disk to memory queue."
                        )

                    # Process new items from the main dispatch queue.
                    try:
                        # Use a timeout to allow the loop to re-check the backlog file periodically
                        new_item = self.sftp_dispatch_q.get(timeout=5.0)

                        if new_item == "STATUS":
                            self.ping_watchdog()
                            continue

                        # Decide where to put the new item
                        if not self.sftp_uploader.upload_q.full():
                            # Rule 1: There's space, enqueue directly.
                            self.sftp_uploader.upload_q.put(new_item)
                        else:
                            # Queue is full, write to disk backlog.
                            logging.warning(
                                f"SFTP Dispatcher: Upload queue is full. Writing {os.path.basename(new_item)} to disk backlog."
                            )
                            with open(self.backlog_file_path, "a") as f:
                                f.write(new_item + "\n")

                    except queue.Empty:
                        # This is normal, just means no new files were created in the last 5 seconds.
                        # The loop will now restart and check the backlog file again.
                        continue

                except Exception as e:
                    logging.exception(
                        f"An error occurred in the SFTP Dispatcher loop: {e}"
                    )
                    time.sleep(self.cfg["general"]["thread_error_backoff_sec"])
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN SFTP DISPATCHER THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "SFTP_DISPATCHER_CRASH", str(e))

    # -----------------
    def sweep_and_enqueue_unuploaded(self):
        """Scans output directories on startup for files that were not yet uploaded."""
        logging.info("Performing startup sweep for un-uploaded files...")
        count = 0
        scan_dirs = [self.events_out_dir, self.timelapse_out_dir]
        for directory in scan_dirs:
            for entry in os.scandir(directory):
                if (
                    entry.is_file()
                    and not entry.name.endswith(".json")
                    and not entry.name.startswith(".")
                ):
                    json_path = os.path.splitext(entry.path)[0] + ".json"
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r") as f:
                                meta = json.load(f)
                            if not meta.get("uploaded_at"):
                                self.sftp_dispatch_q.put(entry.path)
                                self.sftp_dispatch_q.put(json_path)
                                count += 2
                        except (json.JSONDecodeError, IOError):
                            logging.warning(
                                f"Could not read metadata for {entry.name}, skipping upload check."
                            )
                            self.log_health_event(
                                "WARNING",
                                "METADATA_READ_ERROR",
                                "Could not read metadata, skipping upload check.",
                            )
        if count > 0:
            logging.info(f"Found and enqueued {count} previously un-uploaded files.")

    def heartbeat_loop(self):
        """
        Periodically sends a heartbeat ping to a configured URL.
        Crucially, this thread runs INDEPENDENTLY of the main capture schedule
        to signal that the script process is alive at all times.
        """
        logging.info("Heartbeat monitor started.")
        try:
            while self.running.is_set():
                hb_cfg = self.cfg.get("heartbeat")
                url = hb_cfg.get("url")
                active_interval_min = hb_cfg.get("interval_min")
                # Use a different, potentially longer interval for when the system is idle
                idle_interval_min = self.cfg["general"].get(
                    "idle_heartbeat_interval_min"
                )  # e.g., 2 hours

                if not url:
                    # If disabled or URL missing during a reload, wait and check again.
                    time.sleep(60)
                    continue

                # Determine the current state and wait interval
                is_active = self.producer_thread_active()
                current_interval_sec = (
                    active_interval_min if is_active else idle_interval_min
                ) * 60

                # Wait for the determined interval
                for _ in range(current_interval_sec):
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break
                    time.sleep(1)
                if not self.running.is_set():
                    break

                # Send the ping
                try:
                    # You can append a status to the URL for services that support it
                    # For Healthchecks.io, the main URL is enough.
                    response = requests.get(url, timeout=self.cfg["heartbeat"]["request_timeout_sec"])
                    response.raise_for_status()

                    status_msg = "OK (Active)" if is_active else "OK (Idle)"
                    with self.status_lock:
                        self.last_heartbeat_status = (
                            f"{status_msg} ({datetime.now().strftime('%H:%M:%S')})"
                        )
                    logging.info(
                        f"Heartbeat ping sent successfully (State: {'Active' if is_active else 'Idle'})."
                    )

                except requests.exceptions.RequestException as e:
                    with self.status_lock:
                        self.last_heartbeat_status = (
                            f"FAIL ({datetime.now().strftime('%H:%M:%S')})"
                        )
                    logging.warning(f"Failed to send heartbeat ping: {e}")
                    self.log_health_event(
                        "WARNING", "HEARTBEAT", "Failed to send heartbeat ping"
                    )
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN HEARTHBEAT MONITOR THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "HEARTHBEAT_MONITOR_CRASH", str(e))

    # -----------------
    def ntp_sync_loop(self):
        """
        Periodically checks the system clock against an NTP server. If significant
        drift is detected, it triggers the OS's own time service to re-synchronize.
        """

        logging.info("NTP clock monitor started.")
        time.sleep(60)  # Initial delay for network

        try:
            while self.running.is_set():
                try:
                    ntp_cfg = self.cfg.get("ntp")
                    server = ntp_cfg.get("server")
                    interval_sec = ntp_cfg.get("sync_interval_hours") * 3600
                    max_offset = ntp_cfg.get("max_offset_sec")
                    logging.info("Performing periodic NTP clock check...")
                    max_offset = ntp_cfg.get("max_offset_sec")

                    logging.info(
                        "Performing periodic NTP clock check against '%s'...", server
                    )
                    client = ntplib.NTPClient()
                    # version=3: NTP protocol version (RFC 1305), not a tunable.
                    response = client.request(server, version=3, timeout=ntp_cfg["request_timeout_sec"])
                    offset = response.offset
                    logging.info(
                        f"NTP check complete. System clock offset: {offset * 1000:.2f} ms."
                    )

                    if abs(offset) > max_offset:
                        logging.critical("=" * 60)
                        logging.critical("!!! CRITICAL CLOCK DRIFT DETECTED !!!")
                        logging.critical(
                            f"System clock is off by {offset:.3f} seconds, which exceeds the threshold of {max_offset:.3f}s."
                        )
                        logging.critical(
                            "Attempting to force a re-synchronization via the OS time service."
                        )
                        logging.critical("=" * 60)

                        try:
                            # This command tells systemd-timesyncd to re-enable and force a sync.
                            cmd = ["sudo", "/usr/bin/timedatectl", "set-ntp", "true"]
                            result = subprocess.run(
                                cmd,
                                check=True,
                                capture_output=True,
                                text=True,
                                timeout=self.cfg["general"]["hw_query_timeout_sec"],
                            )
                            logging.info(
                                "Successfully triggered OS time synchronization service."
                            )
                            # After triggering, wait a bit for the sync to happen before the next check.
                            time.sleep(60)
                        except subprocess.TimeoutExpired:
                            logging.error("The ntp command has timed out.")
                        except subprocess.CalledProcessError as e:
                            logging.error("Failed to trigger OS time synchronization.")
                            self.log_health_event(
                                "ERROR",
                                "NTP_FAIL",
                                "Failed to trigger OS time synchronization.",
                            )
                            logging.error("Command failed: %s", " ".join(e.cmd))
                            logging.error("Stderr: %s", e.stderr)
                        except FileNotFoundError:
                            logging.error(
                                "Could not find '/usr/bin/timedatectl'. Cannot trigger time sync."
                            )
                    else:
                        logging.info("System clock is within acceptable tolerance.")

                except Exception as e:
                    logging.warning(
                        f"Could not complete NTP query against '{server}': {e}"
                    )
                    self.log_health_event(
                        "WARNING", "NTP_ERROR", "Could not complete NTP query."
                    )

                # Wait for the next interval
                logging.info(
                    f"NTPThread is sleeping for {interval_sec / 3600:.1f} hours."
                )
                for _ in range(interval_sec):
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break
                    time.sleep(1)
        except Exception as e:
            logging.critical("CRITICAL FATAL ERROR IN NTP SYNC THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "NTP_SYNC_CRASH", str(e))

    # -----------------
    def config_reloader_loop(self):
        """
        A dedicated thread that waits for a signal to reload the config file.
        It intelligently compares the new config to the old one, pauses the watchdog,
        and restarts critical threads if necessary.
        """
        logging.info("Smart configuration reloader thread started.")

        CRITICAL_CAPTURE_PARAMS = [
            "width",
            "height",
            "exposure_us",
            "gain",
            "auto_exposure_tuning",
            "red_gain",
            "blue_gain",
        ]
        CRITICAL_CALIB_PARAMS = [
            "dark_frame_path", 
            "bias_frame_path", 
            "flat_frame_path", 
            "auto_dark_enabled", 
            "auto_flat_enabled"
        ]
        CRITICAL_AI_PARAMS = [
            "enabled", 
            "model_path", 
            "label_path"
        ]
        CRITICAL_SENSOR_PARAMS = ["sensor_camera_id", "sensor_ext_id"]
        CRITICAL_POWER_PARAMS = ["pin"]  # Pin for the power monitor

        last_cfg = json.loads(json.dumps(self.cfg))

        try:
            while self.running.is_set():
                self.ping_watchdog()
                if self.reload_config_signal.wait(timeout=10.0):
                    try:
                        logging.warning(
                            "Reload signal received. Performing smart configuration reload."
                        )
                        new_cfg = self.load_and_validate_config(self.config_path)
                        requires_capture_restart = False
                        requires_pipeline_restart = False
                        power_pin_changed = False
                        
                        # 1. Check Capture Hardware
                        for param in CRITICAL_CAPTURE_PARAMS:
                            old_val = last_cfg.get("capture", {}).get(param)
                            new_val = new_cfg.get("capture", {}).get(param)
                            if old_val != new_val:
                                logging.critical(
                                    f"CRITICAL CHANGE DETECTED: 'capture.{param}' changed from '{old_val}' to '{new_val}'."
                                )
                                requires_capture_restart = True
                                break

                        # 2. Check Calibration Files
                        for param in CRITICAL_CALIB_PARAMS:
                            old_val = last_cfg.get("detection", {}).get(param)
                            new_val = new_cfg.get("detection", {}).get(param)
                            if old_val != new_val:
                                logging.critical(f"CRITICAL CHANGE DETECTED: 'detection.{param}' changed from '{old_val}' to '{new_val}'.")
                                requires_pipeline_restart = True
                                break

                        # 3. Check AI Engines
                        for section in ["ai_classification", "ai_observatory"]:
                            for param in CRITICAL_AI_PARAMS:
                                old_val = last_cfg.get(section, {}).get(param)
                                new_val = new_cfg.get(section, {}).get(param)
                                if old_val != new_val:
                                    logging.critical(f"CRITICAL CHANGE DETECTED: '{section}.{param}' changed from '{old_val}' to '{new_val}'.")
                                    requires_pipeline_restart = True
                                    break

                        # 4. Check Thermal Sensors
                        for param in CRITICAL_SENSOR_PARAMS:
                            old_val = last_cfg.get("general", {}).get(param)
                            new_val = new_cfg.get("general", {}).get(param)
                            if old_val != new_val:
                                logging.critical(f"CRITICAL CHANGE DETECTED: 'general.{param}' changed from '{old_val}' to '{new_val}'.")
                                requires_pipeline_restart = True
                                break

                        # 5. Check Power Monitor
                        for param in CRITICAL_POWER_PARAMS:
                            old_val = last_cfg.get("power_monitor", {}).get(param)
                            new_val = new_cfg.get("power_monitor", {}).get(param)
                            if old_val != new_val:
                                logging.critical(
                                    f"CRITICAL CHANGE DETECTED: 'power_monitor.{param}' changed from '{old_val}' to '{new_val}'."
                                )
                                power_pin_changed = True
                                break

                        # Action Phase
                        if power_pin_changed or requires_pipeline_restart:
                            msg = "Critical parameters changed (Power, AI, Sensors, or Calib). A full service restart is required."
                            logging.warning(msg)
                            self.perform_system_action("exit", reason=msg)
                            # The script will terminate here and systemd will restart it cleanly!
                        elif requires_capture_restart:
                            try:
                                # PAUSE WATCHDOG
                                logging.warning("Pausing watchdog for thread restart procedure.")
                                self.watchdog_pause.set()
                                time.sleep(1)  # Grace period

                                logging.warning("Initiating graceful restart of the capture thread...")
                                self.stop_producer_thread()
                                time.sleep(1)

                                # Apply new config BEFORE re-initializing camera
                                with self.status_lock:
                                    self.cfg = new_cfg

                                # Check safety limits on the newly loaded config
                                self._enforce_exposure_safety_limits()

                                # Re-calculate the auto-tuning profile
                                self._apply_resolution_aware_tuning()

                                try:
                                    self.camera.release()
                                    self.camera = CameraCaptureClass(self.cfg["capture"])
                                    self._reconcile_capture_limits_with_sensor()
                                    logging.info("Camera re-initialized successfully with new settings.")
                                except Exception:
                                    logging.critical("Failed to re-initialize camera! This is a fatal error.", exc_info=True)
                                    self.perform_system_action("exit", reason="Fatal camera re-initialization failure")
                            finally:
                                # RESUME WATCHDOG (CRITICAL)
                                logging.info("Resuming watchdog checks.")
                                self.watchdog_pause.clear()
                        else:
                            logging.info("No critical changes detected that require restarts. Applying new settings live.")
                            with self.status_lock:
                                self.cfg = new_cfg

                            # Check safety limits on the newly loaded config
                            self._enforce_exposure_safety_limits()

                        # Update last known config for the next comparison
                        last_cfg = json.loads(json.dumps(new_cfg))
                        logging.info("Smart configuration reload complete.")

                    except Exception as e:
                        logging.exception("Failed to reload configuration: %s", e)
                    finally:
                        self.reload_config_signal.clear()
                        self.config_reloaded_ack.set()

        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN CONFIG RELOADER THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "CONFIG_RELOADER_CRASH", str(e))
