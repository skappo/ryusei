"""
sftp.py
"""
import threading
import os
import time
import queue
import paramiko
import json
import logging
from datetime import datetime

class SFTPUploaderClass:
    def __init__(self, cfg, pipeline_instance):
        self.cfg = cfg
        self.pipeline_instance = pipeline_instance
        self.stop_event = threading.Event()
        self.sftp_password = os.environ.get("SFTP_PASS")
        self.upload_q = queue.Queue(maxsize=cfg.get("max_queue_size"))

        if self.sftp_password:
            logging.info("SFTP password loaded from SFTP_PASS environment variable.")
        else:
            # If the password environment variable is missing, log an error.
            logging.error(
                "SFTP_PASS environment variable is not set. SFTP uploads will fail."
            )

    def _ensure_remote_dir(self, sftp_client):
        remote_dir = self.cfg["remote_dir"]
        try:
            # Check if the directory already exists
            sftp_client.stat(remote_dir)
            logging.info(f"Remote directory '{remote_dir}' already exists.")
        except FileNotFoundError:
            # If it doesn't exist, create it.
            logging.warning(
                f"Remote directory '{remote_dir}' not found. Attempting to create it."
            )
            try:
                # Create directories recursively
                dirs = remote_dir.strip("/").split("/")
                current_dir = ""
                for d in dirs:
                    current_dir = os.path.join(current_dir, d)
                    # For SFTP, paths should always use forward slashes
                    current_dir_posix = current_dir.replace(os.path.sep, "/")
                    if not current_dir_posix.startswith("/"):
                        current_dir_posix = "/" + current_dir_posix
                    try:
                        sftp_client.stat(current_dir_posix)
                    except FileNotFoundError:
                        logging.info(f"Creating remote directory: {current_dir_posix}")
                        sftp_client.mkdir(current_dir_posix)
            except Exception as e:
                logging.error(f"Failed to create remote directory '{remote_dir}': {e}")
                # Re-raise the exception to cause the connection to fail.
                raise

    def connect(self):
        try:
            # Use the higher-level SSHClient which simplifies connection and has a timeout
            client = paramiko.SSHClient()
            # Automatically add the server's host key. For high-security environments,
            # you might want to load known host keys instead.
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            client.connect(
                hostname=self.cfg["host"],
                port=self.cfg["port"],
                username=self.cfg["user"],
                password=self.sftp_password,
                timeout=self.cfg["connect_timeout_sec"],
            )

            sftp = client.open_sftp()
            logging.info("SFTP connection established.")

            self._ensure_remote_dir(sftp)

            # Return both the client and sftp object for proper cleanup later
            return client, sftp

        except Exception as e:
            # This will now catch timeout errors, authentication errors, and name resolution errors.
            msg = f"SFTP connection failed: {e}"
            logging.error(msg)
            self.pipeline_instance.log_health_event(
                "WARNING", "SFTP_connection_failed", msg
            )
            return None, None

    def get_status(self):
        """Returns a tuple of (status_class, status_message) for the dashboard."""
        q_size = self.upload_q.qsize()

        # Check for power loss first, as it's the highest priority status
        # Note: 'DISABLED' means monitoring is not enabled or supported, so we allow activity.
        with self.pipeline_instance.status_lock:
            if self.pipeline_instance.power_status not in ["OK", "DISABLED"]:
                return ("err", f"Paused (Power Loss) - Queue: {q_size}")

        q_max = self.upload_q.maxsize
        if q_max == 0:
            return ("ok", f"Queue: {q_size}")  # Should not happen, but safe

        q_percent = (q_size / q_max) * 100

        if q_percent >= 99:
            return ("err", f"Backlogged - Queue: {q_size}/{q_max} ({q_percent:.0f}%)")
        elif q_percent > 75:
            return ("warn", f"Filling - Queue: {q_size}/{q_max} ({q_percent:.0f}%)")
        else:
            return ("ok", f"Idle/Uploading - Queue: {q_size}")

    def worker_loop(self):
        ssh_client = None
        sftp = None
        was_paused_by_power_loss = False

        INITIAL_RECONNECT_DELAY_SEC = self.cfg["reconnect_initial_sec"]
        MAX_RECONNECT_DELAY_SEC = self.cfg["reconnect_max_sec"]
        current_reconnect_delay = INITIAL_RECONNECT_DELAY_SEC

        while not self.stop_event.is_set():

            # CHECK FOR POWER LOSS
            if self.pipeline_instance.power_status not in ["OK", "DISABLED"]:
                if not was_paused_by_power_loss:
                    logging.warning(
                        "SFTP Uploader: Power loss detected. Pausing all upload activity."
                    )
                    was_paused_by_power_loss = True

                # If a connection exists, close it to be safe during a power outage.
                if sftp:
                    sftp.close()
                    sftp = None

                if ssh_client:
                    ssh_client.close()
                    ssh_client = None

                # pipeline.ping_watchdog()  # OLD: 'pipeline' was never defined in
                # this file/class (self.pipeline_instance is) -- this raised
                # NameError every time a power-loss condition was hit, crashing
                # SftpThread with no heartbeat ever having been sent from here.
                self.stop_event.wait(timeout=10)
                self.pipeline_instance.ping_watchdog()
                continue  # Restart the loop to re-check power status.

            # If we are here, power is OK. Check if it was just restored.
            if was_paused_by_power_loss:
                logging.info(
                    "SFTP Uploader: Power has been restored. Resuming uploads."
                )
                was_paused_by_power_loss = False
                # sftp is already None, so the next block will force a reconnect.
                current_reconnect_delay = INITIAL_RECONNECT_DELAY_SEC

            # GET A CONNECTION
            if sftp is None:
                logging.info("SFTP: No active connection. Attempting to connect...")
                ssh_client, sftp = self.connect()

                if sftp is None:
                    # Connection failed. Wait without touching the queue.
                    logging.warning(
                        f"SFTP: Connection failed. Will retry in {current_reconnect_delay} seconds."
                    )
                    # Backoff can reach MAX_RECONNECT_DELAY_SEC (1800s), far beyond
                    # watchdog_hung_thread_sec: wait in heartbeat-aware slices so a long
                    # network outage is never mistaken for a hung SftpThread. Returns
                    # immediately if stop_event is set.
                    self.pipeline_instance.wait_event_with_heartbeat(
                        self.stop_event, current_reconnect_delay
                    )

                    # Increase the delay for the next attempt, up to the maximum
                    current_reconnect_delay = min(
                        current_reconnect_delay * 2, MAX_RECONNECT_DELAY_SEC
                    )
                    continue
                else:
                    # Reset Backoff Logic on Success
                    logging.info(
                        "SFTP: Connection established successfully. Resetting reconnect delay."
                    )
                    current_reconnect_delay = INITIAL_RECONNECT_DELAY_SEC
                    logging.info("SFTP: Starting to process upload queue.")

            # PROCESS THE QUEUE
            # Now that we know power is OK and we have a connection, we can get a file.
            try:
                local_path = self.upload_q.get(timeout=5)
                if local_path is None:
                    continue

            except queue.Empty:
                # No files to upload.
                continue

            # Attempt the upload.
            try:
                remote_path = os.path.join(
                    self.cfg["remote_dir"], os.path.basename(local_path)
                )
                sftp.put(local_path, remote_path)
                logging.info(f"Uploaded: {os.path.basename(local_path)}")
                self.mark_as_uploaded(local_path)

            except Exception as e:
                # An error occurred during the upload. The connection is likely now broken.

                msg = f"Upload failed for {os.path.basename(local_path)}: {e}"
                logging.error(msg)

                self.pipeline_instance.log_health_event(
                    "ERROR", "SFTP_UPLOAD_FAIL", msg
                )

                logging.warning(
                    "SFTP connection may have been lost. Re-queueing file and forcing reconnect."
                )
                try:
                    self.upload_q.put_nowait(local_path)
                    logging.warning("SFTP connection lost. Re-queued file to memory.")
                except queue.Full:
                    # If memory queue is full, append to the disk backlog
                    logging.warning("SFTP Queue full. Writing failed upload to disk backlog.")
                    try:
                        with open(self.pipeline_instance.backlog_file_path, "a") as f:
                            f.write(local_path + "\n")
                    except Exception as le:
                        logging.error(f"CRITICAL: Could not write to backlog file: {le}")

                if sftp: sftp.close()
                if ssh_client: ssh_client.close()
                sftp, ssh_client = None, None

        # Final cleanup on shutdown.
        if sftp:
            sftp.close()
            logging.info("SFTP connection closed.")

        if ssh_client:
            ssh_client.close()

    def mark_as_uploaded(self, file_path):
        # Marks a file's metadata as uploaded.
        json_path = os.path.splitext(file_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r+") as f:
                    meta = json.load(f)
                    meta["uploaded_at"] = datetime.now().isoformat()
                    self.pipeline_instance.save_json_atomic(json_path, meta)
            except Exception as e:
                logging.error(
                    f"Failed to mark {os.path.basename(json_path)} as uploaded: {e}"
                )

    def stop(self):
        """Signals the SFTP worker to stop."""
        self.stop_event.set()
        # Post a sentinel to unblock the worker_loop if it's waiting on the queue
        try:
            self.upload_q.put_nowait(None)
        except queue.Full:
            pass