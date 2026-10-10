"""
janitor_mixin.py

Disk-space management, log rotation, and archival methods for the
Ryusei pipeline, extracted from the main Pipeline class as a pure move
-- method bodies are unchanged, only their physical location. Composed
into Pipeline via multiple inheritance.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls
self.ping_watchdog (ConcurrencyClass) and self.log_health_event (not yet
extracted, still in Pipeline's own body). Must be provided somewhere in
the composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import glob
import json
import shutil
import logging
import time
import psutil
from datetime import datetime, timedelta

from config_models import (
    EVENT_LOG_FIELDS,
    JANITOR_CLEANUP_PRIORITY,
    JANITOR_PROTECTED_FILES,
    JANITOR_SESSION_PROTECTED_DIRS,
)


class JanitorClass:
    """Disk-space management, log rotation, and archival, composed into Pipeline."""

    def janitor_loop(self):
        """A dedicated thread to periodically manage disk space and rotate logs."""
        logging.info(
            f"Janitor thread started (every {self.cfg['janitor']['check_interval_min']} min; "
            f"every {self.cfg['janitor']['pressure_check_interval_min']} min under storage pressure)."
        )

        # csv_path = self.monitor_out_file
        janitor_cfg = self.cfg.get("janitor")
        janitor_work_timeout_sec = janitor_cfg["janitor_work_timeout_sec"]
        try:
            while self.running.is_set():
                # Cleared at the START of a cycle: a wake raised while this cycle runs
                # stays pending and triggers the next cycle immediately (no lost wakeup).
                self.storage.wake_janitor.clear()
                # Satellite catalogue: refreshed on this cadence (non-blocking, only when due).
                self._maybe_refresh_satellites()
                # Perform the multi-tiered disk space cleanup.
                # cleanup_successful = self.manage_disk_space(
                    # janitor_cfg.get("threshold"), janitor_cfg.get("target")
                # )
                # Wrapped in _run_bounded: manage_disk_space calls shutil.rmtree,
                # os.remove, and psutil.disk_usage repeatedly, none of which have
                # their own timeout. A genuinely unresponsive USB drive -- exactly
                # the condition this method exists to relieve -- could otherwise
                # hang this call indefinitely, freezing janitor_loop with no
                # heartbeat until the watchdog's separate thread_heartbeats check
                # eventually notices. _run_bounded's own ping_watchdog() calls
                # (before and after) keep this thread visibly alive to the
                # watchdog throughout, whether the work finishes quickly or is
                # still legitimately running right up to the timeout.
                cleanup_successful, cleanup_timed_out = self._run_bounded(
                    self.manage_disk_space,
                    janitor_work_timeout_sec,
                    "JanitorThread",
                    janitor_cfg.get("threshold"),
                    janitor_cfg.get("target"),
                 )
                if cleanup_timed_out:# 3. Execute Mass Archive
                    logging.error(
                        f"Janitor: manage_disk_space did not return within "
                        f"{janitor_work_timeout_sec}s -- treating as a failed "
                        f"cleanup rather than assuming success."
                    )
                    cleanup_successful = False

                # Storage backpressure. OLD: a failed cleanup latched emergency_stop_active,
                # which inhibits ALL capture and is never cleared (and is shared with the
                # watchdog's thread-crash emergency, so it can't be) -- one bad hour left a
                # remote station dead until a manual restart. Now the storage governor
                # decides from the live filesystem, sheds low-value writes first, halts
                # only below the reserve, and releases automatically when space returns.
                try:
                    level, used_pct, free_b = self.storage.refresh()
                    if not cleanup_successful:
                        logging.critical(
                            f"Janitor: cleanup could not reach target; storage level {level.name} "
                            f"(used {used_pct:.1f}%, free {free_b / 1e6:.0f}MB)."
                        )
                    refused = self.storage.pop_refusal_summary()
                    if refused:
                        logging.warning(
                            "Janitor: writes refused by the storage governor since last cycle: "
                            + ", ".join(f"{os.path.basename(d) or d}={n}" for d, n in refused.items())
                        )
                except OSError as e:
                    logging.error(f"Janitor: storage governor could not stat the drive: {e}")

                # If ANY single log file exceeds its size limit, we archive ALL of them
                log_max_mb = self.cfg.get("janitor", {}).get("log_rotation_mb")
                csv_max_mb = self.cfg.get("janitor", {}).get("csv_rotation_mb")
                
                pipeline_log = os.path.join(self.general_log_dir, "pipeline.log")
                csv_files = [self.monitor_out_file, self.event_log_out_file, self.health_log_out_file, self.camera_k_file]
                
                trigger_mass_archive = False
                
                # 1. Check pipeline.log
                if os.path.exists(pipeline_log):
                    if (os.path.getsize(pipeline_log) / (1024 * 1024)) >= log_max_mb:
                        logging.warning(f"Janitor: pipeline.log exceeded {log_max_mb}MB. Triggering mass archive.")
                        trigger_mass_archive = True

                # 2. Check all CSV files (Only if pipeline.log didn't already trigger it)
                if not trigger_mass_archive:
                    for csv_path in csv_files:
                        if os.path.exists(csv_path):
                            if (os.path.getsize(csv_path) / (1024 * 1024)) >= csv_max_mb:
                                logging.warning(f"Janitor: {os.path.basename(csv_path)} exceeded {csv_max_mb}MB. Triggering mass archive.")
                                trigger_mass_archive = True
                                break

                # 3. Execute Mass Archive
                # Wrapped in _run_bounded for the same reason as manage_disk_space
                # above -- this uses shutil.move and shutil.copy2 internally (via
                # archive_data_file too), none of which have their own timeout.
                if trigger_mass_archive:
                    # OLD: this direct call ran BEFORE the _run_bounded-wrapped one
                    # below, so archiving happened twice per trigger (CSVs rotated
                    # twice) -- and this copy ran completely unprotected: no
                    # timeout, no bounding, no watchdog ping. On a slow/contended
                    # drive it could hang JanitorThread with no heartbeat for as
                    # long as the copy took, defeating the whole point of wrapping
                    # the call in _run_bounded right below it.
                    # self._force_archive_all_logs()
                    _, archive_timed_out = self._run_bounded(
                       self._force_archive_all_logs,
                        janitor_work_timeout_sec,
                        "JanitorThread",
                    )
                    if archive_timed_out:
                        logging.error(
                            f"Janitor: _force_archive_all_logs did not return "
                            f"within {janitor_work_timeout_sec}s -- abandoning "
                            f"this cycle's archive attempt."
                        )

                j = self.cfg["janitor"]
                interval_min = (
                    j["check_interval_min"] if self.storage.level.name == "OK"
                    else j["pressure_check_interval_min"]
                )
                logging.info(f"Janitor run complete. Next check in {interval_min} min (or sooner if storage pressure rises).")
                self.initial_janitor_check_complete.set()

                # Sleep until the next cycle, or until a write-time check crosses the
                # threshold (the governor sets wake_janitor) -- then clean up NOW rather
                # than up to an hour later. Heartbeat-aware; stop() interrupts it.
                self.wait_event_with_heartbeat(self.storage.wake_janitor, interval_min * 60)
        except Exception as e:
            logging.critical("CRITICAL FATAL ERROR IN JANITOR THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "JANITOR_CRASH", str(e))

    # -----------------
    @staticmethod
    def _session_start_at(start_str, now):
        """
        Datetime of the most recent occurrence of `start_str` ("HH:MM") at or
        before `now`. That is the start of the current session (or of the one
        that just ended, while we're outside the schedule window -- its files
        stay protected until the next session begins, at which point this
        moves forward a day and the previous session becomes deletable).

        Plain datetime arithmetic, so midnight, month-end and year-end
        rollovers need no special handling: at 02:00 on Jan 1 with a 22:50
        start, the candidate for Jan 1 22:50 is in the future, so we step
        back one day to Dec 31 22:50.
        """
        hour, minute = map(int, start_str.split(":"))
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > now:
            candidate -= timedelta(days=1)
        return candidate

    # -----------------
    def _session_cutoff_ts(self):
        """
        Epoch seconds of the current session's start, or None if it can't be
        determined (no start_time configured, i.e. 24/7 mode, and no producer
        session recorded yet) -- in which case nothing is session-protected.
        """
        start_str = self.cfg["general"].get("start_time")
        if start_str:
            return self._session_start_at(start_str, datetime.now()).timestamp()

        started = getattr(self, "session_start_time", None)
        return started.timestamp() if started else None

    # -----------------
    def manage_disk_space(self, threshold, target):
        """
        Performs a multi-tiered cleanup. Returns True on success, False on failure.
        Tier 1: Deletes leftover temporary event frame directories.
        Tier 2: Deletes the oldest timelapse files.
        """
        try:
            # Get the path to monitor from the configuration.
            monitor_path = self.cfg["janitor"].get("monitor_path")

            # Use this path for all disk usage checks.
            disk = psutil.disk_usage(monitor_path)
            self.disk_percent = disk.percent
            if self.disk_percent < threshold:
                return True  # Nothing to do, success.

            logging.warning(
                f"Disk usage on '{monitor_path}' is at {self.disk_percent:.1f}% (threshold is {threshold:.1f}%). Starting prioritized cleanup."
            )
            self.log_health_event(
                "WARNING", "DISK_USAGE_HIGH", f"Disk at {self.disk_percent:.1f}%"
            )

            # --- CLEAN UP LEFTOVER EVENT BACKLOGS ---
            event_dir = self.events_out_dir
            backlog_count = 0
            if os.path.isdir(event_dir):
                with os.scandir(event_dir) as it:
                    for entry in it:
                        # Temporary directories are named 'event_..._frames'
                        if entry.is_dir() and entry.name.endswith("_frames"):
                            try:
                                logging.warning(
                                    "Janitor: Found and deleting leftover event frame directory: %s",
                                    entry.name,
                                )
                                self.log_health_event(
                                    "WARNING",
                                    "LEFTOVER_EVENT",
                                    "Found and deleting leftover event.",
                                )
                                shutil.rmtree(entry.path)
                                backlog_count += 1
                            except Exception:
                                logging.exception(
                                    "Janitor: Error deleting backlog directory %s",
                                    entry.path,
                                )
            if backlog_count > 0:
                logging.info(
                    "Janitor: Tier 1 cleanup removed %d backlog directories.",
                    backlog_count,
                )

            # Re-check usage. If we're already good, we can stop.
            if psutil.disk_usage(monitor_path).percent < target:
                logging.info("Disk space is now sufficient. Cleanup complete.")
                return True

            # 1. Resolve actual paths from the priority list at the top
            cleanup_tiers = []
            for attr_name in JANITOR_CLEANUP_PRIORITY:
                actual_path = getattr(self, attr_name, None)
                if actual_path:
                    cleanup_tiers.append({"name": attr_name, "path": actual_path})

            # Current-session cutoff, computed once so every tier uses the same one.
            session_cutoff_ts = self._session_cutoff_ts()
            session_start_str = (
                datetime.fromtimestamp(session_cutoff_ts).strftime("%Y-%m-%d %H:%M")
                if session_cutoff_ts is not None
                else "n/a"
            )
            session_files_kept = 0

            # CLEAN UP OLDEST FILES, TIER BY TIER
            for tier in cleanup_tiers:
                tier_name = tier["name"]
                tier_path = tier["path"]

                # Check disk space before starting each tier
                if psutil.disk_usage(monitor_path).percent < target:
                    logging.info(
                        f"Disk space is now sufficient ({psutil.disk_usage(monitor_path).percent:.1f}%). Halting cleanup."
                    )
                    break  # Exit the priority loop

                if not tier_path or not os.path.isdir(tier_path):
                    continue

                logging.warning(
                    f"Janitor: Disk space still critical. Deleting from Tier '{tier_name}'..."
                )

                # Scan the tier and snapshot each file's mtime ONCE. A file can vanish
                # between scandir() and stat() (SFTP post-upload handling, the event
                # finalizer moving files); an uncaught FileNotFoundError here used to
                # abort the WHOLE cleanup and report failure. Vanished files are simply
                # skipped -- their space is already free.
                files_in_tier = []
                for entry in os.scandir(tier_path):
                    if not entry.is_file() or entry.name.startswith("."):
                        continue
                    try:
                        files_in_tier.append((entry, entry.stat().st_mtime))
                    except FileNotFoundError:
                        continue

                # Never touch files from the current session in the session-protected
                # directories: anything modified at or after the session start stays,
                # regardless of disk pressure or upload state. mtime is used (not the
                # timestamp in the filename) because it's uniform across every file
                # type in these dirs -- the daylight snapshots use a different name
                # format than timelapse/event files.
                if tier_name in JANITOR_SESSION_PROTECTED_DIRS and session_cutoff_ts is not None:
                    before = len(files_in_tier)
                    files_in_tier = [
                        (e, mt) for (e, mt) in files_in_tier if mt < session_cutoff_ts
                    ]
                    kept = before - len(files_in_tier)
                    if kept:
                        session_files_kept += kept
                        logging.info(
                            f"Janitor ({tier_name}): keeping {kept} file(s) from the "
                            f"current session (since {session_start_str})."
                        )

                files_in_tier.sort(key=lambda em: em[1])  # oldest first (snapshotted mtime)

                deleted_count_in_tier = 0
                for file_entry, _mtime in files_in_tier:
                    # # Skip protected files
                    if file_entry.name in JANITOR_PROTECTED_FILES:
                        continue
                        
                    # Check disk usage on every file to stop as soon as the target is met
                    if psutil.disk_usage(monitor_path).percent < target:
                        break  # Exit the file loop for this tier

                    try:
                        if self.safe_to_delete(file_entry.path):
                            logging.info(
                                f"Janitor ({tier_name}): Deleting old, uploaded file: {file_entry.name}"
                            )
                            os.remove(file_entry.path)
                            json_path = os.path.splitext(file_entry.path)[0] + ".json"
                            if os.path.exists(json_path):
                                os.remove(json_path)
                            deleted_count_in_tier += 1
                        else:
                            logging.info(
                                f"Janitor ({tier_name}): Skipping deletion of un-uploaded file: {file_entry.name}"
                            )
                    except FileNotFoundError:
                        continue  # removed by someone else meanwhile: its space is free already
                    except OSError as e:
                        logging.error(
                            f"Janitor: Failed to delete file {file_entry.path}: {e}"
                        )
                        self.log_health_event(
                            "ERROR",
                            "JANITOR_DELETE_FAIL",
                            f"Failed on {file_entry.name}",
                        )

                if deleted_count_in_tier > 0:
                    logging.info(
                        f"Janitor: Tier '{tier_name}' cleanup removed {deleted_count_in_tier} file(s)."
                    )

            # FINAL CHECK
            final_disk_percent = psutil.disk_usage(monitor_path).percent
            if final_disk_percent < target:
                logging.info(
                    f"Janitor: Cleanup successful. New disk usage on '{monitor_path}' is {final_disk_percent:.1f}%."
                )
                return True
            else:
                logging.critical(
                    f"!!! JANITOR FAILURE: DISK SPACE ON '{monitor_path}' COULD NOT BE FREED SUFFICIENTLY ({final_disk_percent:.1f}%) !!!"
                )
                if session_files_kept:
                    logging.critical(
                        f"Janitor: {session_files_kept} current-session file(s) (since "
                        f"{session_start_str}) were deliberately left untouched -- the "
                        f"remaining space is held by this session's own data."
                    )
                return False

        except Exception:
            logging.exception("An unhandled error occurred in the janitor process.")
            # If we crashed here, the disk was over the threshold and we failed to fix it.
            # Returning False ensures the pipeline triggers the Emergency Stop to protect the OS.
            return False
    # -----------------
    def _on_storage_transition(self, old, new, used_pct, free_bytes):
        """Storage governor level change: one log line + health event per transition, not per write."""
        msg = (
            f"Storage level {old.name} -> {new.name} (used {used_pct:.1f}%, free {free_bytes / 1e6:.0f}MB, "
            f"reserve {self.cfg['janitor']['storage_reserve_mb']}MB)."
        )
        if new.name == "HALT":
            logging.critical(msg + " All artifact writes refused; acquisition stops until space is recovered.")
            self.log_health_event("CRITICAL", "STORAGE_HALT", msg)
        elif new.name == "SHED":
            logging.warning(msg + " Suspending diagnostic writes (debug/rejected/daylight); cleaning up now.")
            self.log_health_event("WARNING", "STORAGE_SHED", msg)
        else:
            logging.info(msg + " Normal writing resumed.")
            self.log_health_event("INFO", "STORAGE_OK", msg)

    # -----------------
    def safe_to_delete(self, file_path):
        """Checks a file's metadata to see if it has been marked as uploaded."""
        # If SFTP is not enabled, it's always safe to delete.
        if not self.sftp_uploader:
            return True

        json_path = os.path.splitext(file_path)[0] + ".json"
        if not os.path.exists(json_path):
            # No metadata, assume it's an old file or an anomaly. Safer to keep.
            logging.warning(
                f"Janitor: Skipping deletion of {os.path.basename(file_path)} because it has no metadata."
            )
            self.log_health_event(
                "WARNING", "JANITOR", "Skipping deletion no metadata."
            )
            return False

        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            logging.warning(
                f"Janitor: Failed to read metadata for {os.path.basename(file_path)}, skipping deletion: {e}"
            )
            self.log_health_event(
                "WARNING", "JANITOR", "Failed to read metadata, skipping deletion."
            )
            return False  # Fail safe: don't delete if metadata is unreadable

        # Check for the 'uploaded_at' key.
        return "uploaded_at" in meta

    # -----------------
    def archive_data_file(self, file_path, max_size_mb, force=False):
        """Archives a generic data file (like a CSV) if it exceeds a max size, or if forced."""
        try:
            if not os.path.exists(file_path):
                return

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # Skip if not forced and file is smaller than the limit
            if not force and file_size_mb < max_size_mb:
                return

            # Safely log or print depending on whether the logger is initialized yet
            msg = f"Archiving file '{os.path.basename(file_path)}' (Forced: {force}, Size: {file_size_mb:.2f}MB)."
            if logging.getLogger().hasHandlers():
                logging.info(msg)
            else:
                print(msg)

            # Archive Logic
            # archive_dir = self.logs_archive
            # os.makedirs(archive_dir, exist_ok=True)

            tstamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            base_name, ext = os.path.splitext(os.path.basename(file_path))
            archive_path = os.path.join(self.logs_archive, f"{base_name}_{tstamp}{ext}")

            # Move the file to the archive folder
            shutil.move(file_path, archive_path)

            # CRITICAL: Re-create the headers for the new, empty CSV files
            if file_path == self.monitor_out_file:
                with open(file_path, "w") as f:
                    f.write(
                        "timestamp,cpu_temp,sensor_temp_cam,sensor_temp_ext,cpu_percent,mem_used_mb,mem_percent,disk_used_mb,disk_percent,gpu_percent,load1,load5,load15,uptime_sec,core_voltage_v\n"
                    )
            elif file_path == self.event_log_out_file:
                with open(file_path, "w") as f:
                    f.write(
                        ",".join(EVENT_LOG_FIELDS) + "\n"
                    )
            elif file_path == self.health_log_out_file:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("event_type,count,level,last_message\n")
            elif file_path == self.camera_k_file:
                with open(file_path, "w") as f:
                    f.write("timestamp, k\n")

        except Exception as e:
            if logging.getLogger().hasHandlers():
                logging.exception("An error occurred during data file archiving.")
            else:
                print(f"Error archiving {file_path}: {e}")

    # -----------------
    def _force_archive_all_logs(self):
        """Forces the archiving of all main log and CSV files for a clean session."""
        print("--- FORCED LOG ARCHIVING TRIGGERED ---")

        # List of all critical CSV files
        files_to_archive = [
            self.monitor_out_file,
            self.event_log_out_file,
            self.health_log_out_file,
            self.camera_k_file,
        ]

        for file_path in files_to_archive:
            if os.path.exists(file_path):
                self.archive_data_file(file_path, max_size_mb=0, force=True)

        # Archive Nightly Summaries (Journals & Keograms) 
        archive_dir = os.path.join(self.general_log_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Use glob to find the dynamically named files
        summaries_to_move = glob.glob(os.path.join(self.general_log_dir, "journal_*")) # Catches .pdf and .txt fallbacks
        summaries_to_move += glob.glob(os.path.join(self.timelapse_out_dir, "keogram_*.png"))
        
        for file_path in summaries_to_move:
            if os.path.exists(file_path):
                try:
                    dest_path = os.path.join(archive_dir, os.path.basename(file_path))
                    shutil.move(file_path, dest_path)
                    
                    log_msg = f"Archived nightly summary to {os.path.basename(dest_path)}"
                    if logging.getLogger().hasHandlers():
                        logging.info(log_msg)
                    else:
                        print(log_msg)
                except Exception as e:
                    if logging.getLogger().hasHandlers():
                        logging.error(f"Failed to archive summary file {file_path}: {e}")

        # --- Special Handling for pipeline.log ---
        # We copy it and empty it, rather than moving it, so Python's active logger doesn't crash!
        pipeline_log = os.path.join(self.general_log_dir, "pipeline.log")
        if os.path.exists(pipeline_log) and os.path.getsize(pipeline_log) > 0:
            # archive_dir = os.path.join(self.general_log_dir, "archive")
            # os.makedirs(archive_dir, exist_ok=True)
            tstamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            archive_path = os.path.join(self.logs_archive, f"pipeline_{tstamp}.log")

            shutil.copy2(pipeline_log, archive_path)
            open(pipeline_log, "w").close()  # Empty the active log file
            
            if logging.getLogger().hasHandlers():
                logging.info(f"Archived pipeline.log to {os.path.basename(archive_path)}")
            else:
                print(f"Archived pipeline.log to {archive_path}")

        print("--- LOG ARCHIVING COMPLETE ---")
