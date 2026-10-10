"""
concurrency_mixin.py

Watchdog, heartbeat, and bounded-execution methods for the Ryusei
pipeline, extracted from the main Pipeline class as a pure move -- method
bodies are unchanged, only their physical location. Composed into Pipeline
via multiple inheritance.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- these methods
call self.log_health_event, self.perform_system_action, and
self.stop_producer_thread, none of which are defined in this file. They
must be provided somewhere else in the composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grepping for module.method/module.attribute patterns before
writing this file.

CLOCK DISCIPLINE: every heartbeat, deadline and elapsed-time measurement in
this file uses time.monotonic(), never time.time(). The wall clock on a Pi
without a battery-backed RTC is stepped by NTP after boot (often by hours,
from the fake-hwclock value); a forward step made every heartbeat look stale
at once and force-restarted a healthy pipeline, a backward step could stall
any deadline loop. Monotonic time cannot jump. time.time() remains correct
only for calendar timestamps (filenames, logs, ephemeris).
"""
import itertools
import logging
import threading
import time
try:
    import sdnotify
except ImportError:
    sdnotify = None

# Process-wide sequence for unique worker-thread names (see _launch_fire_and_forget).
_THREAD_SEQ = itertools.count()


class ConcurrencyClass:
    """Watchdog, heartbeat, and bounded-execution helpers, composed into Pipeline."""

    def watchdog_loop(self):
        """
        The master supervisor thread. It performs two critical functions:
        1. Periodically checks if all other critical threads are alive.
        2. Periodically pings the systemd watchdog to signal script health.

        If any thread has crashed, this loop will exit with an error code,
        triggering a service restart by systemd. If this loop itself freezes,
        systemd will stop receiving pings and restart the service.
        """
        g = self.cfg["general"]
        startup_delay_sec = g["watchdog_startup_delay_sec"]
        check_interval_sec = g["watchdog_check_interval_sec"]
        if sdnotify is None:
            logging.warning("sdnotify library not found. Running without systemd integration.")
            systemd_integration = False
        else:
            n = sdnotify.SystemdNotifier()
            systemd_integration = True

        # Give all other threads a moment to start up before signaling readiness.
        logging.info(
            "Watchdog thread started. Waiting %d seconds before signaling READY.",
            startup_delay_sec,
        )
        for _ in range(startup_delay_sec):
            if not self.running.is_set():
                return
            time.sleep(1)

        # Signal systemd that the service is fully initialized and ready.
        if systemd_integration:
            n.notify("READY=1")
            logging.info("Watchdog Supervisor signaled READY=1 to systemd.")

        # RESET OTA CRASH COUNTER
        # If we made it this far, the code is stable. Reset the bash launcher's crash counter!
        try:
            with open(self.crash_file, "w") as f:
                f.write("0")
            logging.info("OTA Crash counter reset to 0 (Startup survived successfully).")
        except Exception as e:
            logging.error(f"Failed to reset crash counter: {e}")

        last_queue_ping = time.monotonic()
        # Heartbeats keep aging while checks are suspended (config reload, camera
        # restart), because paused threads legitimately stop pinging. Without a
        # rebase, the first check after the pause lifts would judge them against
        # the whole pause duration and could force-exit over nothing. Every
        # heartbeat is therefore aged from no earlier than the end of the last
        # pause (or the start of this loop).
        heartbeat_floor = time.monotonic()
        try:
            while self.running.is_set():
                if self.watchdog_pause.is_set():
                    logging.warning(
                        "Watchdog checks are temporarily suspended for configuration reload."
                    )
                    # We still need to ping the systemd watchdog to prevent it from timing out
                    if systemd_integration:
                        n.notify("WATCHDOG=1")
                        
                    # Wait for the interval and then restart the loop, skipping the thread checks
                    time.sleep(check_interval_sec)
                    heartbeat_floor = time.monotonic()
                    continue

                # Ping systemd to let it know the script is responsive.
                if systemd_integration:
                    n.notify("WATCHDOG=1")

                now = time.monotonic()
                
                # Periodically push a STATUS token into every blocking consumer queue, so
                # consumers idle on get() still wake up and heartbeat.
                if now - last_queue_ping > self.cfg["general"]["queue_status_ping_sec"]:
                    last_queue_ping = now
                    queues_to_ping = [
                        self.event_q, self.event_stack_q, self.event_log_q, 
                        self.timelapse_q, self.health_q, self.sftp_dispatch_q,
                        self.calibration_q
                    ]
                    for q in queues_to_ping:
                        try:
                            # Don't block if the queue is full of real data
                            q.put_nowait("STATUS") 
                        except Exception:
                            pass

                # --- HANG DETECTION FOR ALL THREADS ---
                hung_thread_sec = self.cfg["general"]["watchdog_hung_thread_sec"]
                for t_name, (t_obj, last_ping) in list(self.thread_heartbeats.items()):
                    # A thread that has finished can never ping again. Crashes of the
                    # long-lived workers are caught by the liveness check below; a
                    # finished transient thread (_run_bounded / fire-and-forget worker,
                    # uniquely named) is a normal exit, and its stale entry would
                    # otherwise trip this check hung_thread_sec later. Prune it.
                    if not t_obj.is_alive():
                        if self.thread_heartbeats.get(t_name, (None,))[0] is t_obj:
                            self.thread_heartbeats.pop(t_name, None)
                        continue
                    time_since_ping = now - max(last_ping, heartbeat_floor)

                    if time_since_ping > hung_thread_sec:
                        msg = f"WATCHDOG: HUNG THREAD DETECTED! '{t_name}' has been frozen for {int(time_since_ping)} seconds."
                        
                        logging.error(f"{msg}")
                        self.perform_system_action("exit", reason=msg)

                # --- ZOMBIE FIRE-AND-FORGET THREAD DETECTION ---
                with self.fire_and_forget_lock:
                    tracked = list(self.fire_and_forget_threads.items())

                for fnf_name, (worker, registered_at, own_timeout_sec) in tracked:
                    if not worker.is_alive():
                        # Finished on its own -- clean exit, stop tracking it.
                        with self.fire_and_forget_lock:
                            self.fire_and_forget_threads.pop(fnf_name, None)
                        continue

                    stuck_for = now - registered_at
                    if stuck_for > own_timeout_sec:
                        msg = (
                            f"WATCHDOG: ZOMBIE THREAD DETECTED! '{fnf_name}' has been "
                            f"running {int(stuck_for)}s, past its own {own_timeout_sec}s "
                            f"budget -- this is a genuine stuck operation, not a slow "
                            f"one. Restarting to reclaim it."
                        )
                        logging.error(msg)
                        self.perform_system_action("exit", reason=msg)

                # Check the health of all other threads.
                all_monitored_threads = self.worker_threads + self.control_threads
                for thread in all_monitored_threads:
                    if not thread.is_alive():
                        # EMERGENCY: A THREAD HAS CRASHED
                        msg = f"WATCHDOG: DETECTED CRASHED THREAD: {thread.name}"
                        logging.error(f"{msg}")

                        self.log_health_event("CRITICAL", "THREAD_CRASH", msg)
                        # Stop the data producer.
                        self.stop_producer_thread()
                        # Set emergency mode
                        self.emergency_stop_active.set()

                        self.perform_system_action("exit", reason=msg)
                # If all is well, wait for the next check interval.
                for _ in range(check_interval_sec):
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break
                    time.sleep(1)
        except Exception as e:
            logging.critical("CRITICAL FATAL ERROR IN WATCHDOG THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "WATCHDOG_CRASH", str(e))

    # -----------------

    def ping_watchdog(self):
        """Threads call this function inside their loops to prove they are not frozen."""
        # Keyed by name (a restarted worker reuses its name and simply overwrites its
        # entry); the Thread object is stored so the watchdog can tell a finished
        # thread from a hung one. A single dict assignment is atomic under the GIL.
        t = threading.current_thread()
        self.thread_heartbeats[t.name] = (t, time.monotonic())

    # -----------------
    def _heartbeat_slice_sec(self):
        """Longest a heartbeat-aware wait may block between pings (config-derived)."""
        g = self.cfg["general"]
        return g["watchdog_hung_thread_sec"] * g["watchdog_heartbeat_fraction"]

    # -----------------
    def wait_with_heartbeat(self, waiter, timeout_sec):
        """
        The single primitive for any potentially long blocking wait on a
        watchdog-monitored thread. Blocks for up to timeout_sec (monotonic), in
        slices no longer than _heartbeat_slice_sec(), pinging the watchdog on
        the CALLING thread before every slice and once more on completion.

        waiter(slice_sec) -> bool must block for at most slice_sec and return
        True as soon as the awaited condition holds (Event.wait has exactly
        this contract). Returns True if the condition was met, False on
        timeout. A timeout_sec of 0 or less still evaluates the condition once
        (non-blocking poll).
        """
        deadline = time.monotonic() + max(0.0, timeout_sec)
        slice_sec = self._heartbeat_slice_sec()
        while True:
            self.ping_watchdog()
            remaining = deadline - time.monotonic()
            if waiter(max(0.0, min(slice_sec, remaining))):
                self.ping_watchdog()
                return True
            if remaining <= 0.0:
                return False

    # -----------------
    def wait_event_with_heartbeat(self, event, timeout_sec):
        """Event.wait(timeout_sec) that keeps the calling thread's heartbeat alive."""
        return self.wait_with_heartbeat(event.wait, timeout_sec)

    # -----------------
    def join_with_heartbeat(self, thread, timeout_sec):
        """Thread.join(timeout_sec) that keeps the calling thread's heartbeat alive.
        Returns True if the thread has terminated."""
        def _joined(slice_sec):
            thread.join(slice_sec)
            return not thread.is_alive()
        return self.wait_with_heartbeat(_joined, timeout_sec)
    # -----------------
    # -----------------
    def _launch_fire_and_forget(self, func, timeout_sec, name, *args, **kwargs):
        """
        For operations that are started and genuinely never waited on by any
        caller (e.g. _create_timelapse_video, _finalize_video_package) --
        different from _run_bounded, which is for a caller that wants to
        wait-but-not-forever. Here there is no caller to return control to;
        the thread just needs to be watched for zombie-hood on its own.

        timeout_sec is THIS THREAD'S OWN legitimate maximum duration, not a
        shared global value -- a timelapse video encode can legitimately
        take 600+ seconds, while an astrometry solve should never take more
        than a few. Each fire-and-forget thread gets judged against its own
        realistic budget in watchdog_loop, not one uniform number.

        Returns the started Thread object.
        """
        # Monotonic sequence, not a timestamp: two launches in the same millisecond
        # produced the same name and the second silently overwrote the first's
        # zombie-tracking entry. itertools.count.__next__ is atomic under the GIL.
        unique_name = f"{name}-{next(_THREAD_SEQ)}"
        worker = threading.Thread(target=func, name=unique_name, args=args, kwargs=kwargs, daemon=True)

        with self.fire_and_forget_lock:
            self.fire_and_forget_threads[unique_name] = (worker, time.monotonic(), timeout_sec)

        worker.start()
        return worker

    def _run_bounded(self, func, timeout_sec, heartbeat_name, *args, **kwargs):
        """
        Runs `func` with a hard time ceiling (monotonic). If it doesn't return in
        time, the calling thread stops waiting and moves on -- but the
        abandoned worker thread is registered for zombie detection, so
        watchdog_loop can force a full restart (reclaiming its memory) if
        it turns out to be genuinely, permanently stuck rather than just
        slow. A daemon thread can be abandoned but not truly killed in
        Python; restarting the whole process is how that limitation gets
        resolved without switching to multiprocessing.

        Returns (result, timed_out: bool). result is None if timed_out.
        """
        result_holder = {}

        def _wrapped():
            result_holder["value"] = func(*args, **kwargs)

        unique_name = f"{heartbeat_name}-bounded-{next(_THREAD_SEQ)}"
        worker = threading.Thread(target=_wrapped, name=unique_name, daemon=True)
        worker.start()
        # Heartbeat-aware join: a plain join(timeout_sec) blocked the calling thread
        # with no ping for the full timeout, so any timeout_sec longer than
        # watchdog_hung_thread_sec got the CALLER flagged as hung while it was
        # merely waiting, as designed, on a slow-but-bounded operation.
        self.join_with_heartbeat(worker, timeout_sec)

        if worker.is_alive():
            logging.error(
                f"_run_bounded: '{heartbeat_name}' did not return within "
                f"{timeout_sec}s -- abandoning it and registering it for "
                f"zombie detection."
            )
            zombie_grace_sec = self.cfg["general"]["zombie_thread_timeout_sec"]
            with self.fire_and_forget_lock:
                self.fire_and_forget_threads[unique_name] = (worker, time.monotonic(), zombie_grace_sec)
            return None, True

        return result_holder.get("value"), False

    # -----------------
    # -----------------
