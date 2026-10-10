"""
capture_mixin.py

Camera capture loop, exposure control, calibration, and dark-frame/BPM
methods for the Ryusei pipeline, extracted from the main Pipeline class
as a pure move -- method bodies are unchanged, only their physical
location. Composed into Pipeline via multiple inheritance.

This is the final mixin extracted this session. Three non-contiguous
chunks in the original file: capture_loop through _get_dynamic_accumulate_alpha
(contiguous), _enforce_exposure_safety_limits through
_update_dynamic_dark_if_needed (contiguous), and _enqueue_timelapse_frame
through _reset_camera_failure (contiguous) -- with _validate_output_directories
sitting between the second and third groups, deliberately NOT included
here (a general utility, not capture-specific). Confirmed via an explicit
interleaving check before extraction.

Several methods documented as "capture_mixin candidates" in
detection_mixin.py's docstring from the previous extraction are included
here as promised: _enqueue_timelapse_frame, _get_dynamic_accumulate_alpha,
_heal_live_frame, _update_dynamic_dark_if_needed.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls
self._safe_save_image (FileUtilsClass), self.get_moon_state
(SkyMonitorClass), self.stack_frames (StackingClass), self.ping_watchdog
and self.stop_producer_thread (ConcurrencyClass/Pipeline's own body),
plus self.log_health_event (still directly in Pipeline's own body). All
must be provided somewhere in the composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import glob
import queue
import math
import logging
import time
import cv2
import numpy as np
from datetime import datetime, timezone

from enums import SkyConditionStatus
from constants import QUANTIZATION_SIGMA_DN
from pacing import DropReporter
from flat import load_flat_map


class CaptureClass:
    """Camera capture, exposure control, and calibration, composed into Pipeline."""

    def capture_loop(self):
        logging.info("Capture loop started. Waiting for the 'System Ready' signal...")

        # Wait until the main 'run' method signals that all threads are started.
        # This prevents this thread from producing data before consumers are ready.
        self.system_ready.wait()
        logging.info("'System Ready' signal received. Capture is now active.")

        first_run = True
        consecutive_failures = 0

        self._acq_drops = DropReporter(
            "Acquisition queue", "ACQUISITION_QUEUE",
            lambda: self.cfg["general"]["drop_report_interval_sec"], self.log_health_event,
            hint="processing slower than the camera; the frame rate is being matched",
        )
        try:
            while self.running.is_set() and self.capture_running.is_set():
                max_failures = self.cfg["general"].get("max_camera_failures")
                heartbeat_min = self.cfg["general"].get("idle_heartbeat_interval_min")

                self.ping_watchdog()

                if first_run:
                    logging.info(
                        "Capture loop started. Max failures: %d. Heartbeat: %d min.",
                        max_failures,
                        heartbeat_min,
                    )
                    first_run = False

                # Acquisition pacing: one policy, see _acquisition_pacing().
                pacing, period = self._acquisition_pacing()
                if pacing == "decimate":
                    now = time.monotonic()
                    if now < self.daylight_next_capture_time:
                        time.sleep(min(0.1, self.daylight_next_capture_time - now))
                        continue
                    self.daylight_next_capture_time = now + period

                ts = datetime.now().isoformat()
                with self.camera_lock:
                    frame, raw_frame = self.camera.read()
                    # Read under the same lock as the frame so the two can't belong to
                    # different reads (the sky monitor also reads under this lock).
                    frame_meta = getattr(self.camera, "last_meta", None)

                if frame is None:
                    # Use the new helper to track failures system-wide
                    if not self._register_camera_failure():
                        return  # Terminate this thread if fatal limit reached
                    time.sleep(1)  # Wait a bit longer after a single failure
                    continue

                # If a frame is read successfully, reset the global counter
                self._reset_camera_failure()

                # Save the last raw frame to RAM, tagged with the exposure/gain it was
                # really captured with (frame metadata; falls back to the applied
                # settings for cameras that don't report it, e.g. the simulator).
                self.last_raw_frame = frame
                self.last_raw_frame_bayer10 = raw_frame
                self._remember_frame(
                    frame,
                    (frame_meta or {}).get("exposure_us") or self.current_exposure_us,
                    (frame_meta or {}).get("gain") or self.current_gain,
                )

                try:
                    self.acq_q.put_nowait((ts, frame))
                except queue.Full:
                    # Rare now: under the "sensor" strategy the frame period follows the
                    # measured processing cost. Transients are counted, reported per interval.
                    self._acq_drops.dropped()
                self._acq_drops.tick()

                if pacing == "sensor":
                    self._match_frame_rate_to_processing()
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN CAPTURE LOOP THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "CAPTURE_LOOP_CRASH", str(e))

        logging.info("Capture loop has exited.")

    # -----------------
    def update_operating_mode(self):
        """
        Checks camera's illuminance and dynamically reconfigures the camera
        for either 'Daylight Mode' (auto) or 'Night Mode' (manual) without
        stopping the capture thread.
        """
        lux_threshold = self.cfg.get("daylight", {}).get("lux_threshold")
        night_exposure_us = self.cfg["capture"].get("exposure_us")
        night_gain = self.cfg["capture"].get("gain")

        # Use a lock to ensure the capture_loop pauses while we change settings.
        with self.camera_lock:
            try:
                # OPTIMIZATION: RAM-BASED FAST-PATH LIGHT CHECK
                # If we are in stable NIGHT mode, we check the last frame already in RAM!
                if not self.daylight_mode_active.is_set() and getattr(self, "auto_exposure_converged", False):

                    # Toggle between memory and hardware to guarantee accuracy
                    self.use_ram_for_check = not getattr(self, "use_ram_for_check", False)

                    meta = {}
                    if self.use_ram_for_check:
                        logging.info("Fast Light Check: Using memory frame.")
                        frame = getattr(self, "last_raw_frame", None)
                        current_check_source = "memory"
                    else:
                        logging.info(
                            "Fast Light Check: Forcing hardware frame to ensure accuracy."
                        )
                        try:
                            # Grab the frame AND the metadata directly from the hardware!
                            req = self.camera.picam2.capture_request()
                            img_array = req.make_array("main")
                            meta = req.get_metadata()
                            req.release()

                            if img_array is not None:
                                # frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                frame = img_array
                            else:
                                frame = None
                            current_check_source = "acquired"
                        except Exception as e:
                            logging.warning(f"Hardware frame request failed: {e}")
                            frame = None

                    if frame is not None:
                        # Remove hot pixels before calculating brightness
                        frame = self._heal_live_frame(frame)
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        mean_bright = float(np.mean(gray))

                        if current_check_source == "acquired" and "Lux" in meta and meta["Lux"] > 0:
                            current_lux = meta["Lux"]
                        else:
                            current_lux = self.calculate_lux(mean_bright, self.current_exposure_us, self.current_gain)

                        # Lowered threshold to 150.0 to catch morning twilight safely
                        # if mean_bright < 150.0:    # 300 lux
                        if current_lux < self.cfg["daylight"].get("lux_threshold"):
                            # CALCULATE REAL K FROM THE NIGHT SKY!
                            if (current_check_source == "acquired" and "Lux" in meta and meta["Lux"] > 0):
                                hw_lux = current_lux
                                hw_exp = meta.get("ExposureTime", self.current_exposure_us)
                                hw_gain = meta.get("AnalogueGain", self.current_gain)

                                exposure_sec = hw_exp / 1_000_000.0
                                denominator = hw_lux * exposure_sec * hw_gain

                                if denominator > 0:
                                    self.calibration_k = mean_bright / denominator
                                    self.camera_k_logger()
                                    logging.info(f"Night Calibration: True K updated from deep sky frame (k={self.calibration_k:.2f})")

                                current_lux = hw_lux
                            else:
                                # Memory path (or metadata missing): Calculate lux using our established K
                                current_lux = self.calculate_lux(mean_bright,self.current_exposure_us,self.current_gain,)

                            with self.status_lock:
                                self.session_stats["avg_lux"].append(current_lux)
                                self.last_illuminance = f"{current_lux:.5f}"
                                self.last_light_check_source = current_check_source

                            logging.info(
                                f"Fast Light Check: {current_lux:.5f} Lux (Night confirmed, skipped DAY preset)."
                            )
                            return True
                        else:
                            logging.info("Fast Light Check: Frame saturated (Twilight). Proceeding to full DAY check.")

                # Temporarily switch to fast exposure to get a lux starting point
                self._set_camera_preset("SNAP")
                # time.sleep(2) # Allow sensor to adjust
                frame = None
                metadata = {}

                while self.running.is_set() and self.capture_running.is_set():
                    frame, _ = self.camera.read()

                    if frame is None:
                        # Register failure. If it returns False, we reached max limit.
                        if not self._register_camera_failure():
                            try:
                                self._set_camera_preset("NIGHT")
                            except Exception as e:
                                logging.warning("Could not restore NIGHT preset after camera failure: %s",e)
                            return False  # Abort entirely
                        time.sleep(1)
                        continue  # Retry acquisition

                    # Frame successfully acquired! Reset counter.
                    frame = self._heal_live_frame(frame)
                    self._reset_camera_failure()

                    try:
                        request = self.camera.picam2.capture_request()
                        metadata = request.get_metadata()
                        request.release()
                    except Exception as e:
                        logging.warning(f"Metadata extraction failed: {e}")

                    break  # Break out of the retry loop successfully

                # if frame is None:
                    # return False  # Thread was stopped

                # Capture the hardware's baseline choices
                hw_exp = metadata.get("ExposureTime", self.current_exposure_us)
                hw_gain = metadata.get("AnalogueGain", self.current_gain)

                # Analyze the actual brightness of the hardware's auto-frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_bright = float(np.mean(gray))

                # Save these for perform_auto_exposure() to use!
                self.ae_hw_exposure = hw_exp
                self.ae_hw_gain = hw_gain
                self.ae_hw_brightness = mean_bright

                # Failsafe for missing Lux
                if "Lux" not in metadata:
                    logging.warning(
                        "'Lux' missing from metadata. Estimating manually..."
                    )
                    # Same Lux model as everywhere else (calibrated k when valid).
                    current_lux = self.calculate_lux(mean_bright, hw_exp, hw_gain)
                else:
                    current_lux = metadata["Lux"]

                with self.status_lock:
                    self.session_stats["avg_lux"].append(current_lux)
                    self.last_illuminance = f"{current_lux:.2f}"
                    self.last_light_check_source = "acquired (SNAP preset)"

                # _, moon_in_fov, _, _ = self.get_moon_state()

                logging.info(
                    f"Light Check: {current_lux:.1f} Lux. HW Baseline: Exp={hw_exp}us, Gain={hw_gain:.2f}, Bright={mean_bright:.1f}"
                )

                # CALCULATE CALIBRATION K
                # Formula: K = Brightness / (Lux * Exposure_sec * Gain)
                if frame is not None and current_lux > 0:
                    exposure_sec = hw_exp / 1_000_000.0
                    denominator = current_lux * exposure_sec * hw_gain

                    if denominator > 0:
                        self.calibration_k = mean_bright / denominator
                        self.camera_k_logger()

                logging.info(
                    f"Light level check: Current illuminance is {current_lux:.1f} Lux). (k={self.calibration_k:.2f})"
                )

                # Reconfigure camera for Day or Night
                if current_lux > lux_threshold:
                    # DAYLIGHT MODE
                    if not self.daylight_mode_active.is_set():
                        logging.warning(
                            "Illuminance is HIGH. Switching camera to DAYLIGHT MODE."
                        )
                        self.daylight_mode_active.set()  #
                        self.effective_stack_n = self.daylight_stack_n
                        logging.info(
                            f"Change the stack value from night: {self.cfg["timelapse"]["nighttime_stack_N"]} to day: {self.effective_stack_n}"
                        )

                        with self.status_lock:
                            self.effective_threshold = self.cfg["detection"][
                                "base_threshold"
                            ]
                            self.last_effective_threshold = "N/A (Daylight)"

                        # Clear the queue of old high-gain night frames
                        while not self.timelapse_q.empty():
                            try:
                                self.timelapse_q.get_nowait()
                            except:
                                break

                        self.base_timelapse_interval = self.cfg["timelapse"].get("daymode_interval_sec")
                        self.timelapse_interval_sec = self.base_timelapse_interval

                        # capture_interval_sec only throttles how often the camera is
                        # read (input rate); it's daymode_interval_sec (via the gate
                        # in _enqueue_timelapse_frame) that actually paces the output.
                        # But if capture_interval_sec were ever set >= daymode_interval_sec,
                        # a fresh frame wouldn't even be offered often enough to hit that
                        # target, and the output cadence would silently degrade to
                        # whatever capture_interval_sec is instead. Warn so a
                        # misconfiguration is visible rather than silently discovered.
                        daylight_capture_interval = self.cfg["daylight"]["capture_interval_sec"]
                        if daylight_capture_interval >= self.timelapse_interval_sec:
                            logging.warning(
                                "daylight.capture_interval_sec (%.1fs) >= timelapse.daymode_interval_sec (%.1fs): "
                                "the daytime timelapse cadence will be limited by the capture throttle, not the configured interval.",
                                daylight_capture_interval,
                                self.timelapse_interval_sec,
                            )
                    self._set_camera_preset("DAY")
                else:
                    # NIGHT MODE
                    if self.daylight_mode_active.is_set():
                        logging.info(
                            "Illuminance is LOW. Switching camera to NIGHT MODE."
                        )
                        self.daylight_mode_active.clear()
                        self.effective_stack_n = self.cfg["timelapse"][
                            "nighttime_stack_N"
                        ]
                        logging.info(
                            f"Change the stack value from day: {self.effective_stack_n} to night: {self.cfg["timelapse"]["nighttime_stack_N"]}"
                        )
                        self.base_timelapse_interval = self.cfg["timelapse"].get(
                            "nightmode_interval_sec"
                        )
                        # self.timelapse_interval_sec = self.cfg["timelapse"].get("nightmode_interval_sec")
                        self.timelapse_interval_sec = self.base_timelapse_interval
                        self.current_exposure_us = night_exposure_us
                        self.current_gain = night_gain

                        # New night starting: any previously cached plate
                        # solve is from a prior night (or stale if the
                        # camera moved) and must not be reused -- force a
                        # fresh solve_field_center_roi() this night.
                        with self.astrometry_lock:
                            self.wcs_cache = None
                    self._set_camera_preset("NIGHT")
                return True

            except Exception:
                logging.error(
                    "Failed to set camera controls in update_operating_mode.",
                    exc_info=True,
                )
                # As a fallback, restore night settings to prevent being stuck in a bad state
                try:
                    self.current_exposure_us = night_exposure_us
                    self.current_gain = night_gain
                    self._set_camera_preset("NIGHT")
                except Exception as e:
                    logging.warning(
                        "Could not restore NIGHT preset after auto-exposure: %s", e
                    )
                return False

    # -----------------
    @staticmethod
    def _band_overlap_deg(a, b, lo, hi):
        """Length (deg) of the altitude segment a..b that lies inside [lo, hi]."""
        seg_lo, seg_hi = (a, b) if a <= b else (b, a)
        return max(0.0, min(seg_hi, hi) - max(seg_lo, lo))

    # -----------------
    def _ae_expected_change_stops(self, t_from, t_to):
        """
        Upper bound, in stops (log2 of exposure*gain), on how far the CORRECT
        night exposure can legitimately have moved between two moments, from
        sky geometry alone:

          base                 a plain night: small drift only
        + twilight             ae_twilight_stops_per_deg x degrees of Sun
                               altitude travelled inside the twilight band
        + moon                 ae_moon_stops_per_deg x illuminated fraction x
                               degrees of Moon altitude travelled above
                               ae_moon_alt_min (scattered moonlight)

        The Sun/Moon path is sampled every ae_sample_step_min minutes rather
        than compared end-to-end, so a gap of hours (or a whole day between
        two nights, when both endpoints sit deep in the night) still counts
        the twilight it crossed.

        Returns (allowed_stops, detail) or (None, detail) if it can't be
        computed, in which case the caller must NOT clamp: failing open at
        dusk/dawn is safer than clamping on a bad number.
        """
        cfg = self.cfg["capture"]
        detail = {"base": cfg["ae_max_stops_per_check"], "twilight": 0.0, "moon": 0.0}
        try:
            import ephem

            obs = ephem.Observer()
            obs.lat = str(self.cfg["general"]["latitude"])
            obs.lon = str(self.cfg["general"]["longitude"])

            step_s = cfg["ae_sample_step_min"] * 60.0
            n = max(1, int(math.ceil(max(0.0, t_to - t_from) / step_s)))
            tw_lo, tw_hi = cfg["ae_twilight_sun_alt_min"], cfg["ae_twilight_sun_alt_max"]
            moon_lo = cfg["ae_moon_alt_min"]

            tw_deg = 0.0
            moon_weighted_deg = 0.0
            prev_sun = prev_moon = None
            for i in range(n + 1):
                ts = t_from + (t_to - t_from) * i / n
                # ephem reads the calendar fields as UTC, so hand it naive UTC fields.
                obs.date = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
                sun_alt = math.degrees(float(ephem.Sun(obs).alt))
                moon = ephem.Moon(obs)
                moon_alt = math.degrees(float(moon.alt))
                if prev_sun is not None:
                    tw_deg += self._band_overlap_deg(prev_sun, sun_alt, tw_lo, tw_hi)
                    moon_weighted_deg += float(moon.moon_phase) * self._band_overlap_deg(
                        prev_moon, moon_alt, moon_lo, 90.0
                    )
                prev_sun, prev_moon = sun_alt, moon_alt

            detail["twilight"] = cfg["ae_twilight_stops_per_deg"] * tw_deg
            detail["moon"] = cfg["ae_moon_stops_per_deg"] * moon_weighted_deg
            return detail["base"] + detail["twilight"] + detail["moon"], detail
        except Exception as e:
            logging.warning(f"Auto-exposure: could not compute the expected-change limit ({e}); not clamping.")
            return None, detail

    # -----------------
    def _ae_record_candidate(self, clamped, target, frame=None):
        """
        Remember what this run settled on; _ae_finalize_reference decides whether it
        becomes the reference. `target` is the brightness this run was aiming at
        (stored so a later check can separate a deliberate target change from a
        sky change; see _ae_limit_bounds). `frame` is the run's final verification
        frame, passed only if it was confirmed in sync with the settings, so it can
        seed the in-memory frame the lightweight re-check reads.
        """
        self.ae_candidate = self._ae_make_reference(
            self.current_exposure_us * self.current_gain, target
        )
        self.ae_candidate["clamped"] = clamped
        self.ae_last_run_clamped = clamped
        if frame is not None:
            self._remember_frame(frame, self.current_exposure_us, self.current_gain)

    # -----------------
    def _frame_period_sec(self):
        """
        Seconds between frames as the pipeline actually experiences them: the camera's
        real frame duration (metadata; falls back to the applied exposure), or the
        detection loop's processing time if that is slower -- frames then reach the
        timelapse/calibration consumers at the processing rate, not the sensor rate.
        """
        # max(): right after a settings change (day -> night, or a test shot) the last
        # metadata duration still describes the OLD mode (e.g. 33 ms of daylight
        # free-run) while the sensor is already integrating the new, longer exposure.
        fd_us = max(getattr(self.camera, "_last_frame_duration_us", None) or 0, self.current_exposure_us or 0)
        with self.status_lock:
            loop_ms = self.session_stats.get("loop_time_ms", 0.0) or 0.0
        return max(fd_us / 1_000_000.0, loop_ms / 1000.0)

    # -----------------
    def _frame_wait_budget_sec(self):
        """
        Longest a consumer can legitimately wait for the NEXT frame: the camera sync
        depth (buffers + control delay + the frame itself, see camera_capture.py)
        times the frame period, plus the sync slack.
        """
        cap = self.cfg["capture"]
        depth = cap["raw_buffer_count"] + cap["sync_control_delay_frames"] + 1
        return depth * self._frame_period_sec() + cap["sync_timeout_slack_sec"]

    # -----------------
    def _stack_completion_budget_sec(self):
        """Time for the timelapse stack in flight to finish: its frames plus stacking CPU slack."""
        return self.effective_stack_n * self._frame_period_sec() + self.cfg["capture"]["safe_window_slack_sec"]

    # -----------------
    @staticmethod
    def _ae_make_reference(product, target):
        """
        A trusted auto-exposure operating point: the exposure*gain product, the
        brightness target it was converged/aimed at, and when (wall clock -- it
        feeds the Sun/Moon ephemeris in _ae_expected_change_stops).
        """
        return {"product": float(product), "target": float(target), "t": time.time()}

    # -----------------
    def _remember_frame(self, frame, exposure_us, gain):
        """
        Keep the most recent frame together with the exposure/gain it was really
        captured with. One dict, replaced by a single assignment, so a reader on
        another thread always sees a frame and its settings from the same moment.
        Fed by the capture loop on every read, and by a confirmed-in-sync auto-
        exposure verification frame, so it is populated right after a full check too
        (during a hunt the capture loop can't read: the hunt holds the camera lock).
        """
        self.last_raw_frame_info = {
            "frame": frame,
            "exposure_us": exposure_us,
            "gain": gain,
            "t": time.monotonic(),  # only ever used for frame AGE (see _ae_fast_recheck)
        }

    # -----------------
    def _ae_target(self):
        """Target brightness (raised while the Moon is in the FOV), saturation ceiling, moon impact, in_fov."""
        cfg = self.cfg["capture"]
        moon_impact, in_fov, _, _ = self.get_moon_state()
        base_target = cfg.get("target_sky_brightness")
        target = (base_target + (cfg.get("moon_normal_penalty") * moon_impact) if in_fov else base_target)
        base_max = cfg["max_brightness"]
        effective_max = min(
            255,
            base_max + (cfg.get("moon_max_penalty") * moon_impact if in_fov else 0),
        )
        return target, effective_max, moon_impact, in_fov

    # -----------------
    def _ae_limit_bounds(self, target_now):
        """
        Allowed exposure*gain range around the last trusted check.

        Two independent causes move the correct product, and they are treated
        differently because one is KNOWN and the other only BOUNDED:

          * the target itself (e.g. raised while the Moon is in the FOV): a
            deliberate, exactly known change. Under the controller's own linear
            model (brightness proportional to exposure*gain, the same model its
            correction step ratio = target / brightness uses), it moves the correct
            product by exactly target_now / ref_target. So the window is re-CENTRED
            on ref_product * target_now / ref_target -- not widened: widening by
            |log2(ratio)| would also admit a skewed reading of that size in the
            opposite direction.
          * the sky: unknown, bounded by geometry (_ae_expected_change_stops) --
            the +/- allowed-stops half-width around that centre.

        Returns (bounds, allowed_stops, detail, reference); bounds is None when there
        is no reference yet or the allowance can't be computed (then nothing is clamped).
        """
        ref = getattr(self, "ae_reference", None)
        if not ref:
            return None, None, None, None
        allowed, detail = self._ae_expected_change_stops(ref["t"], time.time())
        if allowed is None:
            return None, None, detail, ref
        target_ratio = target_now / ref["target"]
        detail["target_shift"] = math.log2(target_ratio)
        centre = ref["product"] * target_ratio
        span = 2.0 ** allowed
        return (centre / span, centre * span), allowed, detail, ref

    # -----------------
    def _ae_reschedule_recheck(self):
        self.ae_recheck_due_at = time.monotonic() + self.cfg["capture"]["ae_clamp_recheck_min"] * 60

    # -----------------
    def _ae_reset_background_if_changed(self, old_exp, old_gain, new_exp, new_gain):
        """
        A full check that actually moves exposure/gain leaves the very next frame
        at a different brightness than the motion background was accumulated at
        -- comparing them produces a diff across the whole frame, which reads as
        motion and starts a false event right as the check hands back to
        detection (this is what was happening: the moon FOV control forced a
        big enough jump, with no matching reset, for exactly that to happen).
        _ae_fast_recheck already resets background after its own corrections;
        this brings the full check in line with it. Skipped mid-event for the
        same reason as everywhere else: background is never updated while an
        event is in progress anyway, so nulling it out helps nothing for the
        event and only forces a slow rebuild right when a stable background is
        needed to detect the event cooling down.
        """
        if new_exp == old_exp and new_gain == old_gain:
            return
        if self.event_in_progress.is_set():
            return
        self.background = None
        self.background_reset_time = time.monotonic()

    # -----------------
    def _ae_fast_recheck(self):
        """
        Lightweight follow-up to a CLAMPED full check, so a cloud (or light) that skewed
        it doesn't leave the exposure wrong until the next full check. It costs the
        acquisition nothing while it only looks:

          * brightness comes from the frame already in memory (last_raw_frame_info) --
            no pause, no camera access, no hunt;
          * only if that shows the settings are off does it touch the camera, and then
            only after the current timelapse stack has finished, with a single
            correction computed straight from the brightness ratio.

        Run by the sky monitor thread from its idle wait, so it can never overlap a full
        check; and any full auto-exposure run clears the pending re-check
        (ae_recheck_due_at), so a re-check that would have fallen inside a check is
        simply dropped and the full check does its job.

        The correction obeys the same change limit as a full check. It is applied even
        while the sky reads cloudy (that's what pulls the settings back once the cloud
        has gone), but only adopted as the new trusted reference on a clear sky.
        """
        cfg = self.cfg["capture"]
        self.ae_recheck_due_at = None  # re-armed below if there is still something to follow up

        # Situations where a follow-up doesn't apply or must not touch anything.
        if (
            self.daylight_mode_active.is_set()
            or self.sky_monitor_in_progress.is_set()
            or not self.capture_stable.is_set()
            or not self.producer_thread_active()
            or not getattr(self, "auto_exposure_converged", False)
        ):
            logging.info("Auto-exposure re-check: dropped (day mode / a full check is running or due / acquisition not running).")
            return

        cur_exp, cur_gain = self.current_exposure_us, self.current_gain
        cur_product = cur_exp * cur_gain
        cycle_s = self._frame_period_sec()

        # A usable frame: recent, and captured with the settings that are applied NOW.
        # (Right after a change the camera still hands out frames taken with the old
        # settings for a few frames; their brightness says nothing about the new ones.)
        info = getattr(self, "last_raw_frame_info", None)
        usable = False
        if info and info["frame"] is not None:
            age_ok = (time.monotonic() - info["t"]) <= cfg["ae_recheck_max_frame_age_frames"] * cycle_s
            # Same definition of "taken with these settings" as the flush() sync.
            usable = age_ok and self.camera.settings_match(
                info["exposure_us"], info["gain"], cur_exp, cur_gain
            )
        if not usable:
            self.ae_recheck_due_at = time.monotonic() + cycle_s  # the next frame is one exposure away
            return

        brightness = float(np.mean(cv2.cvtColor(info["frame"], cv2.COLOR_BGR2GRAY)))
        target, _, _, _ = self._ae_target()
        tolerance = cfg.get("tolerance")
        # Same criterion the full check's reference adoption uses (the sky monitor's
        # last status == CLEAR). weather_hold_active is NOT equivalent: it engages only
        # after the state machine's hysteresis, so a sky already reading CLOUDY could
        # still pass as "clear" and a cloud-skewed operating point would become the
        # trusted baseline -- precisely what the reference logic exists to prevent.
        sky_clear = self.last_sky_status == SkyConditionStatus.CLEAR

        if (target - tolerance) <= brightness <= (target + tolerance):
            logging.info(f"Auto-exposure re-check: brightness {brightness:.1f} is on target ({target:.1f} +/-{tolerance}); settings are fine.")
            if sky_clear:
                self.ae_reference = self._ae_make_reference(cur_product, target)
            return

        # Same damping and change limit as a full check's correction step.
        # Floor = uint8 quantisation noise: below it the mean carries no information.
        # The step-limit clamp right below bounds the ratio anyway.
        ratio = target / max(brightness, QUANTIZATION_SIGMA_DN)
        ratio = (
            min(ratio, cfg.get("ae_step_limit_up")) if ratio > 1.0
            else max(ratio, cfg.get("ae_step_limit_down"))
        )
        required = cur_product * ratio
        bounds, allowed, _detail, _ref = self._ae_limit_bounds(target)
        clamped = False
        if bounds is not None:
            limited = min(max(required, bounds[0]), bounds[1])
            clamped = limited != required
            required = limited
        new_gain, new_exp = self._select_gain_exposure(
            required, cfg["min_gain"], cfg["max_gain"], cfg["min_exposure_us"], cfg["max_exposure_us"]
        )

        if new_gain == cur_gain and new_exp == cur_exp:
            logging.info(
                f"Auto-exposure re-check: brightness {brightness:.1f} vs target {target:.1f} but no further move is "
                f"possible ({'change limit' if clamped else 'exposure/gain limits'}); leaving settings as they are."
            )
            if clamped:
                self._ae_reschedule_recheck()  # the limit will have moved by then; look again
            return

        logging.info(
            f"Auto-exposure re-check: brightness {brightness:.1f} vs target {target:.1f} -> "
            f"Exp {cur_exp}us->{new_exp}us, Gain {cur_gain:.1f}->{new_gain:.1f}"
            f"{' (held back by the change limit)' if clamped else ''}. Waiting for a safe window to apply."
        )

        # Apply only once the timelapse stack in flight has finished (and no event is being
        # recorded): wait_for_safe_window() does exactly that and returns False on shutdown.
        if not self.wait_for_safe_window():
            return

        # Things can change while waiting. The re-validation and the apply form ONE
        # critical section under camera_lock -- the lock every camera-settings writer
        # takes -- so no other writer can change the settings between "still what
        # the correction was computed from" and set_controls. (Before, the check ran
        # outside the lock: a full check or a camera re-init landing in that gap
        # would have been overwritten with a correction computed for settings that
        # no longer existed.) An event that started after wait_for_safe_window is
        # also re-checked here: changing exposure mid-event would corrupt its video
        # and its motion background.
        ok = False
        with self.camera_lock:
            unchanged = (
                not self.daylight_mode_active.is_set()
                and not self.sky_monitor_in_progress.is_set()
                and not self.event_in_progress.is_set()
                and self.capture_stable.is_set()
                and self.producer_thread_active()
                and self.current_exposure_us == cur_exp
                and self.current_gain == cur_gain
            )
            if unchanged:
                # Applying the correction syncs the camera (tens of seconds at long
                # exposures) while holding camera_lock: acquisition is paused exactly as
                # in a routine check, so it goes through the same detection gate --
                # which also discards the pre-event frames from before the pause.
                self.capture_stable.clear()
                try:
                    ok, _ = self._set_camera_preset("NIGHT", ExposureTime=new_exp, AnalogueGain=new_gain)
                finally:
                    self.capture_stable.set()
        if not unchanged:
            logging.info("Auto-exposure re-check: conditions changed while waiting; not applying.")
            if clamped:
                self._ae_reschedule_recheck()  # still a skewed-check follow-up to do; look again later
            return
        if not ok:
            logging.warning("Auto-exposure re-check: applying the new settings failed; keeping the old ones.")
            return

        # Same follow-ups a full check leaves behind: dark frame for the new exposure, and a
        # fresh motion background (the whole frame just changed brightness).
        self.last_dark_update_time = float("-inf")  # force a dark refresh on the next frame
        if not self.event_in_progress.is_set():
            self.background = None
            self.background_reset_time = time.monotonic()

        if sky_clear:
            # The APPLIED values: _set_camera_preset may have clamped the request to
            # the sensor's range, and current_* hold what the hardware really runs.
            self.ae_reference = self._ae_make_reference(
                self.current_exposure_us * self.current_gain, target
            )
        if clamped:
            self._ae_reschedule_recheck()

    # -----------------
    def _ae_finalize_reference(self, sky_is_clear):
        """
        Called by the sky monitor once the sky status of this check is known
        (auto-exposure runs BEFORE the cloud analysis, so it can't know).

        The result of a check becomes the reference the NEXT check is limited
        against -- except when it was clamped AND the sky isn't clear: that is
        the signature of a cloud (or a light) skewing the reading, so the last
        trusted reference is kept instead of letting the skewed value become
        the new baseline. A clamped result on a clear sky is a real change
        (city lights switching off, ...) and is adopted, so the settings can
        keep following it check after check.
        """
        cand = getattr(self, "ae_candidate", None)
        self.ae_candidate = None
        if not cand:
            return
        if cand["clamped"]:
            # Whatever the sky state, a clamped check is followed up soon by the
            # lightweight in-memory re-check (see _ae_fast_recheck).
            self._ae_reschedule_recheck()
        if cand["clamped"] and not sky_is_clear:
            logging.info(
                "Auto-exposure: this check was clamped and the sky is not clear -- "
                "keeping the last trusted reference instead of adopting it."
            )
            return
        self.ae_reference = {k: cand[k] for k in ("product", "target", "t")}

    # -----------------
    def perform_auto_exposure(self):
        """
        Mathematical Auto-Exposure with Iterative Verification.
        Attempts to nail the exposure on the first try using Lux & K.
        If the verification frame fails, it gracefully loops to correct and verify again.

        Change limit: when starting from previously converged values, the
        exposure*gain product may not move more than _ae_expected_change_stops()
        away from the last trusted check (self.ae_reference). See
        _ae_finalize_reference for how that reference is maintained.
        """
        logging.info("Performing mathematical auto-exposure with verification...")
        self.ae_candidate = None
        self.ae_last_run_clamped = False
        # A full check supersedes any pending lightweight re-check: if that one's time
        # fell inside (or right before) this check, it is simply dropped and this check
        # does the job. _ae_finalize_reference re-arms it if this check is clamped too.
        self.ae_recheck_due_at = None
        ae_clamped = False
        cfg = self.cfg["capture"]
        # SAVE SAFE FALLBACKS
        safe_night_exp = self.current_exposure_us
        safe_night_gain = self.current_gain

        # Target calculation based on moon interference
        target, effective_max, moon_impact, in_fov = self._ae_target()

        tolerance = cfg.get("tolerance")
        max_attempts = cfg.get("max_attempts")

        min_gain, max_gain = cfg["min_gain"], cfg["max_gain"]
        min_exp, max_exp = cfg["min_exposure_us"], cfg["max_exposure_us"]

        # Reset convergence flag if moon visibility changed since last AE run.
        # A moon rise/set shifts the optimal exposure dramatically; starting
        # from the old converged values would need several iterations to catch up.
        last_ae_moon_in_fov = getattr(self, "_last_ae_moon_in_fov", None)
        if last_ae_moon_in_fov is not None and last_ae_moon_in_fov != in_fov:
            logging.info(
                f"Auto-exposure: Moon FOV state changed ({last_ae_moon_in_fov} -> {in_fov}). "
                f"Resetting convergence to recalculate from hardware baseline."
            )
            self.auto_exposure_converged = False
        self._last_ae_moon_in_fov = in_fov

        # Determine Initial Guess
        if not getattr(self, "auto_exposure_converged", False):
            # FIRST RUN: Grab the Hardware Baseline captured milliseconds ago
            hw_exp = getattr(self, "ae_hw_exposure", self.current_exposure_us)
            hw_gain = getattr(self, "ae_hw_gain", self.current_gain)
            # A black baseline frame (mean below the uint8 quantisation noise) carries no
            # brightness information; floor it there. The resulting product is bounded
            # by the exposure/gain limits in _select_gain_exposure.
            hw_bright = max(getattr(self, "ae_hw_brightness", QUANTIZATION_SIGMA_DN), QUANTIZATION_SIGMA_DN)

            # Calculate Correction Ratio
            ratio = target / hw_bright
            required_product = (hw_exp * hw_gain) * ratio

            # Select the lowest integer gain that keeps exposure
            # within [min_exp, max_exp]. Lower gain = better SNR, and each
            # integer maps to an existing dark frame file.
            new_gain, new_exposure = self._select_gain_exposure(
                required_product, min_gain, max_gain, min_exp, max_exp
            )

            logging.info(
                f"Auto-exposure: First run. Req. Light={required_product:.0f} -> Exp={new_exposure}us, Gain={new_gain}"
            )
        else:
            # SUBSEQUENT RUNS: Start from the last converged values to speed up checks
            new_exposure = self.current_exposure_us
            new_gain = self.current_gain
            logging.info("Auto-exposure: Starting from previous converged values.")

        # CHANGE LIMIT against the last trusted check. Based on self.ae_reference,
        # NOT on auto_exposure_converged: that flag only says which values this
        # check starts iterating from (hardware baseline vs. previous settings)
        # and is reset on a moon-FOV change to seed a better starting guess --
        # it says nothing about whether the last trusted check is still a
        # meaningful reference. A moon-FOV reset mid-session still has one, and
        # the moon moves slowly, so its effect on the target is already folded
        # into the allowance by _ae_expected_change_stops (ae_moon_stops_per_deg),
        # and the step change of the TARGET it causes (moon_normal_penalty) is
        # applied exactly by re-centring the window (see _ae_limit_bounds); it must
        # still be limited like any other check, not treated as an unclamped first run. _ae_limit_bounds() itself returns no bounds when
        # there is no reference yet (a real first run, or straight after startup),
        # and effectively no limit across a day gap (the accumulated twilight
        # allowance covers it) -- so calling it unconditionally is always safe.
        ae_bounds, allowed, detail, ref = self._ae_limit_bounds(target)
        if ae_bounds is not None:
            logging.info(
                f"Auto-exposure: change limit +/-{allowed:.2f} stops vs last trusted check "
                f"(base {detail['base']:.2f} + twilight {detail['twilight']:.2f} + moon {detail['moon']:.2f}), "
                f"centre shifted {detail['target_shift']:+.2f} stops for target {ref['target']:.1f}->{target:.1f} "
                f"-> exposure*gain must stay within {ae_bounds[0]:.0f}..{ae_bounds[1]:.0f} (reference {ref['product']:.0f})."
            )
            start_product = new_exposure * new_gain
            clamped_start = min(max(start_product, ae_bounds[0]), ae_bounds[1])
            if clamped_start != start_product:
                # Not flagged as a "clamped" result: the starting values can be
                # legitimately off-reference (e.g. reset to the config values at
                # the day->night switch); only a correction the limit had to
                # cut short below counts as a skewed reading.
                new_gain, new_exposure = self._select_gain_exposure(
                    clamped_start, min_gain, max_gain, min_exp, max_exp
                )
                logging.warning(
                    f"Auto-exposure: starting values were outside the change limit; "
                    f"starting from Exp={new_exposure}us, Gain={new_gain:.1f} instead."
                )

        sync_retries = self.cfg["capture"]["exposure_sync_retries"]

        # The Verification Loop
        for attempt in range(max_attempts):
            # Each attempt can take tens of seconds (slow camera exposure-sync
            # flush), so a full multi-attempt hunt can run for minutes. Ping here
            # so the watchdog doesn't mistake a legitimately slow hunt for a hang.
            self.ping_watchdog()

            if not self.running.is_set() or not self.capture_running.is_set():
                logging.warning("Auto-exposure aborted: Shutdown initiated.")
                return

            # ANTI-LOOP: Track the values we are testing to detect if we get stuck
            test_exp = new_exposure
            test_gain = new_gain
            logging.info(
                f"Auto-exposure Attempt {attempt + 1}/{max_attempts} — Exp: {new_exposure}us | Gain: {new_gain:.2f}"
            )

            # Apply the settings and get a frame the sensor has CONFIRMED is at the
            # new exposure. A timed-out sync (synced_frame is None) means the frame
            # from camera.read() is still integrating at the previous exposure --
            # measuring it makes the AE controller chase a stale reading and settle
            # one step too aggressive (the underexposed-timelapse bug). Re-apply and
            # wait again; only measure an unsynced frame as a last resort.
            frame = None
            frame_synced = False
            for sync_try in range(sync_retries):
                self.ping_watchdog()
                with self.camera_lock:
                    success, synced_frame = self._set_camera_preset(
                        "NIGHT", ExposureTime=new_exposure, AnalogueGain=new_gain
                    )
                if synced_frame is not None:
                    frame = synced_frame
                    frame_synced = True
                    break
                if not self.running.is_set() or not self.capture_running.is_set():
                    return
                logging.warning(
                    "Auto-exposure: sensor did not confirm exposure sync "
                    "(try %d/%d) — retrying before measuring.",
                    sync_try + 1,
                    sync_retries,
                )

            if frame is None:
                logging.warning(
                    "Auto-exposure: exposure never confirmed after %d tries; "
                    "measuring an unsynced frame (reading may lag one step).",
                    sync_retries,
                )
                with self.camera_lock:
                    frame, _ = self.camera.read()

            if frame is None:
                self._register_camera_failure()
                logging.error(
                    "Auto-exposure: Frame capture failed during verification. Reverting to hardware baseline."
                )
                self.current_exposure_us = safe_night_exp
                self.current_gain = safe_night_gain
                self.snapshot_frame = None
                return

            self._reset_camera_failure()  # <--- Reset on success

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            actual_brightness = float(np.mean(gray))

            logging.info(
                f"Auto-exposure Verification: Actual Brightness={actual_brightness:.1f} (Target={target:.1f} ±{tolerance})"
            )

            # Did we hit the target?
            if (target - tolerance) <= actual_brightness <= (target + tolerance):
                logging.info(
                    f"Auto-exposure: Target accurately acquired on attempt {attempt + 1}. Use: exp {new_exposure}, gain {new_gain}"
                )
                self._ae_reset_background_if_changed(safe_night_exp, safe_night_gain, new_exposure, new_gain)
                self.current_exposure_us = new_exposure
                self.current_gain = new_gain
                self.auto_exposure_converged = True
                self._ae_record_candidate(ae_clamped, target, frame if frame_synced else None)
                # FORCE DARK FRAME RECALCULATION
                self.last_dark_update_time = float("-inf")
                self.snapshot_frame = frame
                return  # Success

            # Calculate Correction for Next Attempt
            if attempt < max_attempts - 1:  # If we have more attempts left
                if actual_brightness > effective_max:
                    # SATURATED: Drastic failsafe reduction
                    backoff = cfg["ae_saturation_backoff"]
                    logging.warning(
                        f"Verification SATURATED ({actual_brightness:.1f}). Backing exposure off x{backoff}."
                    )
                    new_exposure = max(min_exp, int(new_exposure * backoff))
                    if new_exposure == min_exp:
                        new_gain = max(min_gain, new_gain * backoff)
                else:
                    # OFF-TARGET: Proportional ratio correction

                    # Prevent division by tiny numbers near the black level clip
                    safe_brightness = max(actual_brightness, QUANTIZATION_SIGMA_DN)
                    correction_ratio = target / safe_brightness

                    # Dampen the correction to prevent wild oscillation jumps
                    if correction_ratio > 1.0:
                        correction_ratio = min(
                            correction_ratio, cfg.get("ae_step_limit_up")
                        )  # Max x increase per step
                    else:
                        correction_ratio = max(
                            correction_ratio, cfg.get("ae_step_limit_down")
                        )  # Max x decrease per step

                    # Treat Exposure and Gain as a single pooled product
                    current_product = new_exposure * new_gain
                    required_product = current_product * correction_ratio

                    # Change limit: never ask for a product further from the last
                    # trusted check than the sky geometry can explain. (The
                    # SATURATED failsafe above deliberately bypasses this -- a
                    # saturated frame must always be backed off.)
                    if ae_bounds is not None:
                        limited = min(max(required_product, ae_bounds[0]), ae_bounds[1])
                        if limited != required_product:
                            ae_clamped = True
                            logging.warning(
                                f"Auto-exposure: wanted exposure*gain={required_product:.0f} but the change "
                                f"limit allows {ae_bounds[0]:.0f}..{ae_bounds[1]:.0f}; using {limited:.0f}."
                            )
                            required_product = limited

                    new_gain, new_exposure = self._select_gain_exposure(
                        required_product, min_gain, max_gain, min_exp, max_exp
                    )

                    logging.info(
                        f"AE Correction: Req. Light={required_product:.0f} -> Locked Gain {new_gain:.1f}, Adjusted Exp {new_exposure}us"
                    )

                # ANTI-LOOP FAILSAFE
                if new_exposure == test_exp and new_gain == test_gain:
                    logging.warning(
                        f"Auto-exposure: Pinned against "
                        f"{'the change limit' if ae_clamped else 'config limits'}. "
                        f"Cannot reach target {target:.1f}. Settling for {actual_brightness:.1f}"
                    )
                    self._ae_reset_background_if_changed(safe_night_exp, safe_night_gain, new_exposure, new_gain)
                    self.current_exposure_us = new_exposure
                    self.current_gain = new_gain
                    self.auto_exposure_converged = True
                    self._ae_record_candidate(ae_clamped, target, frame if frame_synced else None)
                    # FORCE DARK FRAME RECALCULATION
                    self.last_dark_update_time = float("-inf")
                    self.snapshot_frame = frame
                    return
            else:
                # We reached max_attempts without perfectly hitting the target.
                # We must "settle" for the current test settings so the metadata matches the frame!
                logging.warning(
                    f"Auto-exposure: Max attempts reached without convergence. "
                    f"Settling for Exp={new_exposure}us, Gain={new_gain:.2f}. "
                    f"Last actual brightness={actual_brightness:.1f}."
                )

                # Keep the settings from the final attempt so math stays accurate!
                self._ae_reset_background_if_changed(safe_night_exp, safe_night_gain, new_exposure, new_gain)
                self.current_exposure_us = new_exposure
                self.current_gain = new_gain
                self.auto_exposure_converged = True
                self._ae_record_candidate(ae_clamped, target, frame if frame_synced else None)

                # FORCE DARK FRAME RECALCULATION
                self.last_dark_update_time = float("-inf")
                self.snapshot_frame = frame  # Safe to reuse because variables match!
    # -----------------
    def _select_gain_exposure(self, required_product, min_gain, max_gain, min_exp, max_exp):
        """
        Calculates the optimal Exposure and Gain. 
        Forces Gain to the lowest possible INTEGER to minimize noise and perfectly match the Dark Library.
        """     
        # Calculate the absolute minimum gain needed if we maxed out the exposure
        raw_ideal_gain = required_product / max_exp
        
        # Use ceil() to round UP to the nearest integer (e.g. 1.1 becomes 2.0). 
        # This ensures the resulting exposure drops comfortably below max_exp.
        new_gain_int = math.ceil(raw_ideal_gain) if raw_ideal_gain > 0 else 1
        
        # Clamp to hardware config limits
        new_gain = float(max(min_gain, min(max_gain, new_gain_int)))
        
        # Recalculate exactly how much exposure is needed now that gain is locked to an integer!
        new_exposure = int(max(min_exp, min(max_exp, required_product / new_gain)))
        
        return new_gain, new_exposure    
        # -----------------
    def calculate_lux(self, brightness, exposure_us, gain):
        # If the exposure is not set to auto, the sensor did not return the lux value as metadata.
        # Formula: Lux = Brightness / (K * Exposure_sec * Gain)
        # If for calculate calibration_k the image used is pitch-black the value is too low (0.002)
        # if that the calculate_lux jump to very high value (800) and trick the pipeline into thinking the sun just came up
        # so 0.5 is a floor level for the math
        day = self.cfg["daylight"]
        k_val = (
            self.calibration_k
            if (self.calibration_k and self.calibration_k > day["min_valid_calibration_k"])
            else day["default_calibration_k"]
        )

        exposure_s = exposure_us / 1_000_000.0

        denominator = k_val * exposure_s * gain

        # k, exposure and gain are all validated positive; only a missing reading can be 0.
        if denominator <= 0:
            return 0.0

        lux = brightness / denominator
        return lux

    # -----------------
    def calibration_worker_loop(self):
        """
        A dedicated worker that waits for frames on the calibration_q,
        collects N of them, stacks them, and saves the result.
        """
        logging.info("Calibration worker started, waiting for triggered frames.")

        waiting_since = None  # monotonic time a test shot started waiting for its first frame
        try:
            while self.running.is_set():
                # ---- 1. Idle wait for the first frame of a test shot ----
                # Kept OUTSIDE the session's try/finally on purpose. Before, the idle
                # timeout ("no frame yet") fell through the same `finally` that tears a
                # test session down: a test shot requested during the idle wait was
                # killed as soon as that wait expired, before its first frame.
                wait = self.cfg["general"]["queue_status_ping_sec"]
                if self.is_calibrating.is_set():
                    wait = min(wait, self._frame_wait_budget_sec())
                try:
                    first_frame_item = self.calibration_q.get(timeout=wait)
                except queue.Empty:
                    if self.is_calibrating.is_set():
                        if waiting_since is None:
                            waiting_since = time.monotonic()
                        # Budget: the camera's own frame-wait budget plus the time a
                        # restarted sensor is given to prove itself.
                        budget = self._frame_wait_budget_sec() + self.cfg["general"]["camera_restart_settle_sec"]
                        if time.monotonic() - waiting_since > budget:
                            error_msg = (f"No frame arrived within {budget:.0f}s of starting the test shot. "
                                         f"Check that the camera is streaming (pipeline.log), then try again.")
                            logging.error(f"Calibration worker: {error_msg}")
                            with self.status_lock:
                                self.last_calibration_error = error_msg
                            self._end_test_session()
                            waiting_since = None
                    else:
                        waiting_since = None
                    continue

                if first_frame_item == "STATUS":
                    self.ping_watchdog()
                    continue
                if first_frame_item is None:
                    break
                if not self.is_calibrating.is_set():
                    continue  # stale frame from a session that already ended: never start a new one
                waiting_since = None

                # ---- 2. A test session is running: always end it, whatever happens ----
                try:
                    stack_n = self.effective_stack_n
                    logging.warning(
                        "Calibration worker received first frame. Capturing %d total for test shot...",
                        stack_n,
                    )

                    with self.status_lock:
                        self.last_calibration_error = None

                    frames_for_stack = [first_frame_item]
                    capture_ok = True

                    # Collect the REMAINING frames.
                    for i in range(stack_n - 1):
                        if not self.running.is_set():
                            capture_ok = False
                            break
                        try:
                            # Use a timeout in case the frame stream stops for any reason
                            item = self.calibration_q.get(timeout=self._frame_wait_budget_sec())
                            if item == "STATUS":
                                self.ping_watchdog()
                                item = self.calibration_q.get(timeout=self._frame_wait_budget_sec())
                            frames_for_stack.append(item)
                            logging.info(f"Captured live test frame {i+2}/{stack_n}...")
                        except queue.Empty:
                            error_msg = f"Timed out waiting for frame {i+2}/{stack_n}. Aborting test shot."
                            logging.error(f"Calibration worker: {error_msg}")
                            with self.status_lock:
                                self.last_calibration_error = error_msg
                            capture_ok = False
                            break

                    # Process the collected frames if everything went well.
                    if capture_ok:
                        logging.info(
                            "Stacking %d frames for calibration image...",
                            len(frames_for_stack),
                        )
                        stacked_image, _, _ = self.stack_frames(
                            [f for _, f in frames_for_stack],
                            align=self.cfg["timelapse"]["stack_align"],
                            method="mean",
                            min_features_for_alignment=self.cfg["timelapse"][
                                "min_features_for_alignment"
                            ],
                        )
                        if stacked_image is not None:
                            tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"calibration_shot_{tstamp}.png"
                            full_path = os.path.join(self.calibration_out_dir, filename)
                            self._safe_save_image(full_path, stacked_image)

                            with self.status_lock:
                                self.last_calibration_image_path = full_path
                            logging.info("Calibration image saved to: %s", full_path)
                        else:
                            raise RuntimeError("Frame stacking returned None.")

                except Exception as e:
                    error_msg = f"An unexpected error occurred during test shot: {e}"
                    logging.exception(error_msg)
                    with self.status_lock:
                        self.last_calibration_error = error_msg
                finally:
                    self._end_test_session()

                    while not self.calibration_q.empty():
                        try:
                            self.calibration_q.get_nowait()
                        except queue.Empty:
                            break
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN CALIBRATION THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "CALIBRATION_CRASH", str(e))

    # -----------------
    def _end_test_session(self):
        """Tear down a test-shot capture session (no-op if none is active)."""
        if self.is_calibrating.is_set():
            logging.warning("Calibration worker is terminating the temporary capture session.")
            self.stop_producer_thread()
            self.capture_stable.clear()
            self.is_calibrating.clear()  # Take system out of calibration mode
        while not self.calibration_q.empty():
            try:
                self.calibration_q.get_nowait()
            except queue.Empty:
                break

    # -----------------
    def _acquisition_pacing(self):
        """
        THE single acquisition-rate policy. Returns (strategy, period_s):

          ("decimate", interval)  daylight or weather hold: detection is paused and only
              one frame per daylight.capture_interval_sec is used. The camera keeps
              streaming at its natural rate and the loop simply skips reads, because:
              daytime auto-exposure converges only on a continuous stream, and a sky
              check must be able to sync new settings within a few frames -- with the
              sensor itself paced to minutes, a settings change would wait minutes of
              old frames to drain.
          ("sensor", None)        night acquisition: the SENSOR is paced, through its
              minimum frame duration = max(exposure, measured processing cost x
              capture.processing_headroom) (_night_frame_duration_limits). Frames are
              never produced faster than the pipeline consumes them, sampling stays
              uniform, and no readout/ISP/copy work is spent on frames that would be
              dropped.
          ("free", None)          test shots: consecutive frames, now.

        Timelapse CADENCE (stack size / interval) is a separate thing -- a choice of
        which frames to keep, not a rate limit -- and lives in _enqueue_timelapse_frame.
        """
        if self.is_calibrating.is_set():
            return "free", None
        if self.daylight_mode_active.is_set() or self.weather_hold_active.is_set():
            interval = self.cfg["daylight"]["capture_interval_sec"]
            return ("decimate", interval) if interval > 0 else ("free", None)
        return "sensor", None

    # -----------------
    def _processing_floor_us(self):
        """
        Shortest frame period the pipeline can sustain: the larger of the detection
        loop's and the timelapse stacker's measured per-frame cost, times
        capture.processing_headroom. 0 until something has been measured.
        """
        with self.status_lock:
            det_ms = self.session_stats.get("loop_time_ms", 0.0) or 0.0
            tl_ms = self.session_stats.get("timelapse_ms_per_frame", 0.0) or 0.0
        return int(max(det_ms, tl_ms) * 1000.0 * self.cfg["capture"]["processing_headroom"])

    # -----------------
    def _night_frame_duration_limits(self, exposure_us):
        """
        NIGHT FrameDurationLimits. min = the exposure, or the processing floor when the
        pipeline is slower than that: the sensor then inserts blanking instead of
        producing frames that would only overflow the acquisition queue and be dropped
        at random. That keeps the sampling uniform (a meteor never falls into a random
        hole), and saves the readout, ISP and copy work of frames nobody can use.
        max = min + frame_duration_headroom_us (room for the sensor's minimum blanking).
        """
        min_fd = max(int(exposure_us), self._processing_floor_us())
        return (min_fd, min_fd + int(self.cfg["capture"]["frame_duration_headroom_us"]))

    # -----------------
    def _match_frame_rate_to_processing(self):
        """
        The "sensor" strategy of _acquisition_pacing, applied by the capture loop: if
        the processing floor moved by more than capture.frame_rate_hysteresis since it
        was applied, re-apply the NIGHT frame duration. Frame duration only -- the
        exposure is not touched, so no sync is needed.
        """
        if getattr(self, "_night_preset_active", False) is False:
            return
        set_controls = getattr(getattr(self.camera, "picam2", None), "set_controls", None)
        if set_controls is None:
            return
        wanted = self._night_frame_duration_limits(self.current_exposure_us)
        applied = getattr(self, "_applied_frame_duration", None)
        if applied and abs(wanted[0] - applied[0]) <= self.cfg["capture"]["frame_rate_hysteresis"] * applied[0]:
            return
        with self.camera_lock:
            set_controls({"FrameDurationLimits": wanted})
        self._applied_frame_duration = wanted
        if wanted[0] > self.current_exposure_us:
            logging.info(
                f"Frame rate matched to processing: one frame every {wanted[0] / 1e6:.3f}s "
                f"(exposure {self.current_exposure_us / 1e6:.3f}s; processing {self._processing_floor_us() / 1e3:.0f} ms incl. margin)."
            )
        else:
            logging.info(f"Frame rate back to the exposure: one frame every {wanted[0] / 1e6:.3f}s.")

    # -----------------
    def _set_camera_preset(self, mode="NIGHT", **overrides):
        """
        Camera mode.
        """
        # Determine the target exposure first so we can sync FrameDurationLimits
        target_exp = overrides.get("ExposureTime", self.current_exposure_us)
        target_gain = overrides.get("AnalogueGain", self.current_gain)

        # NIGHT frame-duration window: see _night_frame_duration_limits.

        # Define the base templates using current config/state
        presets = {
            "NIGHT": {
                "ExposureTime": int(target_exp),
                "AnalogueGain": float(target_gain),
                "AeEnable": False,
                "AwbEnable": False,
                "FrameDurationLimits": self._night_frame_duration_limits(target_exp),
                "ColourGains": [
                    self.cfg["capture"].get("red_gain"),
                    self.cfg["capture"].get("blue_gain"),
                ],
            },
            "DAY": {
                "AeEnable": True,
                "AwbEnable": True,
                "FrameDurationLimits": (
                    self.cfg["daylight"]["day_frame_duration_min_us"],
                    self.cfg["daylight"]["day_frame_duration_max_us"],
                ),
            },
            "SNAP": {
                # Short manual exposure — bright enough that the day to not saturate,
                # long enough at night that the ISP can compute a meaningful Lux value.
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": self.cfg["daylight"]["snap_exposure_us"],
                "AnalogueGain": self.cfg["daylight"]["snap_gain"],
                "FrameDurationLimits": (
                    self.cfg["daylight"]["snap_exposure_us"],
                    self.cfg["daylight"]["snap_exposure_us"],
                ),
                "ColourGains": [
                    self.cfg["capture"].get("red_gain"),
                    self.cfg["capture"].get("blue_gain"),
                ],
            },
        }

        # Merge preset with any dynamic overrides
        controls = {**presets.get(mode, presets["NIGHT"]), **overrides}

        # Never request what the sensor cannot deliver. A clamped-by-hardware value
        # can never be confirmed by flush() (guaranteed sync timeout), and storing
        # the unclamped value in current_exposure_us/current_gain below would make
        # every auto-exposure product calculation wrong by the clamp ratio.
        if "ExposureTime" in controls and "AnalogueGain" in controls:
            ach_exp, ach_gain = self.camera.achievable(
                controls["ExposureTime"], controls["AnalogueGain"]
            )
            if ach_exp != int(controls["ExposureTime"]) or ach_gain != float(controls["AnalogueGain"]):
                logging.warning(
                    f"Preset {mode}: request Exp={controls['ExposureTime']}us Gain={controls['AnalogueGain']} "
                    f"is outside the sensor's range; clamped to Exp={ach_exp}us Gain={ach_gain:.3f}."
                )
            controls["ExposureTime"], controls["AnalogueGain"] = ach_exp, ach_gain

        # Re-derive FrameDurationLimits from the (possibly overridden) ExposureTime
        # (exposure or processing floor, whichever is longer; see above).
        if mode == "NIGHT" and "ExposureTime" in controls:
            controls["FrameDurationLimits"] = self._night_frame_duration_limits(controls["ExposureTime"])

        try:
            # Synchronize internal state variables if they were changed
            if "ExposureTime" in controls:
                self.current_exposure_us = controls["ExposureTime"]
            if "AnalogueGain" in controls:
                self.current_gain = controls["AnalogueGain"]

            # Apply to hardware
            self.camera.picam2.set_controls(controls)
            self._night_preset_active = (mode == "NIGHT")
            self._applied_frame_duration = controls.get("FrameDurationLimits") if mode == "NIGHT" else None

            applied_exp_us = controls.get("ExposureTime", self.current_exposure_us)
            applied_gain = controls.get("AnalogueGain", self.current_gain)

            # THE METADATA FLUSH
            if mode == "DAY":
                self.camera.flush()  # Uses the fallback 4-frame loop
                logging.info("Preset: Mode DAY applied. (Auto-Exposure Unlocked)")
                return True, None
            else:
                # We let flush() natively block and return the synced frame.
                fdl = controls.get("FrameDurationLimits")
                synced_frame = self.camera.flush(
                    target_exposure_us=applied_exp_us, target_gain=applied_gain,
                    # the frame period may be longer than the exposure (processing floor)
                    frame_duration_us=fdl[0] if fdl else None,
                )
                logging.info(
                    f"Preset: Mode {mode} applied. (Exp: {applied_exp_us}us, Gain: {applied_gain:.2f})"
                )
                return True, synced_frame

        except Exception as e:
            logging.error(f"Preset: Failed to apply preset {mode}: {e}")
            return False, None

    # -----------------
    def _load_calibration_frames(self):
        """
        Loads Dark, Bias, and Flat frames.
        Pre-calculates the Flat Gain Map for fast runtime application.
        """
        cfg = self.cfg["detection"]

        # Check Calibration Age (Sensor Degradation Warning)
        dark_dir = os.path.join(self.assets_dir, "darks")
        if os.path.exists(dark_dir):
            calib_files = glob.glob(os.path.join(dark_dir, "*.png"))
            if calib_files:
                # Find the newest file in the folder to see when the calibrator was last run
                latest_file = max(calib_files, key=os.path.getmtime)
                age_days = (time.time() - os.path.getmtime(latest_file)) / 86400.0

                if age_days > cfg["calibration_max_age_days"]:
                    msg = f"Dark library is {int(age_days)} days old! Sensor noise changes over time. Please run ryusei_calibrator.py to rebuild it."
                    logging.warning(f"Calibration Maintenance: {msg}")
                    self.log_health_event("WARNING", "CALIB_LIBRARY_OLD", msg)

        # Load Master Dark (Standard subtraction)
        if cfg.get("dark_frame_path") and os.path.exists(cfg["dark_frame_path"]):
            # self.master_dark = cv2.imread(cfg["dark_frame_path"], cv2.IMREAD_UNCHANGED)
            dark_img = cv2.imread(cfg["dark_frame_path"], cv2.IMREAD_UNCHANGED)
            # Safe 16-bit handling for old dark frame
            if dark_img.dtype == np.uint16:
                dark_img = (dark_img / 256.0).astype(np.uint8)

            # self.master_dark = dark_img
            
            # =========================================================
            # --- NEW: BAD PIXEL MAP (BPM) GENERATION ---
            # Identify pixels that are abnormally brighter (>15) than their neighbors.
            # =========================================================
            # d_med = cv2.medianBlur(dark_img, 3)
            # d_diff = cv2.absdiff(dark_img, d_med)
            # self.bad_pixel_mask = (d_diff > 15)
            
            # # Heal the master dark frame by interpolating the bad pixels
            # self.master_dark = np.where(self.bad_pixel_mask, d_med, dark_img)
            
            self.master_dark, self.bad_pixel_mask, self.active_bpm_kernel = self._generate_bpm_and_heal(dark_img)

            logging.info(
                f"Calibration: STATIC Dark loaded. Dynamic Dark logic will be BYPASSED."
            )
            self.use_dynamic_darks = False
            self.active_dark_frame_name = os.path.basename(cfg["dark_frame_path"])
            
        elif self.ds18b20_cam_path:
            # Scenario B: Sensor is healthy. Use Dynamic logic.
            if cfg.get("auto_dark_enabled"):
                self.use_dynamic_darks = True
                # self.use_dynamic_darks = cfg.get("auto_dark_enabled")
                self.master_dark = None  # Will be populated by the first loop iteration
                self.active_dark_frame_name = "Pending Dynamic Update..."
                logging.info("Calibration: DS18B20 found. Dynamic Darks ENABLED.")
            else:
                self.use_dynamic_darks = False
                self.master_dark = None
                self.active_dark_frame_name = "None (Disabled)"
                logging.info("Autodark logic disabled on configuration")
        else:
            # Scenario C: Sensor FAILED and no specific file provided.
            self.use_dynamic_darks = False

            # Look for a 'generic' fallback file in the assets folder
            # fallback_path = os.path.join(base_path, "assets/darks/dark_fallback.png")
            fallback_path = os.path.join(self.assets_dir, "darks", "dark_fallback.png")
            if os.path.exists(fallback_path):
                # self.master_dark = cv2.imread(fallback_path, cv2.IMREAD_UNCHANGED)
                dark_img = cv2.imread(fallback_path, cv2.IMREAD_UNCHANGED)
                if dark_img.dtype == np.uint16:
                    dark_img = (dark_img / 256.0).astype(np.uint8)

                # Same robust BPM + healing as the primary dark path. This fallback used
                # its own hardcoded map (3x3 median, fixed 15 DN threshold): a different
                # hot-pixel definition depending on which dark happened to load, and a
                # threshold blind to the frame's actual noise level.
                self.master_dark, self.bad_pixel_mask, self.active_bpm_kernel = self._generate_bpm_and_heal(dark_img)

                self.active_dark_frame_name = "dark_fallback.png"
                logging.warning(
                    "Hardware: Sensor missing! Using 'dark_fallback.png' as emergency backup."
                )
            else:
                self.master_dark = None
                self.active_dark_frame_name = "None (Missing Sensor & Fallback)"
                logging.critical(
                    "Hardware: Sensor missing AND no fallback dark found! DETECTIONS WILL BE NOISY."
                )
                self.log_health_event(
                    "CRITICAL",
                    "SENSOR_MISSING",
                    "No temperature sensor or dark frame available.",
                )

        # Load Bias (Needed to calibrate the Flat)
        bias_frame = None
        if cfg.get("bias_frame_path") and os.path.exists(cfg["bias_frame_path"]):
            b_img = cv2.imread(cfg["bias_frame_path"], cv2.IMREAD_UNCHANGED)
            
            if b_img is not None:
                # Safe 16-bit downscaling
                if b_img.dtype == np.uint16:
                    b_img = (b_img / 256.0)
                
                bias_frame = b_img.astype(np.float32)
                logging.info(f"Calibration: Master Bias loaded from {cfg['bias_frame_path']}")
            else:
                logging.error(f"Calibration: Failed to read Bias frame at {cfg['bias_frame_path']}")

        # Load and Process Flat Frame
        self.flat_multiplier = None
        if cfg.get("flat_frame_path") and os.path.exists(cfg["flat_frame_path"]):
            try:
                f_img = cv2.imread(cfg["flat_frame_path"], cv2.IMREAD_UNCHANGED)
                
                if f_img is None:
                    raise ValueError(f"Failed to read Flat frame at {cfg['flat_frame_path']}")
                
                # Safe 16-bit downscaling
                if f_img.dtype == np.uint16:
                    f_img = (f_img / 256.0)
                
                raw_flat = f_img.astype(np.float32)
                
                logging.info("Calibration: STATIC Flat loaded. Dynamic Flat logic will be BYPASSED.")
                self.use_dynamic_flats = False

                # Subtract Bias from Flat (Crucial for mathematical accuracy)
                if bias_frame is not None:
                    # Ensure dimensions match
                    if raw_flat.shape == bias_frame.shape:
                        raw_flat = cv2.subtract(raw_flat, bias_frame)
                        # Clip to avoid negative values or divide by zero later
                        raw_flat = np.maximum(raw_flat, 1.0) 
                    else:
                        logging.warning("Bias and Flat dimensions mismatch! Skipping Bias subtraction on Flat.")

                # Calculate mean per channel for color accuracy
                flat_mean = np.mean(raw_flat, axis=(0,1)) # Result is [B_mean, G_mean, R_mean]
                
                # Create the multiplier map
                self.flat_multiplier = np.divide(flat_mean, raw_flat)
                
                logging.info("Calibration: Flat Field Gain Map generated successfully.")
                
            except Exception as e:
                logging.error(f"Failed to process Flat Frame: {e}")
                self.flat_multiplier = None
        else:
            self.use_dynamic_flats = cfg.get("auto_flat_enabled", False)
            logging.info(f"Calibration: No static flat found. Dynamic Flat enabled: {self.use_dynamic_flats}")
            
            # Load the saved Dynamic Flat from yesterday
            if self.use_dynamic_flats:
                npy_path = self.dynamic_flat_path.replace(".png", ".npy")
                if os.path.exists(npy_path):
                    try:
                        cap = self.cfg["capture"]
                        self.flat_multiplier = load_flat_map(npy_path, (cap["height"], cap["width"], 3))
                        logging.info("Calibration: Restored yesterday's Dynamic Flat map from memory.")
                    except (OSError, ValueError) as e:
                        logging.warning(f"Could not load previous dynamic flat memory ({e}). Starting fresh.")
                        self.flat_multiplier = None
    # -----------------
    def camera_k_logger(self):
        try:
            line = f"{datetime.now().isoformat()},{self.calibration_k}\n"

            # Write the header if the file is new (matches janitor rotation).
            write_header = not os.path.exists(self.camera_k_file)
            with open(self.camera_k_file, "a") as f:
                if write_header:
                    f.write("timestamp, k\n")
                f.write(line)
        except Exception as e:
            print(f"[camera_k_logger] Failed to write camera k value: {e}")

    # -----------------
    # 5. ANALYSIS & DETECTION
    # -----------------
    def _get_dynamic_accumulate_alpha(self):
        """
        Calculates the ideal accumulate_alpha based on the CURRENT exposure time.
        Logic: We want the background to adapt to changes in approximately 20 seconds.

        Short Exposure (0.04s) -> Needs tiny alpha (0.002) to smooth noise.
        Long Exposure (4.0s)   -> Needs big alpha (0.2) to prevent ghosting/halos.
        """
        exp_us = self.current_exposure_us or self.cfg["capture"]["exposure_us"]
        exp_seconds = exp_us / 1_000_000.0

        # Calculate raw alpha = actual exposure (sec) / target_time
        alpha = exp_seconds / self.cfg["detection"]["target_adaptation_time"]

        # CLAMP limits to prevent breaking the background model
        min_alpha = self.cfg["detection"]["min_accumulate_alpha"]
        max_alpha = self.cfg["detection"]["max_accumulate_alpha"]
        # 0.001 = Extremely slow update (good for high FPS video)
        # 0.300 = Fast update (good for very long exposures to kill moon ghosts)
        return max(min_alpha, min(max_alpha, alpha))


    def _reconcile_capture_limits_with_sensor(self):
        """
        Intersect the configured exposure/gain search range with the range the
        sensor actually reports. Auto-exposure distributes a required
        exposure*gain product between gain and exposure assuming both config
        ceilings are reachable; if max_gain exceeds the sensor's analogue
        maximum, the hardware clamps silently, the product falls short, and AE
        burns attempts chasing a brightness it can never reach. Call after every
        camera (re)creation. No-op for cameras that report no limits (simulator).
        """
        lim_exp = getattr(self.camera, "exposure_limits_us", None)
        lim_gain = getattr(self.camera, "gain_limits", None)
        if not lim_exp or not lim_gain:
            return
        cap = self.cfg["capture"]
        for key, (lo, hi) in (
            ("min_exposure_us", lim_exp), ("max_exposure_us", lim_exp),
            ("min_gain", lim_gain), ("max_gain", lim_gain),
        ):
            value = cap[key]
            clamped = type(value)(min(max(value, lo), hi))
            if clamped != value:
                logging.warning(
                    f"Capture config: {key}={value} is outside the sensor's range "
                    f"[{lo}, {hi}]; using {clamped} for this session."
                )
                cap[key] = clamped

    # -----------------
    def _enforce_exposure_safety_limits(self):
        """
        Beyond a certain exposure time, _get_dynamic_accumulate_alpha()'s
        alpha (which scales up with exposure to hold the background model's
        real-world adaptation time near target_adaptation_time) saturates at
        max_accumulate_alpha. Past that point, alpha can no longer compensate
        for the falling update rate, so the ACTUAL adaptation time silently
        grows past the target_adaptation_time design goal -- the background
        model responds to genuine sky-brightness changes more sluggishly than
        intended. This is NOT an OpenCV numerical-stability issue --
        cv2.accumulateWeighted's exponential moving average is unconditionally
        stable for any alpha in (0, 1]; nothing here can "break" OpenCV.

        Calculates the exposure at which that saturation begins
        (max_safe_exp_sec = target_adaptation_time * max_accumulate_alpha,
        the inverse of the alpha formula in _get_dynamic_accumulate_alpha)
        and caps the config to it.

        NOTE: this directly limits your maximum capture exposure -- i.e. your
        achievable star SNR -- as a side effect of a motion-detection tuning
        parameter. Raise target_adaptation_time in config to raise this
        ceiling if you need longer integration and can tolerate slower
        background-model response to real sky-brightness changes.
        """
        target_time = self.cfg["detection"]["target_adaptation_time"]
        max_alpha = self.cfg["detection"]["max_accumulate_alpha"]

        # Exposure at which _get_dynamic_accumulate_alpha's alpha saturates
        max_safe_exp_sec = target_time * max_alpha
        max_safe_exp_us = int(max_safe_exp_sec * 1_000_000)

        user_max_exp = self.cfg["capture"]["max_exposure_us"]

        if user_max_exp > max_safe_exp_us:
            logging.warning("=" * 60)
            logging.warning(f"   CONFIGURATION OVERRIDE: EXPOSURE CAP   ")
            logging.warning(
                f"Requested max_exposure_us ({user_max_exp/1e6:.2f} sec) exceeds the exposure "
                f"at which the background-adaptation alpha saturates."
            )
            logging.warning(
                f"Based on target_adaptation_time ({target_time}s) and max_accumulate_alpha ({max_alpha}),"
            )
            logging.warning(
                f"the background model's real-world response time will exceed its "
                f"{target_time:.1f}s design target above {max_safe_exp_sec:.2f} sec exposure."
            )
            logging.warning("=" * 60)

            # Cap the auto-tune dictionary!
            self.cfg["capture"]["max_exposure_us"] = max_safe_exp_us
            # If the pipeline is already running we have to change current_exposure_us also
            self.current_exposure_us = max_safe_exp_us

        # Cap the Starting Exposure
        # If the user's default starting exposure is higher than the safe limit,
        # it will break the first few frames before Auto-Exposure can fix it.
        if self.cfg["capture"]["exposure_us"] > max_safe_exp_us:
            self.cfg["capture"]["exposure_us"] = max_safe_exp_us

            # If the pipeline is already running (live config reload), clamp the active hardware variable too!
            if hasattr(self, "current_exposure_us") and self.current_exposure_us > max_safe_exp_us:
                self.current_exposure_us = max_safe_exp_us
                
    # -----------------

    # -----------------

    # -----------------
    def _generate_bpm_and_heal(self, dark_frame):
        """
        Dynamically calculates the size of hot pixel blobs in the dark frame,
        adjusts the median blur kernel size accordingly, and generates the BPM
        using N-Sigma Statistical Outlier detection (immune to exposure scaling!).

        All thresholds/kernels are pulled from [detection] config (Pydantic-validated)
        rather than hardcoded, since they are sensor/optics dependent and need to be
        re-tunable without editing source.

        NOTE ON THREAD SAFETY: this method only *computes and returns* values. It does
        NOT mutate self.active_bpm_kernel / self.master_dark / self.bad_pixel_mask
        directly. The caller is responsible for assigning all three under the same
        lock, so a concurrent reader (e.g. _heal_live_frame) can never observe a
        kernel size paired with a stale/mismatched mask.
        """
        cfg = self.cfg["detection"]
        bg_kernel = cfg["bpm_bg_blur_kernel"]
        sigma_k = cfg["bpm_sigma_multiplier"]
        sigma_floor = cfg["bpm_sigma_floor"]
        max_blob_px = cfg["bpm_max_blob_px"]
        kernel_min = cfg["bpm_kernel_min_px"]
        kernel_max = cfg["bpm_kernel_max_px"]

        # 1. Fast background approximation to isolate the spikes
        gray = cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY) if len(dark_frame.shape) == 3 else dark_frame
        bg_approx = cv2.blur(gray, (bg_kernel, bg_kernel))

        # 2. Statistical Thresholding
        diff_gray = cv2.absdiff(gray, bg_approx)
        mean_diff, std_diff = cv2.meanStdDev(diff_gray)

        # N-Sigma is the configurable standard for defect detection.
        # We enforce a floor to ignore normal 8-bit read noise.
        dyn_thresh = max(sigma_floor, mean_diff[0][0] + (sigma_k * std_diff[0][0]))

        _, thresh = cv2.threshold(diff_gray, dyn_thresh, 255, cv2.THRESH_BINARY)

        # 3. Measure the physical size of the blobs!
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_blob = 1
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # Ignore massive anomalies (like Amp Glow edges), which are not point defects
            if w < max_blob_px and h < max_blob_px:
                max_blob = max(max_blob, w, h)

        # 4. Calculate the required median kernel size.
        # The window must be > 2x the blob size so healthy pixels outnumber bad ones.
        blur_k = (max_blob * 2) + 1
        if blur_k % 2 == 0: blur_k += 1

        # Clamp to protect live-frame CPU performance (config-defined bounds)
        blur_k = max(kernel_min, min(kernel_max, blur_k))

        logging.info(f"BPM Generator: Dyn_Thresh={dyn_thresh:.1f} | Max blob={max_blob}px | Using kernel {blur_k}x{blur_k}")

        # 5. Generate the final high-precision BPM using the dynamic threshold
        d_med = cv2.medianBlur(dark_frame, blur_k)
        d_diff = cv2.absdiff(dark_frame, d_med)

        # Evaluate each color channel independently against the threshold
        bpm_mask = (d_diff > dyn_thresh)

        healed_dark = np.where(bpm_mask, d_med, dark_frame)
        return healed_dark, bpm_mask, blur_k
    # -----------------
    def _heal_live_frame(self, frame):
        """
        Applies the active Bad Pixel Map to instantly heal a live frame
        using the dynamically calculated kernel size.
        """
        with self.status_lock:
            bpm = getattr(self, "bad_pixel_mask", None)
            blur_k = getattr(self, "active_bpm_kernel", 3)
        if bpm is not None:
            f_med = cv2.medianBlur(frame, blur_k)
            return np.where(bpm, f_med, frame)
        return frame
    # -----------------
    def _update_dynamic_dark_if_needed(self):
        """
        Safely generates the scaled Dark Frame and the Bad Pixel Map.
        Can be called by any thread to ensure the background models are up-to-date.
        """
        # Don't waste CPU if we are in Daylight mode or if dynamic darks are off
        if not self.use_dynamic_darks or self.daylight_mode_active.is_set():
            return
            
        now = time.monotonic()
        if now - self.last_dark_update_time > self.cfg["detection"]["dynamic_dark_refresh_sec"]:
            self.last_dark_update_time = now
            try:
                current_temp = self.current_sensor_temp_cam
                if current_temp is not None:
                    new_dark = self.dark_manager.get_scaled_dark(
                        current_temp,
                        self.current_gain,
                        self.current_exposure_us,
                    )
                    if new_dark is not None:
                        # Recalculate the BPM because hot pixels grow with temperature!
                        healed_dark, bpm, bpm_kernel = self._generate_bpm_and_heal(new_dark)

                        # All three are set together under one lock so a concurrent reader
                        # (e.g. _heal_live_frame, called unlocked from detection_loop /
                        # sky_monitor_loop) can never see a kernel size paired with a
                        # mismatched/stale mask.
                        with self.status_lock:
                            self.master_dark = healed_dark
                            self.bad_pixel_mask = bpm
                            self.active_bpm_kernel = bpm_kernel
                            base_name = f"dark_{int(round(current_temp))}C_G{int(round(self.current_gain))}_5s.png"
                            self.active_dark_frame_name = f"{base_name} (scaled)"
                            logging.info(f"Load darkframe: {base_name}")
                            
                        logging.info(f"DarkFrame: Updated for {current_temp}C | Gain {self.current_gain} | Exp {self.current_exposure_us/1e6}s")
            except Exception as e:
                logging.error(f"DarkFrame: Failed to update: {e}")
    # -----------------

    def _enqueue_timelapse_frame(self, ts, frame, meta_label):
        """
        Timelapse CADENCE: a stack of effective_stack_n consecutive frames, then a pause
        of timelapse_interval_sec. This chooses which frames are kept; it is not a rate
        limit (acquisition pacing in _acquisition_pacing already keeps frames at a rate
        the stacker sustains -- its per-frame cost is part of the processing floor).

        A full queue (a transient: an event being written, a slow drive) drops the frame
        and is counted. RAM stays bounded by the queue size. This replaces the former
        "emergency throttle", a second, hand-tuned controller (three parameters) that
        delayed frames on the same queue the acquisition pacing now governs.
        """
        if time.monotonic() < self.timelapse_next_capture_time:
            return
        reporter = getattr(self, "_tl_drops", None)
        if reporter is None:
            reporter = self._tl_drops = DropReporter(
                "Timelapse queue", "TIMELAPSE_QUEUE",
                lambda: self.cfg["general"]["drop_report_interval_sec"], self.log_health_event,
                hint="stacking slower than the frame rate",
            )
        try:
            self.timelapse_q.put_nowait((ts, frame, meta_label))
        except queue.Full:
            reporter.dropped()
            return
        reporter.tick()
        self.timelapse_frames_queued += 1
        if self.timelapse_frames_queued >= self.effective_stack_n:
            # Stack complete: wait the configured interval before the next one.
            self.timelapse_next_capture_time = time.monotonic() + self.timelapse_interval_sec
            self.timelapse_frames_queued = 0

    # -----------------
    def _register_camera_failure(self):
        """
        Increments the consecutive camera failure counter globally.
        Returns False if the maximum failure limit is reached (fatal), True if safe to retry.
        """
        self.consecutive_camera_failures += 1
        max_failures = self.cfg["general"].get("max_camera_failures")
        logging.warning(
            "Camera read failed (%d/%d consecutive failures).",
            self.consecutive_camera_failures,
            max_failures,
        )

        if self.consecutive_camera_failures >= max_failures:
            logging.critical(
                "Camera has failed %d consecutive times. Setting fatal error flag.",
                max_failures,
            )
            self.camera_fatal_error.set()
            return False

        return True

    # -----------------
    def _reset_camera_failure(self):
        """Resets the consecutive camera failure counter upon a successful frame read."""
        if self.consecutive_camera_failures > 0:
            self.consecutive_camera_failures = 0

    # -----------------
