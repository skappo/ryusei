"""
sky_monitor_mixin.py

Sky analysis, moon state, star detection, and daylight-transition methods
for the Ryusei pipeline, extracted from the main Pipeline class as a pure
move -- method bodies are unchanged, only their physical location.
Composed into Pipeline via multiple inheritance.

NOTE: _combine_mean_stack, stack_frames, and create_alignment_mosaic were
deliberately NOT included here despite being physically interleaved in the
original file between detect_stars and get_moon_state -- they're a
conceptually separate group (image stacking), planned for their own future
stacking_mixin.py, same discipline as leaving _run_bounded out of
astrometry_mixin.py.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- extensive here,
mostly capture/exposure-related methods not yet extracted to their own
mixin: self._heal_live_frame, self._register_camera_failure,
self._reset_camera_failure, self._set_camera_preset,
self._update_dynamic_dark_if_needed, self.perform_auto_exposure,
self.update_operating_mode, plus utility methods: self.get_metadata,
self.save_json_atomic, self.log_health_event, self._safe_save_image,
self._tally_data_write, self._generate_atmosphere_fingerprint,
self._get_chronicle_context. All must be provided somewhere else in the
composed Pipeline class -- currently still directly in Pipeline's own body.

Imports here are exactly what this extracted block actually uses --
verified by grepping for module.method/module.attribute patterns before
writing this file.
"""
import os
import math
import logging
import threading
import time
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple

from enums import SkyConditionStatus, SKY_STATE_MACHINE
from config_models import SKY_SCORE_VALUES
from flat import save_flat_map
from constants import MAD_TO_SIGMA, QUANTIZATION_SIGMA_DN


class SkyMonitorClass:
    """Sky analysis, moon state, and star detection, composed into Pipeline."""

    def sky_monitor_loop(self):
        logging.info("Sky monitor started.")
        try:
            while self.running.is_set():
                check_min = self.cfg["daylight"].get("check_interval_min")

                # Only perform a check if the main capture thread is currently active.
                if self.producer_thread_active():
                    self.sky_monitor_in_progress.set()
                    try:
                        sky_state_changed = False

                        # Only strictly synchronize if in Night Mode.
                        # In Day Mode, we prioritize the check over timelapse continuity.
                        if (
                            not self.daylight_mode_active.is_set()
                            and self.capture_stable.is_set()
                        ):
                            if not self.wait_for_safe_window():
                                continue

                        logging.info(
                            "Sky Monitor: Routine check starting. Pausing data acquisition."
                        )
                        self.capture_stable.clear()
                        # Close the last race: an event may have triggered between the
                        # safe-window decision and the gate engaging just above. Then
                        # give the frames back to detection and go wait for it again.
                        if self.event_in_progress.is_set() and not self.daylight_mode_active.is_set():
                            logging.info("Sky Monitor: An event started just before the check; deferring.")
                            self.capture_stable.set()
                            continue
                        # Check for daylight to determine the primary operating mode.
                        self.update_operating_mode()

                        # If we are in Night Mode, also check for clouds.
                        if not self.daylight_mode_active.is_set():
                            if self.cfg["capture"].get("auto_exposure_tuning"):
                                self.perform_auto_exposure()

                            status, result, frame = self.analyze_celestial_context()
                            # if status is not Error, result is a dictionary with {"stddev": raw_stddev, "norm_stddev": norm_stddev, "mean":roi_mean, "stars": star_count}

                            rules = SKY_STATE_MACHINE[status]

                            # Auto-exposure ran before we knew the sky state; now that we
                            # do, decide whether its result becomes the reference the next
                            # check is limited against (see _ae_finalize_reference).
                            self._ae_finalize_reference(status == SkyConditionStatus.CLEAR)

                            # 1. State Change Logging
                            if status != self.last_sky_status:
                                sky_state_changed = True
                                self.last_sky_status = status

                                # Print the specific message for entering this state
                                if status == SkyConditionStatus.ERROR:
                                    logging.log(
                                        rules["log_level"],
                                        f"{rules['log_msg']} Details: {result.get('message', 'Unknown Error')}",
                                    )
                                    self.log_health_event(
                                        "ERROR",
                                        "SKY_MONITOR_FAIL",
                                        "Sky condition check failed.",
                                    )
                                else:
                                    logging.log(rules["log_level"], rules["log_msg"])

                            # Day / Night mode change
                            if (
                                self.daylight_mode_active.is_set()
                                != self.last_daylight_mode
                            ):
                                logging.info(
                                    f"Sky Monitor: Daylight mode changed from {self.last_daylight_mode} to {self.daylight_mode_active.is_set()}."
                                )
                                sky_state_changed = True
                                self.last_daylight_mode = (
                                    self.daylight_mode_active.is_set()
                                )

                            # 2. Apply Weather Hold
                            if rules["hold_active"]:
                                if not self.weather_hold_active.is_set():
                                    self.weather_hold_active.set()
                                
                                with self.status_lock:    
                                    self.effective_stack_n = 1
                                logging.info("Sky Monitor: Cloudy/Hold active. Reduced stack size to 1.")
                                    
                            else:
                                if self.weather_hold_active.is_set():
                                    self.weather_hold_active.clear()
                                    
                                if not self.daylight_mode_active.is_set():
                                    with self.status_lock:
                                        self.effective_stack_n = self.cfg["timelapse"]["nighttime_stack_N"]
                                    logging.info(f"Sky Monitor: Sky clear. Restored stack size to {self.effective_stack_n}.")

                            # 3. Apply Timelapse Speed Limits
                            target_interval = self.cfg["timelapse"].get(
                                rules["timelapse_speed_key"]
                            )
                            if target_interval is not None:
                                # If clear, drop back to fast interval. If cloudy, slow down.
                                if (
                                    status == SkyConditionStatus.CLEAR
                                    or self.timelapse_interval_sec < target_interval
                                ):
                                    if self.timelapse_interval_sec != target_interval:
                                        logging.info(
                                            f"Sky Monitor: Adjusting timelapse interval to {target_interval}s."
                                        )
                                        with self.status_lock:
                                            self.base_timelapse_interval = target_interval
                                            self.timelapse_interval_sec = target_interval

                            # 4. Failsafe Threshold (for Errors)
                            if rules["failsafe_threshold"]:
                                with self.status_lock:
                                    self.effective_threshold = self.threshold_max
                                    self.last_effective_threshold = (
                                        f"{self.threshold_max} (FAILSAFE)"
                                    )

                            # ASTROMETRY PLATE SOLVER
                            # Only runs if the sky is clear, Astrometry is enabled, 
                            # and we don't already have a cached solution for tonight.
                            if self.cfg["astrometry"].get("enabled"):
                                if self.sky_score >= SKY_SCORE_VALUES["fair"] and frame is not None:
                                    needs_solve = False
                                    with self.astrometry_lock:
                                        needs_solve = (self.wcs_cache is None)
                                        
                                    if needs_solve:
                                        gray_for_solve = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                        # Run the solver in a detached thread so it doesn't stall the Sky Monitor
                                        threading.Thread(
                                            target=self.solve_field_center_roi, args=(gray_for_solve,), name="AstrometrySolver", daemon=True
                                        ).start()

                            # 1. Dynamic Flat Field Sampling
                            if self.use_dynamic_flats:
                                # flat_cfg = self.cfg["detection"]
                                if self.cfg["detection"]["auto_flat_enabled"]:
                                    # SAFETY: Only sample if sky is FAIR (Score > 0.8)
                                    if (
                                        self.sky_score >= SKY_SCORE_VALUES["fair"]
                                        and frame is not None
                                    ):
                                                                                 
                                        # flat_sample_frame = self._heal_live_frame(frame)                                        
                                        # Frame is already healed in the analyze_celestial_context
                                        self.flat_engine.add_sample(frame)
                                        
                                        # Refresh the active detection multiplier every N samples.
                                        # Counted on samples ADDED: len(buffer) stops growing at
                                        # maxlen, so the old len(buffer) % 4 test either fired on
                                        # every sample forever (maxlen divisible by 4) or never
                                        # again once the buffer filled (otherwise).
                                        every = self.cfg["detection"]["auto_flat_refresh_every_samples"]
                                        if self.flat_engine.samples_added % every == 0:
                                            new_flat_map = self.flat_engine.compute()
                                            if new_flat_map is not None:
                                                with self.camera_lock:
                                                    self.flat_multiplier = new_flat_map
                                                logging.info(
                                                    "FlatEngine: Active Flat Multiplier updated."
                                                )
                                                # Save for visual check
                                                debug_flat = (
                                                    np.clip(new_flat_map, 0, 2) * 127
                                                ).astype(np.uint8)
                                                self._safe_save_image(
                                                    self.dynamic_flat_path,
                                                    debug_flat,
                                                )
                                                # Save the map for tomorrow night -- ONLY a real map.
                                                # This save used to sit outside the `is not None`
                                                # check: the first refresh of every session comes
                                                # before auto_flat_min_samples is reached, compute()
                                                # returns None, and np.save(None) replaced the good
                                                # map with a 274-byte pickled None.
                                                try:
                                                    save_flat_map(
                                                        self.dynamic_flat_path.replace(".png", ".npy"),
                                                        new_flat_map,
                                                    )
                                                except (OSError, ValueError) as e:
                                                    logging.error(f"Failed to save dynamic flat memory: {e}")

                                    else:
                                        logging.info(
                                            "FlatEngine: Sampling skipped (Sky quality too low)."
                                        )

                            # 2. AI Observatory Status Check
                            if self.cfg["ai_observatory"]["enabled"]:
                                if frame is not None:
                                    obs_label, obs_conf = self.ai.check_observatory(frame)
                                    logging.info(
                                        f"Observatory AI: {obs_label.upper()} ({obs_conf *100:.1f}%)"
                                    )

                                    with self.status_lock:
                                        self.session_stats["obs_condition"] = obs_label
                                        self.session_stats["obs_confidence"] = obs_conf

                                    # ACTION: If the observatory is compromised, trigger hold
                                    if (
                                        obs_label
                                        in self.cfg["ai_observatory"].get("bad_conditions")
                                        and obs_conf
                                        >= self.cfg["ai_observatory"]["min_confidence"]
                                    ):
                                        if not self.weather_hold_active.is_set():
                                            logging.warning(
                                                f"AI Observatory: {obs_label.upper()} detected. Engaging HOLD."
                                            )
                                            self.weather_hold_active.set()
                                else:
                                    logging.warning("Observatory AI: Skipped (Camera returned an empty frame).")
                                    with self.status_lock:
                                        self.session_stats["obs_condition"] = "N/A (No Frame)"
                                        self.session_stats["obs_confidence"] = 0.0
                            else:
                                # If AI is off, ensure stats show it's not being checked
                                with self.status_lock:
                                    self.session_stats["obs_condition"] = "N/A (AI Off)"
                                    self.session_stats["obs_confidence"] = 0.0

                        # Reset the background model to prevent false triggers.
                        if sky_state_changed:
                            logging.info(
                                "Sky Monitor: Background reset due to state change."
                            )
                            self.background = None
                            self.background_reset_time = time.monotonic()

                        # Signal that the initial check is complete.
                        if not self.capture_stable.is_set():
                            logging.info(
                                "Sky Monitor: Initial check complete. Releasing detection thread."
                            )
                            self.capture_stable.set()
                    finally:
                        self.sky_monitor_in_progress.clear()  # kick the scheduler

                # Wait for the next full interval before checking again. Deadline based
                # (monotonic, not a count of 1s sleeps) because the lightweight exposure re-check
                # below can spend a while waiting for a safe window, and that time must
                # count towards the interval.
                logging.info(
                    "Sky Monitor: Next check in %d minutes or upon request.", check_min
                )
                next_check_at = time.monotonic() + check_min * 60
                while time.monotonic() < next_check_at:
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break  # Allow for fast shutdown
                    if self.request_sky_analysis_trigger.is_set():
                        logging.info(
                            "Sky Monitor: Immediate check requested. Aborting wait and running now."
                        )
                        self.request_sky_analysis_trigger.clear()
                        break

                    # Lightweight auto-exposure re-check after a clamped full check. Runs
                    # here, in the idle wait, so it can never overlap a full check; it reads
                    # the frame already in memory and only touches the camera to apply a
                    # correction at a stack boundary (see _ae_fast_recheck).
                    due = getattr(self, "ae_recheck_due_at", None)
                    if due is not None and time.monotonic() >= due:
                        try:
                            self._ae_fast_recheck()
                        except Exception:
                            logging.exception("Auto-exposure re-check failed; dropping it.")
                            self.ae_recheck_due_at = None
                        # Never leave an already-due time behind, or this would spin.
                        pending = getattr(self, "ae_recheck_due_at", None)
                        if pending is not None and pending <= time.monotonic():
                            self.ae_recheck_due_at = None
                        continue
                    time.sleep(1)
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN SKY MONITOR THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "SKY_MONITOR_CRASH", str(e))
    # -----------------
    def analyze_celestial_context(self,) -> Tuple[SkyConditionStatus, dict, Optional[np.ndarray]]:
        """
        Captures a snapshot, analyzes it, and returns a detailed status.
        """
        logging.info("Capturing sky condition snapshot...")
        # cloudy_interval_time = 60

        # TWO clocks, deliberately:
        #   now_mono -> elapsed time for the clear/cloudy/moon minute counters
        #               (immune to NTP steps; paired with last_check_time);
        #   now_wall -> calendar timestamp of the history points the morning
        #               journal plots (datetime.fromtimestamp needs epoch time).
        # (Regression fixed here: step 3 switched the single now_ts to monotonic,
        # which also stamped star/noise/moon history with seconds-since-boot. The
        # journal then placed those series in January 1970 and the sky score in the
        # present, stretching the axis across 56 years: every tick read "00:00" and
        # the two groups of data sat at opposite edges of the plot.)
        now_mono = time.monotonic()
        now_wall = time.time()
        # Calculate elapsed time since dusk (if 1st check) or since last check
        elapsed_seconds = now_mono - self.session_stats["last_check_time"]
        elapsed_minutes = elapsed_seconds / 60.0

        with self.camera_lock:
            try:
                # Auto-exposure just ran and dialed the camera perfectly to target brightness.
                # We simply take one frame at the current perfect settings.
                logging.info(
                    f"Snapshot Mode: Current NIGHT Settings (Exp {self.current_exposure_us/1000000:.1f}s, Gain {self.current_gain:.2f})"
                )

                # ZERO-LATENCY HANDOFF ROUTING
                frame = getattr(self, "snapshot_frame", None)

                if frame is not None:
                    logging.info(
                        "Snapshot: Using exposed frame from Auto-Exposure handoff."
                    )
                    self.snapshot_frame = (
                        None  # Consume the frame so it isn't reused later
                    )
                else:
                    # If auto-exposure didn't run, check if we can safely use the RAM frame
                    if (
                        getattr(self, "use_ram_for_check", False)
                        and getattr(self, "last_raw_frame", None) is not None
                    ):
                        logging.info("Snapshot: Using recent RAM frame (Zero latency).")
                        frame = self.last_raw_frame
                    else:
                        logging.info("Snapshot: Acquiring fresh hardware frame.")
                        if not self.capture_running.is_set():
                            return (
                                SkyConditionStatus.ERROR,
                                {"message": "Shutdown initiated"},
                                None,
                            )

                        frame, _ = self.camera.read()

                if frame is None:
                    # Register the failure globally!
                    self._register_camera_failure()
                    logging.error("Failed to capture sky condition snapshot frame.")
                    self.log_health_event(
                        "ERROR",
                        "CAPTURE_DAYLIGHT_FAIL",
                        "Failed to capture sky condition snapshot frame.",
                    )
                    return (
                        SkyConditionStatus.ERROR,
                        {"message": "Camera read returned None."},
                        None,
                    )

                # Success!
                self._reset_camera_failure()
                
                # ENSURE BPM EXISTS AND HEAL SNAPSHOT
                # We must heal this frame BEFORE analyzing it, otherwise 
                # hot pixels will be incorrectly counted as stars!

                self._update_dynamic_dark_if_needed()
                frame = self._heal_live_frame(frame)

                # Dynamic red/blue gain calculator
                # 1. Extract the center roi
                h_f, w_f = frame.shape[:2]
                y_start, y_end, x_start, x_end = self._get_roi_coords(h_f, w_f)
                roi_color = frame[y_start:y_end, x_start:x_end]
                
                # 2. Calculate the mean of each channel (OpenCV is BGR)
                b_mean, g_mean, r_mean = cv2.mean(roi_color)[:3]
                
                # 3. Safety Check: Only neutralize if the sky has enough light to measure safely
                min_g_floor = self.cfg["capture"].get("awb_min_green_mean")
                if g_mean < min_g_floor:
                    logging.info(
                        f"Auto White-Balance: skipped, g_mean={g_mean:.2f} below "
                        f"confidence floor ({min_g_floor:}) -- measurement too noisy to trust."
                    )
                else:
                    ratio_bg = b_mean / g_mean
                    ratio_rg = r_mean / g_mean
                
                    curr_b = self.cfg["capture"].get("blue_gain")
                    curr_r = self.cfg["capture"].get("red_gain")
                
                    ideal_b = curr_b / ratio_bg
                    ideal_r = curr_r / ratio_rg
                    
                    # 4. Damping: move only partway toward the computed ideal each cycle
                    damping = self.cfg["capture"]["awb_damping_factor"]

                    smooth_b = curr_b + damping * (ideal_b - curr_b)
                    smooth_r = curr_r + damping * (ideal_r - curr_r)
                    
                    # 5. Apply as long as we're actually NOT at neutral color
                    # balance yet -- checked on the true ratio deviation from
                    # 1.0, NOT on the size of this cycle's damped step. The
                    # step naturally shrinks every cycle as curr approaches
                    # ideal (geometric decay from damping), so gating on step
                    # size alone causes the system to permanently stall with
                    # a real residual bias still outstanding, well before
                    # actual neutral balance is reached -- the step got
                    # small, that's not the same as "we're done".
                    min_ratio_deviation = self.cfg["capture"]["awb_min_ratio_deviation"]
                    still_off_neutral = (
                        abs(ratio_bg - 1.0) > min_ratio_deviation
                        or abs(ratio_rg - 1.0) > min_ratio_deviation
                    )

                    if still_off_neutral:

                        with self.status_lock:
                            self.cfg["capture"]["blue_gain"] = smooth_b
                            self.cfg["capture"]["red_gain"] = smooth_r

                        logging.info(
                            f"Auto White-Balance: Sky B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f} | "
                            f"Ideal -> R:{ideal_r:.2f} B:{ideal_b:.2f} | "
                            f"Applied (damped {damping}) -> R:{smooth_r:.2f} B:{smooth_b:.2f}"
                        )

                        # Apply to the physical hardware -- OUTSIDE status_lock. The preset
                        # syncs the camera (seconds), and computes the frame period from
                        # session_stats, which takes status_lock itself: threading.Lock is
                        # not re-entrant, so calling it under the lock hung this thread
                        # forever (and held the dashboard and every status reader with it).
                        self._set_camera_preset("NIGHT")
                    else:
                        logging.info(
                            f"Auto White-Balance: within {min_ratio_deviation*100:.0f}% of neutral "
                            f"(B/G={ratio_bg:.3f}, R/G={ratio_rg:.3f}) -- holding at R:{curr_r:.2f} B:{curr_b:.2f}"
                        )
                # ------

                # Analysis and Metadata Saving
                today = datetime.now().strftime("%Y-%m-%d_%H-%M")
                results = self.analyze_sky_conditions(frame)

                full_path = os.path.join(
                    self.daylight_out_dir, f"sky_conditions_{today}.png"
                )
                self._safe_save_image(full_path, frame)
                self._tally_data_write(full_path)

                moon_impact, moon_in_fov, moon_alt, moon_dist = self.get_moon_state()

                with self.status_lock:
                    self.last_sky_conditions = results
                    self.last_moon_status = {
                        "impact": moon_impact,
                        "in_fov": moon_in_fov,
                        "alt": moon_alt,
                        "dist": moon_dist,
                    }

                self.session_stats["max_moon_impact"] = max(
                    self.session_stats["max_moon_impact"], moon_impact
                )
                moon_penalty = 0.0
                # Adaptive threshold update
                if self.cfg["detection"]["auto_sensitivity_tuning"]:
                    current_stddev = float(results["stddev"])

                    # Update Noise Baseline (EMA)
                    if self.noise_baseline_stddev is None:
                        self.noise_baseline_stddev = current_stddev
                    else:
                        self.noise_baseline_stddev = (
                            (1 - self.cfg["detection"].get("noise_smoothing_alpha"))
                            * self.noise_baseline_stddev
                        ) + (
                            self.cfg["detection"].get("noise_smoothing_alpha")
                            * current_stddev
                        )

                    # Calculate the Target Threshold (Noise + Moon)
                    base_thr = self.cfg["detection"]["base_threshold"]
                    noise_factor = self.cfg["detection"]["noise_factor"]
                    moon_penalty = self.compute_moon_penalty(
                        moon_impact=moon_impact,
                        moon_alt=moon_alt,
                        moon_in_fov=moon_in_fov,
                    )

                    # This is what the system *wants* the threshold to be
                    target_threshold = int(
                        base_thr
                        + (self.noise_baseline_stddev * noise_factor)
                        + moon_penalty
                    )

                    # Apply Smoothing (Limit change to ±20% of current value)
                    if self.effective_threshold is None:
                        # First run initialization
                        self.effective_threshold = target_threshold
                    else:
                        # Smoothing: don't let it jump more than 20% in one step
                        det = self.cfg["detection"]
                        max_delta = max(
                            det["threshold_min_step"],
                            int(self.effective_threshold * det["threshold_max_step_fraction"]),
                        )

                        if target_threshold > self.effective_threshold:
                            self.effective_threshold = min(
                                self.effective_threshold + max_delta, target_threshold
                            )
                        elif target_threshold < self.effective_threshold:
                            self.effective_threshold = max(
                                self.effective_threshold - max_delta, target_threshold
                            )

                    # Final Clamp (Absolute safety limits)
                    self.effective_threshold = max(
                        self.threshold_min,
                        min(self.threshold_max, self.effective_threshold),
                    )

                    # Update status for Dashboard/ZMQ
                    with self.status_lock:
                        # self.last_sky_conditions = results
#                        self.last_moon_info = (f"{int(moon_impact*100)}% (In FOV: {moon_in_fov})")
                        self.last_effective_threshold = self.effective_threshold

                    logging.info(
                        f"Sky Monitor: Noise StdDev={self.noise_baseline_stddev:.2f} | Moon Impact={moon_impact:.2f} | Effective Threshold={self.effective_threshold}"
                    )
                else:
                    # Static Mode
                    self.effective_threshold = self.cfg["detection"]["static_threshold"]
                    with self.status_lock:
                        self.last_effective_threshold = self.effective_threshold

#                    self.last_moon_info = (f"{int(moon_impact*100)}% (In FOV: {moon_in_fov})")

                meta = self.get_metadata(
                    n_frames=1,
                    mean_brightness=results["mean"],             # Image Brightness
                    effective_threshold=self.effective_threshold,# The threshold chosen
                    stddev=results["stddev"],                    # Noise StdDev
                    image_filename=os.path.basename(full_path)
                )
                meta.update(results)
                # # Add Lunar details
                # meta["moon_impact"] = round(moon_impact, 3)
                # meta["moon_in_fov"] = bool(moon_in_fov) 
                
                # Get Base Config Limits
                cfg_day = self.cfg["daylight"]
                limit_std = cfg_day["stddev_threshold"]
                limit_stars = cfg_day["min_stars"]

                # Apply Lunar Relaxation
                # If the moon is IN FOV, it creates massive contrast (high stddev).
                # We must relax the limit, otherwise the moon itself makes the system think it's "Cloudy".
                if moon_in_fov:
                    # If moon is visible, allow 4x the standard deviation
                    # and accept fewer stars (since the glare hides faint ones)
                    effective_std_limit = limit_std * cfg_day["moon_std_relax_factor"]
                    effective_star_limit = max(
                        cfg_day["moon_min_stars"], int(limit_stars * cfg_day["moon_star_relax_factor"])
                    )
                    self.session_stats["moon_visible_minutes"] += elapsed_minutes
                    logging.info(
                        f"Sky Check: Moon IN FOV. Relaxing limits -> MaxStd: {effective_std_limit:.1f}, MinStars: {effective_star_limit}"
                    )
                else:
                    # Normal Dark Sky
                    effective_std_limit = limit_std
                    effective_star_limit = limit_stars
                    
                meta["effective_std_limit"] = effective_std_limit
                meta["effective_star_limit"] = limit_stars

                # Perform the Check
                score = self.compute_sky_score(
                    raw_stddev=results["stddev"],
                    norm_stddev=results["norm_stddev"],
                    mean_brightness=results["mean"],
                    stars=results["stars"],
                    std_limit=effective_std_limit,
                    star_limit=effective_star_limit,
                )

                meta["sky_score"] = score
                
                # Add for calculate the correct value for MIN_NOISE_SCORE and MIN_STAR_SCORE
                with self.status_lock:
                    self.last_sky_conditions["effective_std_limit"] = effective_std_limit
                    self.last_sky_conditions["effective_star_limit"] = effective_star_limit
                    self.last_sky_conditions["sky_score"] = round(score, 4)
                # -------

                self.save_json_atomic(os.path.splitext(full_path)[0] + ".json", meta)
                logging.info("Saved daylight snapshot and analysis to %s", full_path)

                self.session_stats["avg_sky_score"].append(score)
                self.sky_score = score
                is_clear = (
                    score >= SKY_SCORE_VALUES["fair"]
                )  # Fair start at 0.8, from 0.1 to < 0.8 is cloudy.

                self.session_stats["sky_history"].append(
                    {
                        "t": now_wall,  # one timestamp for all four series of this check
                        "score": round(score, 3),
                        "stddev": round(float(results["stddev"]), 2),
                        "stars": int(results["stars"]),
                    }
                )

                self.session_stats["star_count_history"].append(
                    {"t": now_wall, "val": int(results["stars"])}
                )
                self.session_stats["stddev_history"].append(
                    {"t": now_wall, "val": round(float(results["stddev"]), 2)}
                )
                self.session_stats["moon_penalty_history"].append(
                    {"t": now_wall, "val": round(moon_penalty, 2)}
                )

                logging.info(
                    f"SkyScore={score:.2f} | NormSTD={results['norm_stddev']:.4f} | Stars={results['stars']} | MoonPen={moon_penalty:.1f}"
                )
                logging.info(f"moon_alt={moon_alt:.2f} | moon_in_fov={moon_in_fov}")

                if is_clear:
                    # min_stars = self.cfg["daylight"]["min_stars"]
                    # stddev_thr = self.cfg["daylight"]["stddev_threshold"]

                    logging.info(
                        f"Sky conditions: CLEAR (Stars={results['stars']} >= Min={effective_star_limit}), "
                        f"STDDEV (stddev={results['stddev']:.2f} <= Effective Max={effective_std_limit:.2f})"
                    )

                    self.session_stats["clear_sky_minutes"] += elapsed_minutes
                    return SkyConditionStatus.CLEAR, results, frame
                else:
                    logging.warning(
                        "Sky conditions: CLOUDY/POOR (stddev=%.2f, stars=%d)",
                        results["stddev"],
                        results["stars"],
                    )

                    # FIX: Removed timelapse_interval edits. The State Machine handles this now
                    self.session_stats["cloudy_sky_minutes"] += elapsed_minutes
                    return SkyConditionStatus.CLOUDY, results, frame

            except Exception as e:
                logging.exception("An error occurred during daylight snapshot: %s", e)
                self.session_stats["error_mins"] += elapsed_minutes
                # with self.status_lock:
                    # Informative dashboard message
                    # self.last_moon_info = "SENSOR_ERROR"
                return SkyConditionStatus.ERROR, {"message": str(e)}, None

            finally:
                self.session_stats["last_check_time"] = now_mono
                logging.info("Restored night capture settings.")
    # -----------------
    def analyze_sky_conditions(self, img):
        """
        Analyzes the image to determine sky clarity using unified star detection.
        """
        stats = self.detect_stars(img, build_mask=False)
        
        if self.cfg["general"].get("debug_visualization"):
             logging.info(f"[SkyCheck] Stars={stats['stars']}, StdDev={stats['stddev']:.2f}, NormStdDev={stats['norm_stddev']:.4f}")

        return {
            "stddev": stats["stddev"], 
            "norm_stddev": stats["norm_stddev"], 
            "mean": stats["mean"], 
            "stars": stats["stars"]
        }
    # -----------------
    def compute_moon_penalty(self, moon_impact, moon_alt, moon_in_fov):
        """
        moon_impact: 0..1 (fase * illuminazione)
        moon_alt: gradi sopra l'orizzonte
        moon_in_fov: bool
        """
        if moon_alt <= 0:
            return 0.0

        day = self.cfg["daylight"]
        alt_factor = max(
            day["moon_penalty_alt_floor"],
            min(moon_alt / day["moon_penalty_alt_norm_deg"], 1.0),
        )

        # base scale
        scale = (
            self.cfg["capture"].get("moon_max_penalty")
            if moon_in_fov
            else self.cfg["capture"].get("moon_normal_penalty")
        )

        # NON linear impact
        penalty = scale * (moon_impact ** day["moon_penalty_exponent"]) * alt_factor
        return penalty
    # -----------------
    def compute_sky_score(self, raw_stddev, norm_stddev, mean_brightness, stars, std_limit, star_limit):
        """
        Hybrid Scoring: Uses Raw limit for dark skies, and Normalized texture check for bright skies.
        """
        # Pure saturation check.
        # If the image is completely blown out (pure white), it has 0 standard deviation,
        # which tricks the math into thinking it's a perfectly clean dark sky.
        day = self.cfg["daylight"]
        if mean_brightness > day["sky_saturation_brightness"]:
            return 0.0

        # Star Score (Linear)
        star_score = stars / max(star_limit, 1)

        # Noise Score (Hybrid)
        # The Strict Check (Raw)
        # If Raw Noise is low (e.g. 5.0 vs Limit 10.0), the sky is definitely clear.
        # Denominator floored at the uint8 quantisation noise: no integer-valued
        # measurement can resolve a smaller stddev.
        raw_score = std_limit / max(raw_stddev, QUANTIZATION_SIGMA_DN)

        if raw_score >= 1.0:
            # Pass! The sky is dark and quiet.
            final_noise_score = raw_score
        else:
            # Fail! The sky is noisy (Raw > Limit).
            # NOW we ask: Is it Clouds or just Brightness?

            # If the sky is pitch black (Mean < 5), Normalized math is unstable.
            # We must accept the Raw failure.
            if mean_brightness < day["sky_min_mean_for_texture"]:
                final_noise_score = raw_score
            else:
                # The sky is bright. Check Texture (Normalized).
                # Empirical Rule: Clear sky NormStd is usually < 0.10. Clouds are > 0.15.
                # We set a "Texture Limit" of 0.12
                # Same quantisation floor, expressed relative to the mean (norm = std / mean).
                norm_score = day["sky_texture_limit"] / max(
                    norm_stddev, QUANTIZATION_SIGMA_DN / mean_brightness
                )

                # If norm_score > 1.0, it means the image is smooth (Glare/Light), not lumpy (Clouds).
                # We "Forgive" the high raw noise by taking the better score.
                final_noise_score = max(raw_score, norm_score)

        # Both conditions must independently pass a minimum bar
        if final_noise_score < day["sky_min_noise_score"] or star_score < day["sky_min_star_score"]:
            return min(final_noise_score, star_score)   # penalise hard when either fails badly

        # Weights
        W_STD = day["sky_weight_noise"]
        W_STAR = 1.0 - W_STD

        score = (W_STD * final_noise_score) + (W_STAR * star_score)
        return score
    # -----------------
    def wait_for_safe_window(self):
        """
        Blocks until no event or timelapse stack is active.
        Returns False if shutdown is requested.
        """

        # The window is safe only when BOTH hold at the same moment: no event in
        # progress AND the timelapse stack just completed. They used to be checked
        # one after the other: an event that started while this thread waited (up
        # to minutes) for the stack was never re-checked, so the routine check --
        # which pauses detection and usually changes the exposure -- ran in the
        # middle of it. The event then resumed after a minutes-long gap at a
        # different exposure, never settled, and ran into EVENT TIMEOUT.
        while True:
            if self.event_in_progress.is_set():
                logging.warning("Sky Monitor: Event in progress. Deferring checks.")
                # Unbounded by design (events end on cooldown or their frame cap);
                # heartbeat-aware so a long event is not mistaken for a hang.
                while self.event_in_progress.is_set():
                    if not self.running.is_set():
                        return False  # Abort immediately on shutdown
                    self.ping_watchdog()
                    time.sleep(1.0)
                logging.info("Sky Monitor: Event finished.")

            # Check if Timelapse thread is actually running to avoid unnecessary waiting
            timelapse_alive = any(
                t.name == "TimelapseThread" and t.is_alive() for t in self.worker_threads
            )
            if not timelapse_alive:
                logging.info("Sky Monitor: Timelapse thread not active. Proceeding immediately.")
            else:
                logging.info("Sky Monitor: Waiting for current timelapse stack to finish...")
                self.timelapse_complete_event.clear()
                # Stack frames x the REAL frame period + stacking slack.
                wait_timeout = self._stack_completion_budget_sec()
                event_was_set = self.wait_event_with_heartbeat(self.timelapse_complete_event, wait_timeout)
                if not self.running.is_set():
                    return False
                if not event_was_set:
                    logging.warning(
                        f"Sky Monitor: Timed out ({wait_timeout:.1f}s) waiting for timelapse. Proceeding anyway."
                    )

            if not self.event_in_progress.is_set():
                logging.info("Sky Monitor: Safe window detected.")
                return True
            logging.info("Sky Monitor: An event started while waiting for the stack; waiting for it to end.")

    # -----------------
    def _get_roi_coords(self, h, w):
        """Calculates the exact pixel coordinates for the central ROI based on the configuration."""
        roi_percent = self.cfg["timelapse"]["roi_percent"]
        margin_h = (1.0 - roi_percent) / 2.0
        margin_w = (1.0 - roi_percent) / 2.0
        
        y_start, y_end = int(h * margin_h), int(h * (1.0 - margin_h))
        x_start, x_end = int(w * margin_w), int(w * (1.0 - margin_w))
        
        return y_start, y_end, x_start, x_end
    # -----------------
    def _compute_roi_stats(self, frame, blur=False):
        """
        Extracts the central ROI and returns its statistical properties.
        """
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply blur if requested (useful for noise floor estimation, bad for star counting)
        if blur:
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)  # detector's resolution-aware kernel
            
        h, w = gray.shape
        
        y_start, y_end, x_start, x_end = self._get_roi_coords(h, w)
        
        roi = gray[y_start : y_end, x_start : x_end]
        mean, stddev = cv2.meanStdDev(roi)
        
        mean_val = float(mean[0][0])
        stddev_val = float(stddev[0][0])
        
        # Normalize the stddev relative to the mean level
        norm_stddev = stddev_val / max(mean_val, 1.0)

        return mean_val, norm_stddev, stddev_val, gray, roi, (y_start, y_end, x_start, x_end)
    # -----------------
    def detect_stars(self, img, build_mask=False, save_debug=False, debug_out_dir=None):
        """
        Unified star detection logic using Top-Hat morphology and circularity filtering.
        Used by both SkyMonitor (for scoring) and Timelapse (for ECC alignment).
        """
        mean_val, norm_stddev, stddev_val, gray, roi_img, (y_start, y_end, x_start, x_end) = self._compute_roi_stats(img, blur=False)

        # Early reject: a spread below the uint8 quantisation noise is a flat/blank frame.
        if stddev_val < QUANTIZATION_SIGMA_DN:
            return {"stddev": stddev_val, "norm_stddev": norm_stddev, "mean": mean_val, "stars": 0, "mask": None, "valid": False}

        # --- 1. Top-Hat Transform (Remove light pollution & vignetting) ---
        base_k = self.star_mask_radius * self.top_hat_factor
        k_size = int(base_k) if int(base_k) % 2 != 0 else int(base_k) + 1
        k_size = max(self.cfg["timelapse"]["star_tophat_min_kernel_px"], k_size)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        tophat = cv2.morphologyEx(roi_img, cv2.MORPH_TOPHAT, kernel)

        # --- 2. Thresholding ---
        # Robust noise floor: use the TOPHAT's own statistics (not the pre-tophat
        # ROI stddev — a tophat is img - opening(img), a difference of two noisy
        # quantities with materially different noise variance than the source).
        # Median/MAD instead of mean/std for the same reason _generate_bpm_and_heal
        # uses it: a handful of bright star pixels would otherwise drag mean/std
        # upward and make the threshold self-defeating on star-rich frames.
        th_median = float(np.median(tophat))
        th_mad = float(np.median(np.abs(tophat - th_median)))
        th_sigma = th_mad * MAD_TO_SIGMA  # MAD->sigma consistency constant (matches BPM)

        sigma_mult = self.cfg["timelapse"]["star_sigma_multiplier"]
        thresh_val = th_median + (sigma_mult * th_sigma)

        # Minimum threshold floor to avoid counting black-level noise
        floor = self.cfg["timelapse"]["alignment_threshold_floor"]
        thresh_val = max(thresh_val, floor)

        _, thresh_img = cv2.threshold(tophat.astype(np.float32), thresh_val, 255, cv2.THRESH_BINARY)
        thresh_img = thresh_img.astype(np.uint8)

        # Core DIAMETER of a real star as a fraction of the tracking radius; anything
        # smaller is noise or a hot pixel. Floor 3 px = the smallest symmetric
        # structuring element the opening below can use (structural, not tunable).
        min_star_dim = max(3, int(self.star_mask_radius * self.cfg["timelapse"]["star_core_diameter_fraction"]))
        # --- 3. Morphological Cleanup ---
        clean_size = min_star_dim
        if clean_size % 2 == 0: clean_size += 1
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, clean_kernel)

        # --- 4. Contour Analysis & Shape Filtering ---
        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        star_count = 0
        valid_stars_full_frame = []

        max_star_area = (self.star_mask_radius * self.cfg["timelapse"]["star_max_extent_factor"]) ** 2
        # AREA of a circle with diameter min_star_dim -- NOT the raw linear
        # dimension. The previous code compared a length directly against
        # cv2.contourArea()'s px^2 output, giving an effective floor of ~3 px^2.
        min_star_area = math.pi * (min_star_dim ** 2) / 4.0
        min_circ = self.cfg["daylight"]["min_star_circularity"]
        
        for c in cnts:
            area = cv2.contourArea(c)
            if min_star_area <= area <= max_star_area:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > min_circ:
                    star_count += 1
                    
                    # If we need the mask/debug, offset the coordinates back to full-frame space
                    if build_mask or save_debug:
                        c_full = c.copy()
                        c_full[:, 0, 0] += x_start
                        c_full[:, 0, 1] += y_start
                        valid_stars_full_frame.append(c_full)

        # --- 5. Optional Mask Building ---
        mask = None
        is_valid = False
        
        if build_mask:
            mask = np.zeros_like(gray, dtype=np.uint8)
            # Fill the exact shape of the stars
            cv2.drawContours(mask, valid_stars_full_frame, -1, 255, -1)
            
            # Add a safety radius around the centroids for generous ECC tracking
            for c in valid_stars_full_frame:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(mask, (cx, cy), self.star_mask_radius, 255, -1)
            
            # Dilate slightly for ECC stability
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, dil_kernel, iterations=1)
            
            # Re-enforce ROI boundaries to be safe
            h, w = gray.shape
            mask[:y_start, :] = 0
            mask[y_end:, :] = 0
            mask[:, :x_start] = 0
            mask[:, x_end:] = 0
            
            min_features = self.cfg["timelapse"]["min_features_for_alignment"]
            if star_count >= min_features and cv2.countNonZero(mask) >= self.min_mask_px:
                is_valid = True

        # --- 6. Debug Visualization ---
        if save_debug and debug_out_dir:
            try:
                debug_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(debug_vis, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
                cv2.drawContours(debug_vis, valid_stars_full_frame, -1, (0, 255, 0), 1)
                ts = datetime.now().strftime("%H%M%S")
                out_path = os.path.join(debug_out_dir, f"debug_stack_roi_{ts}.jpg")
                self._safe_save_image(out_path, debug_vis)
            except Exception: pass

        return {
            "stddev": stddev_val,
            "norm_stddev": norm_stddev,
            "mean": mean_val,
            "stars": star_count,
            "mask": mask,
            "valid": is_valid
        }
    # -----------------
    def get_moon_state(self, calc_date=None):
        """Calculates Moon position, phase, and FOV status."""
        try:
            import ephem

            obs = ephem.Observer()
            obs.lat, obs.lon = str(self.cfg["general"]["latitude"]), str(
                self.cfg["general"]["longitude"]
            )
            # pyephem's Observer.date takes the datetime's calendar fields as-is and
            # assumes they are UTC -- it does NOT read/convert via tzinfo. Everywhere
            # else in the pipeline was switched to local time for display/filenames,
            # but this must stay real UTC or the moon alt/az/phase math comes out
            # wrong by the local UTC offset. Any caller that needs to pass an
            # explicit calc_date derived from a local-time value (e.g. re-parsed from
            # a filename) must convert it to true UTC first -- see the startrail
            # generator in timelapse_mixin.py for the correct (DST-safe) way to do
            # that conversion before calling this with calc_date=.
            obs.date = (
                calc_date if calc_date is not None else datetime.now(timezone.utc)
            )

            moon = ephem.Moon(obs)
            # Get raw radians first for math
            m_alt_rad, m_az_rad = float(moon.alt), float(moon.az)
            # Convert to degrees for display
            m_alt_deg, m_az_deg = np.degrees(m_alt_rad), np.degrees(m_az_rad)
            m_phase = moon.moon_phase

            day = self.cfg["daylight"]
            if m_alt_deg <= day["moon_min_alt_deg"]:
                return 0.0, False, m_alt_deg, 180.0

            # Camera Orientation
            c_alt = self.cfg["general"].get("camera_altitude") 
            c_az = self.cfg["general"].get("camera_azimuth")
            h_fov = self.cfg["general"].get("lens_hfov")
            v_fov = self.cfg["general"].get("lens_vfov")
            
            # 3D RECTANGULAR PROJECTION MATH
            # Convert to radians
            c_alt_r, c_az_r = np.radians(c_alt), np.radians(c_az)
            
            # 1. Camera Center Vector (X=East, Y=North, Z=Up)
            cx = np.cos(c_alt_r) * np.sin(c_az_r)
            cy = np.cos(c_alt_r) * np.cos(c_az_r)
            cz = np.sin(c_alt_r)

            # (A former "near-horizon, up vector degenerates" warning was removed: the right
            # vector (cos az, -sin az, 0) is orthogonal to the centre vector for EVERY
            # altitude -- their dot product is cos(alt)*(cos az sin az - sin az cos az) = 0 --
            # so up = right x centre always has unit length and never degenerates.)

            # 2. Moon Vector
            mx = np.cos(m_alt_rad) * np.sin(m_az_rad)
            my = np.cos(m_alt_rad) * np.cos(m_az_rad)
            mz = np.sin(m_alt_rad)
            
            # Absolute distance from center (Dot product of unit vectors)
            dot_prod = np.clip(cx*mx + cy*my + cz*mz, -1.0, 1.0)
            dist = np.degrees(np.arccos(dot_prod))

            # 3. Camera Right Vector (Horizontal plane)
            rx = np.cos(c_az_r)
            ry = -np.sin(c_az_r)
            rz = 0.0

            # 4. Camera Up Vector (Cross product: Right x Center)
            ux = ry * cz - rz * cy
            uy = rz * cx - rx * cz
            uz = rx * cy - ry * cx

            # 5. Project Moon onto Camera's X/Y local axes
            moon_x_proj = mx * rx + my * ry + mz * rz
            moon_y_proj = mx * ux + my * uy + mz * uz

            # Convert projections to degrees off-center
            moon_x_deg = np.degrees(np.arcsin(np.clip(moon_x_proj, -1.0, 1.0)))
            moon_y_deg = np.degrees(np.arcsin(np.clip(moon_y_proj, -1.0, 1.0)))

            # Within the rectangular sensor bounds, padded for glare just off-screen.
            margin = day["moon_fov_glare_margin_deg"]
            in_fov = (abs(moon_x_deg) < (h_fov / 2.0) + margin) and (abs(moon_y_deg) < (v_fov / 2.0) + margin)

            # Impact scaling: phase x altitude above horizon as a fraction of the zenith (90 deg).
            impact = (m_phase * (max(0, m_alt_deg) / 90.0)) ** day["moon_impact_exponent"]
            if in_fov:
                impact = min(1.0, impact + day["moon_in_fov_impact_boost"])

            return round(impact, 3), in_fov, round(m_alt_deg, 1), round(dist, 1)

        except Exception as e:
            logging.error(f"Moon Logic Error: {e}")
            return 0.0, False, 0.0, 180.0
    # -----------------
    def get_cosmic_narrative(self, score, stars, moon_up):
        try:
            stars = int(stars)
        except (ValueError, TypeError):
            stars = 0 # Fallback se è "N/A"

        ctx = self._get_chronicle_context()
        ep = ctx.get("events_pct") if ctx else 50
        trend = ""
        if self.session_stats.get("sky_history"):
            fp = self._generate_atmosphere_fingerprint(
                self.session_stats["sky_history"]
            )
            if fp["trend"] == "improving":
                trend = " The sky is clearing."
            elif fp["trend"] == "degrading":
                trend = " Transparency is fading."

        if score < 0.8:
            return "The sky is veiled. We wait patiently for the curtain to lift."
        if ctx and ep >= 90:
            return f"An exceptional night — top {100-ep}% of all {ctx['nights_total']} nights observed.{trend}"
        if moon_up and score > 1.2:
            return f"The Moon commands the sky, yet the air is crystal clear.{trend}"
        if stars > 40:
            return f"A magnificent night. The deep sky is fully revealed.{trend}"
        if ctx and ep >= 60:
            return f"A good night — above average across {ctx['nights_total']} sessions.{trend}"
        if score >= 1.0:
            return f"The atmosphere is steady. We are watching.{trend}"
        return f"The sky is quiet. Holding vigil.{trend}"

    # -----------------
    # 6. OUTPUTS: EVENTS & TIMELAPSE
    # -----------------
