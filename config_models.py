"""
config_models.py

Pydantic configuration models for the Ryusei pipeline, extracted from the
main pipeline file as a pure move -- no logic changed, only relocated.
These are standalone models with no dependency on Pipeline's runtime state,
which is what makes this extraction low-risk: nothing here references
`self` or anything defined only inside the Pipeline class.

Imports here are exactly what the extracted validators actually use --
verified by grepping the extracted block for cv2./os./psutil. before
writing this file, not assumed.
"""
import os
import socket
import psutil
from typing import Literal, Optional, List
from pydantic import (BaseModel, Field, ValidationError, confloat, conint, conlist, constr, field_validator, model_validator)

# --------------------------- Configuration ---------------------------
DEFAULTS = {
    "capture": {
        "processing_headroom": 1.25,           # NIGHT frame-rate matching: the sensor's minimum frame period is the pipeline's measured per-frame cost (detection loop, timelapse stacking) times this margin, whenever that is longer than the exposure. Frames are then never produced faster than they can be processed.
        "frame_rate_hysteresis": 0.1,          # Re-apply the matched frame period only when the required one differs from the applied one by more than this fraction (avoids chattering).
        "raw_buffer_count": 2,                 # Picamera2 buffer count. Sets the sync pipeline depth (see camera_capture.py). (Was a hidden .get() fallback.)
        "ae_recheck_max_frame_age_frames": 2,  # The AE re-check only trusts an in-memory frame at most this many frame periods old.
        "ae_saturation_backoff": 0.5,          # Exposure*gain multiplier applied when an AE verification frame is saturated (hard back-off, bypasses the change limit).
        "safe_window_slack_sec": 30,           # Slack on the 'current timelapse stack finished' wait: stack frames x frame period + this (covers alignment/stacking CPU time).
        "width": 1280,                      # Horizontal and vertical resolution of the sensor in pixels.
        "height": 720,                      # and vertical resolution of the sensor in pixels.
        "exposure_us": 200000,              # Initial or fixed shutter speed in microseconds.
        "gain": 2.0,                        # Initial or fixed analog sensor gain multiplier.
        "red_gain": 0.98,                   # Fixed white balance multipliers to neutralize sky color.
        "blue_gain": 1.08,
        "auto_exposure_tuning": False,      # Toggles mathematical adjustment of settings to maintain consistent sky brightness.
        "target_sky_brightness": 15,        # Target median brightness on a 0-255 scale
        "min_exposure_us": 100000,          # 0.1 seconds
        "max_exposure_us": 800000,          # 0.8 seconds
        "min_gain": 1.0,                    # The absolute lower limit for analog gain.
        "max_gain": 4.0,                    # The practical maximum before excessive noise, architettural limit
        "tolerance": 2,                     # The allowed brightness deviation (0-255) before the AE loop attempts a correction.
        "max_attempts": 10,                 # Limit on how many iterative frames the AE engine will capture to reach the target.
        "exposure_sync_retries": 3,         # Times the AE loop re-applies a NIGHT exposure and waits for the sensor to confirm the new ExposureTime in metadata before giving up.
        "ae_max_stops_per_check": 1.0,      # Base limit, in stops (factor 2^x on exposure*gain), on how far one auto-exposure may move from the last TRUSTED check on a plain night (no twilight, no moon change). Guards against a passing cloud (or headlights, etc.) during a check leaving settings that are wrong once it's gone.
        "ae_twilight_stops_per_deg": 1.5,   # Extra allowance per degree of sun altitude travelled inside the twilight band since the last trusted check. Real zenith sky brightness changes ~1.3 stops/deg through twilight, so dusk/dawn are never held back.
        "ae_twilight_sun_alt_min": -20.0,   # Sun altitude (deg) where twilight brightening begins (astronomical twilight, with margin).
        "ae_twilight_sun_alt_max": 5.0,     # Sun altitude (deg) up to which the twilight allowance applies.
        "ae_moon_stops_per_deg": 0.1,       # Extra allowance per degree of Moon altitude travelled while above ae_moon_alt_min, scaled by its illuminated fraction (scattered moonlight lifts the whole sky, not only when it's in the FOV).
        "ae_moon_alt_min": -6.0,            # Moon altitude (deg) below which it is ignored.
        "ae_sample_step_min": 10,           # Step, in minutes, used to integrate the Sun/Moon path between two checks (so gaps of hours or a whole day are handled correctly).
        "ae_clamp_recheck_min": 10,         # After a check clamped by the change limit, look again after this many minutes with a LIGHTWEIGHT re-check: it reads the frame already in memory (no acquisition pause) and only touches the camera, at a timelapse-stack boundary, if the settings turn out to be off. Dropped if a full check runs first.
        "frame_duration_headroom_us": 50000, # Slack added to the MAX of FrameDurationLimits for NIGHT so the sensor's minimum vertical blanking always fits on top of the exposure (min limit == exposure). It does NOT affect how soon an exposure change lands: libcamera runs the shortest frame >= the MIN limit, and the latency is (buffer_count + sync_control_delay_frames) frames -- see camera_capture.py. (A raise to 200000 based on the "Flush timed out" logs was reverted: those were a timeout-budget bug.)
        "sync_control_delay_frames": 3,     # Sensor control delay, in frames, before a new ExposureTime/AnalogueGain/vblank takes effect (Raspberry Pi IPA DelayedControls: 2-3 frames depending on sensor; the max is used). Added to buffer_count to size the flush() budget; if CAMSYNC OK lines ever report frames > expected_frames, raise this.
        "sync_timeout_slack_sec": 2.0,      # Additive slack on the physics-derived flush() budget ((buffers + delay) x frame duration) for ISP/scheduler latency. Dominant only for short exposures.
        "sync_exposure_rel_tol": 0.02,      # Relative tolerance for "frame metadata matches the requested ExposureTime". Exposure is quantised to one sensor line (tens of us), <0.1% at night exposures, so 2% is generous yet rejects any real stale frame.
        "sync_gain_rel_tol": 0.05,          # Relative tolerance for "frame metadata matches the requested AnalogueGain". Must exceed the sensor's gain-code quantisation (up to ~2-4% near max analogue gain on IMX219/IMX477 class sensors). Changes smaller than this are indistinguishable from quantisation and are brightness-irrelevant to AE.
        "max_brightness": 240,              # The ceiling for the mean image brightness to prevent sensor saturation.
        "moon_normal_penalty": 15,          # 15 - Penalty when moon is out of FOV
        "moon_max_penalty": 40,             # Penalty when moon is in FOV
        "ae_step_limit_up": 4.0,            # Max multiplier for one AE check
        "ae_step_limit_down": 0.25,         # Min multiplier (0.25 = 4x darker)
        "masterpiece_burn_multiplier": 3.0, # this * detection.base_threshold should be around 45-50. more high, more dark. max 10
        "awb_min_green_mean": 2.0,          # confidence floor before trusting a measurement
        "awb_damping_factor": 0.3,          # fraction of the way to move toward ideal per cycle
        "awb_min_ratio_deviation": 0.02,    # your original commented-out threshold, restored
    },
    "detection": {
        "frame_continuity_max_gap_frames": 2.5,  # DEPRECATED, IGNORED: pre-event history is now cleared explicitly wherever detection pauses (sky-check gate, daylight/weather hold). Kept only so config.json files that already contain it still validate.
        "calibration_library_frames": 50,      # Frames combined (sigma-clipped) into each master bias/dark/flat by the --bias/--dark/--flat CLI.
        "auto_flat_refresh_every_samples": 4,  # Recompute the dynamic flat every N clear-sky samples added.
        "auto_flat_fit_grid_min_cells": 16,    # Minimum cells per axis of the dynamic-flat surface fit grid (keeps the fit well-sampled at low resolution).
        "reject_save_interval_sec": 15,        # Minimum spacing of 'rejected contour' diagnostic snapshots. (Was a hidden .get() fallback: not configurable before.)
        "background_warmup_noise_excess": 0.25, # Detection resumes after a background reset once the difference-image noise is within this fraction of its steady state; the wait in frames is derived from the live accumulate alpha (see detection._background_warmup_sec). 0.25 reproduces the old fixed 5 s at the default 0.2 s exposure.
        "light_shift_contours_per_mpix": 108.507, # Contours per megapixel above which a frame is a global light change (flash, headlights, cloud edge), not motion. Resolution-independent; 108.507 == the old fixed "> 100 contours" at the default 1280x720 (100 / 0.9216 MP = 108.5069).
        "threshold_max_step_fraction": 0.2,    # Auto-sensitivity: max change of the detection threshold per sky check, as a fraction of its current value.
        "threshold_min_step": 2,               # Auto-sensitivity: minimum threshold change per sky check (DN), so small thresholds can still move.
        "dynamic_dark_refresh_sec": 300,       # Period of the temperature-scaled dynamic dark refresh.
        "calibration_max_age_days": 180,       # Warn when the dark library is older than this (sensor dark current/hot pixels drift).
        "dark_reference_exposure_us": 5000000, # Exposure the thermal dark library frames were captured at; darks are scaled from it.
        "event_score_w_ai": 0.5,               # Event score weight: AI classifier term (the four event_score_w_* must sum to 1).
        "event_score_w_aspect": 0.25,          # Event score weight: trail linearity (aspect ratio) term.
        "event_score_w_solidity": 0.15,        # Event score weight: path consistency (solidity) term.
        "event_score_w_brightness": 0.1,       # Event score weight: peak brightness term.
        "event_score_fallback_w_ai": 0.7,      # Weight of the AI term when no trail shape can be measured (brightness gets 1 - this).
        "event_score_aspect_norm": 8.0,        # Aspect ratio that scores 1.0 (a trail this elongated counts as perfectly linear).
        "event_score_trail_percentile": 98.0,  # Percentile of the stacked event image used as the trail/background split.
        "dark_frame_path": None,            # Path to a static PNG used for thermal noise subtraction.
        "bias_frame_path": None,            # Path to a static PNG used for read-noise calibration.
        "flat_frame_path": None,            # Path to a static PNG used to correct lens vignetting.
        "auto_sensitivity_tuning": False,   # Enables the dynamic threshold engine (adjusts sensitivity based on moon/noise).
        "base_threshold": 15,               # used when auto_sensitivity_tuning = true The starting sensitivity value used when auto-tuning is enabled.
        "noise_factor": 2.5,                # <------
        "noise_smoothing_alpha": 0.05,
        "static_threshold": 25,             # used when auto_sensitivity_tuning = false The fixed sensitivity value used when auto-tuning is disabled.
        "min_threshold": 10,                # Hard safety limits for the adaptive threshold engine.
        "max_threshold": 100,
        "target_adaptation_time": 20.0,     # Background model absorption speed, increase to get higher max_exposure_us (60.0 -> Unlocks exposures up to 18 seconds)
        "min_accumulate_alpha": 0.001,
        "max_accumulate_alpha": 0.300,      # Ceiling on the background model's per-frame update rate; scales up with exp to hit target_adaptation_time, but never past this, or a real transient gets absorbed into the background too fast to detect.
        "auto_dark_enabled": False,         # Toggles real-time scaling of dark frames based on sensor temperature.
        "auto_flat_enabled": False,         # Enables automated generation of synthetic flat-field maps from clear sky data.
        "auto_flat_buffer_size": 20,        # Number of clear-sky snapshots to keep in memory for the synthetic flat-field fit.
        "auto_flat_min_samples": 5,         # Minimum required valid snapshots before the engine calculates a new flat map.
        "bpm_bg_blur_kernel": 31,           # Background approximation kernel for hot-pixel isolation (must be odd)
        "bpm_sigma_multiplier": 5.0,        # Statistical outlier threshold for defect detection (5-sigma standard)
        "bpm_sigma_floor": 3.0,             # Floor to ignore normal 8-bit read noise regardless of sigma calc
        "bpm_max_blob_px": 30,              # Ignore contiguous defects larger than this (amp glow, not point defects)
        "bpm_kernel_min_px": 3,             # Minimum median-heal kernel size
        "bpm_kernel_max_px": 15,            # Maximum median-heal kernel size (CPU ceiling for live-frame healing)
        "auto_flat_clip_sigma": 4.0,        # MAD-based robust-sigma reject threshold for per-pixel outliers (stars/planes/clouds)
        "auto_flat_poly_degree": 3,         # Order of the 2D polynomial surface fit; empirically, degree 2 leaves ~10 DN of coherent low-frequency residual in G/B on this sensor (asymmetric/decentered vignetting), degree 3 roughly halves it -- don't raise past 4 casually, it starts refitting real contamination as "shape"
        "auto_flat_fit_grid_fraction": 0.10,# Decimation fraction used for the fit (CPU guard); reconstruction is analytic so this only affects fit noise, not final smoothness
        "auto_flat_fit_oversample_factor": 5.0,  # Minimum (samples / poly terms) ratio required before a fit is trusted
        "auto_flat_mult_min": 0.5,          # Safety floor for the synthetic flat correction multiplier.
        "auto_flat_mult_max": 2.5,          # Safety ceiling for the synthetic flat correction multiplier.
    },
    "astrometry": {
        "enabled": True,
        "tetra3_database": "default_database",  # Ensure you generated this using tetra3!
        "target_half_fov_deg": 25.0,            # Primary center crop radius
        "cross_check_half_fov_deg": 30.0,       # Secondary cross-check radius
        "min_stars_for_solve": 6,               # Tetra3 usually needs a minimum of 5-8
        "match_radius_deg": 0.2,                # Tetra3 tolerance
        "pattern_checking_stars": 8,            # Stars used for checking tetra pattern
        "plate_scale_tolerance_pct": 5.0,       # Cross-check tolerance
        "event_solve_timeout_sec": 5.0,
        "solve_readiness": {
            "max_lux": 0.8,                     # validate against your own avg_lux distribution first
            "min_norm_stddev": 0.13,
            "min_stars_in_roi": 15,             # measured within target_half_fov_deg's crop, not full-frame
            "moon_impact_notable": 0.15,        # above this, a lux failure gets attributed to the moon, not pollution
        },
    },
    "timelapse": {
        "star_core_diameter_fraction": 0.25,   # A star's core diameter as a fraction of star_mask_radius; smaller blobs are noise/hot pixels (floored at the 3 px minimal structuring element).
        "star_max_extent_factor": 4.0,         # Blobs larger than (star_mask_radius x this)^2 px are not stars (planes, glare, clouds).
        "star_tophat_min_kernel_px": 15,       # Lower bound of the top-hat kernel (must be odd); keeps background removal meaningful at low resolution.
        "stack_bias_ref_weight": 0.8,          # Weight of the reference frame in BIAS (unaligned) stacking mode.
        "stack_bias_frame_weight": 0.9,        # Weight of other frames in BIAS mode.
        "ecc_max_iterations": 50,              # findTransformECC iteration cap.
        "ecc_epsilon": 0.001,                  # findTransformECC convergence threshold on the correlation increment.
        "alignment_min_pixels_per_feature": 20, # Star-mask pixels required per alignment feature before ECC is attempted.
        "emergency_throttle_step_sec": 0.5,    # DEPRECATED, IGNORED: the timelapse emergency throttle was replaced by sensor-side acquisition pacing (capture._acquisition_pacing). Kept so existing config.json files still validate.
        "emergency_throttle_recovery_sec": 0.1, # DEPRECATED, IGNORED: the timelapse emergency throttle was replaced by sensor-side acquisition pacing (capture._acquisition_pacing). Kept so existing config.json files still validate.
        "emergency_throttle_recovery_queue_fraction": 0.5, # DEPRECATED, IGNORED: the timelapse emergency throttle was replaced by sensor-side acquisition pacing (capture._acquisition_pacing). Kept so existing config.json files still validate.
        "nighttime_stack_N": 20,            # The number of individual frames combined to create one stacked image.
        "stack_align": True,                # Toggles sub-pixel star alignment (ECC) to eliminate star trailing in stacks.
        "nightmode_interval_sec": 0,        # Idle time between captures during scheduled night hours.
        "daymode_interval_sec": 60,         # Idle time between captures during daylight hours.
        "stack_method": "mean",             # <--- mean remain the only method, mip is used onlyfor masterpiece
        "min_features_for_alignment": 5,    # Minimum number of stars required to attempt sub-pixel ECC stack alignment.
        "roi_percent": 0.7,                 # Central portion of the frame (0.1 to 1.0) analyzed for star detection and scoring.
        "cloudy_interval_sec": 60,          # Cloudy throttle interval
        "alignment_threshold_floor": 3,     # Minimum pixel value to consider for star mask
        "star_sigma_multiplier": 5.0,       # Outlier threshold for stars on the Top-Hat image
        "stack_brightness_boost": 1.2,      # Post-combine brightness multiplier
        "stack_sigma_clip_enabled": True,   # Reject per-pixel temporal outliers (cosmic rays, hot pixels, noise spikes) in MEAN (timelapse) stacks, with robust median/MAD statistics, processed in RAM-budgeted tiles. Transients the detector saw (meteors, planes, satellites) are PROTECTED from rejection via their event footprints, so they still appear in the timelapse exactly where the event stack shows them. NEVER applied to MIP/event stacks -- see stack_frames in stacking.py.
        "stack_sigma_clip_k": 3.0,          # Outlier threshold in robust-sigma (MAD-based) units for the temporal stack
        "stack_sigma_clip_min_frames": 5,   # Minimum frames in the stack before clipping is statistically meaningful
        "stack_sigma_clip_floor": 1.0,      # Robust-sigma floor (0-255 units) to avoid over-rejecting in dead-flat regions where MAD -> 0
        "stack_sigma_clip_ram_fraction": 0.5,  # Max fraction of CURRENTLY AVAILABLE RAM the clip working set may use. Sets the tile height (rows processed at once), computed from exact per-row dtype sizes; falls back to a plain weighted mean only if not even one row fits.
        "stack_event_wait_slack_sec": 5.0,  # Slack on the bounded wait for the detector to finish its pre-event lookback past a stack's last frame ((lookback + 1) frames x exposure + this), so no event starting just after the stack can be missed. Only reached when the detector is stalled or stopped.
        "stack_sigma_clip_peak_multiplier": 3.5,  # DEPRECATED, IGNORED: the tiled clip computes its exact working set from dtype sizes (see _sigma_clip_tile_rows). Kept only so existing config.json files that contain it still validate.
        "cc_floor_normal": 0.60,            # Below this ECC correlation, reject the frame (normal alignment mode)
        "cc_floor_bias": 0.30,              # Below this ECC correlation, reject the frame (bias/degraded mode)
        "cc_perfect_normal": 0.85,          # Correlation at/above which a frame gets full weight (normal mode)
        "cc_perfect_bias": 0.60,            # Correlation at/above which a frame gets full weight (bias mode)
        "weight_floor_normal": 0.4,         # Minimum partial weight for a marginally-correlated frame (normal mode)
        "weight_floor_bias": 0.2,           # Minimum partial weight for a marginally-correlated frame (bias mode)
    },
    "timelapse_video": {
        "brightness_outlier_tolerance_factor": 2.0, # Session average/startrail: frames whose brightness is off the AE target by more than this x capture.tolerance are excluded.
        "encode_timeout_sec": 600,             # ffmpeg hard timeout for the nightly timelapse video. Must be < general.timelapse_video_thread_timeout_sec (validated).
        "enabled": True,
        "video_fps": 10,                    # Frames per second for the resulting output video file.
        "ffmpeg_encoder": "libx264",        # The compression library used for encoding.
        "ffmpeg_crf": 12,                   # Constant Rate Factor; determines quality vs file size (Lower = Better, 12 is near-lossless).
        "ffmpeg_pix_fmt": "yuv444p",        # Chromasubsampling format for video compatibility (e.g., yuv420p or yuv444p).
    },
    "events": {
        "writer_wait_sec": 300,                # How long the file finalizer waits for the event writer's JSON. Must be < general.finalize_video_thread_timeout_sec (validated).
        "video_fps": 4,                     # Frames per second for the resulting output video file.
        "ffmpeg_encoder": "libx264",        # The compression library used for encoding.
        "ffmpeg_crf": 12,                   # Constant Rate Factor; determines quality vs file size (Lower = Better, 12 is near-lossless).
        "ffmpeg_pix_fmt": "yuv444p",        # Chromasubsampling format for video compatibility (e.g., yuv420p or yuv444p).
    },
    "monitor": {
        "interval_sec": 5,                  # Frequency of system health telemetry logs (CPU/RAM/Temp).
        "ram_limit_percent": 85.0,          # Memory usage ceiling that triggers OOM protection (suspends recording).
        "ram_recovery_percent": 70.0,       # Memory level required to resume normal detection after an OOM event.
    },
    "power_monitor": {
        "check_interval_sec": 2,         # Poll period of the power-good GPIO line.
        "enabled": True, 
        "pin": 17,                          # GPIO pin used to monitor the state of the external power supply.
        "shutdown_delay_sec": 60            # Seconds to wait after power loss before initiating a system halt.
    },
    "led_status": {
        "enabled": True, 
        "pin": 26                           # GPIO pin used for the physical status indicator light.
    },
    "daylight": {
        "default_calibration_k": 20.0,         # Lux model constant used until a valid calibration_k exists (Lux = brightness / (k * exposure_s * gain)).
        "min_valid_calibration_k": 0.5,        # calibration_k below this came from a near-black calibration frame and is rejected (it would inflate Lux ~40x).
        "snap_exposure_us": 50000,             # SNAP preset exposure (sky-condition snapshot): short enough not to saturate by day, long enough for a meaningful Lux at night.
        "snap_gain": 2.0,                      # SNAP preset analogue gain.
        "day_frame_duration_min_us": 1000,     # DAY mode FrameDurationLimits minimum (auto-exposure free-run).
        "day_frame_duration_max_us": 33333,    # DAY mode FrameDurationLimits maximum (33333 us = 30 fps).
        "dawn_trigger_moon_floor": 180.0,      # Dawn early detector with the Moon in the FOV: trigger brightness is at least this...
        "dawn_trigger_moon_multiplier": 2.0,   # ...and at least dawn_trigger_brightness x this.
        "moon_std_relax_factor": 2.5,          # Moon in FOV: cloud test stddev limit multiplied by this (lunar glare adds genuine contrast).
        "moon_star_relax_factor": 0.4,         # Moon in FOV: required star count multiplied by this (glare hides faint stars).
        "moon_min_stars": 3,                   # Moon in FOV: never require fewer stars than this.
        "sky_saturation_brightness": 245.0,    # Mean brightness above which a frame is blown out (zero stddev would otherwise read as a perfectly clean sky).
        "sky_min_mean_for_texture": 5.0,       # Below this mean brightness, stddev/mean is numerically unstable: the raw-noise verdict is final.
        "sky_texture_limit": 0.12,             # Normalized-stddev limit separating smooth glare (clear) from lumpy cloud texture.
        "sky_min_noise_score": 0.6,            # Minimum noise sub-score for any positive sky verdict.
        "sky_min_star_score": 0.3,             # Minimum star sub-score for any positive sky verdict.
        "sky_weight_noise": 0.6,               # Weight of the noise sub-score in the sky score (stars get 1 - this).
        "moon_penalty_alt_norm_deg": 60.0,     # Moon altitude at which the brightness penalty reaches its full altitude factor.
        "moon_penalty_alt_floor": 0.2,         # Minimum altitude factor for a risen Moon.
        "moon_penalty_exponent": 1.5,          # Non-linearity of the penalty in moon_impact.
        "moon_min_alt_deg": -5.0,              # Moon below this altitude has no impact.
        "moon_impact_exponent": 1.2,           # Non-linearity of moon impact in (phase x altitude/90).
        "moon_fov_glare_margin_deg": 3.0,      # Moon within this margin outside the FOV edge still counts as in FOV (glare).
        "moon_in_fov_impact_boost": 0.4,       # Impact added when the Moon is in the FOV (capped at 1).
        "enabled": True,
        "stddev_threshold": 7.0,  # <----
        "min_stars": 12,                    # Minimum star count required to classify a clear sky.
        "check_interval_min": 30,           # Time in minutes between routine sky condition checks.
        "lux_threshold": 20.0,              # Light level (Lux) above which the system switches to Daylight Mode.
        "min_star_circularity": 0.6,        # Shape filter (0.0 to 1.0) to distinguish stars from noise or streaks.
        "dawn_trigger_brightness": 50,      # > of detection["base_threshold"]
        "dawn_trigger_consecutive": 2,      # Required count of bright frames before triggering an early dawn transition.
        "capture_interval_sec": 1.0         # Min seconds between camera reads while acquisition is paused (Daylight Mode OR Weather Hold). Long night exposures self-pace, but a bright cloudy/daylight scene drops the exposure to ms and the capture loop then floods the acquisition queue at full sensor fps. 0 disables the throttle.
    },
    "janitor": {
        "monitor_path": "./",               # The root directory checked for disk space availability.
        "log_rotation_mb": 10,              # Maximum size of the primary log file before the Janitor archives it.
        "csv_rotation_mb": 20,              # Maximum size of telemetry CSVs before they are rotated.
        "threshold": 90.0,                  # The disk usage percentage that triggers cleanup and the desired level to reach.
        "target": 75.0,                     # The disk usage percentage the Janitor attempts to reach during a cleanup cycle.
        "storage_reserve_mb": 512,          # Free space that must always remain on the data drive. Below it the storage governor HALTs: every artifact write is refused and acquisition stops until space is recovered (automatic). Also: any single write that would dip below it is refused. Keeps room for logs, the OS and in-flight writes.
        "check_interval_min": 60,           # Routine janitor cycle (cleanup + log rotation) when storage is OK.
        "pressure_check_interval_min": 5,   # Janitor cycle while storage is SHED/HALT, so space freed by uploads or a new session is reclaimed and acquisition resumes promptly. A write-time threshold crossing also wakes the janitor immediately.
        "janitor_work_timeout_sec": 180,    # Hard ceiling for manage_disk_space/_force_archive_all_logs -- neither has its own timeout on the shutil/psutil calls inside, so an unresponsive drive could otherwise hang janitor_loop indefinitely.
    },
    "satellites": {
        "enabled": True,                    # Check every event against sunlit satellites (local catalogue, see satellites.py).
        "groups": ["visual", "active"],     # CelesTrak GP groups merged into the catalogue ("visual": ~150 brightest incl. rocket bodies; "active": all operational satellites).
        "url_template": "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=CSV",  # OMM CSV: TLE cannot carry the 6-digit catalogue numbers in use since 2026-07-11.
        "retry_margin_min": 30,             # Added to CelesTrak's 2-hour update interval: no group is ever requested again sooner than 2 h + this, success or failure (the safe window). Failures back off from there (x2 each time, capped at refresh_hours).
        "refresh_hours": 24,                # Re-download interval per group (>= 2 h: CelesTrak serves each group once per 2-hour update; a second download in that window is refused with HTTP 403).
        "max_catalog_age_days": 7,          # Older catalogues are not used: SGP4 error grows ~1-3 km/day for LEO (more for manoeuvring constellations).
        "request_timeout_sec": 30,          # HTTP timeout per group download.
        "position_tolerance_deg": 3.0,      # Match tolerance with the CONFIGURED pointing (camera_azimuth/altitude, lens model): pointing and lens-distortion error.
        "position_tolerance_solved_deg": 1.0, # Match tolerance when the event was plate-solved (pointing measured, not assumed).
        "time_tolerance_sec": 2.0,          # Uncertainty of a frame's timestamp relative to its exposure (clock + readout latency).
        "min_matching_fraction": 0.8,       # Share of the event's motion frames the satellite must explain.
        "match_timeout_sec": 60,            # Hard ceiling for one event's matching (bounded worker).
    },
    "dashboard": {
        "enabled": True, 
        "host": "0.0.0.0", 
        "port": 5000,
        "archive_nights_per_page": 14,     # Session archive page: nights shown per page.
        "archive_thumb_width": 480,        # Session archive: thumbnail width (px). Full-size PNGs open on click.
        "archive_thumb_jpeg_quality": 85,  # Session archive: thumbnail JPEG quality.
    },
    "heartbeat": {
        "request_timeout_sec": 15,             # HTTP timeout of the heartbeat ping.
        "enabled": True, 
        "url": None,                        # The remote endpoint for heartbeat pings.
        "interval_min": 60                  # Frequency of heartbeat pings in minutes.
     },
    "event_log": {
        "enabled": True
    },
    "health_monitor": {
        "enabled": True, 
    },
    "sftp": {
        "connect_timeout_sec": 30,             # SSH/SFTP connect timeout.
        "reconnect_initial_sec": 30,           # First reconnect delay after a failure (doubles each attempt).
        "reconnect_max_sec": 1800,             # Reconnect back-off ceiling.
        "enabled": False,
        "host": "sftp.myserver.net",        # Credentials and address for the SFTP server.
        "port": 22,
        "user": "myuser",
        "remote_dir": "/remote/path",       # The destination folder path on the SFTP server.
        "max_queue_size": 500,              # Limit on files waiting for upload before switching to a disk-based backlog.
    },
    "ntp": {
        "request_timeout_sec": 15,             # NTP query timeout.
        "enabled": True,
        "server": "pool.ntp.org",           # The address of the NTP server used for clock synchronization.
        "sync_interval_hours": 6,           # Hours between system clock drift checks.
        "max_offset_sec": 2.0,              # Maximum allowed time drift before forcing a system clock sync.
    },
    "ai_classification": {
        "enabled": False,
        "model_path": "models/ryusei_v1.tflite",   # Filesystem path to the TensorFlow Lite (.tflite) weights file.
        "label_path": "models/labels_ryusei.txt",  # Filesystem path to the text file containing the category names.
        "min_confidence": 0.85,                     # The minimum probability score required to accept an AI verdict.
        "input_size": 224,                          # The pixel dimensions (square) required for the AI model's input layer.
    },
    "ai_observatory": {
        "enabled": False,
        "model_path": "models/observatory_status.tflite",   # Filesystem path to the TensorFlow Lite (.tflite) weights file.
        "label_path": "models/labels_observatory.txt",      # Filesystem path to the text file containing the category names.
        "min_confidence": 0.80,                             # Minimum probability score (0.0-1.0) required to trust an AI prediction.
        "bad_conditions": ["rain", "cloud", "obstruction"], #  List of AI labels that will trigger a weather-hold and stop data acquisition.
    },
    "general": {
        "drop_report_interval_sec": 30,        # Dropped-frame reporting: at most one log line and one health event per this interval, with the count, instead of one per frame.
        "record_queue_maxsize": 1000,          # Bound of the small-record queues (event log rows, health events, SFTP paths): backlog tolerated before new records are dropped.
        "scheduler_interval_sec": 60,          # Scheduler evaluation period (schedule window, emergency state).
        "loop_time_ema_alpha": 0.05,           # Smoothing of the detection-loop time estimate (EMA, ~1/alpha frames). It feeds the effective frame period used by every derived wait.
        "hw_query_timeout_sec": 3,             # Timeout for local system/hardware query commands (vcgencmd, timedatectl).
        "thermal_read_interval_sec": 15,       # Period of the DS18B20 temperature reads (sensor conversion takes ~0.75 s at 12-bit).
        "thread_error_backoff_sec": 30,        # Pause after an unexpected error in a service loop (SFTP dispatcher) before retrying.
        "watchdog_startup_delay_sec": 10,      # Seconds the watchdog waits after launch before READY=1 and its first checks (threads starting up).
        "watchdog_check_interval_sec": 5,      # Watchdog supervision period. Must stay well below watchdog_hung_thread_sec (validated).
        "queue_status_ping_sec": 60,           # Period of the STATUS token pushed into blocked consumer queues so idle consumer threads still heartbeat.
        "debug_image_save_interval_sec": 2.0,  # Minimum spacing of detection debug-mosaic disk dumps during an event (debug_level >= 2).
        "png_compression": 3,                  # zlib level for every PNG written (0-9): CPU time vs file size; lossless at any level.
        "shutdown_join_slack_sec": 10.0,       # Slack added to the physics-derived producer-stop budget (in-flight frames x frame duration).
        "thread_shutdown_timeout_sec": 30,     # Per-thread join budget for the remaining worker/control threads at process stop.
        "camera_restart_settle_sec": 10,       # After a camera restart attempt, time before judging whether the producer survived.
        "maintenance_poll_sec": 5,             # Resolution of the maintenance-mode timeout watchdog.
        "debug_level": 0,                   # Verbosity of logic logs (0=Off, 4=Full variable dump).
        "debug_visualization": False,       # Toggles local image saves of the motion detection mask for tuning.
        "debug_align_dump_stride": 5,       # At debug_level >= 3, only write/publish the per-frame alignment mosaic every Nth frame of a stack (1 = every frame, the old behavior). Dumping all ~20 frames of a stack can saturate slow storage badly enough to make the stack -- and anything else writing to the same drive, like the dashboard's config save -- look hung.
        "log_dir": "logs",                  # Folder name where all CSVs and text logs are stored.
        "jpg_quality": 85,                  # Compression level for jpg image.
        "jpg_quality_zmq": 60,              # Compression level for the real-time debug stream.
        "hostname": socket.gethostname(),   # The identifier used for the device in logs and telemetry.
        "zombie_thread_timeout_sec": 60,
        "timelapse_video_thread_timeout_sec": 700,
        "finalize_video_thread_timeout_sec": 360,
        "exit_forensics_timeout_sec": 2.0,  # Upper bound on the all-thread stack dump written just before a watchdog-forced exit. A healthy dump of ~20 threads takes milliseconds; the bound only matters when the log drive or journald is itself the thing that hung, and caps how long that can delay the restart.
        "watchdog_heartbeat_fraction": 0.25, # Heartbeat-aware waits (wait_with_heartbeat) wake at least every watchdog_hung_thread_sec * this fraction to ping. 0.25 = four pings per hang window, so one delayed wake-up can never trip the watchdog.
        "watchdog_hung_thread_sec": 480,    # Seconds without a heartbeat before the watchdog declares a worker thread hung and forces a restart. Must exceed the longest legitimate uninterrupted operation (e.g. a multi-attempt auto-exposure hunt with slow camera sync).
        "location": None,                   # Human-readable name of the observatory site.
        "latitude": 0.0,                    # Exact GPS coordinates used for moon and star position math.
        "longitude": 0.0,                   # Exact East/West coordinate used for lunar position calculations.
        "camera_azimuth": 0.0,              # Horizontal compass heading of the lens center (0=North, 90=East, 180=South, 270=West).
        "camera_altitude": 90.0,            # Vertical angle of the lens center (90=Zenith, 0=Horizon).
        "max_camera_failures": 20,          # Sequential sensor timeouts allowed before flagging a hardware error.
        "max_restart_failures": 3,          # Number of software restart attempts before forcing a full system reboot.
        "idle_heartbeat_interval_min": 5,   # Heartbeat frequency when the system is outside of the active schedule.
        "power_monitor_pin": 17,            # Redundant GPIO pin mapping for power supply checks.
        "lens_hfov": 102,                   # Horizontal Field of View in degrees
        "lens_vfov": 67,                    # Vertical Field of View in degrees
        "lens_projection": "rectilinear",   # Lens mapping: "rectilinear" (r = f tan(theta), normal lenses) or "equidistant" (r = f theta, fisheye). Check: for a rectilinear lens tan(hfov/2)/tan(vfov/2) = width/height.
        "sensor_camera_id": None,           # The unique 1-Wire serial numbers for your DS18B20 temperature probes.
        "sensor_ext_id": None,              # The unique 1-Wire serial numbers for your DS18B20 temperature probes.
        "maintenance_timeout": 300,         # Seconds to wait in Maintenance Mode before automatically resuming.
        "ota_update_url": "",               # Remote URL used to pull the latest version of this script.
        "post_shutdown_script": None,       # Post-Shutdown Script
        "start_time": "15:00",              # The daily window (HH:MM) during which meteor detection is active.
        "end_time": "06:00",                # The time (HH:MM) when the active nightly session concludes.
        "shutdown_time": None,              # Specific time (HH:MM) to perform a full system halt.
    },
}

# --- JANITOR CONFIGURATION ---
# The order in which directories are cleaned (Top = First to be deleted)
JANITOR_CLEANUP_PRIORITY = [
    "calibration_out_dir",
    "debug_out_dir",
    "rejected_out_dir",
    "ai_filtered_out_dir",
    "daylight_out_dir",
    "timelapse_out_dir",
    "events_out_dir"
]

# Directories (attribute names) whose NEW writes the storage governor suspends at
# the SHED level (disk above janitor.threshold): diagnostic data, least valuable
# per unit of space. Timelapse and event data keep flowing until HALT.
STORAGE_SHED_DIRS = [
    "debug_out_dir",
    "rejected_out_dir",
    "daylight_out_dir",
]

# events.csv columns, in order: ONE definition for the writer, the janitor's
# rotation header and every dashboard reader. New columns go at the END only, so
# older files stay readable (missing trailing fields read as empty).
EVENT_LOG_FIELDS = [
    "filename", "timestamp_local", "start_time_local", "duration_sec",
    "num_frames_video", "num_frames_stack", "event_score", "ai_label",
    "status", "ra_deg", "dec_deg", "satellite",
]

JANITOR_PROTECTED_FILES = {             # list of file that must not be deleted
}

# Directories (attribute names, same convention as JANITOR_CLEANUP_PRIORITY) where the
# janitor must never delete anything belonging to the current session, i.e. any file
# modified at or after the most recent general.start_time. Everything older is still
# fair game (oldest first, and only if already uploaded when SFTP is enabled).
JANITOR_SESSION_PROTECTED_DIRS = [
    "ai_filtered_out_dir",
    "daylight_out_dir",
    "timelapse_out_dir",
    "events_out_dir",
]

# Values tuned for optimal balance of sensitivity and RAM safety
RESOLUTION_TUNING_PROFILES = {
    # format: TotalPixels (MP) : {settings}
    0.5: {
        "description": "VGA",
        "min_area": 20,
        "blur": 3,
        "morph": 3,
        "min_changes": 10,
        "max_q": 1000,
        "max_event_frames": 50,
        "max_event_q": 10,
        "radius": 4,
        "std_floor": 2.3,
        "min_mask_px": 30,
        "ecc_gauss": 3,
        "warp_f": 0.06,
        "th_f": 3.0,
        "pre_f": 8,
        "cool_f": 8,
    },
    1.0: {
        "description": "720",
        "min_area": 35,
        "blur": 3,
        "morph": 3,
        "min_changes": 22,
        "max_q": 1000,
        "max_event_frames": 50,
        "max_event_q": 5,
        "radius": 5,
        "std_floor": 1.8,
        "min_mask_px": 50,
        "ecc_gauss": 3,
        "warp_f": 0.05,
        "th_f": 3.0,
        "pre_f": 8,
        "cool_f": 8,
    },
    2.0: {
        "description": "1080",
        "min_area": 100,
        "blur": 3,
        "morph": 5,
        "min_changes": 50,
        "max_q": 350,
        "max_event_frames": 50,
        "max_event_q": 3,
        "radius": 10,
        "std_floor": 0.9,
        "min_mask_px": 50,
        "ecc_gauss": 3,
        "warp_f": 0.04,
        "th_f": 2.5,
        "pre_f": 8,
        "cool_f": 8,
    },
    3.0: {
        "description": "2K",
        "min_area": 150,
        "blur": 3,
        "morph": 7,
        "min_changes": 80,
        "max_q": 250,
        "max_event_frames": 50,
        "max_event_q": 2,
        "radius": 12,
        "std_floor": 0.6,
        "min_mask_px": 150,
        "ecc_gauss": 5,
        "warp_f": 0.03,
        "th_f": 2.0,
        "pre_f": 6,
        "cool_f": 6,
    },
    12.3: {
        "description": "4K",
        "min_area": 350,
        "blur": 5,
        "morph": 9,
        "min_changes": 300,
        "max_q": 60,
        "max_event_frames": 20,
        "max_event_q": 1,
        "radius": 20,
        "std_floor": 0.3,
        "min_mask_px": 300,
        "ecc_gauss": 7,
        "warp_f": 0.02,
        "th_f": 1.5,
        "pre_f": 3,
        "cool_f": 3,
    },
}

# Values tuned for sky_score logic
# | Score     | Stato              |
# | --------- | ------------------ |
# | ≥ 1.3     | Excellent          |
# | 1.0 – 1.3 | Good               |
# | 0.8 – 1.0 | Fair               |
# | < 0.8     | Cloudy             |
SKY_SCORE_VALUES = {"cloudy": 0.1, "fair": 0.8, "good": 1.0, "excellent": 1.3}

# --------------------------- Configuration Schema (Pydantic) ---------------------------
class CaptureConfig(BaseModel):
    processing_headroom: confloat(ge=1.0, le=4.0) = DEFAULTS["capture"]["processing_headroom"]
    frame_rate_hysteresis: confloat(gt=0.0, le=0.5) = DEFAULTS["capture"]["frame_rate_hysteresis"]
    raw_buffer_count: conint(ge=1, le=8) = DEFAULTS["capture"]["raw_buffer_count"]
    ae_recheck_max_frame_age_frames: conint(ge=1, le=20) = DEFAULTS["capture"]["ae_recheck_max_frame_age_frames"]
    ae_saturation_backoff: confloat(gt=0.0, lt=1.0) = DEFAULTS["capture"]["ae_saturation_backoff"]
    safe_window_slack_sec: conint(ge=1, le=3600) = DEFAULTS["capture"]["safe_window_slack_sec"]
    width: conint(gt=0, le=8000) = DEFAULTS["capture"]["width"]
    height: conint(gt=0, le=8000) = DEFAULTS["capture"]["height"]
    exposure_us: conint(gt=0, le=10000000) = DEFAULTS["capture"]["exposure_us"]
    gain: confloat(ge=0, le=6) = DEFAULTS["capture"]["gain"]
    red_gain: confloat(gt=0, le=16) = DEFAULTS["capture"]["red_gain"]
    blue_gain: confloat(gt=0, le=16) = DEFAULTS["capture"]["blue_gain"]
    auto_exposure_tuning: bool = DEFAULTS["capture"]["auto_exposure_tuning"]
    # gt=0: the target is a divisor (AE correction ratio, change-limit re-centring); 0 is meaningless.
    target_sky_brightness: conint(gt=0, le=255) = DEFAULTS["capture"]["target_sky_brightness"]
    min_exposure_us: conint(gt=0) = DEFAULTS["capture"]["min_exposure_us"]
    max_exposure_us: conint(gt=0) = DEFAULTS["capture"]["max_exposure_us"]
    min_gain: confloat(ge=1.0) = DEFAULTS["capture"]["min_gain"]
    max_gain: confloat(ge=1.0, le=6.0) = DEFAULTS["capture"]["max_gain"]
    tolerance: conint(gt=0, le=10) = DEFAULTS["capture"]["tolerance"]
    max_attempts: conint(gt=0, le=10) = DEFAULTS["capture"]["max_attempts"]
    exposure_sync_retries: conint(ge=1, le=10) = DEFAULTS["capture"]["exposure_sync_retries"]
    ae_max_stops_per_check: confloat(gt=0.0, le=8.0) = DEFAULTS["capture"]["ae_max_stops_per_check"]
    ae_twilight_stops_per_deg: confloat(ge=0.0, le=5.0) = DEFAULTS["capture"]["ae_twilight_stops_per_deg"]
    ae_twilight_sun_alt_min: confloat(ge=-90.0, le=0.0) = DEFAULTS["capture"]["ae_twilight_sun_alt_min"]
    ae_twilight_sun_alt_max: confloat(ge=-30.0, le=30.0) = DEFAULTS["capture"]["ae_twilight_sun_alt_max"]
    ae_moon_stops_per_deg: confloat(ge=0.0, le=1.0) = DEFAULTS["capture"]["ae_moon_stops_per_deg"]
    ae_moon_alt_min: confloat(ge=-30.0, le=30.0) = DEFAULTS["capture"]["ae_moon_alt_min"]
    ae_sample_step_min: conint(ge=1, le=60) = DEFAULTS["capture"]["ae_sample_step_min"]
    ae_clamp_recheck_min: conint(ge=1, le=60) = DEFAULTS["capture"]["ae_clamp_recheck_min"]
    frame_duration_headroom_us: conint(ge=0, le=1000000) = DEFAULTS["capture"]["frame_duration_headroom_us"]
    sync_control_delay_frames: conint(ge=0, le=10) = DEFAULTS["capture"]["sync_control_delay_frames"]
    sync_timeout_slack_sec: confloat(gt=0.0, le=60.0) = DEFAULTS["capture"]["sync_timeout_slack_sec"]
    sync_exposure_rel_tol: confloat(gt=0.0, le=0.25) = DEFAULTS["capture"]["sync_exposure_rel_tol"]
    sync_gain_rel_tol: confloat(gt=0.0, le=0.25) = DEFAULTS["capture"]["sync_gain_rel_tol"]
    max_brightness: conint(gt=0, le=240) = DEFAULTS["capture"]["max_brightness"]
    moon_normal_penalty: conint(ge=0, le=100) = DEFAULTS["capture"]["moon_normal_penalty"]
    moon_max_penalty: conint(ge=0, le=100) = DEFAULTS["capture"]["moon_max_penalty"]
    ae_step_limit_up: confloat(gt=1.0, le=10.0) = DEFAULTS["capture"]["ae_step_limit_up"]
    ae_step_limit_down: confloat(gt=0.0, lt=1.0) = DEFAULTS["capture"]["ae_step_limit_down"]
    masterpiece_burn_multiplier: confloat(ge=1.0, le=10.0) = DEFAULTS["capture"]["masterpiece_burn_multiplier"]
    awb_min_green_mean: confloat(ge=0, le=5) = DEFAULTS["capture"]["awb_min_green_mean"]
    awb_damping_factor: confloat(ge=0, le=1) = DEFAULTS["capture"]["awb_damping_factor"]
    awb_min_ratio_deviation: confloat(ge=0, le=0.1) = DEFAULTS["capture"]["awb_min_ratio_deviation"]

    @model_validator(mode="after")
    def validate_logical_bounds(self):
        """
        Enforces logical consistency between min/max and target limits.
        """
        # 1. Exposure Range
        if self.min_exposure_us >= self.max_exposure_us:
            raise ValueError(
                f"Configuration Error: 'min_exposure_us' ({self.min_exposure_us}) "
                f"must be strictly less than 'max_exposure_us' ({self.max_exposure_us})."
            )
        # 2. Gain Range
        if self.min_gain >= self.max_gain:
            raise ValueError(
                f"Configuration Error: 'min_gain' ({self.min_gain}) "
                f"must be strictly less than 'max_gain' ({self.max_gain})."
            )
        # 3. Brightness Targets (Target vs Saturation Limit)
        if self.target_sky_brightness >= self.max_brightness:
            raise ValueError(
                f"Configuration Error: 'target_sky_brightness' ({self.target_sky_brightness}) "
                f"must be less than the saturation limit 'max_brightness' ({self.max_brightness})."
            )
        # 4. Twilight band of the auto-exposure change limit. Inverted or empty, the
        # band-overlap integral in _ae_expected_change_stops is identically zero: the
        # twilight allowance silently vanishes, and every dusk/dawn check gets clamped
        # to the plain-night base limit while the sky brightens/darkens ~1.3 stops per
        # degree of Sun altitude -- badly exposed twilight timelapses, no error anywhere.
        if self.ae_twilight_sun_alt_min >= self.ae_twilight_sun_alt_max:
            raise ValueError(
                f"Configuration Error: 'ae_twilight_sun_alt_min' ({self.ae_twilight_sun_alt_min}) "
                f"must be strictly less than 'ae_twilight_sun_alt_max' ({self.ae_twilight_sun_alt_max})."
            )
        return self

class DetectionConfig(BaseModel):
    @model_validator(mode="after")
    def validate_event_score_weights(self):
        total = (self.event_score_w_ai + self.event_score_w_aspect
                 + self.event_score_w_solidity + self.event_score_w_brightness)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Configuration Error: detection.event_score_w_* must sum to 1 (got {total:.4f}); "
                f"the event score is a convex combination in [0, 1]."
            )
        return self

    frame_continuity_max_gap_frames: confloat(gt=1.0, le=100.0) = DEFAULTS["detection"]["frame_continuity_max_gap_frames"]  # deprecated, ignored
    calibration_library_frames: conint(ge=3, le=500) = DEFAULTS["detection"]["calibration_library_frames"]
    auto_flat_refresh_every_samples: conint(ge=1) = DEFAULTS["detection"]["auto_flat_refresh_every_samples"]
    auto_flat_fit_grid_min_cells: conint(ge=2) = DEFAULTS["detection"]["auto_flat_fit_grid_min_cells"]
    reject_save_interval_sec: confloat(gt=0.0) = DEFAULTS["detection"]["reject_save_interval_sec"]
    background_warmup_noise_excess: confloat(gt=0.0, le=1.0) = DEFAULTS["detection"]["background_warmup_noise_excess"]
    light_shift_contours_per_mpix: confloat(gt=0.0) = DEFAULTS["detection"]["light_shift_contours_per_mpix"]
    threshold_max_step_fraction: confloat(gt=0.0, le=1.0) = DEFAULTS["detection"]["threshold_max_step_fraction"]
    threshold_min_step: conint(ge=1, le=255) = DEFAULTS["detection"]["threshold_min_step"]
    dynamic_dark_refresh_sec: conint(ge=1, le=86400) = DEFAULTS["detection"]["dynamic_dark_refresh_sec"]
    calibration_max_age_days: conint(ge=1) = DEFAULTS["detection"]["calibration_max_age_days"]
    dark_reference_exposure_us: conint(gt=0) = DEFAULTS["detection"]["dark_reference_exposure_us"]
    event_score_w_ai: confloat(ge=0.0, le=1.0) = DEFAULTS["detection"]["event_score_w_ai"]
    event_score_w_aspect: confloat(ge=0.0, le=1.0) = DEFAULTS["detection"]["event_score_w_aspect"]
    event_score_w_solidity: confloat(ge=0.0, le=1.0) = DEFAULTS["detection"]["event_score_w_solidity"]
    event_score_w_brightness: confloat(ge=0.0, le=1.0) = DEFAULTS["detection"]["event_score_w_brightness"]
    event_score_fallback_w_ai: confloat(ge=0.0, le=1.0) = DEFAULTS["detection"]["event_score_fallback_w_ai"]
    event_score_aspect_norm: confloat(gt=1.0) = DEFAULTS["detection"]["event_score_aspect_norm"]
    event_score_trail_percentile: confloat(gt=50.0, lt=100.0) = DEFAULTS["detection"]["event_score_trail_percentile"]
    dark_frame_path: Optional[str] = DEFAULTS["detection"]["dark_frame_path"]
    bias_frame_path: Optional[str] = DEFAULTS["detection"]["bias_frame_path"]
    flat_frame_path: Optional[str] = DEFAULTS["detection"]["flat_frame_path"]
    auto_sensitivity_tuning: bool = DEFAULTS["detection"]["auto_sensitivity_tuning"]
    base_threshold: conint(ge=0, le=255) = DEFAULTS["detection"]["base_threshold"]
    noise_factor: confloat(ge=0.0) = DEFAULTS["detection"]["noise_factor"]
    noise_smoothing_alpha: confloat(ge=0.0) = DEFAULTS["detection"]["noise_smoothing_alpha"]
    static_threshold: conint(ge=0, le=255) = DEFAULTS["detection"]["static_threshold"]
    min_threshold: conint(ge=0, le=255) = DEFAULTS["detection"]["min_threshold"]
    max_threshold: conint(ge=0, le=255) = DEFAULTS["detection"]["max_threshold"]
    target_adaptation_time: confloat(gt=0.0) = DEFAULTS["detection"]["target_adaptation_time"]
    min_accumulate_alpha: confloat(gt=0.0, lt=1.0) = DEFAULTS["detection"]["min_accumulate_alpha"]
    max_accumulate_alpha: confloat(gt=0.0, le=1.0) = DEFAULTS["detection"]["max_accumulate_alpha"]
    auto_dark_enabled: bool = DEFAULTS["detection"]["auto_dark_enabled"]
    auto_flat_enabled: bool = DEFAULTS["detection"]["auto_flat_enabled"]
#    auto_flat_interval_min: conint(ge=1, le=1440) = DEFAULTS["detection"]["auto_flat_interval_min"]
    auto_flat_buffer_size: conint(ge=5, le=100) = DEFAULTS["detection"]["auto_flat_buffer_size"]
    auto_flat_min_samples: conint(ge=3, le=100) = DEFAULTS["detection"]["auto_flat_min_samples"]
#    auto_flat_blur_sigma: conint(ge=1, le=201) = DEFAULTS["detection"]["auto_flat_blur_sigma"]
    bpm_bg_blur_kernel: conint(ge=3, le=201) = DEFAULTS["detection"]["bpm_bg_blur_kernel"]
    bpm_sigma_multiplier: confloat(gt=0.0) = DEFAULTS["detection"]["bpm_sigma_multiplier"]
    bpm_sigma_floor: confloat(ge=0.0) = DEFAULTS["detection"]["bpm_sigma_floor"]
    bpm_max_blob_px: conint(ge=1) = DEFAULTS["detection"]["bpm_max_blob_px"]
    bpm_kernel_min_px: conint(ge=3) = DEFAULTS["detection"]["bpm_kernel_min_px"]
    bpm_kernel_max_px: conint(ge=3) = DEFAULTS["detection"]["bpm_kernel_max_px"]
    auto_flat_clip_sigma: confloat(ge=1.0, le=5.0) = DEFAULTS["detection"]["auto_flat_clip_sigma"]
    auto_flat_fit_grid_fraction: confloat(ge=0.01, le=0.5) = DEFAULTS["detection"]["auto_flat_fit_grid_fraction"]
    auto_flat_poly_degree: conint(ge=1, le=6) = DEFAULTS["detection"]["auto_flat_poly_degree"]
    auto_flat_fit_oversample_factor: confloat(ge=1.0) = DEFAULTS["detection"]["auto_flat_fit_oversample_factor"]
    auto_flat_mult_min: confloat(ge=0.1, lt=1.0) = DEFAULTS["detection"]["auto_flat_mult_min"]
    auto_flat_mult_max: confloat(gt=1.0, le=5.0) = DEFAULTS["detection"]["auto_flat_mult_max"]

    @field_validator("bpm_bg_blur_kernel")
    @classmethod
    def bpm_bg_blur_kernel_must_be_odd(cls, v):
        if v % 2 == 0:
            raise ValueError(
                f"Configuration Error: 'bpm_bg_blur_kernel' ({v}) must be odd (cv2.blur/medianBlur kernels require odd dimensions)."
            )
        return v

    @model_validator(mode="after")
    def validate_bpm_kernel_bounds(self):
        if self.bpm_kernel_min_px > self.bpm_kernel_max_px:
            raise ValueError(
                f"Configuration Error: 'bpm_kernel_min_px' ({self.bpm_kernel_min_px}) "
                f"cannot exceed 'bpm_kernel_max_px' ({self.bpm_kernel_max_px})."
            )
        return self

    @field_validator(
        "dark_frame_path", "bias_frame_path", "flat_frame_path", mode="before"
    )
    @classmethod
    def allow_empty_str_for_optional_path(cls, v):
        # If the input is an empty string, convert it to None before other validation.
        if v == "" or str(v).lower() == "none":
            return None
        return v

    @model_validator(mode="after")
    def validate_alpha_logic(self):
        """
        Enforces logical consistency for the background adaptation math.
        """
        if self.min_accumulate_alpha >= self.max_accumulate_alpha:
            raise ValueError(
                f"Configuration Error: 'min_accumulate_alpha' ({self.min_accumulate_alpha}) "
                f"must be strictly less than 'max_accumulate_alpha' ({self.max_accumulate_alpha})."
            )
        return self

    @model_validator(mode="after")
    def validate_flat_logic(self):
        """Ensures the flat engine can actually trigger."""
        if self.auto_flat_min_samples > self.auto_flat_buffer_size:
            raise ValueError(
                f"Configuration Error: 'auto_flat_min_samples' ({self.auto_flat_min_samples}) "
                f"cannot be greater than 'auto_flat_buffer_size' ({self.auto_flat_buffer_size})."
            )

        robust_limit = (self.auto_flat_buffer_size / 2.0) + 1.0
        if self.auto_flat_min_samples >= robust_limit:
            raise ValueError(
                f"Configuration Error: To maintain statistical robustness, 'auto_flat_min_samples' "
                f"({self.auto_flat_min_samples}) must be strictly less than (auto_flat_buffer_size / 2) + 1 "
                f"({robust_limit:.1f}). Please increase your buffer size or lower your minimum samples."
            )
        return self

class SolveReadinessConfig(BaseModel):
    max_lux: confloat(gt=0.0) = DEFAULTS["astrometry"]["solve_readiness"]["max_lux"]
    min_norm_stddev: confloat(gt=0.0) = DEFAULTS["astrometry"]["solve_readiness"]["min_norm_stddev"]
    min_stars_in_roi: conint(ge=3) = DEFAULTS["astrometry"]["solve_readiness"]["min_stars_in_roi"]
    moon_impact_notable: confloat(ge=0.0, le=1.0) = DEFAULTS["astrometry"]["solve_readiness"]["moon_impact_notable"]

class AstrometryConfig(BaseModel):
    enabled: bool = DEFAULTS["astrometry"]["enabled"]
    tetra3_database: str = DEFAULTS["astrometry"]["tetra3_database"]
    target_half_fov_deg: confloat(gt=0.0) = DEFAULTS["astrometry"]["target_half_fov_deg"]
    cross_check_half_fov_deg: confloat(gt=0.0) = DEFAULTS["astrometry"]["cross_check_half_fov_deg"]
    min_stars_for_solve: conint(ge=3) = DEFAULTS["astrometry"]["min_stars_for_solve"]
    match_radius_deg: confloat(gt=0.0) = DEFAULTS["astrometry"]["match_radius_deg"]
    pattern_checking_stars: conint(ge=3) = DEFAULTS["astrometry"]["pattern_checking_stars"]
    plate_scale_tolerance_pct: confloat(gt=0.0) = DEFAULTS["astrometry"]["plate_scale_tolerance_pct"]
    event_solve_timeout_sec: confloat(gt=0.0) = DEFAULTS["astrometry"]["event_solve_timeout_sec"]
    solve_readiness: SolveReadinessConfig = Field(default_factory=SolveReadinessConfig)

    @model_validator(mode="after")
    def validate_crop_ordering(self):
        """Cross-check crop must be strictly larger than the solve crop, or the
        linearity check is comparing a window against itself."""
        if self.cross_check_half_fov_deg <= self.target_half_fov_deg:
            raise ValueError(
                f"Configuration Error in [astrometry]: cross_check_half_fov_deg "
                f"({self.cross_check_half_fov_deg}) must exceed target_half_fov_deg "
                f"({self.target_half_fov_deg}) — otherwise the linearity check is meaningless."
            )
        return self

    @model_validator(mode="after")
    def validate_readiness_logic(self):
        """Ensure the advisory star count is logically higher or equal to the solver floor."""
        if self.solve_readiness.min_stars_in_roi < self.min_stars_for_solve:
            raise ValueError(
                f"Configuration Error in [astrometry]: solve_readiness.min_stars_in_roi "
                f"({self.solve_readiness.min_stars_in_roi}) cannot be lower than "
                f"min_stars_for_solve ({self.min_stars_for_solve}). Advisory would be misleading."
            )
        return self

class TimelapseConfig(BaseModel):
    @field_validator("star_tophat_min_kernel_px")
    def validate_tophat_kernel_odd(cls, v):
        if v % 2 == 0:
            raise ValueError(f"timelapse.star_tophat_min_kernel_px ({v}) must be odd (morphological kernels need a centre pixel).")
        return v

    nighttime_stack_N: conint(gt=0, le=100) = DEFAULTS["timelapse"]["nighttime_stack_N"
    ]
    star_core_diameter_fraction: confloat(gt=0.0, le=2.0) = DEFAULTS["timelapse"]["star_core_diameter_fraction"]
    star_max_extent_factor: confloat(gt=0.0) = DEFAULTS["timelapse"]["star_max_extent_factor"]
    star_tophat_min_kernel_px: conint(ge=3) = DEFAULTS["timelapse"]["star_tophat_min_kernel_px"]
    stack_bias_ref_weight: confloat(gt=0.0, le=1.0) = DEFAULTS["timelapse"]["stack_bias_ref_weight"]
    stack_bias_frame_weight: confloat(gt=0.0, le=1.0) = DEFAULTS["timelapse"]["stack_bias_frame_weight"]
    ecc_max_iterations: conint(ge=1, le=1000) = DEFAULTS["timelapse"]["ecc_max_iterations"]
    ecc_epsilon: confloat(gt=0.0, lt=1.0) = DEFAULTS["timelapse"]["ecc_epsilon"]
    alignment_min_pixels_per_feature: conint(ge=1) = DEFAULTS["timelapse"]["alignment_min_pixels_per_feature"]
    emergency_throttle_step_sec: confloat(gt=0.0) = DEFAULTS["timelapse"]["emergency_throttle_step_sec"]
    emergency_throttle_recovery_sec: confloat(gt=0.0) = DEFAULTS["timelapse"]["emergency_throttle_recovery_sec"]
    emergency_throttle_recovery_queue_fraction: confloat(gt=0.0, lt=1.0) = DEFAULTS["timelapse"]["emergency_throttle_recovery_queue_fraction"]
    stack_align: bool = DEFAULTS["timelapse"]["stack_align"]
    nightmode_interval_sec: int = DEFAULTS["timelapse"]["nightmode_interval_sec"]
    daymode_interval_sec: int = DEFAULTS["timelapse"]["daymode_interval_sec"]
    stack_method: str = DEFAULTS["timelapse"]["stack_method"]
    min_features_for_alignment: conint(ge=1, le=10) = DEFAULTS["timelapse"]["min_features_for_alignment"]
    roi_percent: confloat(ge=0.4, le=0.8) = DEFAULTS["timelapse"]["roi_percent"]
    cloudy_interval_sec: conint(ge=0) = DEFAULTS["timelapse"]["cloudy_interval_sec"]
    alignment_threshold_floor: conint(ge=1, le=50) = DEFAULTS["timelapse"]["alignment_threshold_floor"]
    star_sigma_multiplier: confloat(gt=0.0) = DEFAULTS["timelapse"]["star_sigma_multiplier"]
    stack_brightness_boost: confloat(ge=1.0, le=5.0) = DEFAULTS["timelapse"]["stack_brightness_boost"]
    stack_sigma_clip_enabled: bool = DEFAULTS["timelapse"]["stack_sigma_clip_enabled"]
    stack_sigma_clip_k: confloat(gt=0.0) = DEFAULTS["timelapse"]["stack_sigma_clip_k"]
    stack_sigma_clip_min_frames: conint(ge=3) = DEFAULTS["timelapse"]["stack_sigma_clip_min_frames"]
    stack_sigma_clip_floor: confloat(ge=0.0) = DEFAULTS["timelapse"]["stack_sigma_clip_floor"]
    stack_sigma_clip_ram_fraction: confloat(gt=0.0, le=1.0) = DEFAULTS["timelapse"]["stack_sigma_clip_ram_fraction"]
    stack_sigma_clip_peak_multiplier: confloat(ge=1.0, le=10.0) = DEFAULTS["timelapse"]["stack_sigma_clip_peak_multiplier"]  # deprecated, ignored
    stack_event_wait_slack_sec: confloat(gt=0.0, le=120.0) = DEFAULTS["timelapse"]["stack_event_wait_slack_sec"]
    cc_floor_normal: confloat(ge=0.0, le=1.0) = DEFAULTS["timelapse"]["cc_floor_normal"]
    cc_floor_bias: confloat(ge=0.0, le=1.0) = DEFAULTS["timelapse"]["cc_floor_bias"]
    cc_perfect_normal: confloat(ge=0.0, le=1.0) = DEFAULTS["timelapse"]["cc_perfect_normal"]
    cc_perfect_bias: confloat(ge=0.0, le=1.0) = DEFAULTS["timelapse"]["cc_perfect_bias"]
    weight_floor_normal: confloat(ge=0.0, le=1.0) = DEFAULTS["timelapse"]["weight_floor_normal"]
    weight_floor_bias: confloat(ge=0.0, le=1.0) = DEFAULTS["timelapse"]["weight_floor_bias"]

    @model_validator(mode="after")
    def validate_correlation_logic(self):
        """Enforces that 'perfect' correlation is strictly above the rejection floor for both modes."""
        if self.cc_floor_normal >= self.cc_perfect_normal:
            raise ValueError(
                f"Configuration Error in [timelapse]: cc_floor_normal ({self.cc_floor_normal}) "
                f"must be strictly less than cc_perfect_normal ({self.cc_perfect_normal})."
            )
        if self.cc_floor_bias >= self.cc_perfect_bias:
            raise ValueError(
                f"Configuration Error in [timelapse]: cc_floor_bias ({self.cc_floor_bias}) "
                f"must be strictly less than cc_perfect_bias ({self.cc_perfect_bias})."
            )
        return self

class TimelapseVideoConfig(BaseModel):
    brightness_outlier_tolerance_factor: confloat(ge=1.0) = DEFAULTS["timelapse_video"]["brightness_outlier_tolerance_factor"]
    encode_timeout_sec: conint(ge=1) = DEFAULTS["timelapse_video"]["encode_timeout_sec"]
    enabled: bool = DEFAULTS["timelapse_video"]["enabled"]
    video_fps: conint(gt=0, le=30) = DEFAULTS["timelapse_video"]["video_fps"]
    ffmpeg_encoder: str = DEFAULTS["timelapse_video"]["ffmpeg_encoder"]
    ffmpeg_crf: conint(ge=0, le=51) = DEFAULTS["timelapse_video"]["ffmpeg_crf"]
    ffmpeg_pix_fmt: str = DEFAULTS["timelapse_video"]["ffmpeg_pix_fmt"]

class EventsConfig(BaseModel):
    writer_wait_sec: conint(ge=1) = DEFAULTS["events"]["writer_wait_sec"]
    video_fps: conint(gt=0, le=30) = DEFAULTS["events"]["video_fps"]
    ffmpeg_encoder: str = DEFAULTS["events"]["ffmpeg_encoder"]
    ffmpeg_crf: conint(ge=0, le=51) = DEFAULTS["events"]["ffmpeg_crf"]
    ffmpeg_pix_fmt: str = DEFAULTS["events"]["ffmpeg_pix_fmt"]

class MonitorConfig(BaseModel):
    interval_sec: conint(gt=0, le=60) = DEFAULTS["monitor"]["interval_sec"]
    ram_limit_percent: confloat(ge=50.0, le=95.0) = DEFAULTS["monitor"]["ram_limit_percent"]
    ram_recovery_percent: confloat(ge=40.0, le=90.0) = DEFAULTS["monitor"]["ram_recovery_percent"]

class PowerMonitorConfig(BaseModel):
    check_interval_sec: conint(ge=1, le=60) = DEFAULTS["power_monitor"]["check_interval_sec"]
    enabled: bool = DEFAULTS["power_monitor"]["enabled"]
    pin: int = DEFAULTS["power_monitor"]["pin"]
    shutdown_delay_sec: conint(ge=0) = DEFAULTS["power_monitor"]["shutdown_delay_sec"]

class LedStatusConfig(BaseModel):
    enabled: bool = DEFAULTS["led_status"]["enabled"]
    pin: int = DEFAULTS["led_status"]["pin"]

class DaylightConfig(BaseModel):
    @model_validator(mode="after")
    def validate_day_frame_duration(self):
        if self.day_frame_duration_min_us >= self.day_frame_duration_max_us:
            raise ValueError(
                f"Configuration Error: daylight.day_frame_duration_min_us ({self.day_frame_duration_min_us}) "
                f"must be less than day_frame_duration_max_us ({self.day_frame_duration_max_us})."
            )
        return self

    default_calibration_k: confloat(gt=0.0) = DEFAULTS["daylight"]["default_calibration_k"]
    min_valid_calibration_k: confloat(gt=0.0) = DEFAULTS["daylight"]["min_valid_calibration_k"]
    snap_exposure_us: conint(gt=0) = DEFAULTS["daylight"]["snap_exposure_us"]
    snap_gain: confloat(ge=1.0) = DEFAULTS["daylight"]["snap_gain"]
    day_frame_duration_min_us: conint(gt=0) = DEFAULTS["daylight"]["day_frame_duration_min_us"]
    day_frame_duration_max_us: conint(gt=0) = DEFAULTS["daylight"]["day_frame_duration_max_us"]
    dawn_trigger_moon_floor: confloat(gt=0.0, le=255.0) = DEFAULTS["daylight"]["dawn_trigger_moon_floor"]
    dawn_trigger_moon_multiplier: confloat(ge=1.0) = DEFAULTS["daylight"]["dawn_trigger_moon_multiplier"]
    moon_std_relax_factor: confloat(ge=1.0) = DEFAULTS["daylight"]["moon_std_relax_factor"]
    moon_star_relax_factor: confloat(gt=0.0, le=1.0) = DEFAULTS["daylight"]["moon_star_relax_factor"]
    moon_min_stars: conint(ge=1) = DEFAULTS["daylight"]["moon_min_stars"]
    sky_saturation_brightness: confloat(gt=0.0, le=255.0) = DEFAULTS["daylight"]["sky_saturation_brightness"]
    sky_min_mean_for_texture: confloat(gt=0.0) = DEFAULTS["daylight"]["sky_min_mean_for_texture"]
    sky_texture_limit: confloat(gt=0.0) = DEFAULTS["daylight"]["sky_texture_limit"]
    sky_min_noise_score: confloat(ge=0.0) = DEFAULTS["daylight"]["sky_min_noise_score"]
    sky_min_star_score: confloat(ge=0.0) = DEFAULTS["daylight"]["sky_min_star_score"]
    sky_weight_noise: confloat(ge=0.0, le=1.0) = DEFAULTS["daylight"]["sky_weight_noise"]
    moon_penalty_alt_norm_deg: confloat(gt=0.0, le=90.0) = DEFAULTS["daylight"]["moon_penalty_alt_norm_deg"]
    moon_penalty_alt_floor: confloat(ge=0.0, le=1.0) = DEFAULTS["daylight"]["moon_penalty_alt_floor"]
    moon_penalty_exponent: confloat(gt=0.0) = DEFAULTS["daylight"]["moon_penalty_exponent"]
    moon_min_alt_deg: confloat(ge=-90.0, le=90.0) = DEFAULTS["daylight"]["moon_min_alt_deg"]
    moon_impact_exponent: confloat(gt=0.0) = DEFAULTS["daylight"]["moon_impact_exponent"]
    moon_fov_glare_margin_deg: confloat(ge=0.0, le=45.0) = DEFAULTS["daylight"]["moon_fov_glare_margin_deg"]
    moon_in_fov_impact_boost: confloat(ge=0.0, le=1.0) = DEFAULTS["daylight"]["moon_in_fov_impact_boost"]
    enabled: bool = DEFAULTS["daylight"]["enabled"]
    stddev_threshold: float = DEFAULTS["daylight"]["stddev_threshold"]
    min_stars: conint(gt=10, le=50) = DEFAULTS["daylight"]["min_stars"]
    check_interval_min: conint(ge=1, le=60) = DEFAULTS["daylight"]["check_interval_min"]
    lux_threshold: float = DEFAULTS["daylight"]["lux_threshold"]
    min_star_circularity: confloat(ge=0.1, le=1.0) = DEFAULTS["daylight"]["min_star_circularity"]
    dawn_trigger_brightness: confloat(ge=0.0, le=255.0) = DEFAULTS["daylight"]["dawn_trigger_brightness"]
    dawn_trigger_consecutive: conint(ge=1, le=10) = DEFAULTS["daylight"]["dawn_trigger_consecutive"]
    capture_interval_sec: confloat(ge=0.0, le=60.0) = DEFAULTS["daylight"]["capture_interval_sec"]

class JanitorConfig(BaseModel):
    monitor_path: str = DEFAULTS["janitor"]["monitor_path"]
    log_rotation_mb: conint(gt=5, le=20) = DEFAULTS["janitor"]["log_rotation_mb"]
#    log_backup_count: conint(gt=0, le=10) = DEFAULTS["janitor"]["log_backup_count"]
    csv_rotation_mb: conint(gt=5, le=20) = DEFAULTS["janitor"]["csv_rotation_mb"]
    threshold: confloat(gt=0.0, lt=100.0) = DEFAULTS["janitor"]["threshold"]
    target: confloat(gt=0.0, lt=100.0) = DEFAULTS["janitor"]["target"]
    storage_reserve_mb: conint(ge=64) = DEFAULTS["janitor"]["storage_reserve_mb"]
    check_interval_min: conint(ge=1, le=1440) = DEFAULTS["janitor"]["check_interval_min"]
    pressure_check_interval_min: conint(ge=1, le=1440) = DEFAULTS["janitor"]["pressure_check_interval_min"]
    janitor_work_timeout_sec: confloat(gt=0.0) = DEFAULTS["janitor"]["janitor_work_timeout_sec"]

    @field_validator("monitor_path")
    def validate_monitor_path(cls, v):
        if not v:
            raise ValueError("Janitor monitor_path cannot be empty.")
        if not os.path.exists(v):
            raise ValueError(f"Janitor monitor_path '{v}' does not exist.")
        if not os.path.isdir(v):
            raise ValueError(f"Janitor monitor_path '{v}' is not a directory.")
        return v

    @model_validator(mode="after")
    def validate_disk_logic(self):
        # Ensures that the cleanup trigger threshold is higher than the target cleanup level.
        if self.threshold <= self.target:
            raise ValueError(
                f"Configuration Error: Janitor 'threshold' ({self.threshold}%) "
                f"must be greater than 'target' ({self.target}%)."
            )
        if self.pressure_check_interval_min > self.check_interval_min:
            raise ValueError(
                f"Configuration Error: Janitor 'pressure_check_interval_min' ({self.pressure_check_interval_min}) "
                f"must not exceed 'check_interval_min' ({self.check_interval_min}): under storage "
                f"pressure the janitor must run at least as often as normally."
            )
        return self

class SatellitesConfig(BaseModel):
    enabled: bool = DEFAULTS["satellites"]["enabled"]
    groups: conlist(str, min_length=1) = DEFAULTS["satellites"]["groups"]
    url_template: str = DEFAULTS["satellites"]["url_template"]
    retry_margin_min: conint(ge=0, le=240) = DEFAULTS["satellites"]["retry_margin_min"]
    refresh_hours: confloat(ge=2.0) = DEFAULTS["satellites"]["refresh_hours"]
    max_catalog_age_days: confloat(gt=0.0, le=30.0) = DEFAULTS["satellites"]["max_catalog_age_days"]
    request_timeout_sec: conint(ge=1, le=600) = DEFAULTS["satellites"]["request_timeout_sec"]
    position_tolerance_deg: confloat(gt=0.0, le=20.0) = DEFAULTS["satellites"]["position_tolerance_deg"]
    position_tolerance_solved_deg: confloat(gt=0.0, le=20.0) = DEFAULTS["satellites"]["position_tolerance_solved_deg"]
    time_tolerance_sec: confloat(ge=0.0, le=60.0) = DEFAULTS["satellites"]["time_tolerance_sec"]
    min_matching_fraction: confloat(gt=0.0, le=1.0) = DEFAULTS["satellites"]["min_matching_fraction"]
    match_timeout_sec: conint(ge=1, le=600) = DEFAULTS["satellites"]["match_timeout_sec"]

    @field_validator("url_template")
    def validate_url_template(cls, v):
        if "{group}" not in v or not v.startswith("https://") or "FORMAT=CSV" not in v.upper():
            raise ValueError(
                "satellites.url_template must be an https:// URL containing '{group}' and FORMAT=CSV "
                "(OMM CSV; the TLE format cannot represent catalogue numbers >= 100000)."
            )
        return v

    @model_validator(mode="after")
    def validate_refresh_vs_safe_window(self):
        if self.refresh_hours * 60 < 120 + self.retry_margin_min:
            raise ValueError(
                f"Configuration Error: satellites.refresh_hours ({self.refresh_hours}) must be at least the safe "
                f"window, 2 h + retry_margin_min ({self.retry_margin_min} min)."
            )
        return self

    @model_validator(mode="after")
    def validate_refresh_vs_age(self):
        if self.refresh_hours >= self.max_catalog_age_days * 24:
            raise ValueError(
                "Configuration Error: satellites.refresh_hours must be shorter than max_catalog_age_days, "
                "or the catalogue would expire before its next refresh."
            )
        return self

class DashboardConfig(BaseModel):
    enabled: bool = DEFAULTS["dashboard"]["enabled"]
    host: str = DEFAULTS["dashboard"]["host"]
    port: conint(gt=1023, lt=65536) = DEFAULTS["dashboard"]["port"]
    archive_nights_per_page: conint(ge=1, le=366) = DEFAULTS["dashboard"]["archive_nights_per_page"]
    archive_thumb_width: conint(ge=64, le=1920) = DEFAULTS["dashboard"]["archive_thumb_width"]
    archive_thumb_jpeg_quality: conint(ge=10, le=100) = DEFAULTS["dashboard"]["archive_thumb_jpeg_quality"]

class HeartbeatConfig(BaseModel):
    request_timeout_sec: conint(ge=1, le=300) = DEFAULTS["heartbeat"]["request_timeout_sec"]
    enabled: bool = DEFAULTS["heartbeat"]["enabled"]
    url: Optional[str] = DEFAULTS["heartbeat"]["url"]
    interval_min: conint(gt=10, le=60) = DEFAULTS["heartbeat"]["interval_min"]

    @field_validator("url", mode="before")
    @classmethod
    def allow_empty_str_for_url(cls, v):
        if v == "" or str(v).lower() == "none":
            return None
        return v

class EventLogConfig(BaseModel):
    enabled: bool = DEFAULTS["event_log"]["enabled"]

class HealthMonitorConfig(BaseModel):
    enabled: bool = DEFAULTS["health_monitor"]["enabled"]

class SftpConfig(BaseModel):
    @model_validator(mode="after")
    def validate_reconnect_backoff(self):
        if self.reconnect_initial_sec > self.reconnect_max_sec:
            raise ValueError(
                f"Configuration Error: sftp.reconnect_initial_sec ({self.reconnect_initial_sec}) "
                f"must not exceed reconnect_max_sec ({self.reconnect_max_sec})."
            )
        return self

    connect_timeout_sec: conint(ge=1, le=600) = DEFAULTS["sftp"]["connect_timeout_sec"]
    reconnect_initial_sec: conint(ge=1) = DEFAULTS["sftp"]["reconnect_initial_sec"]
    reconnect_max_sec: conint(ge=1) = DEFAULTS["sftp"]["reconnect_max_sec"]
    enabled: bool = DEFAULTS["sftp"]["enabled"]
    host: str = DEFAULTS["sftp"]["host"]
    port: int = DEFAULTS["sftp"]["port"]
    user: str = DEFAULTS["sftp"]["user"]
    remote_dir: str = DEFAULTS["sftp"]["remote_dir"]
    max_queue_size: int = DEFAULTS["sftp"]["max_queue_size"]

class NtpConfig(BaseModel):
    request_timeout_sec: conint(ge=1, le=300) = DEFAULTS["ntp"]["request_timeout_sec"]
    enabled: bool = DEFAULTS["ntp"]["enabled"]
    server: str = DEFAULTS["ntp"]["server"]
    sync_interval_hours: conint(ge=0) = DEFAULTS["ntp"]["sync_interval_hours"]
    max_offset_sec: float = DEFAULTS["ntp"]["max_offset_sec"]

class AIClassificationConfig(BaseModel):
    enabled: bool = DEFAULTS["ai_classification"]["enabled"]
    model_path: str = DEFAULTS["ai_classification"]["model_path"]
    label_path: str = DEFAULTS["ai_classification"]["label_path"]
    min_confidence: confloat(ge=0.0, le=1.0) = DEFAULTS["ai_classification"]["min_confidence"]
#    quarantine_non_meteors: bool = DEFAULTS["ai_classification"]["quarantine_non_meteors"]
    input_size: conint(gt=32, le=1024) = DEFAULTS["ai_classification"]["input_size"]

    @field_validator("model_path", "label_path")
    @classmethod
    def validate_paths(cls, v):
        # We don't strictly require the files to exist at the moment of validation
        # (to allow for remote setup), but we ensure they aren't empty strings.
        if not v or v.strip() == "":
            raise ValueError("AI model and label paths cannot be empty.")
        return v

class AIObservatoryConfig(BaseModel):
    enabled: bool = DEFAULTS["ai_observatory"]["enabled"]
    model_path: str = DEFAULTS["ai_observatory"]["model_path"]
    label_path: str = DEFAULTS["ai_observatory"]["label_path"]
    min_confidence: confloat(ge=0.0, le=1.0) = DEFAULTS["ai_observatory"]["min_confidence"]
    bad_conditions: List[str] = DEFAULTS["ai_observatory"]["bad_conditions"]

    @field_validator("bad_conditions", mode="before")
    @classmethod
    def parse_bad_conditions(cls, v):
        # If the web dashboard sends it as a text string (e.g. "['rain', 'fog']")
        if isinstance(v, str):
            # Remove brackets and quotes
            cleaned = (
                v.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            )
            # Split by comma into a proper list, ignoring empty spaces
            return [item.strip() for item in cleaned.split(",") if item.strip()]
        return v

class GeneralConfig(BaseModel):
    drop_report_interval_sec: conint(ge=1, le=3600) = DEFAULTS["general"]["drop_report_interval_sec"]
    record_queue_maxsize: conint(ge=10, le=100000) = DEFAULTS["general"]["record_queue_maxsize"]
    scheduler_interval_sec: conint(ge=1, le=600) = DEFAULTS["general"]["scheduler_interval_sec"]
    loop_time_ema_alpha: confloat(gt=0.0, le=1.0) = DEFAULTS["general"]["loop_time_ema_alpha"]
    hw_query_timeout_sec: conint(ge=1, le=60) = DEFAULTS["general"]["hw_query_timeout_sec"]
    thermal_read_interval_sec: conint(ge=1, le=3600) = DEFAULTS["general"]["thermal_read_interval_sec"]
    thread_error_backoff_sec: conint(ge=1, le=3600) = DEFAULTS["general"]["thread_error_backoff_sec"]
    watchdog_startup_delay_sec: conint(ge=0, le=600) = DEFAULTS["general"]["watchdog_startup_delay_sec"]
    watchdog_check_interval_sec: conint(ge=1, le=300) = DEFAULTS["general"]["watchdog_check_interval_sec"]
    queue_status_ping_sec: conint(ge=1, le=3600) = DEFAULTS["general"]["queue_status_ping_sec"]
    debug_image_save_interval_sec: confloat(gt=0.0, le=3600.0) = DEFAULTS["general"]["debug_image_save_interval_sec"]
    png_compression: conint(ge=0, le=9) = DEFAULTS["general"]["png_compression"]
    shutdown_join_slack_sec: confloat(gt=0.0, le=600.0) = DEFAULTS["general"]["shutdown_join_slack_sec"]
    thread_shutdown_timeout_sec: conint(ge=1, le=600) = DEFAULTS["general"]["thread_shutdown_timeout_sec"]
    camera_restart_settle_sec: conint(ge=1, le=300) = DEFAULTS["general"]["camera_restart_settle_sec"]
    maintenance_poll_sec: conint(ge=1, le=300) = DEFAULTS["general"]["maintenance_poll_sec"]
    debug_visualization: bool = DEFAULTS["general"]["debug_visualization"]
    debug_align_dump_stride: conint(ge=1) = DEFAULTS["general"]["debug_align_dump_stride"]
    debug_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="0=Off, 1=Stats, 2=Images, 3=Full/Stream, 4=Variables dump",
    )
    log_dir: str = DEFAULTS["general"]["log_dir"]

    @field_validator("log_dir")
    def validate_log_dir(cls, v):
        # Must stay a relative sub-folder of janitor.monitor_path: an empty value would
        # dump logs into the data root (where the janitor tiers live), an absolute path
        # would make os.path.join discard monitor_path entirely, and ".." would escape it.
        v = v.strip()
        if not v:
            raise ValueError("general.log_dir cannot be empty.")
        if os.path.isabs(v):
            raise ValueError(f"general.log_dir '{v}' must be relative to janitor.monitor_path.")
        norm = os.path.normpath(v)
        if norm == "." or norm.split(os.sep)[0] == "..":
            raise ValueError(f"general.log_dir '{v}' must be a sub-folder inside janitor.monitor_path.")
        return norm

    jpg_quality: conint(gt=0, le=100) = DEFAULTS["general"]["jpg_quality"]
    jpg_quality_zmq: conint(gt=0, le=100) = DEFAULTS["general"]["jpg_quality_zmq"]
    hostname: str = DEFAULTS["general"]["hostname"]
    zombie_thread_timeout_sec: int = DEFAULTS["general"]["zombie_thread_timeout_sec"]
    timelapse_video_thread_timeout_sec: int = DEFAULTS["general"]["timelapse_video_thread_timeout_sec"]
    finalize_video_thread_timeout_sec: int = DEFAULTS["general"]["finalize_video_thread_timeout_sec"]
    watchdog_hung_thread_sec: conint(gt=0) = DEFAULTS["general"]["watchdog_hung_thread_sec"]
    # le=0.5 guarantees at least two pings per hang window: with a single ping per window,
    # scheduler jitter alone could push the gap past watchdog_hung_thread_sec.
    exit_forensics_timeout_sec: confloat(gt=0.0, le=30.0) = DEFAULTS["general"]["exit_forensics_timeout_sec"]
    watchdog_heartbeat_fraction: confloat(gt=0.0, le=0.5) = DEFAULTS["general"]["watchdog_heartbeat_fraction"]
    location: Optional[str] = DEFAULTS["general"]["location"]
    latitude: confloat(ge=-90.0, le=90.0) = DEFAULTS["general"]["latitude"]
    longitude: confloat(ge=-180.0, le=180.0) = DEFAULTS["general"]["longitude"]
    max_camera_failures: conint(gt=0) = DEFAULTS["general"]["max_camera_failures"]
    max_restart_failures: conint(gt=0, le=50) = DEFAULTS["general"]["max_restart_failures"]
    idle_heartbeat_interval_min: conint(gt=0) = DEFAULTS["general"]["idle_heartbeat_interval_min"]
    power_monitor_pin: int = DEFAULTS["general"]["power_monitor_pin"]
    camera_azimuth: confloat(ge=0.0, le=360.0) = DEFAULTS["general"]["camera_azimuth"]
    camera_altitude: confloat(ge=0.0, le=90.0) = DEFAULTS["general"]["camera_altitude"]
    lens_hfov: conint(ge=0, le=360) = DEFAULTS["general"]["lens_hfov"]
    lens_vfov: conint(ge=0, le=360) = DEFAULTS["general"]["lens_vfov"]
    lens_projection: Literal["rectilinear", "equidistant"] = DEFAULTS["general"]["lens_projection"]
    sensor_camera_id: Optional[str] = DEFAULTS["general"]["sensor_camera_id"]
    sensor_ext_id: Optional[str] = DEFAULTS["general"]["sensor_ext_id"]
    maintenance_timeout: conint(gt=0) = DEFAULTS["general"]["maintenance_timeout"]
    ota_update_url: Optional[str] = DEFAULTS["general"]["ota_update_url"]
    post_shutdown_script: Optional[str] = DEFAULTS["general"]["post_shutdown_script"]
    shutdown_time: Optional[constr(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")] = DEFAULTS["general"]["shutdown_time"]
    start_time: Optional[constr(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")] = DEFAULTS["general"]["start_time"]
    end_time: Optional[constr(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")] = DEFAULTS["general"]["end_time"]

    @field_validator(
        "location",
        "shutdown_time",
        "start_time",
        "end_time",
        "post_shutdown_script",
        "ota_update_url", 
        "sensor_camera_id",
        "sensor_ext_id",
        mode="before",
    )
    @classmethod
    def allow_empty_str_for_optionals(cls, v):
        # This single validator handles location and time fields
        if v == "" or str(v).lower() in ["none", "null"]:
            return None
        return v

class MainConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    astrometry: AstrometryConfig = Field(default_factory=AstrometryConfig)
    timelapse: TimelapseConfig = Field(default_factory=TimelapseConfig)
    timelapse_video: TimelapseVideoConfig = Field(default_factory=TimelapseVideoConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    power_monitor: PowerMonitorConfig = Field(default_factory=PowerMonitorConfig)
    led_status: LedStatusConfig = Field(default_factory=LedStatusConfig)
    daylight: DaylightConfig = Field(default_factory=DaylightConfig)
    janitor: JanitorConfig = Field(default_factory=JanitorConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    satellites: SatellitesConfig = Field(default_factory=SatellitesConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    event_log: EventLogConfig = Field(default_factory=EventLogConfig)
    health_monitor: HealthMonitorConfig = Field(default_factory=HealthMonitorConfig)
    sftp: SftpConfig = Field(default_factory=SftpConfig)
    ntp: NtpConfig = Field(default_factory=NtpConfig)
    ai_classification: AIClassificationConfig = Field(
        default_factory=AIClassificationConfig
    )
    ai_observatory: AIObservatoryConfig = Field(default_factory=AIObservatoryConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)

    @model_validator(mode="after")
    def check_cross_section_timing(self):
        """
        Timing budgets that live in different sections but constrain each other.
        Each violation is a guaranteed runtime failure, so it is rejected at load.
        """
        g, cap = self.general, self.capture
        # 1. A legitimate camera sync (step 5 physics: buffers + control delay + the
        #    new frame, each up to max_exposure long, plus slack) must never outlast the
        #    hang window, or a normal long-exposure AE step force-restarts the process.
        flush_budget = (
            (cap.raw_buffer_count + cap.sync_control_delay_frames + 1) * cap.max_exposure_us / 1e6
            + cap.sync_timeout_slack_sec
        )
        if g.watchdog_hung_thread_sec <= flush_budget:
            raise ValueError(
                f"Configuration Error: general.watchdog_hung_thread_sec ({g.watchdog_hung_thread_sec}) must exceed "
                f"the worst-case camera sync time ({flush_budget:.1f}s = (raw_buffer_count + "
                f"sync_control_delay_frames + 1) x max_exposure_us + sync_timeout_slack_sec)."
            )
        # 2. The watchdog must be able to look at least twice inside one hang window.
        if g.watchdog_check_interval_sec * 2 > g.watchdog_hung_thread_sec:
            raise ValueError(
                f"Configuration Error: general.watchdog_check_interval_sec ({g.watchdog_check_interval_sec}) "
                f"must be at most half of watchdog_hung_thread_sec ({g.watchdog_hung_thread_sec})."
            )
        # 3. Work budgets must fit inside the zombie budgets of the threads running them,
        #    or a normal slow encode is reported (and restarted) as a zombie thread.
        if self.timelapse_video.encode_timeout_sec >= g.timelapse_video_thread_timeout_sec:
            raise ValueError(
                f"Configuration Error: timelapse_video.encode_timeout_sec ({self.timelapse_video.encode_timeout_sec}) "
                f"must be less than general.timelapse_video_thread_timeout_sec ({g.timelapse_video_thread_timeout_sec})."
            )
        if self.events.writer_wait_sec >= g.finalize_video_thread_timeout_sec:
            raise ValueError(
                f"Configuration Error: events.writer_wait_sec ({self.events.writer_wait_sec}) must be less than "
                f"general.finalize_video_thread_timeout_sec ({g.finalize_video_thread_timeout_sec})."
            )
        return self

    @model_validator(mode="after")
    def check_stack_n_memory_usage(self):
        # After all individual fields are validated,
        # check if the configured nighttime_stack_N value is safe for the system's available memory.
        # This validator runs for every field, but we only need to execute
        # our logic once all the necessary data is available.
        # We check if 'capture' and 'timelapse' are present in the 'values.data' dict.
        # if 'capture' in values.data and 'timelapse' in values.data:
        capture_cfg = self.capture
        timelapse_cfg = self.timelapse

        # 1. Estimate memory per frame (width * height * 3 bytes/pixel)
        bytes_per_frame = capture_cfg.width * capture_cfg.height * 6

        # 2. Get available system memory
        try:
            # Check the current available memory
            available_mem_bytes = psutil.virtual_memory().available
        except Exception:
            # If psutil fails, fallback to a safe, low default 2GB
            available_mem_bytes = 2 * 1024 * 1024 * 1024

        # 3. Calculate a safe maximum stack size
        # We'll use a conservative limit: don't let the image buffer
        # consume more than 25% of the *available* RAM.
        safe_mem_for_stack = available_mem_bytes * 0.25
        safe_max_stack_n = int(safe_mem_for_stack / bytes_per_frame)
        safe_max_stack_n = max(5, safe_max_stack_n)  # Failsafe minimum

        # 4. Perform the validation check
        if timelapse_cfg.nighttime_stack_N > safe_max_stack_n:
            mem_per_frame_mb = bytes_per_frame / (1024 * 1024)
            configured_usage_mb = timelapse_cfg.nighttime_stack_N * mem_per_frame_mb
            available_mem_mb = available_mem_bytes / (1024 * 1024)

            error_msg = (
                f"Configuration failed: timelapse.nighttime_stack_N ({timelapse_cfg.nighttime_stack_N}) is too high for the available system RAM. "
                f"Estimated usage: {configured_usage_mb:.0f} MB. "
                f"Available RAM: {available_mem_mb:.0f} MB. "
                f"Recommended maximum nighttime_stack_N for this resolution is ~{safe_max_stack_n}."
            )
            raise ValueError(error_msg)

        return self  # Must return the value for the next validator

