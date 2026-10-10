"""
detection_mixin.py

Core motion-detection loop and event-scoring methods for the Ryusei
pipeline, extracted from the main Pipeline class as a pure move -- method
bodies are unchanged, only their physical location. Composed into
Pipeline via multiple inheritance.

Two non-contiguous single-method chunks in the original file: detection_loop
(huge, ~575 lines, stands alone) and _calculate_event_score (far away,
many unrelated methods -- health monitoring, watchdog loops, thermal
throttling -- sit between them and are deliberately NOT included here).
Confirmed via an explicit interleaving check before extraction.

This is the most cross-cutting code in the whole file -- it calls into
nearly every other mixin already extracted this session.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls
self._run_bounded (ConcurrencyClass), self.get_event_astro_data
(AstrometryClass), self.publish_debug (DashboardClass), self._safe_save_image
(FileUtilsClass), plus self._enqueue_timelapse_frame,
self._get_dynamic_accumulate_alpha, self._heal_live_frame,
self._update_dynamic_dark_if_needed, self.log_health_event -- all still
directly in Pipeline's own body (candidates for a future capture_mixin).
All must be provided somewhere in the composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import math
import queue
import logging
import time
import cv2
import numpy as np

from camera_capture import RAW_BIT_DEPTH
from datetime import datetime
from collections import deque


class DetectionClass:
    """Core motion-detection loop and event scoring, composed into Pipeline."""

    def detection_loop(self):
        logging.info("Detection loop started.")

        # Rate limiter for debug saves
        last_debug_save = float("-inf")  # monotonic; -inf = never saved
        debug_save_interval = self.cfg["general"]["debug_image_save_interval_sec"]

        buffer_frames = self.pre_event_frames
        cooldown_frames = self.event_cooldown_frames
        recent_frames = deque(maxlen=buffer_frames)
        cnts = []
        # Timelapse clip protection: an event's span reaches back over the whole
        # pre-event buffer, so the registry needs this lookback to know when a
        # frame can no longer be claimed by a future trigger (see event_footprints.py).
        self.event_footprints.configure(lookback_frames=buffer_frames)
        footprint_id = None

        # State Management for creating two different packages
        current_event_video_frames = []
        current_event_stack_frames = []
        event_cooldown_counter = 0

        max_event_frames = self.effective_max_event_frames
        current_event_frame_count = 0
        current_event_bbox = None
        current_event_astro_data = None
        # Per-frame motion boxes of the event: (ts, [x0, y0, x1, y1]) for every frame with
        # detected motion. The satellite matcher compares this track, frame by frame,
        # with the positions of sunlit satellites at those exact times.
        current_event_track = []

        local_loop_ema = 0.0

        def _truncate_event(reason):
            """
            Close the event in progress at this point THROUGH the normal exit path
            (video, stack, CSV row, upload) and reset the state machine. Used when
            the frame stream stops being one continuous sequence: daylight/weather
            hold engaging, or a gap in the frame stream.
            """
            nonlocal current_event_video_frames, current_event_stack_frames, current_event_frame_count
            nonlocal current_event_bbox, current_event_astro_data, event_cooldown_counter, footprint_id
            nonlocal current_event_track
            logging.warning(
                f"Event in progress closed early ({reason}). Dispatching packages "
                f"(Video: {len(current_event_video_frames)}, Stack: {len(current_event_stack_frames)})."
            )
            self.log_health_event(
                "WARNING", "EVENT_TRUNCATED",
                f"Event closed early by {reason} ({current_event_frame_count} frames).",
            )
            self._dispatch_event(
                current_event_video_frames, current_event_stack_frames,
                current_event_bbox, current_event_astro_data, False,
                track=current_event_track,
            )
            current_event_track = []
            current_event_video_frames = []
            current_event_stack_frames = []
            current_event_frame_count = 0
            current_event_bbox = None
            current_event_astro_data = None
            event_cooldown_counter = 0
            if footprint_id is not None:
                self.event_footprints.close(footprint_id)
                footprint_id = None
            recent_frames.clear()  # same reset as a normal event end
            self.event_in_progress.clear()

        if not hasattr(self, "background_reset_time"):
            self.background_reset_time = float("-inf")

        gate_logged = False
        try:
            while self.running.is_set():
                
                self.ping_watchdog()

                # Pre-check gate: until the SkyMonitor has validated exposure and sky
                # (capture_stable), frames are not trustworthy for detection. A test
                # shot is exempt: it must get frames whatever the sky state.
                if not self.capture_stable.is_set() and not self.is_calibrating.is_set():
                    try:
                        item = self.acq_q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if item is None:
                        # The stop sentinel must be honoured here too: discarding it left
                        # a paused pipeline with an unstoppable DetectionThread
                        # (THREAD_JOIN_TIMEOUT -> reboot).
                        logging.info("Detection loop received sentinel while waiting for the sky check, exiting.")
                        break
                    # Drop stale FRAMES queued for the timelapse, but keep its control
                    # tokens (FLUSH/STATUS/None): the old full drain could swallow the
                    # end-of-session FLUSH and lose the partial stack.
                    kept = []
                    while True:
                        try:
                            q_item = self.timelapse_q.get_nowait()
                        except queue.Empty:
                            break
                        if not isinstance(q_item, tuple):
                            kept.append(q_item)
                    for tok in kept:
                        try:
                            self.timelapse_q.put_nowait(tok)
                        except queue.Full:
                            break
                    if not gate_logged:
                        logging.info("Detection waiting for the sky check; discarding frames until then.")
                        gate_logged = True
                        # Acquisition is paused for the check: the frames buffered before
                        # it can no longer precede anything captured after it.
                        if recent_frames:
                            logging.info(f"Discarding {len(recent_frames)} pre-event frame(s) captured before the pause.")
                        recent_frames.clear()
                    continue
                if gate_logged:
                    logging.info("Sky check done: detection is processing frames.")
                    gate_logged = False

                try:
                    item = self.acq_q.get(timeout=1.0)
                    if item is None:
                        logging.info(
                            "Detection loop received sentinel. Draining complete, exiting."
                        )
                        break
                    ts, frame = item
                except queue.Empty:
                    continue

                process_start = time.perf_counter()

                # Dynamic Scaled Dark Frame                          
                # Update the Dark Frame and BPM.
                self._update_dynamic_dark_if_needed()

                # LIVE FRAME HEALING (BPM INTERPOLATION)
                frame = self._heal_live_frame(frame)
                                    
                # APPLICATION OF THE DARK FRAME
                if self.master_dark is not None:
                    # frame_tmp = cv2.cvtColor(self.master_dark, cv2.COLOR_RGB2BGR)   #   <--- da togliere  TODO
                    # frame = cv2.subtract(frame, frame_tmp)   #   <--- da togliere  TODO
                    frame = cv2.subtract(frame, self.master_dark)   #   <--- da ripristinare  TODO

                # Fetch current cooldown limit from profile
                cooldown_frames = self.event_cooldown_frames

                # Flat Field Correction
                if self.flat_multiplier is not None:
                    try:                     
                        frame_f = frame.astype(np.float32)
                        frame_f = cv2.multiply(frame_f, self.flat_multiplier)
                        frame = np.clip(frame_f, 0, 255).astype(np.uint8)
                    except Exception:
                        pass

                # Calibration Queue
                if self.is_calibrating.is_set():
                    try:
                        self.calibration_q.put_nowait((ts, frame))
                    except queue.Full:
                        logging.warning("Calibration queue is full.")
                    self.event_footprints.mark_evaluated(ts)  # no event can involve this frame
                    continue

                with self.status_lock:
                    effective_threshold = self.effective_threshold

                if effective_threshold is None or not isinstance(
                    effective_threshold, (int, float)
                ):
                    logging.warning(
                        "Invalid effective_threshold (%r), falling back to base_threshold",
                        effective_threshold,
                    )
                    effective_threshold = self.cfg["detection"]["base_threshold"]
                    self.effective_threshold = self.cfg["detection"]["base_threshold"]

                if (
                    self.daylight_mode_active.is_set()
                    or self.weather_hold_active.is_set()
                ):
                    if self.event_in_progress.is_set():
                        # Daylight/weather hold engaged mid-event. Every frame from
                        # here on takes the early `continue` below and never reaches
                        # the event state machine further down, so event_in_progress
                        # would otherwise stay set forever -- freezing any later
                        # SkyMonitorThread check that waits on it via
                        # wait_for_safe_window(), which cascades into the scheduler
                        # (which pauses while sky_monitor_in_progress is set) never
                        # evaluating end_time/shutdown_time again. Force-close the
                        # partial event now so nothing is left waiting on it.
                        # Close it THROUGH the normal exit path: every frame in these
                        # buffers was captured before the hold engaged, i.e. valid night
                        # data (a dawn fireball is exactly the event that ends this way).
                        # Discarding it lost the video, the stack, the CSV row and the
                        # SFTP upload.
                        reason = "daylight" if self.daylight_mode_active.is_set() else "weather hold"
                        _truncate_event(f"{reason} engaged")
                    # Detection is paused (daylight / weather hold): no pre-event history
                    # may survive into the session that follows.
                    recent_frames.clear()
                    self._enqueue_timelapse_frame(ts, frame, "N/A (Paused)")
                    if self.cfg["general"].get("debug_level") >= 1:
                        debug_payload = {
                            "Daylight_mode": self.daylight_mode_active.is_set(),
                            "weather_hold_active": self.weather_hold_active.is_set(),
                        }
                        self.publish_debug("detection_stats", debug_payload, image=None)
                        # If it's daylight, we are just draining the queue to keep it fresh.
                        # Add a tiny sleep to prevent 100% CPU usage on one core.
                        if self.daylight_mode_active.is_set():
                            time.sleep(0.02)
                    self.event_footprints.mark_evaluated(ts)  # paused: no event can involve this frame
                    continue

                # -------- Night Mode Logic ----------
                # Append pre-event buffer
                if not self.event_in_progress.is_set():
                    recent_frames.append((ts, frame, effective_threshold))

                current_alpha = self._get_dynamic_accumulate_alpha()
                local_bg = self.background

                # now_ts = time.time()
                in_cooldown = (time.monotonic() - self.background_reset_time) < self._background_warmup_sec()
                skip_motion_math = (local_bg is None) or in_cooldown

                # gray is needed in every case: for the motion math, and during the
                # warm-up after a reset, which exists precisely so the model can LEARN
                # (the old code skipped the update in warm-up too, so the model stayed a
                # single frame for the whole wait and the wait achieved nothing).
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
                
                if local_bg is None:
                    # Initialize the background safely
                    self.background = gray.astype("float32")
                # Background LEARNING happens after the motion decision (see below),
                # never before: blending this frame in first put the trigger frame's
                # meteor into the background (alpha x its brightness: ~30% at a 6 s
                # exposure), which was then frozen for the whole event. Every later
                # frame "moved" against that ghost, the event never cooled down and
                # always ran into EVENT TIMEOUT.

                self._enqueue_timelapse_frame(ts, frame, effective_threshold)

                # Motion Math
                detected = False
                massive_light_shift = False
                raw_contour_count = 0
                rejected_cnts_list = []
                rejection_stats = {"area": 0, "total": 0}
                cnts = []  # Reset each frame so a low-motion frame doesn't draw stale contours in the debug mosaic

                if not skip_motion_math:
                    diff = cv2.absdiff(gray.astype(np.float32), local_bg)

                    _, th = cv2.threshold(
                        diff, int(effective_threshold), 255, cv2.THRESH_BINARY
                    )
                    th = th.astype(np.uint8)

                    morph_kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
                    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph_kernel)

                    motion_pixels = cv2.countNonZero(th)
                    if motion_pixels >= self.min_changes_required:
                        cnts, _ = cv2.findContours(
                            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        raw_contour_count = len(cnts)
                        rejection_stats["total"] = raw_contour_count
                        
                        # Contour DENSITY, not a raw count: the same flash or headlight sweep
                        # produces proportionally more contours at higher resolution, so a
                        # fixed count meant a different physical criterion per sensor mode.
                        contours_per_mpix = raw_contour_count / (th.shape[0] * th.shape[1] / 1e6)
                        if contours_per_mpix > self.cfg["detection"]["light_shift_contours_per_mpix"]:
                            massive_light_shift = True

                            # Only reset when there's no event to disturb. Background
                            # is never updated while event_in_progress is set anyway
                            # (see the accumulateWeighted guard above), so nulling it
                            # out mid-event helps nothing for the event currently
                            # being tracked and only forces a slow single-frame
                            # rebuild once it ends -- right when a stable background
                            # is what's needed to cleanly detect the quiet (no-motion)
                            # frames that let an event cool down naturally.
                            if not self.event_in_progress.is_set():
                                logging.warning(
                                    f"Massive light shift detected ({raw_contour_count} contours). Resetting background."
                                )
                                self.background = None
                                self.background_reset_time = time.monotonic()
                            else:
                                logging.warning(
                                    f"Massive light shift detected ({raw_contour_count} contours) "
                                    "during an event -- leaving background untouched."
                                )

                            if self.event_in_progress.is_set() and not self.memory_critical.is_set():
                                event_cooldown_counter += 1
                                current_event_video_frames.append(
                                    (ts, frame, effective_threshold)
                                )
                                current_event_frame_count += 1

                            # NOTE: deliberately no `continue` here (there used to be
                            # one). It let a burst of these -- e.g. during dawn/cloud
                            # brightness swings -- run an event all the way to its
                            # hard frame cap without the shared is_cooldown_end /
                            # is_timeout_end check below ever running on these frames,
                            # since that check sits after the EVENT STATE MACHINE
                            # block this would have skipped past. massive_light_shift
                            # makes that block skip just the increment (already done
                            # above, right here) while still falling through to the
                            # termination check every other frame gets.
                        else:
                            accepted_cnts = 0
                            frame_bbox = None  # Tracks motion in THIS specific frame
                            for c in cnts:
                                x, y, w, h = cv2.boundingRect(c)
                                bbox_area = w * h
                                actual_area = max(cv2.contourArea(c), bbox_area)

                                if actual_area < self.min_area:
                                    rejection_stats["area"] += 1
                                    rejected_cnts_list.append(c)
                                else:
                                    accepted_cnts += 1
                                    if frame_bbox is None:
                                        frame_bbox = [x, y, x + w, y + h]
                                    else:
                                        frame_bbox[0] = min(frame_bbox[0], x)
                                        frame_bbox[1] = min(frame_bbox[1], y)
                                        frame_bbox[2] = max(frame_bbox[2], x + w)
                                        frame_bbox[3] = max(frame_bbox[3], y + h)

                            detected = accepted_cnts > 0

                # REJECTED EVENT DIAGNOSTIC SNAPSHOT
                if not detected and len(rejected_cnts_list) > 0:
                    now = time.monotonic()
                    save_interval = self.cfg["detection"]["reject_save_interval_sec"]
                    if now - getattr(self, "last_reject_save_time", float("-inf")) > save_interval:
                        self.last_reject_save_time = now
                        # rej_img = (frame >> 8).astype(np.uint8)
                        rej_img = frame.copy()
                        thickness = max(1, int(self.cfg["capture"]["width"] / 1000))
                        cv2.drawContours(
                            rej_img, rejected_cnts_list, -1, (0, 0, 255), thickness
                        )

                        rej_name = f"rejected_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jpg"
                        rej_path = os.path.join(self.rejected_out_dir, rej_name)
                        logging.info(
                            f"Diagnostics: Filtered {len(rejected_cnts_list)} noise artifact(s). Saved snapshot to {rej_name}"
                        )
                        self._safe_save_image(
                            rej_path,
                            rej_img,
                            [
                                int(cv2.IMWRITE_JPEG_QUALITY),
                                self.cfg["general"]["jpg_quality"],
                            ],
                        )

                with self.status_lock:
                    self.session_stats["rejected_contours"] += rejection_stats["area"]
                    self.session_stats["total_events"] += raw_contour_count

                # ZMQ DEBUG PUBLISHER
                try:
                    debug_level = self.cfg["general"]["debug_level"]
                    if debug_level >= 1 and not skip_motion_math:
                        mean_val, std_dev = cv2.meanStdDev(diff)
                        noise_sigma = float(std_dev[0][0])
                        debug_payload = {
                            "noise_sigma": round(noise_sigma, 2),
                            "threshold": effective_threshold,
                            "contours_total": raw_contour_count,
                            "contours_accepted": raw_contour_count
                            - rejection_stats["area"],
                            "rejected": rejection_stats,
                            "event_active": self.event_in_progress.is_set(),
                            "process_time_ms": round(local_loop_ema, 2),
                        }
                        debug_img = None
                        if (debug_level >= 2) and not in_cooldown:
                            h, w = gray.shape
                            s = 0.5
                            new_size = (int(w * s), int(h * s))

                            def ensure_color(img):
                                if len(img.shape) == 2:
                                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                return img

                            d1 = cv2.resize(ensure_color(gray), new_size)
                            d2 = cv2.resize(ensure_color(diff), new_size)
                            d3 = cv2.resize(ensure_color(th), new_size)
                            d4 = cv2.resize(ensure_color(frame), new_size)
                            cv2.drawContours(d4, cnts, -1, (0, 0, 65535), 1)
                            top = np.hstack((d1, d2))
                            bot = np.hstack((d3, d4))
                            # debug_img = (np.vstack((top, bot)) >> 8).astype(np.uint8)
                            debug_img = np.vstack((top, bot)).astype(np.uint8)

                            # Disk dump at level >= 2 ("Images"); ZMQ streaming of
                            # this same debug_img is separately gated at level >= 3
                            # inside publish_debug().
                            if self.event_in_progress.is_set():
                                now = time.monotonic()
                                if now - last_debug_save > debug_save_interval:
                                    fname = os.path.join(
                                        self.debug_out_dir,
                                        f"det_{datetime.now().strftime('%H%M%S_%f')}.jpg",
                                    )
                                    self._safe_save_image(fname, debug_img)
                                    last_debug_save = now
                        self.publish_debug(
                            "detection_stats", debug_payload, image=debug_img
                        )
                except Exception as e:
                    logging.error(f"ZMQ Debug Error: {e}")

                # EVENT STATE MACHINE
                if massive_light_shift:
                    # Already accounted for above (cooldown/frame-count increment,
                    # background handling) -- don't double-count it here.
                    pass
                elif detected:
                    if self.memory_critical.is_set():
                        pass
                    elif not self.event_in_progress.is_set():
                        logging.info(
                            "New event started at %s (contours=%d)",
                            ts,
                            raw_contour_count,
                        )
                        self.event_in_progress.set()
                        current_event_video_frames = list(recent_frames)
                        current_event_stack_frames = [
                            (ts, frame.copy(), effective_threshold)
                        ]
                        current_event_frame_count = len(current_event_video_frames)
                        # Set the initial coordinates
                        current_event_bbox = frame_bbox
                        current_event_track = [(ts, list(frame_bbox))] if frame_bbox else []
                        # Span starts at the oldest pre-event frame: those frames are
                        # part of the event even though they were below threshold.
                        footprint_id = self.event_footprints.open(
                            current_event_video_frames[0][0], ts, frame_bbox
                        )

                        # Astrometry solve, ONCE per event (here, not on every
                        # subsequent frame of an ongoing event) -- uses the
                        # 10-bit raw Bayer green channel, which avoids the
                        # MAD-collapses-to-zero quantization failure the
                        # 8-bit fallback path hits on dim event frames.
                        current_event_astro_data = None
                        if self.cfg["astrometry"].get("enabled"):
                            with self.status_lock:
                                raw_for_event = (
                                    self.last_raw_frame_bayer10.copy()
                                    if self.last_raw_frame_bayer10 is not None
                                    else None
                                )
                            if raw_for_event is not None:
                                if self.cfg["general"]["debug_level"] >= 2:
                                    ts_str = ts.strftime("%Y%m%dT%H%M%S") if hasattr(ts, "strftime") else str(ts).replace(":", "-")
                                    debug_raw_path = os.path.join(
                                        self.debug_out_dir, f"debug_raw_event_{ts_str}.png"
                                    )
                                    try:
                                        cv2.imwrite(debug_raw_path, raw_for_event)
                                        logging.info(f"Debug: saved raw event frame to {debug_raw_path}")
                                    except Exception as e:
                                        logging.error(f"Debug: failed to save raw event frame: {e}")

                                # event_bbox was computed on the MAIN stream's
                                # resolution; the raw stream comes back at a
                                # DIFFERENT (smaller, binned) resolution --
                                # confirmed 1536x864 raw vs 1920x1080 main, a
                                # clean 0.8x scale on both axes. Scale the bbox
                                # into raw-stream coordinates before using it,
                                # computed from actual dimensions, not a
                                # hardcoded 0.8 -- so this stays correct if
                                # either resolution ever changes.
                                raw_h, raw_w = raw_for_event.shape[:2]
                                main_h, main_w = frame.shape[:2]
                                scale_x = raw_w / main_w
                                scale_y = raw_h / main_h
                                scaled_bbox = None
                                if frame_bbox is not None:
                                    scaled_bbox = [
                                        frame_bbox[0] * scale_x, frame_bbox[1] * scale_y,
                                        frame_bbox[2] * scale_x, frame_bbox[3] * scale_y,
                                    ]

                                # Green sub-array: RGGB Bayer tile, full
                                # 10-bit precision, no demosaic needed
                                # for star detection.
                                green10 = raw_for_event[0::2, 1::2]
                                solve_timeout_sec = self.cfg["astrometry"]["event_solve_timeout_sec"]
                                current_event_astro_data, _timed_out = self._run_bounded(
                                    self.get_event_astro_data,
                                    solve_timeout_sec,
                                    "DetectionThread",
                                    green10,
                                    event_bbox=scaled_bbox,
                                    bit_depth=RAW_BIT_DEPTH,
                                )

                        with self.status_lock:
                            self.session_stats["accepted_events"] += 1
                    else:
                        current_event_video_frames.append(
                            (ts, frame, effective_threshold)
                        )
                        current_event_stack_frames.append(
                            (ts, frame, effective_threshold)
                        )
                        current_event_frame_count += 1
                        if frame_bbox:
                            current_event_track.append((ts, list(frame_bbox)))
                            current_event_bbox[0] = min(
                                current_event_bbox[0], frame_bbox[0]
                            )
                            current_event_bbox[1] = min(
                                current_event_bbox[1], frame_bbox[1]
                            )
                            current_event_bbox[2] = max(
                                current_event_bbox[2], frame_bbox[2]
                            )
                            current_event_bbox[3] = max(
                                current_event_bbox[3], frame_bbox[3]
                            )

                    event_cooldown_counter = 0

                elif self.event_in_progress.is_set():
                    if not self.memory_critical.is_set():
                        event_cooldown_counter += 1
                        current_event_video_frames.append(
                            (ts, frame, effective_threshold)
                        )
                        current_event_frame_count += 1

                is_cooldown_end = event_cooldown_counter > cooldown_frames
                is_timeout_end = current_event_frame_count >= max_event_frames
                is_memory_kill = self.memory_critical.is_set()

                if self.event_in_progress.is_set() and (
                    is_cooldown_end or is_timeout_end or is_memory_kill
                ):

                    if is_memory_kill:
                        logging.error(
                            f"EVENT FORCIBLY TERMINATED: RAM Critical. Halting early after {current_event_frame_count} frames."
                        )
                        self.log_health_event(
                            "WARNING",
                            "EVENT_TRUNCATED",
                            "RAM Critical. Event cut short.",
                        )
                        # self.background = None
                        self.background_reset_time = time.monotonic()
                    elif is_timeout_end:
                        logging.error(
                            f"EVENT TIMEOUT: Forcibly terminated after {current_event_frame_count} frames."
                        )
                        self.log_health_event(
                            "ERROR",
                            "EVENT_TIMEOUT",
                            f"Forced termination after {current_event_frame_count} frames.",
                        )
                        # self.background = None
                        self.background_reset_time = time.monotonic()

                    if is_cooldown_end:
                        logging.info(
                            f"Event finished naturally. Dispatching packages (Video: {len(current_event_video_frames)}, Stack: {len(current_event_stack_frames)})."
                        )
                    else:
                        logging.info(
                            f"Event aborted. Dispatching packages (Video: {len(current_event_video_frames)}, Stack: {len(current_event_stack_frames)})."
                        )

                    self._dispatch_event(
                        current_event_video_frames,
                        current_event_stack_frames,
                        current_event_bbox,
                        current_event_astro_data,
                        is_timeout_end,
                        track=current_event_track,
                    )
                    current_event_track = []

                    # RESET STATE FOR NEXT EVENT
                    if footprint_id is not None:
                        self.event_footprints.extend(footprint_id, ts)  # include the closing frame
                        self.event_footprints.close(footprint_id)
                        footprint_id = None
                    self.event_in_progress.clear()
                    current_event_video_frames = []
                    current_event_stack_frames = []
                    event_cooldown_counter = 0
                    current_event_frame_count = 0
                    current_event_bbox = None
                    current_event_astro_data = None
                    recent_frames.clear()

                    self.background = None
                    self.background_reset_time = time.monotonic()

                # Background learning: only from frames judged motion-free, outside events,
                # and only into the model this frame was compared against (another thread
                # may have reset it meanwhile).
                if (
                    local_bg is not None
                    and (skip_motion_math or not detected)   # warm-up, or a motion-free frame
                    and not self.event_in_progress.is_set()
                    and self.background is local_bg
                ):
                    cv2.accumulateWeighted(gray.astype(np.float32), local_bg, current_alpha)

                # Timelapse clip protection bookkeeping for THIS frame. Every frame of
                # a still-open event (detections, cooldown and light-shift frames alike)
                # extends the footprint in time -- a meteor's persistent train can
                # linger after motion stops; frames with motion also grow its bbox.
                if footprint_id is not None and self.event_in_progress.is_set():
                    self.event_footprints.extend(
                        footprint_id, ts, frame_bbox if detected else None
                    )
                self.event_footprints.mark_evaluated(ts)

                # CALC PROCESS TIME
                try:
                    process_end = time.perf_counter()
                    duration_ms = (process_end - process_start) * 1000.0

                    if local_loop_ema == 0.0:
                        local_loop_ema = duration_ms  # first frame seeds the average
                    else:
                        a_ema = self.cfg["general"]["loop_time_ema_alpha"]
                        local_loop_ema = (a_ema * duration_ms) + ((1.0 - a_ema) * local_loop_ema)

                    with self.status_lock:
                        self.session_stats["loop_time_ms"] = local_loop_ema
                except Exception as e:
                    logging.error(f"Detection Timer Math Error: {e}")

        except Exception as e:
            logging.critical("CRITICAL FATAL ERROR IN DETECTION THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "DETECTION_CRASH", str(e))

    # -----------------
    def _background_warmup_sec(self):
        """
        How long motion detection must wait after a background reset.

        After a reset the background is a single frame, so the difference image
        carries its noise twice: sigma_diff^2 = sigma^2 * (1 + (1 - alpha)^(2k))
        after k accumulateWeighted updates with rate alpha (the transient term of
        the EMA started from one frame). Detection resumes once sigma_diff is within
        a fraction eps (detection.background_warmup_noise_excess) of its steady
        state sigma:
            (1 - alpha)^(2k) <= (1 + eps)^2 - 1
            k >= ln((1 + eps)^2 - 1) / (2 ln(1 - alpha))
        converted to time with the real frame period. alpha is the live dynamic
        accumulate alpha, so the wait is correct at every exposure. (The old fixed
        5 s was ~25 frames at a 0.2 s exposure but less than ONE frame at 6 s,
        i.e. the next frame was compared against a single-frame background.)
        """
        alpha = self._get_dynamic_accumulate_alpha()
        eps = self.cfg["detection"]["background_warmup_noise_excess"]
        target = (1.0 + eps) ** 2 - 1.0
        if alpha >= 1.0 or target >= 1.0:
            return 0.0
        k = math.ceil(math.log(target) / (2.0 * math.log(1.0 - alpha)))
        return max(k, 0) * self._frame_period_sec()

    # -----------------
    def _dispatch_event(self, video_frames, stack_frames, event_bbox, astro_data, is_timeout_end, track=None):
        """
        Hand a finished event to the downstream pipeline: nightly masterpiece update,
        then the video writer (event_q) and the stacker/classifier/logger
        (event_stack_q). The ONE exit path for every event, however it ended --
        natural cooldown, timeout, RAM kill, or a daylight/weather hold engaging
        mid-event -- so no ending can silently discard captured data.
        """
        # DIVINE MASTERPIECE LOGIC
        if len(stack_frames) > 0:
            try:
                # We use the first frame of the stack as a representative sample
                sample_frame = stack_frames[0][1]
                gray_sample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                current_brightness = np.mean(gray_sample)

                # If the sky is brighter than 3x your target, it is likely dawn.
                # Reaching this limit means we stop adding to the Masterpiece for the night.
                target = self.cfg["capture"].get("target_sky_brightness")
                burn_limit = target * self.cfg["capture"].get(
                    "masterpiece_burn_multiplier"
                )

                if current_brightness > burn_limit:
                    logging.warning(
                        f"Masterpiece: Transition to Dawn detected ({current_brightness:.1f} > {burn_limit}). Freezing canvas to prevent overexposure."
                    )
                else:
                    frames_for_mip = [
                        f[1] for f in stack_frames
                    ]
                    event_mip = np.maximum.reduce(frames_for_mip)

                    if self.nightly_masterpiece is None:
                        self.nightly_masterpiece = event_mip.copy()
                    else:
                        self.nightly_masterpiece = np.maximum(
                            self.nightly_masterpiece, event_mip
                        )

                    temp_mip_path = self.mip_path.replace(".png", ".tmp.png")

                    if self._safe_save_image(
                        temp_mip_path, self.nightly_masterpiece
                    ):
                        os.replace(temp_mip_path, self.mip_path)
                    logging.debug(
                        "Nightly Masterpiece updated with new cosmic path."
                    )
            except Exception as e:
                logging.error(f"Failed to update Masterpiece: {e}")

        # DISPATCH TO QUEUES
        if (
            len(video_frames) > 0
            and len(stack_frames) > 0
        ):
            try:
                # Always built: besides the CSV row it carries what the stacker needs
                # (bbox, astrometry, track, exposure). Whether it is LOGGED is decided
                # by event_log.enabled at the logging step (event.py).
                log_summary = {}
                start_dt = datetime.fromisoformat(
                    video_frames[0][0]
                )
                end_dt = datetime.fromisoformat(
                    video_frames[-1][0]
                )
                tstamp = end_dt.strftime("%Y%m%dT%H%M%S")

                log_summary = {
                    "timestamp_local": end_dt.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "start_time_local": start_dt.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "duration_sec": (end_dt - start_dt).total_seconds(),
                    "num_frames_video": len(video_frames),
                    "num_frames_stack": len(stack_frames),
                    "filename": f"event_{tstamp}",
                    "tstamp": tstamp,
                    "event_bbox": event_bbox,
                    "astro_data": astro_data,
                    # For the satellite matcher (event.py): per-frame motion boxes
                    # and the exposure the frames were taken with.
                    "track": list(track or []),
                    "exposure_us": self.current_exposure_us,
                }

                self.event_q.put_nowait(video_frames)
                self.event_stack_q.put_nowait(
                    (
                        stack_frames,
                        is_timeout_end,
                        log_summary,
                    )
                )

            except queue.Full:
                logging.warning(
                    "Event queue is FULL! Data dropped to protect RAM."
                )
                self.log_health_event(
                    "WARNING", "QUEUE_FULL", "Event dropped to protect RAM."
                )

    def _calculate_event_score(self, stacked_image, ai_label, ai_conf):
        """
        Generates a continuous confidence score (0.0 to 1.0) for post-processing.
        Combines AI Classification, Trajectory Linearity (Aspect Ratio),
        Path Consistency (Solidity), and Brightness Profile.
        """
        # 1. Base AI Score (Inverse penalty if it's not a meteor)
        # ai_score = ai_conf if ai_label == "meteor" else (1.0 - ai_conf)
        # ai_score = ai_conf if "meteor" in ai_label.lower() else (1.0 - ai_conf)
        ai_score = ai_conf if ai_label.lower() == "meteor" else (1.0 - ai_conf)
        det = self.cfg["detection"]

        try:
            # Prepare image for geometric analysis
            gray = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)

            # 2. Brightness Profile
            # Maximum intensity on the stacked image (how bright was the meteor?)
            brightness_score = min(np.max(gray) / 255.0, 1.0)

            # 3. Thresholding for Shape Analysis
            # A high percentile of the stack isolates the bright streak from the sky.
            thresh_val = np.percentile(gray, det["event_score_trail_percentile"])
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            # Opening with the smallest symmetric structuring element (3x3): removes
            # 1-px speckle (hot pixels, noise) without eroding a trail only a few px
            # wide. A structural minimum, not a tunable -- a larger element would
            # start deleting faint meteors.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            cnts, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not cnts:
                # If no clear streak remains, heavily penalize the geometric score
                w_ai = det["event_score_fallback_w_ai"]
                return round((ai_score * w_ai) + (brightness_score * (1.0 - w_ai)), 3)

            # Find the largest object (presumably the meteor streak)
            c = max(cnts, key=cv2.contourArea)

            # 4. Trajectory Linearity (Aspect Ratio)
            # minAreaRect calculates the width/height of diagonal streaks perfectly
            rect = cv2.minAreaRect(c)
            cx, cy = rect[0]  # Extract center coordinates
            w, h = rect[1]
            if min(w, h) == 0:
                aspect_ratio = 1.0
            else:
                aspect_ratio = max(w, h) / min(w, h)

            # Elongation normalised so event_score_aspect_norm:1 scores 1.0 (meteors are typically > 5:1).
            aspect_score = min(aspect_ratio / det["event_score_aspect_norm"], 1.0)

            # 5. Path Consistency (Solidity)
            # Area / Convex Hull Area. A straight line is very solid (~0.8 - 1.0).
            # A fluttering moth creates a jagged, hollow shape (~0.2 - 0.5).
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity_score = (area / hull_area) if hull_area > 0 else 0.0

            # FINAL COMPOSITE MATH
            # Convex combination (weights validated to sum to 1).
            composite = (
                (ai_score * det["event_score_w_ai"])
                + (aspect_score * det["event_score_w_aspect"])
                + (solidity_score * det["event_score_w_solidity"])
                + (brightness_score * det["event_score_w_brightness"])
            )

            return round(float(composite), 3)

        except Exception as e:
            logging.error(f"Event score calculation failed: {e}")
            return round(float(ai_score), 3)  # Fallback to pure AI score
