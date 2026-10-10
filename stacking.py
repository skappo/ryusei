"""
stacking_mixin.py

Frame-stacking methods (mean-stack combination, sigma-clipping, ECC
alignment) for the Ryusei pipeline, extracted from the main Pipeline
class as a pure move -- method bodies are unchanged, only their physical
location. Composed into Pipeline via multiple inheritance.

Deliberately kept separate from sky_monitor_mixin.py despite being
physically interleaved with it in the original file -- see that file's
docstring for the same discipline applied in reverse.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls
self._safe_save_image, self.publish_debug, self.detect_stars (defined
in SkyMonitorClass, extracted last turn), and self.ping_watchdog (defined
in ConcurrencyClass). Must be provided somewhere in the composed Pipeline
class.

Imports here are exactly what this extracted block actually uses --
verified by grepping for module.method/module.attribute patterns AND by
running pyflakes before finalizing, per the methodology correction from
the previous extraction (regex search alone missed a real dependency
there).
"""
import os
import gc
import logging
import time
import cv2
import numpy as np
import psutil
from datetime import datetime

from enums import SkyConditionStatus
from event_footprints import as_datetime
from constants import MAD_TO_SIGMA, REC601_LUMA_BGR


class StackingClass:
    """Frame-stacking and alignment methods, composed into Pipeline."""

    # Gaussian consistency constant (constants.py): a property of the normal
    # distribution, not a tunable.
    _MAD_TO_SIGMA = MAD_TO_SIGMA

    @staticmethod
    def _valid_region(warp, y0, y1, w, h):
        """
        Bool (rows, w) mask of destination pixels [y0, y1) that a frame warped with
        cv2.warpAffine(..., INTER_NEAREST + WARP_INVERSE_MAP) actually covers:
        dst(q) = src(M q), valid iff the nearest source pixel lies inside the frame.
        Computed analytically per tile, so no full-frame (H, W) float mask per frame
        has to be kept for the whole stack (that was N x H x W x 4 bytes: ~166 MB
        for 20 frames at 1920x1080 on a Pi). warp None = identity (all valid).
        """
        if warp is None:
            return np.ones((y1 - y0, w), dtype=bool)
        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        sx = warp[0, 0] * xs + warp[0, 1] * ys + warp[0, 2]
        sy = warp[1, 0] * xs + warp[1, 1] * ys + warp[1, 2]
        return (sx > -0.5) & (sx < w - 0.5) & (sy > -0.5) & (sy < h - 0.5)

    # -----------------
    @staticmethod
    def _map_rect_to_aligned(rect, warp, margin, w, h):
        """
        Axis-aligned rectangle (x0, y0, x1, y1, exclusive end) in a frame's ORIGINAL
        pixel coordinates -> its bounding rectangle in the ALIGNED (reference) frame,
        dilated by `margin`, clipped to the image. With WARP_INVERSE_MAP a source
        point p satisfies p = A q + t, so q = A^-1 (p - t); the four corners are
        mapped and their bounding box taken (a superset of the rotated rectangle).
        """
        x0, y0, x1, y1 = rect
        if warp is not None:
            A = warp[:, :2].astype(np.float64)
            t = warp[:, 2].astype(np.float64)
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)
            q = (corners - t) @ np.linalg.inv(A).T
            x0, y0 = q.min(axis=0)
            x1, y1 = q.max(axis=0)
        return (
            int(max(0, np.floor(x0 - margin))),
            int(max(0, np.floor(y0 - margin))),
            int(min(w, np.ceil(x1 + margin))),
            int(min(h, np.ceil(y1 + margin))),
        )

    # -----------------
    def _sigma_clip_tile_rows(self, n, w):
        """
        Largest tile height (rows) whose working set fits the configured fraction of
        the RAM available right now. Derived from dtype sizes of what one tile row
        really allocates -- no empirical multiplier:
          per (pixel, channel, frame): stack f32 (4) + deviation/weight f32 (4) + keep bool (1)
          per (pixel, frame):          valid bool (1) + protect bool (1)
          per (pixel, channel):        median, sigma, k*sigma, weight sum, numerator,
                                       fallback, safe divisor, quotient, final np.where
                                       result, per-frame uint8->f32 cast: 10 x f32 (40)
          per pixel:                   x/y grid + mapped source x/y for the valid region: 4 x f32 (16)
        np.median runs with overwrite_input=True on the deviation buffer, so it adds
        no hidden full-size copy.
        """
        bytes_per_row = w * (3 * n * (4 + 4 + 1) + n * (1 + 1) + 3 * 10 * 4 + 4 * 4)
        budget = psutil.virtual_memory().available * self.cfg["timelapse"]["stack_sigma_clip_ram_fraction"]
        return int(budget // bytes_per_row), bytes_per_row

    # -----------------
    def _combine_mean_stack(self, frames_list, weights_list, warps_list, h, w, protect_rects=None):
        """
        Combines already-aligned uint8 frames into one float32 weighted mean, with
        per-pixel temporal sigma-clipping when enabled.

        SCOPE: 'mean' (timelapse / calibration) stacks only. Never MIP/event stacks:
        there the transient IS the signal.

        CLIPPING (robust, tiled): per pixel and channel, median and MAD over the N
        frames; samples further than k * max(MAD * 1.4826, floor) from the median
        are rejected (cosmic rays, hot pixels, noise spikes). The frame is processed
        in horizontal tiles sized from currently available RAM
        (_sigma_clip_tile_rows), so peak memory is a computed quantity instead of
        a whole-stack (N, H, W, 3) allocation that could swap-thrash the Pi.

        TRANSIENT PROTECTION: protect_rects[i] lists rectangles (aligned coords)
        where frame i overlaps a transient the detector saw (event_footprints.py).
        Samples there are never rejected, so a meteor or plane crossing during the
        stack appears in the timelapse exactly as it does in the event stack --
        at 1/N weight, i.e. what the sky really did -- while clipping still cleans
        everything else. This is what allows stack_sigma_clip_enabled to be on.

        ALIGNMENT BORDERS: a pixel not covered by every frame (warp edge strip)
        is excluded from rejection and falls back to the weighted mean over the
        frames that do cover it: the median of a partly-empty sample is meaningless.

        Falls back to a plain weighted mean if clipping is disabled, there are
        fewer than stack_sigma_clip_min_frames frames, or not even one tile row
        fits the RAM budget.
        """
        cfg = self.cfg["timelapse"]
        n = len(frames_list)
        if protect_rects is None:
            protect_rects = [()] * n

        rows = 0
        if cfg["stack_sigma_clip_enabled"]:
            if n < cfg["stack_sigma_clip_min_frames"]:
                logging.info(
                    f"Stack sigma-clip skipped (too_few_frames): frames={n} < "
                    f"{cfg['stack_sigma_clip_min_frames']}. Plain weighted mean."
                )
            else:
                rows, bytes_per_row = self._sigma_clip_tile_rows(n, w)
                if rows < 1:
                    logging.warning(
                        f"Stack sigma-clip skipped (insufficient_ram): one tile row needs "
                        f"{bytes_per_row / 1e6:.2f}MB, budget "
                        f"{psutil.virtual_memory().available * cfg['stack_sigma_clip_ram_fraction'] / 1e6:.1f}MB. "
                        f"Plain weighted mean."
                    )
        if rows >= 1:
            return self._sigma_clip_tiled(
                frames_list, weights_list, warps_list, h, w, protect_rects, min(rows, h)
            )

        # --- Plain weighted mean (no rejection); one frame's valid mask at a time ---
        accumulator = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        for frame, weight, warp in zip(frames_list, weights_list, warps_list):
            valid = self._valid_region(warp, 0, h, w, h).astype(np.float32) * np.float32(weight)
            accumulator += frame.astype(np.float32) * valid[..., None]
            weight_map += valid
        safe_weight = np.where(weight_map > 0, weight_map, 1.0)
        stacked_float = accumulator / safe_weight[..., None]
        return np.where(weight_map[..., None] > 0, stacked_float, 0.0).astype(np.float32)

    # -----------------
    def _sigma_clip_tiled(self, frames_list, weights_list, warps_list, h, w, protect_rects, rows):
        cfg = self.cfg["timelapse"]
        n = len(frames_list)
        k = np.float32(cfg["stack_sigma_clip_k"])
        floor = np.float32(cfg["stack_sigma_clip_floor"])
        wts = np.asarray(weights_list, dtype=np.float32)  # (N,)

        # Frame axis LAST: np.median partitions along a contiguous axis (cache-friendly
        # on the Pi's small caches); buffers are allocated once and reused per tile.
        st = np.empty((rows, w, 3, n), dtype=np.float32)
        dv = np.empty_like(st)
        keep = np.empty(st.shape, dtype=bool)
        vt = np.empty((rows, w, n), dtype=bool)
        pt = np.empty((rows, w, n), dtype=bool)
        out = np.empty((h, w, 3), dtype=np.float32)

        rejected = 0
        rescued = 0
        samples = 0
        for y0 in range(0, h, rows):
            self.ping_watchdog()
            y1 = min(y0 + rows, h)
            r = y1 - y0
            s, d, kp, v, p = st[:r], dv[:r], keep[:r], vt[:r], pt[:r]

            p[:] = False
            for i in range(n):
                s[..., i] = frames_list[i][y0:y1]
                v[..., i] = self._valid_region(warps_list[i], y0, y1, w, h)
                for (rx0, ry0, rx1, ry1) in protect_rects[i]:
                    a, b = max(ry0, y0), min(ry1, y1)
                    if a < b:
                        p[a - y0:b - y0, rx0:rx1, i] = True
            eligible = v.all(axis=-1)  # (r, W): covered by every frame

            # Robust location and scale; the deviation buffer doubles as the
            # median's in-place scratch space, then is recomputed (cheap arithmetic,
            # no extra full-size buffer).
            np.copyto(d, s)
            med = np.median(d, axis=-1, overwrite_input=True)             # (r, W, 3)
            np.subtract(s, med[..., None], out=d)
            np.abs(d, out=d)
            sigma = np.median(d, axis=-1, overwrite_input=True)           # MAD, (r, W, 3)
            sigma *= np.float32(self._MAD_TO_SIGMA)
            np.maximum(sigma, floor, out=sigma)
            np.subtract(s, med[..., None], out=d)
            np.abs(d, out=d)

            np.less_equal(d, (k * sigma)[..., None], out=kp)              # True = inlier
            # Statistics via per-pixel reductions: no extra full-size bool temporaries.
            n_elig = int(np.count_nonzero(eligible))
            inliers = np.count_nonzero(kp, axis=(2, 3))                   # (r, W)
            rejected += 3 * n * n_elig - int(inliers[eligible].sum())
            samples += 3 * n * n_elig

            kp |= p[:, :, None, :]        # transients seen by the detector are never rejected
            rescued += int((np.count_nonzero(kp, axis=(2, 3)) - inliers)[eligible].sum())
            kp |= ~eligible[:, :, None, None]  # border pixels: no rejection, weighted mean only
            kp &= v[:, :, None, :]        # uncovered samples carry no weight

            np.multiply(kp, wts, out=d)   # d now = per-sample weight
            wsum = d.sum(axis=-1)
            np.multiply(s, d, out=s)
            num = s.sum(axis=-1)
            fallback = np.where(eligible[..., None], med, np.float32(0.0))
            out[y0:y1] = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), fallback)

        n_rects = sum(len(r_) for r_ in protect_rects)
        logging.info(
            f"Stack sigma-clip applied: k={float(k)}, frames={n}, tile_rows={rows}, "
            f"rejected_fraction={rejected / max(samples, 1):.5f}, "
            f"protected_rects={n_rects}, rescued_samples={rescued}"
        )
        del st, dv, keep, vt, pt
        gc.collect()
        return out

    # -----------------
    def _event_protect_rects(self, timestamps, warps, h, w):
        """
        Per-frame protected rectangles (aligned coordinates) from the detector's
        event footprints.

        First waits until the detector can no longer attribute any of these frames
        to an event (EventFootprintRegistry.wait_covered: last frame + the pre-event
        lookback evaluated). The wait is heartbeat-aware and bounded by the time
        that lookback physically takes -- (lookback + 1) frames at the current
        exposure -- plus stack_event_wait_slack_sec; if the detector has stopped
        (end of session, daylight), it simply proceeds with what is registered.

        Margin: the detector's bbox comes from a motion mask produced after a
        Gaussian blur of blur_size and a morphological close of morph_size, so the
        object's real footprint can extend up to that radius beyond it; both sizes
        are the resolution-aware values the detector itself runs with.
        """
        reg = self.event_footprints
        frame_period_sec = self.current_exposure_us / 1_000_000.0
        timeout = (reg.lookback_frames + 1) * frame_period_sec + self.cfg["timelapse"]["stack_event_wait_slack_sec"]
        ts_dt = [as_datetime(t) for t in timestamps]
        last_ts = max(ts_dt)
        if not self.wait_with_heartbeat(lambda s: reg.wait_covered(last_ts, s), timeout):
            logging.info(
                f"Stack protection: detector did not cover the lookback within {timeout:.1f}s "
                f"(stopped or paused); using the footprints registered so far."
            )
        events = reg.footprints(min(ts_dt), last_ts)
        if not events:
            return None

        margin = self.blur_size + self.morph_size
        rects = []
        for t, warp in zip(ts_dt, warps):
            rects.append([
                self._map_rect_to_aligned(bbox, warp, margin, w, h)
                for (t0, t1, bbox) in events
                if t0 <= t <= t1
            ])
        logging.info(
            f"Stack protection: {len(events)} event footprint(s) overlap this stack; "
            f"{sum(1 for r in rects if r)} frame(s) carry protected regions (margin {margin}px)."
        )
        return rects

    # -----------------
    def stack_frames(self, frames, align=True, method="mean", min_features_for_alignment=None,
                     is_event_stack=False, frame_timestamps=None):
        """
        A method to align and stack a list of frames.
        It uses an iterative approach for 'mean' methods.

        frame_timestamps (optional, same length/order as frames): the capture ts of
        each frame. When given for a 'mean' stack with sigma-clipping enabled, the
        detector's event footprints for these frames are protected from rejection
        (see _combine_mean_stack and event_footprints.py). Without them (e.g. the
        calibration shot), clipping runs unprotected.

        min_features_for_alignment: None = timelapse.min_features_for_alignment.
        """
        if min_features_for_alignment is None:
            min_features_for_alignment = self.cfg["timelapse"]["min_features_for_alignment"]
        if not frames:
            return None, "aborted", 0

#        stack_dump_dir = None
        stack_dump_prefix = None
        debug_mode = self.cfg["general"].get("debug_level") >= 2

        # Setup disk dumping if Level = 2
        if debug_mode:
            timestamp = datetime.now().strftime("%H%M%S")
            stack_type = "event" if is_event_stack else "timelapse"
            stack_dump_prefix = f"stack_{stack_type}_{timestamp}"

        try:
            # Basic frame parameters
            h, w = frames[0].shape[:2]
            half_stack = len(frames) // 2
            first_frame_gray = cv2.cvtColor(frames[half_stack], cv2.COLOR_BGR2GRAY)

            effective_method = method

            # 4. PRE-ALIGNMENT CHECKS & INITIALIZATION
            perform_alignment = align

            # MIP event stacks are never aligned. The whole point of MIP is to
            # preserve whatever moving/transient object triggered the event --
            # and that object is exactly what breaks star-based ECC correlation
            # on the frames it actually appears in. Confirmed live: every MIP
            # event across a full night's log had findTransformECC crash
            # ("Iterations do not converge... images may be uncorrelated or
            # non-overlapped") on frames clustered right around where the
            # trigger fired, discarding precisely the frames the stack exists
            # to capture -- resulting in a PNG of empty sky next to a video
            # that clearly shows the object. An event's span is short enough
            # that the star field doesn't drift enough to need correcting
            # anyway: TimelapseThread's own alignment stats over much longer
            # (~minutes) gaps between regular stacks sit at cc_min ~0.97-0.99
            # all night, on the same sky.
            if method == "mip":
                perform_alignment = False
                # logging.info(
                    # "Stack MIP mode | alignment skipped by design (event span "
                    # "too short for background drift; the moving object is what "
                    # "would break ECC) | frames=%d",
                    # len(frames),
                # )

            star_mask = None
            bias_mode = False
            bias_reasons = []
            anchors_used = 0

            # UNCONDITIONAL INITIALIZATION ensures safety if mask logic is skipped
            star_meta = {"stddev": 0.0, "anchor_points": 0}

            if perform_alignment:
                # STAR MASK CREATION
                if (
                    self.last_sky_status != SkyConditionStatus.CLOUDY
                    and not self.daylight_mode_active.is_set()
                ):
                    # stats = self.detect_stars(first_frame_gray, build_mask=True, save_debug=debug_mode, debug_out_dir=stack_dump_dir)
                    stats = self.detect_stars(first_frame_gray, build_mask=True, save_debug=debug_mode, debug_out_dir=self.debug_out_dir)
                    star_mask = stats["mask"]
                    star_meta = {"stddev": stats["stddev"], "anchor_points": stats["stars"]}
                else:
                    reason = (
                        "CLOUDY"
                        if self.last_sky_status == SkyConditionStatus.CLOUDY
                        else "DAYLIGHT"
                    )
                    logging.info(f"Alignment skipped: SkyMonitor reports {reason}.")

                if star_mask is None:
                    bias_mode = True
                    perform_alignment = False
                    bias_reasons.append("no_star_mask")
                else:
                    anchors_used = star_meta.get("anchor_points", 0)

                    if anchors_used < min_features_for_alignment:
                        bias_mode = True
                        perform_alignment = False
                        bias_reasons.append(f"low_anchor_count:{anchors_used}")

                    elif star_meta.get("stddev", 0) < self.alignment_contrast_floor:
                        bias_mode = True
                        perform_alignment = False
                        bias_reasons.append(
                            f"low_contrast:{star_meta.get('stddev',0):.2f}"
                        )

                    else:
                        bias_mode = False
                        perform_alignment = True

                # Ensure there is enough physical pixel area for the matrix math to converge
                min_pixel_area = min_features_for_alignment * self.cfg["timelapse"]["alignment_min_pixels_per_feature"]
                if star_mask is None or cv2.countNonZero(star_mask) < min_pixel_area:
                    perform_alignment = False

                self.publish_debug(
                    "stack_bias",
                    {
                        "enabled": bias_mode,
                        "reasons": bias_reasons,
                        "anchors": anchors_used,
                        "stddev": star_meta.get("stddev", 0),
                        "frames": len(frames),
                    },
                )

                if bias_mode:
                    logging.warning(
                        "Stack BIAS mode | reasons=%s | anchors=%d | stddev=%.2f | frames=%d",
                        ",".join(bias_reasons),
                        anchors_used,
                        star_meta.get("stddev", 0),
                        len(frames),
                    )
                else:
                    logging.info(
                        "Stack NORMAL mode | anchors=%d | stddev=%.2f | frames=%d",
                        anchors_used,
                        star_meta.get("stddev", 0),
                        len(frames),
                    )

                with self.status_lock:
                    if "bias_history" in self.session_stats:
                        self.session_stats["bias_history"].append(
                            {"t": time.time(), "bias": bias_mode}
                        )

            # If alignment is manually disabled in config, ensure bias fallback state is used
            if not align:
                bias_mode = True
                bias_reasons.append("alignment_disabled_in_config")

            used_method = "bias mode" if bias_mode else effective_method
            brightness_boost = self.cfg["timelapse"]["stack_brightness_boost"]

            # 5. ALIGNMENT & ACCUMULATION LOOP
            cc_values = []
            weight_values = []

            # Loop is now un-indented so it executes even if perform_alignment is False!
            # mean for stack - mip for event
            if effective_method in ["mean", "mip"]:
                # MEAN needs the full per-frame stack retained so sigma-clipping can
                # compute per-pixel robust statistics across it (see
                # _combine_mean_stack). MIP only ever needs a running max, which is
                # associative -- no need to retain every frame, keeping the event
                # stacker's memory footprint small. NOTE: sigma-clipping is
                # deliberately never applied to MIP -- see _combine_mean_stack docstring.
                # mean_warps: per-frame warp matrix (None = identity). Valid regions and
                # protected rectangles are derived from it analytically at combine time,
                # instead of storing an (H, W) float mask per frame.
                mean_frames, mean_weights, mean_warps, mean_idx = [], [], [], []
                accumulator = np.zeros((h, w, 3), dtype=np.float32)  # MIP running max only
                best_luma = np.zeros((h, w), dtype=np.float32)
                actual_weight = 0.0
                actual_frames_stacked = 0

                ref_weight = self.cfg["timelapse"]["stack_bias_ref_weight"] if bias_mode else 1.0

                # Add reference frame
                if effective_method == "mean":
                    mean_frames.append(frames[half_stack])
                    mean_weights.append(ref_weight)
                    mean_warps.append(None)
                    mean_idx.append(half_stack)
                else:
                    # MIP: Just keep the absolute brightest pixel
                    # accumulator = np.maximum(accumulator, frames[half_stack].astype(np.float32))
                    ref_frame_f = frames[half_stack].astype(np.float32)
                    accumulator = ref_frame_f.copy()
                    # OpenCV is BGR: Blue=0, Green=1, Red=2
                    best_luma = (REC601_LUMA_BGR[0] * ref_frame_f[..., 0] +
                                 REC601_LUMA_BGR[1] * ref_frame_f[..., 1] +
                                 REC601_LUMA_BGR[2] * ref_frame_f[..., 2])

                actual_weight += ref_weight
                actual_frames_stacked += 1

                for i in range(len(frames)):
                    # Per-frame alignment (ECC) plus, at debug_level >= 3, a synchronous
                    # disk write + ZMQ publish of an alignment mosaic for every frame --
                    # on slow storage this can push a single stack well past
                    # watchdog_hung_thread_sec with no other checkpoint in between,
                    # making a merely slow stack look like a hung TimelapseThread.
                    self.ping_watchdog()

                    if i == half_stack:
                        continue

                    frame_to_add = frames[i]
                    should_add = True
                    weight = self.cfg["timelapse"]["stack_bias_frame_weight"] if bias_mode else 1.0
                    alignment_meta = {
                        "frame": i,
                        "success": False,
                        "cc": 0.0,
                        "warp_norm": 0.0,
                    }

                    # Initialize fresh matrix at top of loop
                    warp_matrix = np.eye(2, 3, dtype=np.float32)

                    if perform_alignment:
                        try:
                            gray = cv2.cvtColor(frame_to_add, cv2.COLOR_BGR2GRAY)

                            warp_mode = (cv2.MOTION_TRANSLATION if bias_mode else cv2.MOTION_EUCLIDEAN)
                            tl_ecc = self.cfg["timelapse"]
                            criteria = (
                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                tl_ecc["ecc_max_iterations"],
                                tl_ecc["ecc_epsilon"],
                            )

                            k = max(3, self.star_mask_radius * 2 - 1)
                            if k % 2 == 0:
                                k += 1
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                            ref_prep = cv2.morphologyEx(first_frame_gray, cv2.MORPH_TOPHAT, kernel)
                            cur_prep = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

                            ref_ecc = cv2.normalize(ref_prep, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
                            cur_ecc = cv2.normalize(cur_prep, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
                            gauss_size = ((self.ecc_gauss_size + 2) if bias_mode else self.ecc_gauss_size)  # next odd kernel (+2 keeps it odd)

                            cc, warp_matrix = cv2.findTransformECC(
                                ref_ecc,
                                cur_ecc,
                                warp_matrix,
                                warp_mode,
                                criteria,
                                inputMask=star_mask,
                                gaussFiltSize=gauss_size,
                            )

                            warp_diff = warp_matrix - np.eye(2, 3, dtype=np.float32)
                            warp_norm = float(np.linalg.norm(warp_diff))
                            # MAX_WARP_PIXELS = min(w, h) * self.max_warp_factor

                            if warp_norm > min(w, h) * self.max_warp_factor:
                                logging.warning(f"Frame {i}: warp too large ({warp_norm:.1f}px), rejecting.")
                                should_add = False
                                alignment_meta["error"] = "excessive_warp"

                            alignment_meta.update({"success": True, "cc": float(cc), "warp_norm": warp_norm})

                            tl_cfg = self.cfg["timelapse"]
                            cc_floor = tl_cfg["cc_floor_bias"] if bias_mode else tl_cfg["cc_floor_normal"]
                            cc_perfect = tl_cfg["cc_perfect_bias"] if bias_mode else tl_cfg["cc_perfect_normal"]

                            if cc < cc_floor:
                                logging.warning(
                                    f"Frame {i}: Correlation too low ({cc:.3f}), rejecting."
                                )
                                should_add = False
                                alignment_meta["error"] = "low_correlation"
                                weight = 0.0
                            elif cc < cc_perfect:
                                weight_floor = tl_cfg["weight_floor_bias"] if bias_mode else tl_cfg["weight_floor_normal"]
                                weight = weight_floor + (1.0 - weight_floor) * (
                                    (cc - cc_floor) / (cc_perfect - cc_floor)
                                )
                            else:
                                weight = 1.0

                            if should_add:
                                frame_to_add = cv2.warpAffine(
                                    frame_to_add,
                                    warp_matrix,
                                    (w, h),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                )

                            cc_values.append(cc)
                            weight_values.append(weight)

                            # OLD BEHAVIOR (dumped every single frame): writing a JPEG
                            # to disk + ZMQ-publishing a mosaic for all ~20 frames of a
                            # stack was heavy enough on slow storage to saturate it,
                            # making the stack itself (and anything else writing to the
                            # same drive, like the dashboard's config save) look hung to
                            # the watchdog. Kept here, commented, in case the stride
                            # ever needs to go back to dumping every frame.
                            # if self.cfg["general"].get("debug_level") >= 3:
                            #     aligned_gray = cv2.cvtColor(frame_to_add, cv2.COLOR_BGR2GRAY)
                            #     diff = cv2.absdiff(first_frame_gray, aligned_gray)
                            #     mosaic = self.create_alignment_mosaic(first_frame_gray, gray, aligned_gray, diff)
                            #
                            #     if mosaic is not None:
                            #         # if stack_dump_dir:
                            #         self._safe_save_image(os.path.join(self.debug_out_dir,f"{stack_dump_prefix}_align_{i:03d}_cc{int(cc*100)}.jpg",), mosaic,)
                            #         self.publish_debug("alignment_mosaic", {"frame_index": i, "correlation": float(cc)}, image=mosaic)

                            dump_stride = self.cfg["general"]["debug_align_dump_stride"]
                            if self.cfg["general"].get("debug_level") >= 3 and i % dump_stride == 0:
                                aligned_gray = cv2.cvtColor(frame_to_add, cv2.COLOR_BGR2GRAY)
                                diff = cv2.absdiff(first_frame_gray, aligned_gray)
                                mosaic = self.create_alignment_mosaic(first_frame_gray, gray, aligned_gray, diff)

                                if mosaic is not None:
                                    self._safe_save_image(os.path.join(self.debug_out_dir,f"{stack_dump_prefix}_align_{i:03d}_cc{int(cc*100)}.jpg",), mosaic,)
                                    self.publish_debug("alignment_mosaic", {"frame_index": i, "correlation": float(cc)}, image=mosaic)

                        except cv2.error as e:
                            logging.warning(f"Frame {i}: ECC Alignment crashed. {e}")
                            should_add = False
                            alignment_meta["error"] = "cv2_error"
                            # (A former "weight = 0.8 if bias_mode" here was dead: the
                            # frame is not added, so its weight is never read.)
                            # Failsafe reset if corrupted during try block
                            warp_matrix = np.eye(2, 3, dtype=np.float32)

                    self.publish_debug("alignment", alignment_meta)

                    if should_add:
                        weight = np.clip(weight, 0.0, 1.0)

                        if effective_method == "mean":
                            mean_frames.append(frame_to_add)
                            mean_weights.append(weight)
                            mean_warps.append(warp_matrix.copy() if perform_alignment else None)
                            mean_idx.append(i)
                        elif effective_method == "mip":
                            ones = np.ones((h, w), dtype=np.float32)
                            if perform_alignment:
                                # Safely apply the validated warp_matrix
                                valid_mask = cv2.warpAffine(ones, warp_matrix, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
                            else:
                                valid_mask = ones
                            # MIP: Mask out the black edges from the warp, then take the maximum
                            # Track best luma seen and the FULL RGB triplet that produced it,
                            # so channels stay physically coherent (no cross-frame color mixing).
                            # 1. Apply the alignment mask to prevent edge artifacts
                            masked_frame = frame_to_add.astype(np.float32) * valid_mask[..., None].astype(np.float32)
                            # accumulator = np.maximum(accumulator, masked_frame)                          
                            # 2. Calculate Luma of the incoming frame
                            frame_luma = (REC601_LUMA_BGR[0] * masked_frame[..., 0] +
                                          REC601_LUMA_BGR[1] * masked_frame[..., 1] +
                                          REC601_LUMA_BGR[2] * masked_frame[..., 2])
                            # 3. Find exactly which pixels are brighter than our current best
                            update_mask = frame_luma > best_luma
                            # 4. Update the accumulator and the best_luma tracker simultaneously
                            accumulator[update_mask] = masked_frame[update_mask]
                            best_luma[update_mask] = frame_luma[update_mask]

                        actual_weight += weight
                        actual_frames_stacked += 1

                if actual_frames_stacked == 0 or actual_weight <= 0:
                    return None, "aborted", 0

                if effective_method == "mean":
                    protect_rects = None
                    if frame_timestamps is not None and self.cfg["timelapse"]["stack_sigma_clip_enabled"]:
                        protect_rects = self._event_protect_rects(
                            [frame_timestamps[i] for i in mean_idx], mean_warps, h, w
                        )
                    stacked_float = self._combine_mean_stack(
                        mean_frames, mean_weights, mean_warps, h, w, protect_rects
                    )
                else:
                    # MIP is already perfectly calculated, no division required!
                    stacked_float = accumulator

                logging.info(f"Stacked frames: {actual_frames_stacked}/{len(frames)}")

                # Inputs are uint8 (0-255) end to end, and stacked_float is a
                # weighted/clipped combination of them, so it is already bounded
                # to [0, 255] -- normalize on that fixed, known range rather than
                # a re-measured per-stack max (a re-measured max would make
                # brightness inconsistent frame-to-frame, since it'd rescale each
                # stack to its own peak instead of a common physical reference).
                final_float = stacked_float / 255.0
                final_float = np.clip(final_float, 0.0, 1.0).astype(np.float32)

                final_float = np.clip(final_float * brightness_boost, 0.0, 1.0)

                if cc_values:
                    logging.info(
                        "Stack align stats | bias=%s | cc_med=%.3f | cc_min=%.3f | weight_mean=%.2f",
                        bias_mode,
                        float(np.median(cc_values)),
                        float(np.min(cc_values)),
                        float(np.mean(weight_values)),
                    )

                info = {
                    "anchors": anchors_used,
                    "stddev": star_meta.get("stddev", 0),
                    "bias_reasons": bias_reasons if bias_mode else [],
                    "actual_frames_stacked": actual_frames_stacked,
                }

                final_uint8 = (final_float * 255).astype(np.uint8)
                return final_uint8, used_method, info

            else:
                return None, "aborted", 0

        except Exception as e:
            logging.error(f"Stacking Crash: {e}", exc_info=True)
            return None, "error", 0

    # -----------------
    def create_alignment_mosaic(self, ref, inp, aligned, diff):
        """Creates a 4-panel diagnostic image for alignment debugging."""
        try:
            h, w = ref.shape[:2]
            scale = 0.5

            # Helper to resize and convert to color if needed
            def prep(img, label):
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                res = cv2.resize(img, (int(w * scale), int(h * scale)))
                cv2.putText(
                    res, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                return res

            # Normalize diff for visualization (amplify differences)
            # diff_8 = (diff >> 8).astype(np.uint8)
            # diff_vis = cv2.normalize(diff_8, None, 0, 255, cv2.NORM_MINMAX)
            diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

            top = np.hstack((prep(ref, "Reference"), prep(inp, "Input")))
            bot = np.hstack((prep(aligned, "Aligned"), prep(diff_vis, "Diff Score")))
            return np.vstack((top, bot))
        except Exception:
            return None

    # -----------------
