"""
astrometry_mixin.py

    _get_tetra3_solver
    _extract_star_centroids
    _crop_center_roi
    assess_solve_readiness
    solve_field_center_roi
    get_event_astro_data

Plate-solving methods for the Ryusei pipeline, extracted from the main
Pipeline class as a pure move -- method bodies are unchanged, only their
physical location. Composed into Pipeline via multiple inheritance
(class Pipeline(AstrometryClass, ...)); self. resolution happens against
the fully composed instance at runtime, regardless of which mixin file a
given method physically lives in.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- these methods
call self._compute_roi_stats, self.calculate_lux, and self.get_moon_state,
none of which are defined in this file. They must be provided somewhere
else in the composed Pipeline class (currently still in the main class
body, pending their own future mixin extraction). This mixin cannot be
unit-tested in isolation without stubbing those three.

Imports here are exactly what this extracted block actually uses --
verified by grepping for module.method/module.attribute patterns before
writing this file.
"""
import os
import math
import logging
import cv2
import numpy as np
import tetra3
from datetime import datetime
from constants import MAD_TO_SIGMA, IQR_TO_SIGMA


class AstrometryClass:
    """Plate-solving methods, composed into Pipeline via multiple inheritance."""

    def _get_tetra3_solver(self):
        with self.astrometry_lock:
            if getattr(self, "_tetra3_instance", None) is None:
                db_name = self.cfg["astrometry"]["tetra3_database"]
                if not db_name.endswith('.npz'):
                    db_name += '.npz'
                
                db_dir = os.path.join(self.assets_dir, "db")
                os.makedirs(db_dir, exist_ok=True)
                
                db_path = os.path.join(db_dir, db_name)
                
                if not os.path.exists(db_path):
                    logging.error(f"Astrometry: Star database NOT FOUND at {db_path}!")
                    logging.error("Please run the tetra3 generation script to build the database.")
                    logging.error(
                        f"Astrometry Config Advice: Ensure that 2x 'target_half_fov_deg' ({2 * self.cfg['astrometry']['target_half_fov_deg']}°) "
                        f"does not exceed the 'max_fov' (65°) used in your build_db.py script, or patterns will not match."
                    )
                    return None
                    
                logging.info(f"Astrometry: loading Tetra3 star-pattern database from {db_path}")
                self._tetra3_instance = tetra3.Tetra3(load_database=db_path)

            return self._tetra3_instance
    # -----------------
    def _extract_star_centroids(self, gray_roi, bit_depth=8):
        """
        bit_depth: the MEANINGFUL bit depth of gray_roi's data -- 8 for the
        standard event_stack/timelapse path (0-255), 10 for the raw Bayer
        green-channel path (0-1023, even though it's stored in a uint16
        container). Passed explicitly by the caller rather than inferred
        from dtype: a uint16 CONTAINER doesn't tell you the DATA only uses
        10 of those bits, and guessing from gray_roi.max() would be fragile
        on a genuinely dark ROI. The caller always knows this unambiguously.
        """
        kernel_size = max(3, int(self.star_mask_radius * self.top_hat_factor))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray_roi, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

        th_median = float(np.median(tophat))
        th_mad = float(np.median(np.abs(tophat - th_median)))
        th_sigma = th_mad * MAD_TO_SIGMA

        if th_sigma <= 0.0:
            # MAD collapses to exactly zero whenever >50% of pixels share the
            # same discrete value -- routine on a dim, quantized tophat,
            # where the whole background compresses into a few integer
            # levels. MAD assumes an approximately continuous distribution;
            # heavy quantization breaks that assumption outright, not just
            # approximately.
            #
            # IQR-based spread only collapses if >75% of pixels share one
            # value -- a much harder condition to hit even on heavily
            # quantized dim data. 0.7413 is the standard IQR->sigma
            # consistency constant for Gaussian data, the direct IQR
            # analogue of MAD's 1.4826.
            p25, p75 = np.percentile(tophat, [25, 75])
            th_sigma = (p75 - p25) * IQR_TO_SIGMA
            logging.info(
                f"Astrometry: MAD collapsed to 0 (quantization on a dim frame) -- "
                f"fell back to IQR-based sigma={th_sigma:.3f}."
            )

        sigma_mult = self.cfg["timelapse"]["star_sigma_multiplier"]
        floor_8bit = self.cfg["timelapse"]["alignment_threshold_floor"]
        # Scale the configured floor (calibrated in 8-bit terms) to whatever
        # bit depth this specific call is actually working in, so a fixed
        # "3" doesn't become meaninglessly permissive against 10-bit data
        # (where it would represent under a quarter of the equivalent
        # significance it has at 8-bit).
        floor = floor_8bit * ((2 ** bit_depth) - 1) / 255.0
        thresh_val = max(th_median + (sigma_mult * th_sigma), floor)

        _, thresh_img = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
        thresh_img = thresh_img.astype(np.uint8)

        # Same star-size model as SkyMonitor.detect_stars (shared config).
        min_star_dim = max(3, int(self.star_mask_radius * self.cfg["timelapse"]["star_core_diameter_fraction"]))
        clean_size = min_star_dim if min_star_dim % 2 == 1 else min_star_dim + 1
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, clean_kernel)

        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_star_area = math.pi * (min_star_dim ** 2) / 4.0
        max_star_area = (self.star_mask_radius * self.cfg["timelapse"]["star_max_extent_factor"]) ** 2
        min_circ = self.cfg["daylight"]["min_star_circularity"]

        centroids = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_star_area <= area <= max_star_area):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < min_circ:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            mask = np.zeros(tophat.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            flux = float(cv2.sumElems(cv2.bitwise_and(tophat, tophat, mask=mask))[0])
            centroids.append((cx, cy, flux))

        centroids.sort(key=lambda t: t[2], reverse=True)
        return centroids
    # -----------------
    def _crop_center_roi(self, frame_gray, half_fov_deg):
        h, w = frame_gray.shape[:2]
        hfov = self.cfg["general"]["lens_hfov"]
        vfov = self.cfg["general"]["lens_vfov"]
        px_per_deg_h = w / hfov
        px_per_deg_v = h / vfov

        crop_w = min(w, int(2 * half_fov_deg * px_per_deg_h))
        crop_h = min(h, int(2 * half_fov_deg * px_per_deg_v))
        x0 = max(0, (w - crop_w) // 2)
        y0 = max(0, (h - crop_h) // 2)
        roi = frame_gray[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return roi, x0, y0
    # -----------------
    def assess_solve_readiness(self, frame_gray):
        """
        Advisory check: estimates whether conditions look favorable for
        solve_field_center_roi() BEFORE attempting the (comparatively
        expensive) Tetra3 solve. Does not block the solve -- returns
        diagnostics so the caller can warn with a REASON, not just "may fail".

        Deliberately computed on the SAME center-ROI the solver actually
        uses (not the full frame) -- a star count or contrast measurement
        from outside that window doesn't describe what the solver will see.
        """
        cfg = self.cfg["astrometry"]
        roi, x0, y0 = self._crop_center_roi(frame_gray, cfg["target_half_fov_deg"])

        mean_val, norm_stddev, stddev_val, *_ = self._compute_roi_stats(roi, blur=False)
        current_lux = self.calculate_lux(mean_val, self.current_exposure_us, self.current_gain)
        centroids = self._extract_star_centroids(roi)
        star_count = len(centroids)

        moon_impact, moon_in_fov, _, _ = self.get_moon_state()

        readiness_cfg = self.cfg["astrometry"]["solve_readiness"]
        checks = {
            "lux": (current_lux < readiness_cfg["max_lux"], current_lux, readiness_cfg["max_lux"]),
            "contrast": (norm_stddev > readiness_cfg["min_norm_stddev"], norm_stddev, readiness_cfg["min_norm_stddev"]),
            "star_count": (star_count > readiness_cfg["min_stars_in_roi"], star_count, readiness_cfg["min_stars_in_roi"]),
        }

        failed = [name for name, (ok, _, _) in checks.items() if not ok]
        is_clean_dark_sky = not failed

        if not is_clean_dark_sky:
            reasons = []
            if "lux" in failed:
                # Distinguish moonlight from pollution -- same lux failure,
                # different actionable meaning, since moon_impact is already
                # tracked and lux alone can't tell them apart.
                if moon_in_fov and moon_impact > readiness_cfg["moon_impact_notable"]:
                    reasons.append(f"moon-bright sky (lux={current_lux:.3f}, moon_impact={moon_impact:.2f})")
                else:
                    reasons.append(f"elevated sky background, lux={current_lux:.3f} (limit {readiness_cfg['max_lux']})")
            if "contrast" in failed:
                reasons.append(f"low contrast, norm_stddev={norm_stddev:.3f} (need >{readiness_cfg['min_norm_stddev']})")
            if "star_count" in failed:
                reasons.append(f"only {star_count} stars in solve ROI (need >{readiness_cfg['min_stars_in_roi']})")
            logging.warning("Astrometry: solve may fail: %s.", "; ".join(reasons))

        return {
            "is_clean_dark_sky": is_clean_dark_sky,
            "lux": current_lux,
            "norm_stddev": norm_stddev,
            "star_count": star_count,
            "failed_checks": failed,
        }
    # -----------------
    def solve_field_center_roi(self, frame_gray, update_cache=True, bit_depth=8):
        cfg = self.cfg["astrometry"]

        try:
            assess_readiness = self.assess_solve_readiness(frame_gray)
            logging.info(f"Frame result: "
                f"is_clean_dark_sky = {assess_readiness['is_clean_dark_sky']} - "
                f"lux = {assess_readiness['lux']:.3f} - "
                f"norm_stddev = {assess_readiness['norm_stddev']:.3f} - "
                f"star_count = {assess_readiness['star_count']} - "
                f"failed_checks = {assess_readiness['failed_checks']}"
            )
            
            solver = self._get_tetra3_solver()

            def _solve_crop(half_fov_deg):
                roi, x0, y0 = self._crop_center_roi(frame_gray, half_fov_deg)
                centroids = self._extract_star_centroids(roi, bit_depth=bit_depth)
                
                if len(centroids) < cfg["min_stars_for_solve"]:
                    logging.warning(
                        "Astrometry: only %d stars in %.0f deg crop (need %d) -- skipping solve.",
                        len(centroids), 2 * half_fov_deg, cfg["min_stars_for_solve"],
                    )
                    return None, roi.shape, (x0, y0), roi
                
                # Tetra3 explicitly requires (Y, X) for Row/Col matrix matching!
                star_yx = [(c[1], c[0]) for c in centroids]
                
                # Debug visualization (Only if debug level is 2 or higher)
                if self.cfg["general"]["debug_level"] >= 2:
                    debug_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    for x, y, _ in centroids:
                        cv2.circle(debug_roi, (int(x), int(y)), 2, (0, 255, 0), 1)
                    cv2.imwrite(os.path.join(self.debug_out_dir, f"debug_stars_crop_{half_fov_deg}deg.png"), debug_roi)
                
                logging.info(f"Astrometry: Passing {len(star_yx)} stars to Tetra3 for a Blind Solve.")
                result = solver.solve_from_centroids(
                    star_yx,
                    size=roi.shape,
                    fov_estimate=None,   # BLIND SOLVE
                    match_radius=cfg["match_radius_deg"],
                    pattern_checking_stars=cfg["pattern_checking_stars"],
                )
                return result, roi.shape, (x0, y0), roi

            primary, primary_shape, primary_origin, primary_roi = _solve_crop(cfg["target_half_fov_deg"])
            if primary is None or primary.get("RA") is None:
                logging.warning("Astrometry: primary center-crop solve failed.")
                return None

            cross, cross_shape, _, _ = _solve_crop(cfg["cross_check_half_fov_deg"])
            # Starts False: linearity is only "agreed" once the cross-check
            # actually ran and confirmed it. An inconclusive cross-check
            # (too few stars) must NOT be treated the same as a confirmed
            # agreement -- consumers of linearity_verified/valid_beyond_crop_deg
            # must never trust extrapolation that was never actually validated.
            plate_scale_agrees = False
            if cross is not None and cross.get("RA") is not None:
                primary_scale = primary["FOV"] / primary_shape[1]
                cross_scale = cross["FOV"] / cross_shape[1]
                pct_diff = 100.0 * abs(primary_scale - cross_scale) / primary_scale
                if pct_diff > cfg["plate_scale_tolerance_pct"]:
                    logging.warning(
                        "Astrometry: plate scale disagrees by %.1f%% between %.0f deg and "
                        "%.0f deg crops (tolerance %.1f%%).",
                        pct_diff, 2 * cfg["target_half_fov_deg"], 2 * cfg["cross_check_half_fov_deg"],
                        cfg["plate_scale_tolerance_pct"],
                    )
                else:
                    plate_scale_agrees = True
            else:
                logging.info("Astrometry: cross-check crop solve inconclusive (too few stars).")

            result_dict = {
                "timestamp": datetime.now().isoformat(),
                "ra_deg": primary["RA"],
                "dec_deg": primary["Dec"],
                "roll_deg": primary["Roll"],
                "fov_deg": primary["FOV"],
                "plate_scale_deg_per_px": primary["FOV"] / primary_shape[1],
                "roi_origin_px": primary_origin,
                "roi_shape_px": primary_shape,
                "linearity_verified": plate_scale_agrees,
                "valid_beyond_crop_deg": (
                    2 * cfg["target_half_fov_deg"] if plate_scale_agrees else None
                ),
            }
            logging.info(f"Astrometry: Solve Successful! (RA: {primary['RA']:.2f}, Dec: {primary['Dec']:.2f}, Roll: {primary['Roll']:.2f})")

            if update_cache:
                with self.astrometry_lock:
                    self.wcs_cache = result_dict

            return result_dict

        except Exception as e:
            logging.error("Astrometry: solve_field_center_roi failed: %s", e)
            return None

    def get_event_astro_data(self, event_frame_gray, event_bbox=None, bit_depth=8):
        """
        Resolves RA/Dec for an event via a FRESH, independent solve -- no
        propagation against a stale reference, no ECC-against-an-old-frame,
        no drift/viability reasoning needed, because this solves for *now*,
        on *now's* data. Solves are cheap (~50-80ms measured); events happen
        far less often than the periodic 30-min sky-monitor check, so there
        is no real cost concern here.

        event_bbox, if provided (full-frame pixel coords: [x0,y0,x1,y1],
        same as detection_loop's current_event_bbox), gets masked out of
        the frame before star extraction -- the meteor's own bright streak
        would otherwise be picked up as a spurious high-flux "star" by
        _extract_star_centroids, wasting a pattern_checking_stars slot on
        a point that corresponds to no real catalog star.

        bit_depth: passed straight through to _extract_star_centroids via
        solve_field_center_roi -- 10 for the raw Bayer green-channel path,
        8 for the standard event_stack/timelapse fallback path. Must match
        what event_frame_gray actually contains, or the star-detection
        threshold will be miscalibrated against the wrong dynamic range.

        Deliberately does NOT touch self.wcs_cache (update_cache=False) --
        this is a self-contained result for this one event, decoupled from
        the shared nightly sky-monitor cache.
        """
        frame_for_solve = event_frame_gray
        if event_bbox is not None:
            frame_for_solve = event_frame_gray.copy()
            x0, y0, x1, y1 = [int(v) for v in event_bbox]
            h, w = frame_for_solve.shape[:2]
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)
            if x1 > x0 and y1 > y0:
                frame_for_solve[y0:y1, x0:x1] = 0
                
        return self.solve_field_center_roi(frame_for_solve, update_cache=False, bit_depth=bit_depth)
