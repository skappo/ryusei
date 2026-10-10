#!/usr/bin/env python3
import cv2
import numpy as np
import math
import logging
import threading
import sys
import os
from datetime import datetime, timezone

# ==========================================
# --- FIX: THE NUMPY 2.0 MONKEY PATCH ---
# Tricks tetra3 into working flawlessly on the newest NumPy versions!
np.math = math
# ==========================================

# Set up basic logging so we can see your function's outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class AstrometryTester:
    def __init__(self):
        # 1. Mock the Pipeline Configuration
        self.cfg = {
            "astrometry": {
                "enabled": True,
                "tetra3_database": "default_database",  # Ensure you generated this using tetra3!
                "hfov_deg": 102.0,                      # From your Ryusei config
                "vfov_deg": 67.0,                       # From your Ryusei config
                "target_half_fov_deg": 25.0,            # Primary center crop radius
                "cross_check_half_fov_deg": 30.0,       # Secondary cross-check radius
                "min_stars_for_solve": 6,               # Tetra3 usually needs a minimum of 5-8
                "fov_max_error_deg": 5.0,               # Max error for FOV guessing
                "match_radius_deg": 0.05,               # Tetra3 tolerance
                "pattern_checking_stars": 8,            # Stars used for checking tetra pattern
                "plate_scale_tolerance_pct": 5.0        # Cross-check tolerance
            },
            "timelapse": {
                "star_sigma_multiplier": 5.0,
                "alignment_threshold_floor": 3
            },
            "daylight": {
                "min_star_circularity": 0.6
            }
        }
        
        # 2. Mock Pipeline State Variables
        self.astrometry_lock = threading.Lock()
        self.wcs_cache = None
        
        # Mock 1080p/4K Resolution Tuning variables 
        # (Change these if testing a different resolution)
        self.star_mask_radius = 10
        self.top_hat_factor = 2.5
        
    # =====================================================================
    # YOUR EXACT FUNCTIONS BELOW
    # =====================================================================
    
    def _get_tetra3_solver(self):
        with self.astrometry_lock:
            if getattr(self, "_tetra3_instance", None) is None:
                import tetra3  # deferred import
                logging.info("Astrometry: loading Tetra3 star-pattern database (one-time cost)...")
                self._tetra3_instance = tetra3.Tetra3(
                    load_database=self.cfg["astrometry"]["tetra3_database"]
                )
            return self._tetra3_instance

    def _extract_star_centroids(self, gray_roi):
        kernel_size = max(3, int(self.star_mask_radius * self.top_hat_factor))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray_roi, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

        th_median = float(np.median(tophat))
        th_mad = float(np.median(np.abs(tophat - th_median)))
        th_sigma = th_mad * 1.4826
        sigma_mult = self.cfg["timelapse"].get("star_sigma_multiplier", 5.0)
        floor = self.cfg["timelapse"].get("alignment_threshold_floor", 3)
        thresh_val = max(th_median + (sigma_mult * th_sigma), floor)

        _, thresh_img = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
        thresh_img = thresh_img.astype(np.uint8)

        min_star_dim = max(3, int(self.star_mask_radius * 0.25))
        clean_size = min_star_dim if min_star_dim % 2 == 1 else min_star_dim + 1
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, clean_kernel)

        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_star_area = math.pi * (min_star_dim ** 2) / 4.0
        max_star_area = (self.star_mask_radius * 4) ** 2
        min_circ = self.cfg["daylight"].get("min_star_circularity", 0.6)

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

    def _crop_center_roi(self, frame_gray, half_fov_deg):
        h, w = frame_gray.shape[:2]
        hfov = self.cfg["astrometry"]["hfov_deg"]
        vfov = self.cfg["astrometry"]["vfov_deg"]
        px_per_deg_h = w / hfov
        px_per_deg_v = h / vfov

        crop_w = min(w, int(2 * half_fov_deg * px_per_deg_h))
        crop_h = min(h, int(2 * half_fov_deg * px_per_deg_v))
        x0 = max(0, (w - crop_w) // 2)
        y0 = max(0, (h - crop_h) // 2)
        roi = frame_gray[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return roi, x0, y0

    def solve_field_center_roi(self, frame_gray):
        cfg = self.cfg["astrometry"]
        if not cfg.get("enabled", True):
            return None

        if frame_gray is None or frame_gray.ndim != 2:
            logging.error("Astrometry: solve_field_center_roi requires a single-channel grayscale frame.")
            return None

        try:
            solver = self._get_tetra3_solver()

            def _solve_crop(half_fov_deg):
                roi, x0, y0 = self._crop_center_roi(frame_gray, half_fov_deg)
                centroids = self._extract_star_centroids(roi)
                
                if len(centroids) < cfg["min_stars_for_solve"]:
                    logging.warning(
                        "Astrometry: only %d stars in %.0f deg crop (need %d) -- skipping solve.",
                        len(centroids), 2 * half_fov_deg, cfg["min_stars_for_solve"],
                    )
                    return None, roi.shape, (x0, y0)
                
                # =========================================================
                # --- FIX 1: COORDINATE REVERSAL ---
                # Tetra3 explicitly requires (Y, X) for Row/Col matrix matching!
                # =========================================================
                star_yx = [(c[1], c[0]) for c in centroids]
                
                # Draw extracted stars on ROI for debug visualization
                debug_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                for x, y, _ in centroids:
                    cv2.circle(debug_roi, (int(x), int(y)), 2, (0, 255, 0), 1)
                cv2.imwrite(f"debug_stars_crop_{half_fov_deg}deg.png", debug_roi)
                
                # =========================================================
                # --- FIX 2: THE BLIND SOLVE ---
                # Setting fov_estimate to None forces Tetra3 to search everything.
                # Once we know the real FOV, we can put the strict limits back!
                # =========================================================
                logging.info(f"Passing {len(star_yx)} stars to Tetra3 for a Blind Solve...")
                result = solver.solve_from_centroids(
                    star_yx,
                    size=roi.shape,
                    fov_estimate=None,   # <--- BLIND SOLVE
                    match_radius=cfg["match_radius_deg"],
                    pattern_checking_stars=cfg["pattern_checking_stars"],
                )
                return result, roi.shape, (x0, y0)

            primary, primary_shape, primary_origin = _solve_crop(cfg["target_half_fov_deg"])
            if primary is None or primary.get("RA") is None:
                logging.warning("Astrometry: primary center-crop solve failed.")
                return None

            cross, cross_shape, _ = _solve_crop(cfg["cross_check_half_fov_deg"])
            plate_scale_agrees = True
            if cross is not None and cross.get("RA") is not None:
                primary_scale = primary["FOV"] / primary_shape[1]   
                cross_scale = cross["FOV"] / cross_shape[1]          
                pct_diff = 100.0 * abs(primary_scale - cross_scale) / primary_scale
                if pct_diff > cfg["plate_scale_tolerance_pct"]:
                    plate_scale_agrees = False
                    logging.warning(
                        "Astrometry: plate scale disagrees by %.1f%% between %.0f deg and "
                        "%.0f deg crops (tolerance %.1f%%).",
                        pct_diff, 2 * cfg["target_half_fov_deg"], 2 * cfg["cross_check_half_fov_deg"],
                        cfg["plate_scale_tolerance_pct"],
                    )
            else:
                logging.info("Astrometry: cross-check crop solve inconclusive (too few stars).")

            with self.astrometry_lock:
                self.wcs_cache = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                return dict(self.wcs_cache)

        except Exception as e:
            logging.error("Astrometry: solve_field_center_roi failed: %s", e)
            return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_astrometry.py <path_to_timelapse_image.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    # Load image
    logging.info(f"Loading test image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Failed to load image. Is it a valid image file?")
        sys.exit(1)

    # Convert to grayscale as expected by the solver
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Instantiate our mocked pipeline and run the solve!
    tester = AstrometryTester()
    logging.info("Starting plate solve routine...")
    result = tester.solve_field_center_roi(gray)

    print("\n" + "="*50)
    if result:
        print("ðŸŽ‰ SOLVE SUCCESSFUL!")
        print(f"RA (Right Ascension): {result['ra_deg']:.3f}Â°")
        print(f"Dec (Declination):    {result['dec_deg']:.3f}Â°")
        print(f"Roll:                 {result['roll_deg']:.3f}Â°")
        print(f"FOV Calculated:       {result['fov_deg']:.3f}Â°")
        print(f"Plate Scale:          {result['plate_scale_deg_per_px']:.5f} deg/px")
        print(f"Linearity Verified:   {result['linearity_verified']}")
        print(f"Valid Beyond Crop:    {result['valid_beyond_crop_deg']}Â°")
    else:
        print("âŒ SOLVE FAILED.")
    print("="*50 + "\n")
