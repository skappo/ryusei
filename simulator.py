"""
simulator.py
"""
import time
# from config_models import DEFAULTS
import logging
import numpy as np
import cv2 
import random
from datetime import datetime
from typing import Optional, Tuple

from camera_capture import metadata_matches

class FrameSimulatorClass:
    """
    Ultra-Realistic Sky + Sensor Simulator (v11.0 Corrected)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.width = cfg.get("width")
        self.height = cfg.get("height")
        self.frame_count = 0
        self.exposure_us = cfg.get("exposure_us")

        # self.sim_params  = SimParams()           # FIX #14 — dataclass, not raw dict
        self.sim_params = {
            # Sky base
            "day_brightness_range": (200, 220),
            "night_brightness_range": (20, 35),
            "day_noise_sigma": 2.0,
            "night_noise_sigma": 6.0,
            # Atmosphere multipliers
            "rayleigh_day_boost": 1.5,
            "rayleigh_night_rgb": (0.2, 0.3, 0.8),
            # Sensor artefacts
            "hot_pixel_ratio": 0.000007,
            # Atmosphere / optics
            "light_pollution_grad": 2.0,
            "rayleigh_strength": 10.0,
            "mie_strength": 15.0,
            "vignette_strength": 0.3,
            "amp_glow_strength": 5.0,
            # Stars
            "star_count": 1200,
            "star_rotation_speed": 0.002,
            "pole_distance_from_center": 800.0,
            # Extinction
            "extinction_k": 0.3,
            # Moon
            "moon_radius_pixels": 40,
            "moon_glow_radius": 120,
            "moon_glow_strength": 0.4,
            "moon_earthshine_strength": 0.15,
            "moon_phase_override": 0.35,
            "moon_initial_offset_y": 400.0,
            "moon_arc_period": 10800,
            "moon_arc_peak_frac": 0.10,
            "moon_arc_horizon_frac": 0.90,
            # Clouds
            "cloud_layers": [
                {
                    "speed": (0.6, 0.2),
                    "blobs": 8,
                    "opacity": (0.1, 0.25),
                    "blur": 201,
                    "reseed": 18000,
                },
                {
                    "speed": (0.3, 0.1),
                    "blobs": 6,
                    "opacity": (0.2, 0.45),
                    "blur": 151,
                    "reseed": 24000,
                },
                {
                    "speed": (0.1, 0.05),
                    "blobs": 4,
                    "opacity": (0.4, 0.70),
                    "blur": 101,
                    "reseed": 30000,
                },
            ],
            "cloud_grey": (65.0, 60.0, 60.0),
            # Events
            "satellite_interval": 600,
            "airplane_interval": 900,
            "streak_duration": 20,
            # Meteors
            "meteor_interval": 300,
            "meteor_duration": 8,
            "meteor_trail_length": 20,
            "meteor_speed_min": 15.0,
            "meteor_speed_max": 35.0,
            "meteor_thickness_min": 2,
            "meteor_thickness_max": 5,
            # Frame drops
            "failure_rate": 0.1,  # 0.01 = 1% chance per frame to return None
        }

        self.force_fatal_failure = False  # Track if we want to simulate a dead sensor

        self.is_day_time = False
        self.ae_enabled = False

        # Define `parent` BEFORE the inner class so the closure is valid
        parent = self

        class DummyControls:
            def set_controls(self_, controls):
                if "AeEnable" in controls:
                    parent.ae_enabled = controls["AeEnable"]
                    logging.info(
                        f"SIM: Camera Control 'AeEnable' set to {parent.ae_enabled}"
                    )
                if "ExposureTime" in controls:
                    parent.exposure_us = controls["ExposureTime"]
                    logging.info(
                        f"SIM: Camera Control 'ExposureTime' set to {parent.exposure_us}us"
                    )

            def capture_request(self_):
                return parent

            def get_metadata(self_):
                lux = 500.0 if parent.is_day_time else 0.05
                return {
                    "Lux": lux,
                    "ExposureTime": parent.exposure_us,
                    "AnalogueGain": 8.0,
                }

        self.picam2 = DummyControls()

        # Pre-allocated buffers
        self.frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.layer_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)

        self._init_static_maps()
        self._init_stars()
        self._init_clouds()
        self._init_streaks()
        self._init_meteors()
        self._init_noise_bank()
        self._init_fpn_hotpixels()

        self.moon_phase_data = self._get_moon_phase_fraction()

        logging.warning("=" * 60)
        logging.warning(
            f"=== SKY SIMULATOR v11.0 STARTED ({self.width}x{self.height}) ==="
        )
        logging.warning(f"=== LOGGING: Enabled (Meteors/Satellites/Day-Night) ===")
        logging.warning("=" * 60)

    def _init_static_maps(self):
        p = self.sim_params
        h, w = self.height, self.width

        y_grad = np.linspace(1, 0, h, dtype=np.float32).reshape(-1, 1)
        x_grad = np.linspace(0, 1, w, dtype=np.float32)

        xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        r2 = (xx**2 + yy**2).astype(np.float32)

        self.vignette_map = (1 - p["vignette_strength"] * r2)[..., np.newaxis]
        glow = np.tile(x_grad**3, (h, 1))
        self.amp_glow_map = (glow * p["amp_glow_strength"]).astype(np.float32)[
            ..., np.newaxis
        ]
        self.rayleigh_map = ((y_grad**2) * p["rayleigh_strength"]).astype(np.float32)[
            ..., np.newaxis
        ]
        self.mie_map = (np.exp(-(y_grad * 4)) * p["mie_strength"]).astype(np.float32)[
            ..., np.newaxis
        ]
        self.pollution_map = np.linspace(
            0, p["light_pollution_grad"], h, dtype=np.float32
        ).reshape(-1, 1)[..., np.newaxis]

        temp_dome = np.zeros((h, w), dtype=np.float32)
        cv2.circle(temp_dome, (w // 2, h + 100), w // 2, 1.0, -1)
        temp_dome = cv2.GaussianBlur(temp_dome, (0, 0), w * 0.05)
        self.city_dome_map = (temp_dome * 30.0).astype(np.float32)[..., np.newaxis]

    def _init_noise_bank(self):
        # 60 frames eliminates visible cycling at normal frame rates
        bank_size = 60
        self.noise_bank = np.random.normal(
            0, 1.0, (bank_size, self.height, self.width, 3)
        ).astype(np.float32)
        self.noise_idx = 0

    def _init_stars(self):
        p = self.sim_params
        n = p["star_count"]
        max_dim = max(self.width, self.height) * 1.5

        self.stars_rel_pos = np.column_stack(
            [
                np.random.uniform(-max_dim, max_dim, n),
                np.random.uniform(-max_dim, max_dim, n),
            ]
        ).astype(np.float32)

        mag = np.random.pareto(2.0, n) + 1.0
        self.stars_brightness = np.clip(
            255 * (10 ** ((4.0 - mag) / 2.5)), 20, 255
        ).astype(np.float32)

        # Renamed masks to reflect true RGB appearance
        #   White/neutral  : default (no mask)
        #   Warm yellow     : [1.0, 0.95, 0.7]   → reddish-yellow tint
        #   Cool red/orange : [1.0, 0.7, 0.5]    → orange-red
        #   Blue-white      : [0.8, 0.9, 1.0]    → blueish
        #   Deep blue       : [0.5, 0.6, 1.0]    → deep blue (O/B-type)
        colors = np.ones((n, 3), dtype=np.float32)
        r = np.random.rand(n)

        mask_warm_yellow = (r >= 0.65) & (r < 0.80)
        mask_orange_red = (r >= 0.80) & (r < 0.90)
        mask_blue_white = (r >= 0.90) & (r < 0.97)
        mask_deep_blue = r >= 0.97

        # Colors in BGR order
        colors[mask_warm_yellow] = [0.70, 0.95, 1.0]  # warm yellow
        colors[mask_orange_red] = [0.50, 0.70, 1.0]  # orange-red
        colors[mask_blue_white] = [1.00, 0.90, 0.80]  # blue-white
        colors[mask_deep_blue] = [1.00, 0.60, 0.50]  # deep blue

        self.stars_colors = colors * self.stars_brightness[:, np.newaxis]

        radii = np.random.choice([1, 2, 3], n, p=[0.70, 0.25, 0.05])
        self.star_rads = radii

        # Pre-compute stamp masks; avoids per-star cv2.circle in render loop
        self.star_stamps: dict[int, np.ndarray] = {}
        for r_val in [2, 3]:
            stamp = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r_val + 1, 2 * r_val + 1)
            ).astype(np.float32)
            self.star_stamps[r_val] = stamp

    def _init_clouds(self):
        self.cloud_layers = []
        for params in self.sim_params["cloud_layers"]:
            layer = np.zeros((self.height, self.width), dtype=np.float32)
            self._generate_cloud_layer(layer, params)
            self.cloud_layers.append(
                {
                    "data": layer,
                    "opacity": np.random.uniform(*params["opacity"]),
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                    "params": params,
                    "last_reseed": 0,
                }
            )

    def _generate_cloud_layer(self, layer: np.ndarray, params: dict):
        """
        True seamless tiling via 3x3 oversized canvas + center crop.

        Strategy:
          1. Paint blobs on a 3x3 grid of tiles (9 copies of the same pattern).
          2. Blur the whole oversized canvas — blur never touches the borders of
             the CENTER tile because it is surrounded by identical content.
          3. Crop only the center tile and resize to frame dimensions.
          4. np.roll can now scroll in any direction forever with zero seams.
        """
        h, w = self.height, self.width
        h4, w4 = h // 4, w // 4

        # Step 1: paint blobs onto a single tile
        single = np.zeros((h4, w4), dtype=np.uint8)
        for _ in range(params["blobs"]):
            cx = np.random.randint(0, w4)
            cy = np.random.randint(0, h4)
            size = np.random.randint(w4 // 8, w4 // 3)
            cv2.circle(single, (cx, cy), size, 255, -1)

        # Step 2: tile it 3x3 so blur has identical neighbours on every side
        tiled = np.tile(single, (3, 3))

        # Step 3: blur the whole 3x3 canvas
        k = params["blur"] // 4 * 2 + 1
        blurred = cv2.GaussianBlur(tiled, (k, k), 0)

        # Step 4: crop center tile — its edges are now seamless
        center = blurred[h4 : h4 * 2, w4 : w4 * 2]
        resized = cv2.resize(center, (w, h), interpolation=cv2.INTER_LINEAR)
        layer[:] = resized.astype(np.float32) / 255.0
        layer *= np.random.uniform(*params["opacity"])

    def _init_streaks(self):
        self.streak_active = False
        self.streak_type = None
        self.streak_frame = 0
        self.streak_pos = np.array([0.0, 0.0])
        self.streak_vec = np.array([0.0, 0.0])
        self.streak_color = (255, 255, 255)
        self.streak_thickness = 1
        self.streak_dash = False

    def _init_meteors(self):
        self.meteor_active = False
        self.meteor_frame = 0
        self.meteor_trail = []
        self.meteor_pos = np.array([0.0, 0.0])
        self.meteor_vec = np.array([0.0, 0.0])
        self.meteor_peak = 255.0
        self.meteor_thick = 2
        self.meteor_col = (255, 255, 255)

    def _init_fpn_hotpixels(self):
        p = self.sim_params
        self.fpn = np.random.normal(0, 3.0, (self.height, self.width, 3)).astype(
            np.float32
        )
        hot_count = int(self.width * self.height * p["hot_pixel_ratio"])
        ys = np.random.randint(0, self.height, hot_count)
        xs = np.random.randint(0, self.width, hot_count)
        self.hot_pixel_map = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.hot_pixel_map[ys, xs] = 200.0

    def _update_environment_state(self):
        new_state = 6 <= datetime.now().hour < 18
        if new_state != self.is_day_time:
            self.is_day_time = new_state
            label = "DAY (Blue Sky)" if new_state else "NIGHT (Stars Visible)"
            logging.info("SIM: ------------------------------------------------")
            logging.info(f"SIM: Environment Transition -> {label}")
            logging.info("SIM: ------------------------------------------------")

    def read(self) -> np.ndarray:
        # --- FAILURE LOGIC START ---
        # 1. Check for a permanent "Dead Sensor" state
        if self.force_fatal_failure:
            return None

        # 2. Check for a random "Hardware Glitch"
        if random.random() < self.sim_params.get("failure_rate"):
            logging.warning(
                f"SIM: [Glitch] Simulated sensor timeout at frame {self.frame_count}"
            )
            return None
        # --- FAILURE LOGIC END ---

        start = time.monotonic()
        self.frame_count += 1

        if self.frame_count % 300 == 0:
            mode = "Day" if self.is_day_time else "Night"
            logging.info(f"SIM: [Status] Frame {self.frame_count} | Mode: {mode}")

        self._update_environment_state()

        self._render_base_sky()
        self._render_noise()

        if not self.is_day_time:
            self._render_stars()
            self._render_moon()
            self._render_clouds()
            self._render_streaks()
            self._render_meteors()
            self.frame_buffer += self.hot_pixel_map

        self._render_vignette()

        # Output directly as BGR — no colour conversion needed
        frame_bgr = np.clip(self.frame_buffer, 0, 255).astype(np.uint8)

        # Respect simulated exposure time
        elapsed = time.monotonic() - start
        delay = max(0.0, self.exposure_us / 1e6 - elapsed)
        if delay > 0.001:
            time.sleep(delay)

        # return frame_u16
        return frame_bgr

    def _render_base_sky(self):
        """Fill frame with base sky brightness and atmospheric gradients."""
        p = self.sim_params
        if self.is_day_time:
            base_val = np.random.uniform(*p["day_brightness_range"])
            self.frame_buffer[:] = base_val
            # FIX #16 — rayleigh_day_boost is now a named param, not a magic 1.5
            self.frame_buffer[..., 0] += (
                self.rayleigh_map[..., 0] * p["rayleigh_day_boost"]
            )
        else:
            base_val = np.random.uniform(*p["night_brightness_range"])
            self.frame_buffer[:] = base_val
            self.frame_buffer += self.pollution_map
            # FIX #16 — rayleigh_night_rgb is now a named param
            self.frame_buffer += self.rayleigh_map * np.array(
                p["rayleigh_night_rgb"], dtype=np.float32
            )
            self.frame_buffer += self.mie_map
            self.frame_buffer += self.amp_glow_map
            self.frame_buffer += self.city_dome_map

    def _render_noise(self):
        """Add read noise (cycling bank) and fixed-pattern noise."""
        p = self.sim_params
        noise_sigma = (
            p["day_noise_sigma"] if self.is_day_time else p["night_noise_sigma"]
        )
        self.noise_idx = (self.noise_idx + 1) % len(self.noise_bank)
        self.frame_buffer += self.noise_bank[self.noise_idx] * noise_sigma
        self.frame_buffer += self.fpn

    def _render_stars(self):
        """
        Vectorised star rendering with:
         - airmass-based atmospheric extinction
         - altitude-dependent twinkle sigma
         - pre-computed stamp masks for r>1 stars
         - direct star_rads indexing, no np.isin
        """
        p = self.sim_params
        h, w = self.height, self.width
        theta = self.frame_count * p["star_rotation_speed"]
        c, s = np.cos(theta), np.sin(theta)
        pole = p["pole_distance_from_center"]

        rx = self.stars_rel_pos[:, 0]
        ry = self.stars_rel_pos[:, 1] + pole

        rot_x = rx * c - ry * s + (w / 2)
        rot_y = rx * s + ry * c + (h / 2) - pole

        valid_mask = (rot_x >= 0) & (rot_x < w) & (rot_y >= 0) & (rot_y < h)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return

        vx = rot_x[valid_indices].astype(np.int32)
        vy = rot_y[valid_indices].astype(np.int32)

        # Airmass-based extinction (replaces simple quadratic)
        zenith_norm = np.clip(vy / h, 0.0, 0.999)  # 0 = zenith, ~1 = horizon
        cos_z = np.cos(zenith_norm * (np.pi / 2))
        airmass = 1.0 / np.maximum(cos_z, 0.05)  # avoid division by zero near 90°
        extinction = np.exp(-p["extinction_k"] * (airmass - 1.0)).astype(np.float32)

        # Ttwinkle sigma scales with altitude (more near horizon)
        twinkle_sigma = (0.05 + 0.25 * zenith_norm).astype(np.float32)
        twinkle = np.random.normal(1.0, twinkle_sigma).astype(np.float32)

        current_colors = (
            self.stars_colors[valid_indices] * (extinction * twinkle)[:, np.newaxis]
        )

        # Direct index instead of np.isin (O(n) vs O(n log n))
        rads = self.star_rads[valid_indices]
        is_small = rads == 1

        # Small (1-px) stars — simple scatter
        if np.any(is_small):
            sx = vx[is_small]
            sy = vy[is_small]
            self.frame_buffer[sy, sx] += current_colors[is_small]

        # Large stars — stamp-based
        for r_val in [2, 3]:
            mask_r = rads == r_val
            if not np.any(mask_r):
                continue
            stamp = self.star_stamps[r_val]  # pre-computed circular mask
            half = r_val
            lx = vx[mask_r]
            ly = vy[mask_r]
            lc = current_colors[mask_r]

            for i in range(len(lx)):
                x0, y0 = lx[i] - half, ly[i] - half
                x1, y1 = x0 + stamp.shape[1], y0 + stamp.shape[0]

                # Clamp to frame bounds
                fx0, fy0 = max(x0, 0), max(y0, 0)
                fx1, fy1 = min(x1, w), min(y1, h)
                sx0, sy0 = fx0 - x0, fy0 - y0
                sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)

                if fx1 <= fx0 or fy1 <= fy0:
                    continue

                patch = stamp[sy0:sy1, sx0:sx1, np.newaxis]
                colour = lc[i].reshape(1, 1, 3)
                self.frame_buffer[fy0:fy1, fx0:fx1] += patch * colour

    def _render_moon(self):
        """
        Draw moon disk + soft radial glow + earthshine.
        - Glow is a proper Gaussian radial falloff added onto the frame (not a
          solid filled circle which creates dark rings).
        - Moon disk has a minimum brightness so it is always visible even at
          new moon phase (earthshine keeps it dimly lit).
        - Arc sweeps left-to-right so the moon rises on the left, peaks at
          zenith, and sets on the right — matching real sky behaviour.
        """
        p = self.sim_params
        illum, waxing = self.moon_phase_data
        h, w = self.height, self.width
        r = p["moon_radius_pixels"]

        # Arc parameter: 0.0 (left/rise) -> 1.0 (right/set)
        t = self.frame_count % p["moon_arc_period"] / p["moon_arc_period"]

        # Horizontal sweep left to right with glow-radius margin
        margin = p["moon_glow_radius"] + r
        cx = int(margin + (w - 2 * margin) * t)

        # Vertical sine arc: derive pixel positions from frame height fractions
        peak_y = p["moon_arc_peak_frac"] * h
        horizon_y = p["moon_arc_horizon_frac"] * h
        arc_y = horizon_y + (peak_y - horizon_y) * np.sin(t * np.pi)
        cy = int(np.clip(arc_y, margin, h - margin))

        # --- Soft radial glow via Gaussian falloff added onto frame ---
        # Build a small glow patch and add it (no hard edges, no dark rings)
        gr = p["moon_glow_radius"]
        gsize = gr * 2 + 1
        sigma = gr / 2.5

        # Gaussian kernel
        ax = np.arange(gsize, dtype=np.float32) - gr
        gauss = np.exp(-(ax**2) / (2 * sigma**2))
        patch = np.outer(gauss, gauss)
        patch /= patch.max()

        # Scale by illumination — minimum glow even at new moon (earthshine)
        glow_illum = max(illum, 0.15)
        # BGR: slightly cool-white glow
        glow_bgr = np.stack(
            [
                patch * 220 * p["moon_glow_strength"] * glow_illum,
                patch * 235 * p["moon_glow_strength"] * glow_illum,
                patch * 255 * p["moon_glow_strength"] * glow_illum,
            ],
            axis=-1,
        ).astype(np.float32)

        # Paste glow patch onto frame (additive blend)
        y0, x0 = cy - gr, cx - gr
        y1, x1 = cy + gr + 1, cx + gr + 1
        # Clip to frame bounds
        fy0, fx0 = max(y0, 0), max(x0, 0)
        fy1, fx1 = min(y1, h), min(x1, w)
        py0, px0 = fy0 - y0, fx0 - x0
        py1, px1 = py0 + (fy1 - fy0), px0 + (fx1 - fx0)
        self.frame_buffer[fy0:fy1, fx0:fx1] += glow_bgr[py0:py1, px0:px1]

        # --- Moon disk with proper phase terminator ---
        #
        # This implementation uses a SINGLE continuous illumination sweep:
        #
        #  phase 0.00 → new moon:      fully dark disk (earthshine only)
        #  phase 0.25 → crescent:      thin lit sliver on RIGHT
        #  phase 0.50 → half moon:     right half lit
        #  phase 0.75 → gibbous:       most lit, thin dark sliver on LEFT
        #  phase 1.00 → full moon:     fully lit disk
        #
        # Key rules:
        #  - Illumination ALWAYS grows from RIGHT → LEFT

        phase = p["moon_phase_override"] if p["moon_phase_override"] >= 0 else illum

        earthshine = p["moon_earthshine_strength"] * (1.0 - illum)

        # Colours (BGR)
        dark_val = max(8, int(earthshine * 255))
        dark_col = (dark_val, dark_val, dark_val)

        lit_b = min(255, int(235))
        lit_g = min(255, int(245))
        lit_r = min(255, int(255))
        lit_col = (lit_b, lit_g, lit_r)

        # Patch canvas — work at pixel level, paste onto frame at the end
        pad = r + 2
        psize = pad * 2 + 1
        patch = np.zeros((psize, psize, 3), dtype=np.float32)
        pc = (pad, pad)

        # Step 1: full dark disk
        cv2.circle(patch, pc, r, dark_col, -1, cv2.LINE_AA)

        # Always light the right half first
        cv2.ellipse(patch, pc, (r, r), 0, -90, 90, lit_col, -1, cv2.LINE_AA)

        if phase < 0.5:
            # Crescent: terminator cuts into the right lit half
            # term_rx shrinks from r→0 as phase goes 0→0.5
            term_rx = int(r * (1.0 - phase * 2))
            if term_rx > 0:
                cv2.ellipse(
                    patch, pc, (term_rx, r), 0, -90, 90, dark_col, -1, cv2.LINE_AA
                )

        elif phase > 0.5:
            # Gibbous: extend lit area into the left dark half
            # term_rx grows from 0→r as phase goes 0.5→1.0
            term_rx = int(r * ((phase - 0.5) * 2))
            if term_rx > 0:
                cv2.ellipse(
                    patch, pc, (term_rx, r), 0, 90, 270, lit_col, -1, cv2.LINE_AA
                )

        # phase == 0.5 → exactly right half lit, no terminator needed

        # Subtle limb darkening ring
        limb_col = (int(lit_b * 0.55), int(lit_g * 0.55), int(lit_r * 0.55))
        cv2.circle(patch, pc, r, limb_col, 1, cv2.LINE_AA)

        # Paste patch onto frame buffer.
        # Use a disk mask so the moon OVERWRITES whatever is beneath it
        # (stars, noise) — the moon is an opaque body that occludes everything.
        py0, px0 = cy - pad, cx - pad
        py1, px1 = py0 + psize, px0 + psize
        fy0, fx0 = max(py0, 0), max(px0, 0)
        fy1, fx1 = min(py1, h), min(px1, w)
        sy0, sx0 = fy0 - py0, fx0 - px0
        sy1, sx1 = sy0 + (fy1 - fy0), sx0 + (fx1 - fx0)
        if fy1 > fy0 and fx1 > fx0:
            # Build a binary disk mask for the moon circle
            disk_mask = np.zeros((psize, psize), dtype=np.float32)
            cv2.circle(disk_mask, pc, r, 1.0, -1, cv2.LINE_AA)
            m = disk_mask[sy0:sy1, sx0:sx1, np.newaxis]  # (h, w, 1) in [0,1]

            # Inside disk: overwrite frame with moon patch (occludes stars)
            # Outside disk: keep frame as-is, blend in any glow (already added above)
            fb = self.frame_buffer[fy0:fy1, fx0:fx1]
            self.frame_buffer[fy0:fy1, fx0:fx1] = (
                fb * (1.0 - m) + patch[sy0:sy1, sx0:sx1] * m
            )

    def _render_clouds(self):
        """Drift and blend all cloud layers."""
        for layer_info in self.cloud_layers:
            self._process_cloud_layer(layer_info)

    def _render_streaks(self):
        """Trigger and advance satellite / airplane streaks."""
        self._process_streaks()

    def _render_meteors(self):
        """Trigger and advance meteor events."""
        p = self.sim_params
        if not self.meteor_active and random.random() < 1.0 / p["meteor_interval"]:
            self._start_meteor()
        if self.meteor_active:
            self._draw_meteor()

    def _render_vignette(self):
        """Apply lens vignette darkening at frame edges."""
        self.frame_buffer *= self.vignette_map

    def _process_cloud_layer(self, layer_info: dict):
        params = layer_info["params"]
        layer_info["offset_x"] += params["speed"][0]
        layer_info["offset_y"] += params["speed"][1]

        if self.frame_count - layer_info["last_reseed"] > params["reseed"]:
            self._generate_cloud_layer(layer_info["data"], params)
            layer_info["last_reseed"] = self.frame_count
            logging.info("SIM: Reseeding cloud layer.")

        # Np.roll wraps the tileable canvas: whatever drifts off the
        # right edge reappears on the left (and same vertically). No seams because
        # the canvas was generated with tiling neighbours in _generate_cloud_layer.
        ox = int(layer_info["offset_x"]) % self.width
        oy = int(layer_info["offset_y"]) % self.height
        drifted = np.roll(layer_info["data"], shift=(oy, ox), axis=(0, 1))

        # Blend toward grey cloud colour instead of multiplying to black
        cloud_colour = np.array(self.sim_params["cloud_grey"], dtype=np.float32)
        alpha = drifted[..., np.newaxis]
        self.frame_buffer = self.frame_buffer * (1.0 - alpha) + cloud_colour * alpha

    def _process_streaks(self):
        p = self.sim_params
        if not self.streak_active:
            if random.random() < 1.0 / p["satellite_interval"]:
                self._start_streak("satellite")
            elif random.random() < 1.0 / p["airplane_interval"]:
                self._start_streak("airplane")
            return

        # Airplane strobes cycle red → green → white (realistic nav lights)
        if self.streak_type == "airplane":
            strobe_colours = [  # BGR
                (50, 50, 255),  # red
                (50, 255, 50),  # green
                (255, 255, 255),  # white
            ]
            self.streak_color = strobe_colours[(self.streak_frame // 3) % 3]

        if not (self.streak_dash and (self.streak_frame % 6 < 3)):
            p1 = tuple(self.streak_pos.astype(int))
            p2 = tuple((self.streak_pos + self.streak_vec * 2).astype(int))
            cv2.line(
                self.frame_buffer, p1, p2, self.streak_color, self.streak_thickness
            )

        self.streak_pos += self.streak_vec
        self.streak_frame += 1

        if self.streak_frame > p["streak_duration"]:
            self.streak_active = False
            logging.info(f"SIM: {self.streak_type.capitalize()} object exited view.")

    def _start_streak(self, s_type: str):
        logging.info(f"SIM: >>> GENERATING {s_type.upper()} STREAK <<<")
        self.streak_active = True
        self.streak_type = s_type
        self.streak_frame = 0
        self.streak_pos[:] = [
            np.random.uniform(0, self.width),
            np.random.uniform(0, self.height),
        ]
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(8, 20)
        self.streak_vec[:] = [np.cos(angle) * speed, np.sin(angle) * speed]

        if s_type == "satellite":
            self.streak_color = (255, 255, 255)  # white BGR
            self.streak_thickness = 1
            self.streak_dash = False
        else:  # airplane
            # Initial colour; will cycle each frame via strobe logic
            self.streak_color = (50, 50, 255)  # red BGR
            self.streak_thickness = 3
            self.streak_dash = True

    def _start_meteor(self):
        logging.info("SIM: ☄️  >>> TRIGGERING METEOR EVENT <<<")
        self.meteor_active = True
        self.meteor_frame = 0
        self.meteor_trail = []
        p = self.sim_params

        edge = random.choice(["top", "left", "right"])
        if edge == "top":
            x, y = np.random.uniform(0, self.width), 0.0
            vx = np.random.uniform(-8, 8)
            vy = np.random.uniform(p["meteor_speed_min"], p["meteor_speed_max"])
        elif edge == "left":
            x, y = 0.0, np.random.uniform(0, self.height * 0.6)
            vx = np.random.uniform(10, p["meteor_speed_max"])
            vy = np.random.uniform(-10, 10)
        else:
            x, y = float(self.width), np.random.uniform(0, self.height * 0.6)
            vx = np.random.uniform(-p["meteor_speed_max"], -10)
            vy = np.random.uniform(-10, 10)

        self.meteor_pos[:] = [x, y]
        self.meteor_vec = np.array([vx, vy], dtype=np.float32)
        self.meteor_peak = np.random.uniform(200, 255)
        self.meteor_thick = np.random.randint(
            p["meteor_thickness_min"], p["meteor_thickness_max"] + 1
        )

        c = random.random()
        # BGR order
        if c < 0.70:
            self.meteor_col = (255, 255, 255)  # white
        elif c < 0.90:
            self.meteor_col = (200, 255, 200)  # greenish-white
        else:
            self.meteor_col = (200, 220, 255)  # warm white

    def _draw_meteor(self):
        p = self.sim_params
        self.meteor_frame += 1
        prev_pos = self.meteor_pos.copy()
        self.meteor_pos += self.meteor_vec

        progress = self.meteor_frame / p["meteor_duration"]
        head_bright = self.meteor_peak * (1 - progress)

        if head_bright > 0:
            cv2.line(
                self.frame_buffer,
                tuple(prev_pos.astype(int)),
                tuple(self.meteor_pos.astype(int)),
                self.meteor_col,
                self.meteor_thick + 1,
            )

        self.meteor_trail.append(
            {
                "p1": prev_pos,
                "p2": self.meteor_pos.copy(),
                "b": head_bright * 1.2,
                "age": 0,
            }
        )
        # Age-out stale trail segments
        self.meteor_trail = [
            t for t in self.meteor_trail if t["age"] < p["meteor_trail_length"]
        ]

        for t in self.meteor_trail:
            t["age"] += 1
            fade = 1.0 - (t["age"] / p["meteor_trail_length"])
            if fade > 0 and t["b"] > 0:
                col = tuple(ch * fade * t["b"] / 255 for ch in self.meteor_col)
                cv2.line(
                    self.frame_buffer,
                    tuple(t["p1"].astype(int)),
                    tuple(t["p2"].astype(int)),
                    col,
                    self.meteor_thick,
                )

        if self.meteor_frame > p["meteor_duration"] and not self.meteor_trail:
            self.meteor_active = False
            logging.info("SIM: Meteor event ended.")

    def _get_moon_phase_fraction(self) -> Tuple[float, bool]:
        # Use override if set; otherwise derive from real date
        p = self.sim_params
        if p["moon_phase_override"] >= 0:
            frac = p["moon_phase_override"]
        else:
            now = datetime.now()
            ref = datetime(2000, 1, 6, 18, 14, 0)
            days = (now - ref).total_seconds() / 86400.0
            cycle = 29.530_588_853
            frac = (days % cycle) / cycle
        illum = (1 - np.cos(frac * 2 * np.pi)) / 2
        return float(illum), frac < 0.5

    def get_metadata(self) -> dict:
        return {"Lux": 500.0 if self.is_day_time else 0.5}

    def release(self):
        pass

    def flush(self, target_exposure_us=None, target_gain=None, timeout_sec=None, frame_duration_us=None):
        """Simulated flush. In the simulator, there is no hardware queue to drain."""
        pass

    def achievable(self, exposure_us, gain):
        """No physical sensor: every request is achievable (same interface as the real camera)."""
        return int(exposure_us), (None if gain is None else float(gain))

    def settings_match(self, actual_exp_us, actual_gain, target_exp_us, target_gain):
        return metadata_matches(self.cfg, actual_exp_us, actual_gain, target_exp_us, target_gain)