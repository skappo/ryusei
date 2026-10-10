"""
flat.py
"""
import os

import numpy as np
import cv2
import logging
from collections import deque
from constants import MAD_TO_SIGMA
# from config_models import DEFAULTS

class DynamicFlatEngineClass:
    """Computes a synthetic flat field using a temporal median of clear sky snapshots."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cfg = pipeline.cfg["detection"]
        self.buffer = deque(maxlen=self.cfg["auto_flat_buffer_size"])
        self.samples_added = 0  # monotonic count; len(buffer) saturates at maxlen

    def add_sample(self, frame):
        # Store as 8-bit to save RAM; vignetting is visible enough in 8-bit
        self.buffer.append(frame.copy())
        self.samples_added += 1
        logging.info(
            f"FlatEngine: Sample added. Buffer: {len(self.buffer)}/{self.buffer.maxlen}"
        )

    @staticmethod
    def _poly_design_matrix(x_norm, y_norm, degree):
        """
        Builds a 2D polynomial design matrix (1, x, y, x^2, xy, y^2, ...) up to
        `degree` on normalized coordinates in [-1, 1]. Normalization keeps the
        least-squares system well-conditioned regardless of sensor resolution.
        """
        terms = []
        for total_order in range(degree + 1):
            for i in range(total_order + 1):
                j = total_order - i
                terms.append((x_norm ** i) * (y_norm ** j))
        return np.stack(terms, axis=1).astype(np.float32)

    def compute(self):
        if len(self.buffer) < self.cfg.get("auto_flat_min_samples"):
            return None

        logging.info(
            "FlatEngine: Computing synthetic flat (sigma-clipped stack + polynomial surface fit)..."
        )

        # 1. Stack samples
        stack = np.array(list(self.buffer), dtype=np.float32)  # (N, H, W[, C])

        # 2. ROBUST STATISTICS: per-pixel MAD-based sigma clipping.
        #    A plain median (the old approach) is NOT an outlier rejector by itself
        #    at N in the 5-20 range: it only guarantees >50% clean samples per pixel.
        #    A slow-moving cloud edge, a plane trail, or moon glow drifting across
        #    several consecutive frames survives a plain median easily and shows up
        #    as exactly the local "blobs" you're seeing. Sigma-clipping around the
        #    median using a robust (MAD-derived) sigma estimate rejects those
        #    contaminated samples per-pixel before collapsing the stack.
        median_frame = np.median(stack, axis=0)
        abs_dev = np.abs(stack - median_frame)
        mad = np.median(abs_dev, axis=0)
        # Floor: float32 machine epsilon relative to the ~1.0-normalised flat, i.e. the
        # smallest meaningful spread (guards the division, never binds on real data).
        robust_sigma = np.maximum(mad * MAD_TO_SIGMA, np.finfo(np.float32).eps)

        clip_k = self.cfg["auto_flat_clip_sigma"]
        keep_mask = abs_dev <= (clip_k * robust_sigma)

        counts = np.maximum(np.sum(keep_mask, axis=0), 1)

        clean_frame = (np.sum(np.where(keep_mask, stack, 0.0), axis=0) / counts).astype(np.float32)

        # 3. Safe Dark Subtraction (unchanged - this was already correct)
        with self.pipeline.status_lock:
            current_dark = self.pipeline.master_dark

        if current_dark is not None:
            clean_frame = clean_frame - current_dark.astype(np.float32)

        # Floor at 1.0 to prevent division by zero
        clean_frame = np.maximum(clean_frame, 1.0)

        # 4. GLOBAL SMOOTH SURFACE FIT - replaces the decimate/blur/upscale/blur pipeline.
        #    Real optical vignetting (natural cos^4 falloff + mechanical cutoff) is a
        #    smooth, low-order, near-radially-symmetric function of position relative
        #    to the optical axis. It is PHYSICALLY INCAPABLE of producing a local blob.
        #    A blur is a low-pass filter, not a smoothness model: any residual
        #    structure wider than the blur kernel survives untouched, and cv2.INTER_CUBIC
        #    upscaling from a small grid actively rings/overshoots around whatever
        #    structure remains in the decimated image, which is the mottling you're
        #    seeing on the right and left edges. A least-squares 2D polynomial fit has
        #    no such failure mode: it is a global model, so it cannot represent local
        #    blobs at all, by construction, regardless of what noise or minor
        #    contamination is left in `clean_frame`.
        h, w = clean_frame.shape[:2]
        channels = 1 if clean_frame.ndim == 2 else clean_frame.shape[2]

        min_cells = self.cfg["auto_flat_fit_grid_min_cells"]
        grid_w = max(min_cells, int(round(w * self.cfg["auto_flat_fit_grid_fraction"])))
        grid_h = max(min_cells, int(round(h * self.cfg["auto_flat_fit_grid_fraction"])))
        # INTER_AREA still used here deliberately: it properly averages (decimates)
        # rather than subsamples, keeping the fit input representative.
        small = cv2.resize(clean_frame, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

        degree = self.cfg["auto_flat_poly_degree"]
        n_terms = (degree + 1) * (degree + 2) // 2
        min_required = int(n_terms * self.cfg["auto_flat_fit_oversample_factor"])
        if grid_w * grid_h < min_required:
            logging.warning(
                f"FlatEngine: Fit grid ({grid_w}x{grid_h}={grid_w*grid_h} px) is below the "
                f"oversample floor ({min_required} px) for a degree-{degree} surface. "
                f"Skipping this cycle rather than publishing an underdetermined fit."
            )
            return None

        ys, xs = np.mgrid[0:grid_h, 0:grid_w]
        xs_n = (xs.astype(np.float32) / max(grid_w - 1, 1)) * 2 - 1
        ys_n = (ys.astype(np.float32) / max(grid_h - 1, 1)) * 2 - 1
        design = self._poly_design_matrix(xs_n.ravel(), ys_n.ravel(), degree)

        full_ys, full_xs = np.mgrid[0:h, 0:w]
        full_xs_n = (full_xs.astype(np.float32) / max(w - 1, 1)) * 2 - 1
        full_ys_n = (full_ys.astype(np.float32) / max(h - 1, 1)) * 2 - 1
        full_design = self._poly_design_matrix(full_xs_n.ravel(), full_ys_n.ravel(), degree)

        small_flat = small.reshape(grid_h * grid_w, -1)
        surface = np.empty((h * w, small_flat.shape[1]), dtype=np.float32)
        for c in range(small_flat.shape[1]):
            coeffs, *_ = np.linalg.lstsq(design, small_flat[:, c], rcond=None)
            surface[:, c] = full_design @ coeffs

        blurred = surface.reshape(h, w) if channels == 1 else surface.reshape(h, w, channels)

        # 5. Calculate mean per channel
        flat_mean = np.mean(blurred, axis=(0, 1))

        if np.any(flat_mean <= 0):
            return None

        # 6. Calculate Multiplier Map safely
        multiplier = flat_mean / blurred

        # Clamp to prevent extreme ghosting
        multiplier = np.clip(
            multiplier, self.cfg["auto_flat_mult_min"], self.cfg["auto_flat_mult_max"]
        ).astype(np.float32)

        return multiplier


# ---------------------------------------------------------------- persistence

def save_flat_map(path, flat_map):
    """
    Persist a flat-field gain map ATOMICALLY: written to <path>.tmp, flushed to the
    card, then renamed over <path>. A crash or power cut mid-write leaves the
    previous map intact instead of a truncated file. Only real numeric maps are
    accepted -- never None or object arrays (np.save(None) writes a 274-byte pickled
    object array that np.load then refuses).
    """
    arr = np.asarray(flat_map)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number) or arr.ndim != 3:
        raise ValueError(f"not a flat map (dtype {arr.dtype}, shape {arr.shape})")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr.astype(np.float32, copy=False), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_flat_map(path, expected_shape):
    """
    Load a saved gain map, or raise ValueError saying why it cannot be used: wrong
    type, a resolution other than the current capture size (the map would be applied
    to the wrong pixels), or non-finite / non-positive gains.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3 or tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"saved map is {arr.shape}, capture is {tuple(expected_shape)}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"saved map has dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("saved map contains non-finite or non-positive gains")
    return arr.astype(np.float32, copy=False)

