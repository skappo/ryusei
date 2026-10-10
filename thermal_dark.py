"""
thermal_dark.py
"""
import os
import cv2
import numpy as np

class ThermalDarkManagerClass:
    def __init__(self, dark_dir, ref_exp_us):
        self.dark_dir = dark_dir
        self.ref_exp_us = ref_exp_us
        self.master_bias = None
        self.bias_library = {}  # Cache: {gain: image}
        self.dark_cache = {}  # Cache: {(temp, gain): thermal_growth_image}

    def _get_bias(self, gain):
        g_int = int(gain)
        if g_int not in self.bias_library:
            path = os.path.join(self.dark_dir, f"bias_G{g_int}.png")
            if os.path.exists(path):
                # self.bias_library[g_int] = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # For use the old bias files at 16 bit safely
                if img.dtype == np.uint16:
                    img = img / 256.0
                self.bias_library[g_int] = img.astype(np.float32)
        return self.bias_library.get(g_int)

    def get_scaled_dark(self, temp, gain, exp_us):
        it_temp = int(round(temp))
        it_gain = int(round(gain))

        bias = self._get_bias(it_gain)
        if bias is None:
            return None

        cache_key = (it_temp, it_gain)
        if cache_key not in self.dark_cache:
            path = os.path.join(self.dark_dir, f"dark_{it_temp}C_G{it_gain}_5s.png")
            if os.path.exists(path):
                # raw_dark = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                raw_dark = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # For use the old dark frame files at 16 bit safely
                if raw_dark.dtype == np.uint16:
                    raw_dark = raw_dark / 256.0

                raw_dark = raw_dark.astype(np.float32)
                # Store isolated Thermal Growth (Dark - Bias)
                self.dark_cache[cache_key] = cv2.subtract(raw_dark, bias)
            else:
                return None

        thermal_part = self.dark_cache[cache_key]
        scale = exp_us / self.ref_exp_us

        # Result = Bias(Gain) + (Thermal(Temp, Gain) * TimeScale)
        synthetic = cv2.add(bias, thermal_part * scale)
        return np.clip(synthetic, 0, 255).astype(np.uint8)