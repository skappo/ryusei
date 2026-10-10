"""
neural.py
"""
import cv2
import os
import logging
import numpy as np
# from config_models import DEFAULTS
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    tflite = None

class NeuralObserverClass:
    def __init__(self, cfg_event, cfg_obs):
        self.cfg_event = cfg_event
        self.cfg_obs = cfg_obs

        # Load the two separate interpreters
        self.event_interp = self._load_model(
            self.cfg_event["enabled"], self.cfg_event["model_path"]
        )
        self.obs_interp = self._load_model(
            self.cfg_obs["enabled"], self.cfg_obs["model_path"]
        )

        # Load the two separate label sets
        self.event_labels = self._load_labels(self.cfg_event["label_path"])
        self.obs_labels = self._load_labels(self.cfg_obs["label_path"])

    def _load_model(self, enabled, path):
        if not enabled or not os.path.exists(path):
            return None
        try:
            interp = tflite.Interpreter(model_path=path)
            interp.allocate_tensors()
            logging.info(f"AI: Model loaded from {path}")
            return interp
        except Exception as e:
            logging.error(f"AI: Critical model load failure ({path}): {e}")
            return None

    def _load_labels(self, path):
        """Loads labels and safely strips Google Teachable Machine numeric prefixes."""
        labels = []
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    # If Google added "0 meteor", split by space and take the second part
                    if " " in line and line.split(" ")[0].isdigit():
                        clean_label = line.split(" ", 1)[1]
                    else:
                        clean_label = line
                    labels.append(clean_label)
        return labels

    @staticmethod
    def _model_input_size(interpreter, configured=None):
        """
        Square input side the model was TRAINED at, read from its input tensor
        ([1, H, W, C]). Feeding any other size makes set_tensor fail (or, with a
        resize, silently changes what the network sees), so the model is the
        authority; a configured value that disagrees is reported, not used.
        """
        if interpreter is None:
            return configured
        h, w = (int(v) for v in interpreter.get_input_details()[0]["shape"][1:3])
        if h != w:
            raise ValueError(f"AI: model input is not square ({w}x{h}); crop logic assumes square inputs.")
        if configured is not None and configured != h:
            logging.warning(f"AI: configured input_size={configured} disagrees with the model ({h}); using the model's.")
        return h

    def classify_event(self, image, event_bbox=None):
        """Used by EventStacker to separate meteors from non-meteors."""
        size = self._model_input_size(self.event_interp, self.cfg_event["input_size"])
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pass the coordinates to the smart crop
        cropped_img = self._smart_crop_event(img_rgb, size, event_bbox)
        return self._run_inference(self.event_interp, self.event_labels, cropped_img)

    def check_observatory(self, image):
        """Used by SkyMonitor to detect Rain/Fog/Obstructions."""
        if self.obs_interp is None:
            return "unknown", 0.0
        size = self._model_input_size(self.obs_interp)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use Center Crop to look at the zenith and ignore distorted horizon edges
        cropped_img = self._center_crop(img_rgb, size)
        return self._run_inference(self.obs_interp, self.obs_labels, cropped_img)

    def _smart_crop_event(self, img_rgb, size, event_bbox=None):
        """Resizes preserving aspect ratio, then crops a square around the brightest object."""
        h, w = img_rgb.shape[:2]

        # 1. Resize so the shorter side is exactly 'size' (the model's input side)
        scale = size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))

        # 2. Find the center of the meteor using the coordinates from the DetectionThread!
        if event_bbox is not None:
            # Calculate the center point of the original 1080p/4K bounding box
            cx_orig = (event_bbox[0] + event_bbox[2]) // 2
            cy_orig = (event_bbox[1] + event_bbox[3]) // 2

            # Scale the center point down to match the new resized image
            cx = int(cx_orig * scale)
            cy = int(cy_orig * scale)
        else:
            # Fallback to the center of the image if coordinates are missing
            cx, cy = new_w // 2, new_h // 2

        # 3. Calculate crop bounds to put the meteor perfectly in the middle of the 224x224 square
        x_start = cx - size // 2
        y_start = cy - size // 2

        # 4. Clamp bounds so we don't accidentally try to crop "off the edge" of the screen
        x_start = max(0, min(x_start, new_w - size))
        y_start = max(0, min(y_start, new_h - size))

        return resized[y_start : y_start + size, x_start : x_start + size]

    def _center_crop(self, img_rgb, size):
        """Resizes preserving aspect ratio, then crops the exact center."""
        h, w = img_rgb.shape[:2]
        scale = size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))

        x_start = (new_w - size) // 2
        y_start = (new_h - size) // 2
        return resized[y_start : y_start + size, x_start : x_start + size]

    def _run_inference(self, interpreter, labels, img_square):
        """Takes an RGB square at the model's input size and runs the math."""
        if not interpreter or not labels or img_square is None:
            return "unknown", 0.0
        try:
            # Normalize to 0.0 -> 1.0
            img_array = img_square.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            interpreter.set_tensor(
                interpreter.get_input_details()[0]["index"], img_array
            )
            interpreter.invoke()

            # Parse Result
            output = interpreter.get_tensor(
                interpreter.get_output_details()[0]["index"]
            )[0]
            idx = np.argmax(output)
            return labels[idx], float(output[idx])
        except Exception as e:
            logging.error(f"AI Inference Error: {e}")
            return "error", 0.0