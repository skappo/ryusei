"""
file_utils_mixin.py

Small, standalone file and JSON utility methods for the Ryusei pipeline,
extracted from the main Pipeline class as a pure move -- method bodies
are unchanged, only their physical location. Composed into Pipeline via
multiple inheritance.

Three non-contiguous chunks in the original file: _safe_save_image
(alone), read_last_lines + get_file_list + is_safe_path + get_latest_file
(contiguous with each other), and save_json_atomic (a classmethod,
interleaved with three other config-related classmethods that are NOT
part of this group and are deliberately left in Pipeline's own body).
Confirmed via an explicit interleaving check before extraction.

No cross-mixin dependencies -- these methods don't reference any other
self.method() or self.attribute at all, genuinely standalone utilities.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import json
import logging
import cv2
from datetime import datetime
from collections import deque


class FileUtilsClass:
    """Small, standalone file and JSON utility methods, composed into Pipeline."""

    def _safe_save_image(self, filepath, image, params=None):
        """
        Safely saves an image to disk, ensuring the directory exists and
        validating the output.
        """
        # Storage backpressure (storage_governor.py). The raw array size is an upper
        # bound on the encoded PNG/JPEG. Refusals are counted by the governor and
        # reported per janitor cycle, not logged per frame.
        storage = getattr(self, "storage", None)
        if storage is not None and not storage.admits(filepath, getattr(image, "nbytes", 0)):
            return False
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            success = (
                cv2.imwrite(filepath, image, params)
                if params
                else cv2.imwrite(filepath, image)
            )

            if success:
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    return True
                else:
                    logging.error(
                        f"Image save failed: File {filepath} was reported saved but is missing or empty."
                    )
            else:
                logging.error(
                    f"Image save failed: cv2.imwrite returned False for {filepath}"
                )

        except Exception as e:
            logging.exception(f"Critical error saving image to {filepath}: {e}")

        return False

    def read_last_lines(self, filepath, num_lines=50):
        """
        Efficiently reads the last N lines from a file.
        Returns a list of strings, or an empty list if the file doesn't exist.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Use a deque for efficient appending to the front
                lines = deque(maxlen=num_lines)
                for line in f:
                    lines.append(line.strip())
                return list(lines)
        except FileNotFoundError:
            return ["Log file not found."]
        except Exception as e:
            return [f"Error reading log file: {e}"]

    # -----------------
    def get_file_list(self, path):
        """Scans a directory and returns a list of file details."""
        files = []
        if not os.path.isdir(path):
            return files

        for entry in os.scandir(path):
            # Skip directories and hidden files
            if entry.is_file() and not entry.name.startswith("."):
                try:
                    stat = entry.stat()
                    files.append(
                        {
                            "name": entry.name,
                            "path": entry.path,
                            "size_mb": f"{stat.st_size / (1024*1024):.2f} MB",
                            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )
                except FileNotFoundError:
                    continue  # File might have been deleted between scan and stat

        # Sort files by modification time, newest first
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return files

    # -----------------
    def is_safe_path(self, base_dirs, path_to_check):
        """
        CRITICAL SECURITY FUNCTION: Checks if a file path is safely within one of
        the allowed base directories to prevent path traversal attacks.
        """
        try:
            # Resolve the absolute path, following symlinks, to prevent '..' tricks
            # and symlink-based escapes from an allowed directory.
            resolved_path = os.path.realpath(path_to_check)

            # Check if the resolved path starts with any of the allowed base directories
            for base in base_dirs:
                # if resolved_path.startswith(os.path.abspath(base)):
                if resolved_path.startswith(os.path.realpath(base) + os.sep):
                    return True
        except Exception:
            return False
        return False

    # -----------------
    def get_latest_file(self, path, extension=".png"):
        """Finds the most recently modified file with a given extension in a directory."""
        files = []
        if not os.path.isdir(path):
            return None

        for entry in os.scandir(path):
            if entry.is_file() and entry.name.lower().endswith(extension):
                try:
                    files.append(entry.path)
                except FileNotFoundError:
                    continue

        if not files:
            return None

        # Return the file with the latest modification time
        return max(files, key=os.path.getmtime)

    @classmethod
    def save_json_atomic(cls, path, data):
        try:
            temp_path = path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=4)
            os.rename(temp_path, path)
        except Exception as e:
            logging.error("Failed to save JSON atomically to %s: %s", path, e)
