"""
pacing.py

Shared, rate-limited reporting of frames dropped because a queue was full.

One line and one health event per reporting interval, with the count -- never one
per frame (a mismatch at 20 fps used to write hundreds of identical lines a minute).
Used for every frame queue that can overflow (acquisition, timelapse).
"""
import logging
import time


class DropReporter:
    def __init__(self, label, health_type, interval_getter, health_cb, hint=""):
        self.label = label                  # e.g. "Acquisition queue"
        self.health_type = health_type      # e.g. "ACQUISITION_QUEUE"
        self._interval = interval_getter    # -> seconds (live config)
        self._health = health_cb            # log_health_event(level, type, msg)
        self._hint = hint
        self.count = 0
        self._since = None                  # monotonic time of the first unreported drop
        self._last_report = float("-inf")

    def dropped(self):
        if self.count == 0:
            self._since = time.monotonic()
        self.count += 1
        self.tick()

    def tick(self):
        """Report pending drops if the interval since the last report has passed."""
        now = time.monotonic()
        if not self.count or now - self._last_report < self._interval():
            return
        span = now - self._since
        logging.warning(
            f"{self.label} full: {self.count} frame(s) dropped"
            + (f" over {span:.0f}s" if span >= 1 else "")
            + (f" ({self._hint})" if self._hint else "") + "."
        )
        self._health("WARNING", self.health_type, f"{self.count} frames dropped.")
        self.count, self._last_report = 0, now
