"""
camera_capture.py

Picamera2 wrapper: frame acquisition plus exposure/gain synchronisation.

SYNC PHYSICS (why flush() is built the way it is)
-------------------------------------------------
After set_controls(), frames that still carry the OLD exposure keep arriving
for a number of frames fixed by the pipeline, not by time:

  * buffer_count requests are already queued in libcamera with the old
    controls; each completes as a full frame at the OLD frame duration;
  * the sensor applies new exposure/gain/vblank with its own control delay
    (DelayedControls in the Raspberry Pi IPA, typically 2-3 frames).

So the time to the first new-settings frame is
    (buffer_count + control_delay_frames) * old_frame_duration + new_frame_duration
which scales with EXPOSURE. The previous timeout (2 frames + 10 s) is shorter
than that for multi-second exposures -- e.g. 5 s exposures need ~25 s, the
old budget gave 20 s -- which reproduces the "Flush timed out ... (20-25s)"
log lines exactly. FrameDurationLimits headroom does not enter this equation:
libcamera runs the shortest frame >= the MIN limit that fits the exposure, so
raising the MAX does not make a change land sooner.

The match predicate is relative (exposure and gain quantisation are relative
in nature) and is evaluated against the ACHIEVABLE target -- the request
clamped to the sensor's reported control ranges -- because a target the
sensor cannot produce can never be confirmed and would always time out.
"""
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
import collections
import time
import logging
import numpy as np


def metadata_matches(capture_cfg, actual_exp_us, actual_gain, target_exp_us, target_gain):
    """
    True if frame metadata corresponds to the target settings within the
    configured RELATIVE tolerances (capture.sync_exposure_rel_tol /
    capture.sync_gain_rel_tol). The single definition of "this frame was taken
    with these settings", used by flush() and by the auto-exposure re-check,
    for the real camera and the simulator alike.

    Relative, not absolute: the old fixed +/-0.5 gain window was 50% at gain
    1.0 (so a stale frame from a 1.0 -> 1.4 change was accepted as synced) yet
    only ~2% at gain 22; the old max(5 ms, 10%) exposure window was 25% at a
    20 ms exposure. Both let stale frames through at the low end.
    """
    if actual_exp_us is None or not target_exp_us:
        return False
    exp_ok = abs(actual_exp_us - target_exp_us) <= capture_cfg["sync_exposure_rel_tol"] * target_exp_us
    if target_gain is None:
        return exp_ok
    if actual_gain is None or target_gain <= 0:
        return False
    gain_ok = abs(actual_gain - target_gain) <= capture_cfg["sync_gain_rel_tol"] * target_gain
    return exp_ok and gain_ok


# Raw stream format and the bit depth it implies (one definition for every consumer
# of the raw Bayer frame, e.g. the event astrometry's 10-bit star extraction).
RAW_FORMAT = "SRGGB10"
RAW_BIT_DEPTH = 10


class CameraCaptureClass:
    def __init__(self, cfg):
        # NOTE: Despite the "RGB888" label, libcamera/picamera2 delivers this buffer
        # in physical BGR byte order (a known libcamera naming quirk, not a bug).
        # capture_array() output is ALREADY BGR — safe to feed directly into cv2.*
        # functions (imwrite, mean, etc.) with NO cv2.cvtColor() conversion.
        # Verified empirically against a known-color reference; do not "fix" this
        # by adding a COLOR_RGB2BGR call, it will re-invert the channels.
        self.cfg = cfg
        self.picam2 = Picamera2()
        main_config = self.picam2.create_still_configuration(
            main={"size": (cfg["width"], cfg["height"]), "format": "RGB888"},
            raw={"format": RAW_FORMAT},
            buffer_count=cfg["raw_buffer_count"]
        )
        self.picam2.configure(main_config)

        # What the pipeline really allocated (picamera2 may adjust the request).
        self.buffer_count = int(main_config.get("buffer_count", cfg["raw_buffer_count"]))

        # Sensor-reported control ranges for the configured mode: (min, max, default).
        ctrl = self.picam2.camera_controls
        self.exposure_limits_us = (int(ctrl["ExposureTime"][0]), int(ctrl["ExposureTime"][1]))
        self.gain_limits = (float(ctrl["AnalogueGain"][0]), float(ctrl["AnalogueGain"][1]))
        logging.info(
            f"Camera: sensor limits ExposureTime {self.exposure_limits_us[0]}..{self.exposure_limits_us[1]}us, "
            f"AnalogueGain {self.gain_limits[0]:.3f}..{self.gain_limits[1]:.3f}, "
            f"buffer_count={self.buffer_count}."
        )

        red_gain = cfg.get("red_gain")
        blue_gain = cfg.get("blue_gain")

        init_exp, init_gain = self.achievable(cfg.get("exposure_us"), cfg.get("gain"))
        controls = {
            "ExposureTime": init_exp,
            "AnalogueGain": init_gain,
            "AwbEnable": False,
            "ColourGains": [red_gain, blue_gain],
        }
        self.picam2.set_controls(controls)

        # Frame duration of the most recent frame seen (from metadata); the OLD
        # frame duration that sizes the next flush() budget. None until a frame
        # arrives, in which case flush() falls back to the target exposure.
        self._last_frame_duration_us = None

        self.picam2.start()
        # Drain exactly the frames that can still carry pre-configuration state (the
        # pipeline depth, see flush()), instead of a fixed 2 s sleep that was too long
        # at short exposures and too short at long ones.
        self.flush()
        logging.info("CameraCaptureClass initialized successfully.")

    # -----------------
    def achievable(self, exposure_us, gain):
        """Clamp a request to the sensor's reported ranges (what it can really deliver)."""
        exp = int(min(max(int(exposure_us), self.exposure_limits_us[0]), self.exposure_limits_us[1]))
        g = None if gain is None else float(min(max(float(gain), self.gain_limits[0]), self.gain_limits[1]))
        return exp, g

    # -----------------
    def settings_match(self, actual_exp_us, actual_gain, target_exp_us, target_gain):
        """See metadata_matches()."""
        return metadata_matches(self.cfg, actual_exp_us, actual_gain, target_exp_us, target_gain)

    # -----------------
    def pipeline_depth_frames(self):
        """Frames that can still carry old settings after a change (see module docstring)."""
        return self.buffer_count + self.cfg["sync_control_delay_frames"]

    # -----------------
    def _note_frame_meta(self, meta):
        fd = meta.get("FrameDuration")
        if fd:
            self._last_frame_duration_us = int(fd)

    # -----------------
    @property
    def is_streaming(self):
        """True while the sensor is started (Picamera2.started)."""
        return bool(getattr(self.picam2, "started", False))

    # -----------------
    def ensure_streaming(self):
        """
        Start the sensor if a previous stop (pause, end of session, test-shot
        teardown) left it stopped, then drain the frames in flight. Idempotent.
        Callers hold camera_lock.
        """
        if not self.is_streaming:
            self.picam2.start()
            logging.info("Camera: sensor restarted for a new capture session.")
            self.flush()

    # -----------------
    def read(self):
        # Side channel (kept off the return value so existing callers are untouched):
        # the exposure/gain this frame was ACTUALLY captured with, straight from the
        # frame metadata. Readers that need to know what a frame's brightness means
        # (the lightweight auto-exposure re-check) can't trust the currently applied
        # settings, because right after a change the camera still delivers frames
        # taken with the old ones for a few frames.
        self.last_meta = None
        if not self.is_streaming:
            # capture_request() on a stopped sensor never returns: the caller would
            # block forever (unstoppable thread). Fail fast instead; the capture loop
            # counts this as a read failure.
            logging.warning("Frame capture refused: the camera is stopped.")
            return None, None
        try:
            request = self.picam2.capture_request()
            try:
                img_array = request.make_array("main")
                raw_bayer = request.make_array("raw").view(np.uint16)  # new: 10-bit, single most-recent buffer
                meta = request.get_metadata()
                self.last_meta = {
                    "exposure_us": meta.get("ExposureTime"),
                    "gain": meta.get("AnalogueGain"),
                }
                self._note_frame_meta(meta)
            finally:
                request.release()  # always released, even on exception

            if img_array is None:
                self.last_meta = None
                return None, None
            # bgr_8bit = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array, raw_bayer
        except Exception as e:
            self.last_meta = None
            logging.warning("Frame capture failed: %s", e)
            return None, None

    # -----------------
    def flush(self, target_exposure_us=None, target_gain=None, timeout_sec=None, frame_duration_us=None):
        """
        Drain frames until one is confirmed (by metadata) to carry the target
        settings, and return it. Returns None on timeout or error.

        Every call emits one machine-greppable "CAMSYNC" log line:
            CAMSYNC result=OK|TIMEOUT|ERROR ... frames= seconds= expected_frames=
        On TIMEOUT it adds the requested vs. achievable targets, the sensor
        limits, the metadata trace of the last frames, and a verdict:
            NO_CHANGE_SEEN    every frame still carried the first frame's (old)
                              settings -> the queued old frames were not even
                              drained: budget far too short, or the controls
                              were never applied;
            STILL_CONVERGING  values changed and were still changing -> the
                              budget was too short (pipeline depth x frame
                              duration);
            STABLE_MISMATCH   values changed, then settled outside tolerance ->
                              target unreachable / tolerance tighter than the
                              sensor's quantisation (axis and error reported);
            NO_FRAMES         no frame completed at all within the budget.
        """
        if target_exposure_us is None:
            # No target: drain exactly the frames that can still be in flight.
            for _ in range(self.pipeline_depth_frames()):
                try:
                    req = self.picam2.capture_request()
                    try:
                        self._note_frame_meta(req.get_metadata())
                    finally:
                        req.release()
                except Exception:
                    pass
            return None

        req_exp, req_gain = target_exposure_us, target_gain
        target_exposure_us, target_gain = self.achievable(target_exposure_us, target_gain)

        depth = self.pipeline_depth_frames()
        old_fd_us = self._last_frame_duration_us or target_exposure_us
        # New frame period: the requested minimum frame duration (it can exceed the
        # exposure when the frame rate is matched to processing), else the exposure.
        new_fd_us = max(target_exposure_us, frame_duration_us or 0)
        expected_frames = depth + 1
        if timeout_sec is None:
            timeout_sec = (
                depth * max(old_fd_us, new_fd_us) + new_fd_us
            ) / 1e6 + self.cfg["sync_timeout_slack_sec"]

        # Only the tail matters for the verdict; depth+1 frames covers one full turnover.
        trace = collections.deque(maxlen=expected_frames)
        first_settings = None  # (exp, gain) of the first drained frame: the OLD settings
        start_time = time.monotonic()
        frames = 0

        while time.monotonic() - start_time < timeout_sec:
            try:
                req = self.picam2.capture_request()
                try:
                    img_array = req.make_array("main")
                    meta = req.get_metadata()
                finally:
                    req.release()
            except Exception as e:
                logging.warning(
                    f"CAMSYNC result=ERROR frames={frames} "
                    f"seconds={time.monotonic() - start_time:.2f} error={e!r}"
                )
                return None

            frames += 1
            self._note_frame_meta(meta)
            actual_exp = meta.get("ExposureTime")
            actual_gain = meta.get("AnalogueGain")
            trace.append((actual_exp, actual_gain, meta.get("FrameDuration")))
            if first_settings is None:
                first_settings = (actual_exp, actual_gain)

            if self.settings_match(actual_exp, actual_gain, target_exposure_us, target_gain):
                logging.info(
                    f"CAMSYNC result=OK frames={frames} seconds={time.monotonic() - start_time:.2f} "
                    f"expected_frames<={expected_frames} budget={timeout_sec:.1f}s "
                    f"exp={actual_exp}us gain={actual_gain:.3f} old_fd={old_fd_us}us"
                )
                return img_array

        logging.warning(self._timeout_report(
            frames, time.monotonic() - start_time, timeout_sec, expected_frames,
            req_exp, req_gain, target_exposure_us, target_gain, old_fd_us, trace,
            first_settings,
        ))
        return None

    # -----------------
    def _timeout_report(self, frames, elapsed, budget, expected_frames, req_exp, req_gain,
                        tgt_exp, tgt_gain, old_fd_us, trace, first_settings):
        if not trace:
            verdict = "NO_FRAMES"
            detail = ""
        else:
            last_exp, last_gain, _ = trace[-1]
            settled = len(trace) >= 2 and trace[-1][:2] == trace[-2][:2]
            if trace[-1][:2] == first_settings:
                verdict = "NO_CHANGE_SEEN"
            elif settled:
                verdict = "STABLE_MISMATCH"
            else:
                verdict = "STILL_CONVERGING"
            exp_err = (last_exp - tgt_exp) / tgt_exp if last_exp is not None else float("nan")
            gain_err = (
                (last_gain - tgt_gain) / tgt_gain
                if (tgt_gain and last_gain is not None) else float("nan")
            )
            detail = (
                f" exp_rel_err={exp_err:+.4f} (tol {self.cfg['sync_exposure_rel_tol']})"
                f" gain_rel_err={gain_err:+.4f} (tol {self.cfg['sync_gain_rel_tol']})"
            )
        trace_str = ";".join(
            f"{e}/{'-' if g is None else f'{g:.3f}'}/{fd}" for e, g, fd in trace
        )
        return (
            f"CAMSYNC result=TIMEOUT verdict={verdict} frames={frames} seconds={elapsed:.2f} "
            f"budget={budget:.1f}s expected_frames<={expected_frames} "
            f"(buffers={self.buffer_count} + delay={self.cfg['sync_control_delay_frames']}) "
            f"requested={req_exp}us/{req_gain} achievable={tgt_exp}us/{tgt_gain} "
            f"limits_exp={self.exposure_limits_us} limits_gain={self.gain_limits} "
            f"old_fd={old_fd_us}us{detail} trace[exp/gain/fd]={trace_str}"
        )

    # -----------------
    def release(self):
        try:
            self.picam2.stop()
            logging.info("Camera stream stopped.")
            self.picam2.close()
            logging.info("Camera device closed and resources released.")
        except Exception as e:
            # Log an error but don't crash if the camera is already in a bad state
            logging.error(f"An error occurred during camera release: {e}")
