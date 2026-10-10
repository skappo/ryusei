"""
timelapse_mixin.py

Timelapse capture loop, end-of-session video creation, and timelapse
metadata methods for the Ryusei pipeline, extracted from the main
Pipeline class as a pure move -- method bodies are unchanged, only their
physical location. Composed into Pipeline via multiple inheritance.

These four methods were originally (mistakenly) bundled into
event_mixin.py during that extraction -- moved here to their correct,
intended home. See event_mixin.py's docstring for that correction.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls methods
from SkyMonitorClass (_compute_roi_stats, get_moon_state), StackingClass
(stack_frames), ConcurrencyClass (_launch_fire_and_forget, ping_watchdog),
plus self._tally_data_write, self._safe_save_image, self.save_json_atomic,
self.log_health_event, self.get_metadata -- all still directly in
Pipeline's own body. All must be provided somewhere in the composed
Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import subprocess
import queue
import logging
import time
import cv2
import numpy as np
from datetime import datetime, timezone


class TimelapseClass:
    """Timelapse capture, video creation, and metadata, composed into Pipeline."""

    def timelapse_loop(self):
        first_run = True
        buffer = []

        try:
            while self.running.is_set():
                try:
                    cfg = self.cfg["timelapse"]
                    # interval_sec = self.timelapse_interval_sec
                    if self.cfg["general"].get("debug_visualization"):
                        logging.info(f"Timelapse started at {time.time()}")

                    if first_run:
                        logging.info(
                            "Timelapse worker started (stack_N=%d, method=%s, align=%s)",
                            self.effective_stack_n,
                            cfg["stack_method"],
                            cfg["stack_align"],
                        )
                        first_run = False

                    item = self.timelapse_q.get()
                    
                    if item == "STATUS":
                        self.ping_watchdog()
                        continue

                    # Handle Shutdown (None) AND Schedule End ("FLUSH")
                    if item is None or item == "FLUSH":
                        if item is None:
                            logging.info("Timelapse loop received sentinel.")
                        else:
                            logging.info(
                                "End of schedule reached. Flushing partial timelapse stack..."
                            )

                        if buffer:
                            logging.info(
                                f"Processing partial stack of {len(buffer)} frames before exiting..."
                            )
                            frames_to_stack = [f for (_, f, _) in buffer]
                            stacked_image, final_method, stack_info = self.stack_frames(
                                frames_to_stack,
                                align=cfg["stack_align"],
                                method=cfg["stack_method"],
                                min_features_for_alignment=cfg[
                                    "min_features_for_alignment"
                                ],
                                is_event_stack=False,
                                frame_timestamps=[t for (t, _, _) in buffer],
                            )
                            self.event_footprints.prune(buffer[-1][0])
                            if stacked_image is not None and final_method is not None:
                                tstamp = datetime.now().strftime(
                                    "%Y%m%dT%H%M%S"
                                )
                                # Name it 'final' if shutting down, or standard if just flushing
                                out_name = (
                                    f"timelapse_final_{tstamp}.png"
                                    if item is None or item == "FLUSH"
                                    else f"timelapse_{tstamp}.png"
                                )
                                full_path = os.path.join(
                                    self.timelapse_out_dir, out_name
                                )

                                self._safe_save_image(
                                    full_path,
                                    stacked_image,
                                    [int(cv2.IMWRITE_PNG_COMPRESSION), self.cfg["general"]["png_compression"]],
                                )
                                self._tally_data_write(full_path)
                                self.save_timelapse_metadata(
                                    full_path, final_method, buffer, stack_info
                                )

                                if self.sftp_uploader and self.running.is_set():
                                    logging.info(
                                        f"Queueing {os.path.basename(full_path)} and its metadata for upload."
                                    )
                                    self.sftp_dispatch_q.put_nowait(full_path)
                                    self.sftp_dispatch_q.put_nowait(
                                        os.path.splitext(full_path)[0] + ".json"
                                    )

                            buffer.clear()

                        self.timelapse_flush_done.set()  # the partial stack (if any) is on disk
                        if item is None:
                            logging.info("Draining complete, exiting.")
                            break  # Exit the thread permanently
                        else:
                            self.timelapse_complete_event.set()
                            continue  # Keep thread alive for tomorrow night

                    buffer.append(item)

                except queue.Empty:
                    continue

                if len(buffer) >= self.effective_stack_n:
                    frames_to_stack = [f for (_, f, _) in buffer]
                    
                    # Dawn/sunrise early detector 
                    # We only care about this if we are currently stuck in NIGHT mode
                    if not self.daylight_mode_active.is_set():
                        first_frame = frames_to_stack[0]
                        # roi_mean, _, _, _, _, _, _ = self._compute_roi_stats(first_frame)
                        roi_mean, *_ = self._compute_roi_stats(first_frame)

                        # Moon awarness
                        with self.status_lock:
                            moon_in_fov = self.last_moon_status.get("in_fov", False)
                        
                        # Moon in the FOV: raise the trigger (daylight.dawn_trigger_moon_*).
                        base_trigger = self.cfg["daylight"]["dawn_trigger_brightness"]
                        cfg_day = self.cfg["daylight"]
                        effective_trigger = (
                            max(cfg_day["dawn_trigger_moon_floor"], base_trigger * cfg_day["dawn_trigger_moon_multiplier"])
                            if moon_in_fov else base_trigger
                        )
                        
                        if roi_mean > effective_trigger:
                            self.bright_timelapse_count += 1
                            logging.info(f"Dawn Early Detector: Bright sky ROI detected ({roi_mean:.1f} > {effective_trigger:.1f}). Count: {self.bright_timelapse_count}/{self.cfg['daylight']['dawn_trigger_consecutive']}")
                            
                            if self.bright_timelapse_count >= self.cfg["daylight"]["dawn_trigger_consecutive"]:
                                logging.warning("Dawn Early Detector: Consecutive bright stacks detected! Trigger Sky Monitor.")
                                self.request_sky_analysis_trigger.set()
                                self.bright_timelapse_count = 0  # Reset after triggering
                        else:
                            self.bright_timelapse_count = 0  # Reset if it gets dark again
                    else:
                        self.bright_timelapse_count = 0 # Ensure it stays reset during the day
                    # -----------------

                    if len(frames_to_stack) == 1:
                        stacked_image = frames_to_stack[0]
                        final_method = "single_frame"
                        stack_info = {
                            "stars": 0,
                            "stddev": 0,
                            "actual_frames_stacked": 1,
                        }
                    else:
                        t_stack = time.monotonic()
                        stacked_image, final_method, stack_info = self.stack_frames(
                            frames_to_stack,
                            align=cfg["stack_align"],
                            method=cfg["stack_method"],
                            min_features_for_alignment=cfg[
                                "min_features_for_alignment"
                            ],
                            is_event_stack=False,
                            frame_timestamps=[t for (t, _, _) in buffer],
                        )
                        # Stacking cost per frame: one input of the capture frame-rate
                        # matching (capture._processing_floor_us). Smoothed like the
                        # detection loop time.
                        ms_per_frame = (time.monotonic() - t_stack) * 1000.0 / max(1, len(frames_to_stack))
                        a_ema = self.cfg["general"]["loop_time_ema_alpha"]
                        with self.status_lock:
                            prev = self.session_stats.get("timelapse_ms_per_frame", 0.0) or 0.0
                            self.session_stats["timelapse_ms_per_frame"] = (
                                ms_per_frame if prev == 0.0 else a_ema * ms_per_frame + (1.0 - a_ema) * prev
                            )
                    # Every later stack only holds later frames: closed footprints that
                    # ended before this stack's last frame can never be needed again.
                    self.event_footprints.prune(buffer[-1][0])

                    if stacked_image is not None and final_method is not None:
                        tstamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                        out_name = f"timelapse_{tstamp}.png"
                        full_path = os.path.join(self.timelapse_out_dir, out_name)
                        self._safe_save_image(
                            full_path,
                            stacked_image,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), self.cfg["general"]["png_compression"]],
                        )
                        logging.info("Saved timelapse stack to %s", full_path)
                        self._tally_data_write(full_path)
                        self.save_timelapse_metadata(
                            full_path, final_method, buffer, stack_info
                        )

                        if self.sftp_uploader and self.running.is_set():
                            logging.info(
                                f"Queueing {os.path.basename(full_path)} and its metadata for upload."
                            )
                            self.sftp_dispatch_q.put(full_path)
                            self.sftp_dispatch_q.put(
                                os.path.splitext(full_path)[0] + ".json"
                            )

                    buffer.clear()

                    # Signal to sky_monitor that timelapse has finish to stack self.effective_stack_n frames and should take the control of the camera
                    logging.info("Timelapse: Stack complete. Signaling safe window.")
                    self.timelapse_complete_event.set()

                    if self.cfg["general"].get("debug_visualization"):
                        logging.info(f"Timelapse finished at {time.time()}")
        except Exception as e:
            logging.critical("CRITICAL FATAL ERROR IN TIMELAPSE THREAD!", exc_info=True)
            self.log_health_event("CRITICAL", "TIMELAPSE_CRASH", str(e))

    # -----------------
    def start_timelapse_video_creation(self):
        """
        Launches the timelapse video creation process in a new, non-blocking
        daemon thread.
        """
        if not self.cfg["timelapse_video"].get("enabled"):
            return

        logging.info("Scheduler has triggered end-of-session timelapse video creation.")
        # Run the potentially long video encoding process in a separate thread
        # so it doesn't block the scheduler or the main shutdown sequence.
        # Timeout matches the internal ffmpeg subprocess.run(timeout=600)
        # budget plus margin for the startrail/stack generation that runs
        # before it -- this is the thread's own real lifetime, not an
        # arbitrary short number.
        self._launch_fire_and_forget(
            self._create_timelapse_video,
            self.cfg["general"]["timelapse_video_thread_timeout_sec"],
            "TimelapseVideoThread",
        )

    # -----------------
    def _create_timelapse_video(self):
        """
        Gathers all timelapse images from the last session to generate End-of-Session
        summary images (Startrail & Average Stack), and optionally encodes an MP4 video.
        """
        logging.info("End-of-Session media compilation thread started.")
        # Wait for the timelapse thread to write the session's final partial stack
        # (event-driven; the old fixed 10 s sleep could be too short on a slow drive
        # or needlessly long otherwise). Bounded by the physical stack budget.
        budget = self._stack_completion_budget_sec()
        if not self.timelapse_flush_done.wait(timeout=budget):
            logging.warning(
                f"Session media: final timelapse stack not confirmed within {budget:.0f}s; "
                f"compiling with the frames on disk."
            )

        session_start_ts = self.session_start_time.timestamp()

        # 1. Find all timelapse PNGs from the current session
        image_files = []
        for entry in os.scandir(self.timelapse_out_dir):
            if (
                entry.is_file()
                and entry.name.startswith("timelapse_")
                and entry.name.endswith(".png")
            ):
                try:
                    if entry.stat().st_mtime >= session_start_ts:
                        image_files.append(entry.path)
                except FileNotFoundError:
                    continue  # File might have been deleted

        if len(image_files) < 2:
            logging.warning(
                f"Not enough timelapse images ({len(image_files)}) found for this session to create media."
            )
            return

        # 2. Sort files chronologically (by filename, which contains the timestamp)
        image_files.sort()
        logging.info(
            f"Found {len(image_files)} timelapse images. Compiling session summaries..."
        )

        # GENERATE STARTRAIL & AVERAGE STACK
        try:
            logging.info(
                "Generating Star Trail and Average Stack (this may take a minute)..."
            )

            base_target = self.cfg["capture"].get("target_sky_brightness")
            tolerance = self.cfg["capture"].get("tolerance")

            trail_img = None
            mean_acc = None
            valid_count = 0

            keogram_cols = []

            for path in image_files:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # Extract the center column for the Keogram
                # We take a 3-pixel wide slice and average it to reduce sensor noise,
                # then reshape it back to a 1-pixel wide column.
                _, w = img.shape[:2]
                center_slice = img[:, (w//2)-1 : (w//2)+2].mean(axis=1, keepdims=True).astype(np.uint8)
                keogram_cols.append(center_slice)
                
                # Average Stack: Accumulate EVERYTHING for extreme noise reduction
                if mean_acc is None:
                    mean_acc = img.astype(np.float64)
                else:
                    mean_acc += img.astype(np.float64)
                valid_count += 1

                # Startrail: Filter anomalies (Headlights, Lightning, Thick Clouds)
                try:
                    # Parse timestamp from filename (local wall-clock, no tz suffix).
                    fname = os.path.basename(path)
                    t_str = fname.replace("timelapse_", "").replace(".png", "")
                    dt_local = datetime.strptime(t_str, "%Y%m%dT%H%M%S")

                    # get_moon_state()/ephem need a real UTC instant, not the local
                    # wall-clock value -- convert via time.mktime, which applies the
                    # system's DST rules for that specific date (a fixed UTC offset
                    # would be wrong on the other side of a DST transition).
                    dt_utc = datetime.fromtimestamp(
                        time.mktime(dt_local.timetuple()), tz=timezone.utc
                    )

                    # Calculate EXACT historical target for that specific minute
                    moon_impact, in_fov, _, _ = self.get_moon_state(calc_date=dt_utc)
                    historical_target = (
                        base_target
                        + (self.cfg["capture"].get("moon_normal_penalty") * moon_impact)
                        if in_fov
                        else base_target
                    )

                    # Get the actual 8-bit equivalent brightness of the frame
                    gray = (
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if len(img.shape) == 3
                        else img
                    )
                    actual_brightness = float(np.mean(gray))

                    # Bounds widened for natural intra-check drift (timelapse_video.brightness_outlier_tolerance_factor)
                    factor = self.cfg["timelapse_video"]["brightness_outlier_tolerance_factor"]
                    upper_bound = historical_target + (tolerance * factor)
                    lower_bound = historical_target - (tolerance * factor)

                    if lower_bound <= actual_brightness <= upper_bound:
                        if trail_img is None:
                            trail_img = img.copy()
                        else:
                            trail_img = np.maximum(trail_img, img)
                    else:
                        # Frame is rejected from Startrail (but kept in Average Stack)
                        logging.info(
                            f"Startrail: Rejected {fname} (Bright: {actual_brightness:.1f}, Target: {historical_target:.1f})"
                        )

                except Exception as e:
                    # Fallback: if filename parsing fails, include it anyway
                    if trail_img is None:
                        trail_img = img.copy()
                    else:
                        trail_img = np.maximum(trail_img, img)

            # Finalize and Save
            if valid_count > 0:
                mean_img = (mean_acc / valid_count).clip(0, 255).astype(np.uint8)
                tstamp = self.session_start_time.strftime("%Y%m%d")

                avg_path = os.path.join(
                    self.timelapse_out_dir, f"session_avg_{tstamp}.png"
                )
                self._safe_save_image(
                    avg_path, mean_img, [int(cv2.IMWRITE_PNG_COMPRESSION), self.cfg["general"]["png_compression"]]
                )
                if self.sftp_uploader and self.running.is_set():
                    self.sftp_dispatch_q.put(avg_path)
                    
                # KEOGRAM GENERATION
                if keogram_cols:
                    # Glue all the 1-pixel columns together left-to-right
                    keogram_raw = np.hstack(keogram_cols)
                    
                    # Stretch it horizontally into a timeline
                    keo_h = keogram_raw.shape[0]
                    # Same width as the timelapse frames it summarises.
                    keo_w = self.cfg["capture"]["width"]
                    keogram_stretched = cv2.resize(keogram_raw, (keo_w, keo_h), interpolation=cv2.INTER_LINEAR)
                    
                    keo_path = os.path.join(self.timelapse_out_dir, f"keogram_{tstamp}.png")
                    self._safe_save_image(keo_path, keogram_stretched, [int(cv2.IMWRITE_PNG_COMPRESSION), self.cfg["general"]["png_compression"]])
                    
                    if self.sftp_uploader and self.running.is_set():
                        self.sftp_dispatch_q.put(keo_path)

                if trail_img is not None:
                    self._safe_save_image(
                        self.startrail_path, trail_img, [int(cv2.IMWRITE_PNG_COMPRESSION), self.cfg["general"]["png_compression"]]
                    )
                    if self.sftp_uploader and self.running.is_set():
                        self.sftp_dispatch_q.put(self.startrail_path)

                logging.info(
                    "Successfully generated filtered Star Trail and Session Average images."
                )

        except Exception as e:
            logging.error(f"Failed to generate session summary images: {e}")

        # VIDEO CREATION
        if not self.cfg["timelapse_video"].get("enabled"):
            logging.info(
                "Video creation is disabled in config. Compilation thread finished."
            )
            return

        # Create a temporary file list for FFmpeg (most reliable method)
        filelist_path = os.path.join(self.general_log_dir, "timelapse_filelist.txt")
        with open(filelist_path, "w") as f:
            for path in image_files:
                # FFmpeg's concat demuxer needs a specific format
                f.write(f"file '{os.path.abspath(path)}'\n")

        # Construct and run the FFmpeg command
        try:
            cfg = self.cfg["timelapse_video"]
            tstamp = self.session_start_time.strftime("%Y%m%d")
            output_path = os.path.join(
                self.timelapse_out_dir, f"timelapse_video_{tstamp}.mp4"
            )

            # Storage backpressure. Size estimate = total size of the PNG inputs: at the
            # configured near-lossless CRF the H.264 output is normally well below it
            # (a conservative estimate, not a strict bound). Checked before ffmpeg
            # starts, so a full drive never leaves a truncated video.
            input_bytes = 0
            for p in image_files:
                try:
                    input_bytes += os.path.getsize(p)
                except OSError:
                    pass
            if not self.storage.admits(output_path, input_bytes):
                logging.error(
                    f"Timelapse video not created: storage governor refused ~{input_bytes / 1e6:.0f}MB "
                    f"(level {self.storage.level.name}). The stacked frames are kept."
                )
                return

            ff_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-f",
                "concat",  # Use the concatenation demuxer
                "-safe",
                "0",  # Allow absolute paths in the file list
                "-r",
                str(cfg["video_fps"]),  # Input framerate (how fast to read images)
                "-i",
                filelist_path,  # The input file list
                "-c:v",
                cfg["ffmpeg_encoder"],
            ]
            if self.half_power:
                ff_cmd.extend(
                    [
                        "-preset",
                        "ultrafast",
                        "-threads",
                        "2",
                    ]
                )
            ff_cmd.extend(
                [
                    "-crf", str(cfg["ffmpeg_crf"]),
                    "-pix_fmt", cfg["ffmpeg_pix_fmt"],
                ]
            )

            # If using 4:4:4, we must explicitly tell libx264 to use the High 4:4:4 profile
            if cfg["ffmpeg_pix_fmt"] == "yuv444p" and cfg["ffmpeg_encoder"] == "libx264":
                ff_cmd.extend(["-profile:v", "high444"])
                
            ff_cmd.append(output_path)

            logging.info(
                f"Encoding end-of-session timelapse video to {os.path.basename(output_path)}..."
            )
            try:
                subprocess.run(
                    ff_cmd, check=True, capture_output=True, text=True, timeout=self.cfg["timelapse_video"]["encode_timeout_sec"]
                )
            except subprocess.TimeoutExpired:
                logging.error(
                    "Timelapse video creation timed out after 10 minutes. FFmpeg may be frozen."
                )
                self.log_health_event(
                    "ERROR", "FFMPEG_TIMEOUT", "Timelapse video encoding timed out."
                )
                return

            logging.info(
                f"Successfully created timelapse video: {os.path.basename(output_path)}"
            )

            # Also queue this final video for upload
            if self.sftp_uploader and self.running.is_set():
                logging.info(f"Queueing {os.path.basename(output_path)} for upload.")
                self.sftp_dispatch_q.put(output_path)

        except subprocess.CalledProcessError as e:
            logging.error("--- TIMELAPSE FFMPEG FAILED ---")
            logging.error("FFmpeg Command: %s", " ".join(e.cmd))
            logging.error("FFmpeg stderr:\n%s", e.stderr)
        except Exception as e:
            logging.exception(
                "An error occurred during timelapse video creation: %s", e
            )
        finally:
            # Clean up the temporary file list
            if os.path.exists(filelist_path):
                os.remove(filelist_path)

    # -----------------
    def save_timelapse_metadata(
        self, img_path, method_used, frames_buffer, stack_info=None
    ):
        if not frames_buffer:
            return

        # Calculate brightness from the first frame in the buffer
        first_frame = frames_buffer[0][1]
        roi_mean, *_ = self._compute_roi_stats(first_frame)
        mean_brightness = float(roi_mean)

        # Calculate the average threshold from all frames in the buffer
        thresholds = [t for (_, _, t) in frames_buffer if isinstance(t, (int, float))]
        avg_threshold = np.mean(thresholds) if thresholds else None

        # feat_count = stack_info.get("stars") if stack_info else 0
        stddev_val = stack_info.get("stddev") if stack_info else 0
        bias_list = stack_info.get("bias_reasons", []) if stack_info else []

        n_frames = len(frames_buffer)
        image_filename = os.path.basename(img_path)

        meta = self.get_metadata(
            n_frames=n_frames,
            mean_brightness=mean_brightness,
            effective_threshold=avg_threshold,
            stddev=stddev_val,
            bias_reasons=bias_list,
            stack_info=stack_info,
            image_filename=image_filename,
        )

        meta["sky_status"] = (
            self.last_sky_status.name if self.last_sky_status else "UNKNOWN"
        )
        meta["stack_method"] = method_used
        meta["alignment"] = bool(method_used != "bias mode")

        # Add for calculate the correct value for MIN_NOISE_SCORE and MIN_STAR_SCORE
        sky_cond = self.last_sky_conditions  # dict populated by analyze_celestial_context
        meta["norm_stddev"]           = sky_cond.get("norm_stddev", 0.0)
        meta["stars"]                 = sky_cond.get("stars", 0)
        meta["effective_std_limit"]   = sky_cond.get("effective_std_limit",self.cfg["daylight"]["stddev_threshold"])
        meta["effective_star_limit"]  = sky_cond.get("effective_star_limit", self.cfg["daylight"]["min_stars"])
        meta["sky_score"]             = round(self.sky_score, 4)
        # ------
        json_path = img_path.rsplit(".", 1)[0] + ".json"
        self.save_json_atomic(json_path, meta)
        logging.info("Saved timelapse metadata to %s", json_path)

    # -----------------
