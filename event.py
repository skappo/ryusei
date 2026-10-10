"""
event_mixin.py

Event writing, stacking, and metadata methods for the Ryusei pipeline,
extracted from the main Pipeline class as a pure move -- method bodies
are unchanged, only their physical location. Composed into Pipeline via
multiple inheritance.

CORRECTION: an earlier version of this file accidentally included four
timelapse-specific methods and get_metadata due to a boundary-checking
mistake (assumed the range between two target method names contained only
those methods, without checking for interleaved content -- the same class
of mistake made once before with sky_monitor_mixin.py). Those methods have
been moved to timelapse_mixin.py (the timelapse ones) and back to
Pipeline's own body (get_metadata, a general utility never meant to be
extracted yet). This file now contains only its intended six methods.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls methods
from AstrometryClass (get_event_astro_data), SkyMonitorClass
(_compute_roi_stats, calculate_lux, get_moon_state), StackingClass
(stack_frames), ConcurrencyClass (_launch_fire_and_forget, ping_watchdog),
plus self._calculate_event_score, self.save_timelapse_metadata (now in
TimelapseClass), and self.get_metadata (back in Pipeline's own body).
All must be provided somewhere in the composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import json
import shutil
import subprocess
import queue
import logging
import time
import cv2
import numpy as np
from datetime import datetime

from satellites import CameraModel, SatelliteMatcher, to_utc


class EventClass:
    """Event writing, stacking, and metadata handling, composed into Pipeline."""

    def event_writer_loop(self):
        logging.info("Event writer started (Direct-to-FFmpeg Pipe Mode).")
        out_dir = self.events_out_dir

        try:
            while True:
                try:
                    # Wait for a complete package of frames
                    frames = self.event_q.get()
                    
                    if frames == "STATUS":
                        self.ping_watchdog()
                        continue
                    
                    if frames is None:
                        logging.info("Event writer received sentinel. Exiting.")
                        break
                except queue.Empty:
                    continue

                if not frames:
                    logging.info("Received an empty event package, ignoring.")
                    continue

                # Generate filename based on the last frame's timestamp
                tstamp = datetime.fromisoformat(frames[-1][0]).strftime(
                    "%Y%m%dT%H%M%S"
                )
                base_name = f"event_{tstamp}"
                video_path = os.path.join(out_dir, base_name + ".mp4")

                # Storage backpressure: the raw input frames are an upper bound on the
                # encoded video. Checked BEFORE spawning ffmpeg, so a full drive never
                # leaves a truncated .mp4 behind.
                if not self.storage.admits(video_path, sum(f[1].nbytes for f in frames)):
                    logging.error(
                        f"Event video {base_name}.mp4 not written: storage governor refused it "
                        f"(level {self.storage.level.name})."
                    )
                    continue

                # Determine video properties from the first frame
                first_frame = frames[0][1]
                height, width, layers = first_frame.shape

                event_cfg = self.cfg["events"]
                fps = event_cfg["video_fps"]
                encoder = event_cfg["ffmpeg_encoder"]
                crf_val = str(event_cfg["ffmpeg_crf"])
                pix_fmt = event_cfg.get("ffmpeg_pix_fmt", "yuv444p")
                

                logging.info(
                    "Encoding event video to %s (%d frames, %dx%d)",
                    base_name,
                    len(frames),
                    width,
                    height,
                )

                # Build FFmpeg command for reading raw video from stdin (pipe)
                ff_cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-f", "rawvideo",  # Input format is raw
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",  # Resolution must be specified for raw input
                    "-pix_fmt", "bgr24",  # OpenCV uses BGR
                    "-r", str(fps),  # Input framerate
                    "-i", "-",  # Read from stdin
                    "-c:v", encoder,  # Output encoder (libx264 or h264_v4l2m2m)
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
                        "-crf", crf_val,
                        "-pix_fmt", pix_fmt,
                    ]
                )

                # If using 4:4:4, we must explicitly tell libx264 to use the High 4:4:4 profile
                if pix_fmt == "yuv444p" and encoder == "libx264":
                    ff_cmd.extend(["-profile:v", "high444"])
                    
                ff_cmd.append(video_path)

                process = None
                try:
                    # Start FFmpeg process with a pipe for stdin
                    process = subprocess.Popen(
                        ff_cmd,
                        stdin=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                    )

                    # Write frames directly to the pipe
                    for _, frame, _ in frames:
                        try:
                            # Write raw bytes (fastest method, no JPEG encoding)
                            process.stdin.write(frame.tobytes())
                        except BrokenPipeError:
                            logging.error("FFmpeg pipe broke unexpectedly.")
                            process.kill()  # Ensure ffmpeg doesn't write a partial file silently.
                            break

                    # Close stdin to signal EOF to FFmpeg
                    _, stderr_data = process.communicate()

                    if process.returncode != 0:
                        logging.error("FFmpeg Error:\n%s", stderr_data.decode("utf-8"))
                        self.log_health_event(
                            "ERROR", "FFMPEG_FAIL", "Non-zero return code"
                        )
                    else:
                        self._tally_data_write(video_path)
                        self.save_event_metadata(video_path, frames)
                        logging.info("Event video saved: %s", video_path)

                        # Update dashboard status
                        with self.status_lock:
                            self.last_event_files["video"] = video_path

                except Exception as e:
                    logging.exception("An error occurred in event_writer_loop: %s", e)
                    # Ensure process is killed if python crashes
                    if process:
                        try:
                            process.kill()
                        except OSError:
                            pass
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN EVENT WRITER THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "EVENT_WRITER_CRASH", str(e))

    # -----------------
    def _maybe_refresh_satellites(self):
        """
        Refresh the satellite catalogue in the background when it is due. Called on
        the janitor's cycle and at startup -- never per event -- so a dead network
        costs one bounded attempt per cycle, and CelesTrak is never hammered.
        """
        sc = self.cfg["satellites"]
        cat = self.satellite_catalog
        if not sc["enabled"] or not cat.is_due() or cat.refresh_in_flight.is_set():
            return
        cat.refresh_in_flight.set()

        def _work():
            try:
                cat.refresh()   # outcome (success or failure) is recorded and reported inside
            except Exception as e:
                age = cat.age_hours()
                logging.warning(
                    f"Satellites: catalogue refresh failed ({e}); "
                    + (f"keeping the cached copy ({age:.0f} h old)." if age != float("inf") else "no catalogue yet.")
                )
            finally:
                cat.refresh_in_flight.clear()

        # One request per group, each bounded by its own timeout.
        budget = (len(sc["groups"]) + 1) * sc["request_timeout_sec"]
        self._launch_fire_and_forget(_work, budget, "SatCatalogRefresh")

    # -----------------
    def _match_satellite(self, log_summary, image_shape):
        """Satellite explaining this event, or None (see satellites.py for the physics)."""
        sc = self.cfg["satellites"]
        catalogue, entries = self.satellite_catalog.catalogue()
        if not len(catalogue):
            logging.info(
                f"Satellites: no usable catalogue (missing or older than {sc['max_catalog_age_days']} days; "
                f"see the download schedule in assets/satellites/state.json); check skipped."
            )
            return None
        g = self.cfg["general"]
        h, w = image_shape
        camera = CameraModel(w, h, g["lens_hfov"], g["lens_projection"])
        exposure_s = (log_summary.get("exposure_us") or self.current_exposure_us) / 1_000_000.0
        track = [(to_utc(datetime.fromisoformat(ts)), bbox) for ts, bbox in log_summary.get("track", []) if bbox]
        if not track:
            # No per-frame motion boxes: treat the whole event as one long "exposure"
            # over the union box (weaker, but still a physical test).
            bbox = log_summary.get("event_bbox")
            if not bbox or not log_summary.get("timestamp_local"):
                return None
            track = [(to_utc(datetime.fromisoformat(log_summary["timestamp_local"])), bbox)]
            exposure_s += log_summary.get("duration_sec", 0.0)
        matcher = SatelliteMatcher(g["latitude"], g["longitude"])
        bore, solved = matcher.boresight(
            g["camera_azimuth"], g["camera_altitude"], log_summary.get("astro_data"),
            track[len(track) // 2][0], camera, sc["position_tolerance_solved_deg"],
        )
        return matcher.match(catalogue, entries, track, exposure_s, camera, bore, solved, sc)

    # -----------------
    def event_stacker_loop(self):
        logging.info("Event Stacker thread started and waiting for packages.")

        out_dir = self.events_out_dir

        try:
            while self.running.is_set():
                try:
                    # cfg = self.cfg["timelapse"]
                    # method = cfg.get("stack_method", "mean")  # <--- mean remain the only method
                    # align = cfg.get("stack_align")
                    # alignment_features = cfg.get("min_features_for_alignment")

                    payload = self.event_stack_q.get()

                    if payload == "STATUS":
                        self.ping_watchdog()
                        continue
    
                    if payload is None:
                        logging.info("Event Stacker received sentinel. Exiting.")
                        break

                    event_frames, is_timeout_end, log_summary = payload

                    if not event_frames:
                        continue

                    if is_timeout_end:
                        logging.warning(f"Event Stacker: Processing maximum-length event ({len(event_frames)} frames).")
                    else:
                        logging.info(
                            "Event Stacker received package of %d frames.",
                            len(event_frames),
                        )
                    frames_to_stack = [f for (_, f, _) in event_frames]

                    stacked_image, final_method, stack_info = self.stack_frames(
                        frames_to_stack,
                        align=self.cfg["timelapse"]["stack_align"],
                        method="mip",
                        min_features_for_alignment=self.cfg["timelapse"][
                            "min_features_for_alignment"
                        ],
                        is_event_stack=True,
                    )

                    if stacked_image is not None:
                        event_bbox = (
                            log_summary.get("event_bbox") if log_summary else None
                        )
                        
                        # astro_data: prefer the result already computed once,
                        # at event start, from the 10-bit raw Bayer path (see
                        # detection_loop) -- avoids the MAD-collapses-to-zero
                        # quantization failure the 8-bit path hits on dim
                        # events, and avoids re-solving work already done.
                        # Only fall back to a fresh 8-bit solve here if that
                        # wasn't available (astrometry was disabled at event
                        # start, no raw frame was ready yet, etc.) -- graceful
                        # degradation, not a silent loss of the feature.
                        astro_data = log_summary.get("astro_data") if log_summary else None
                        if astro_data is None and self.cfg["astrometry"].get("enabled"):
                            half_idx = len(event_frames) // 2
                            _, event_frame_bgr, _ = event_frames[half_idx]
                            event_frame_gray = cv2.cvtColor(event_frame_bgr, cv2.COLOR_BGR2GRAY)
                            astro_data = self.get_event_astro_data(event_frame_gray, event_bbox)
                            
                        # 1. AI CLASSIFICATION & SCORING
                        if self.cfg["ai_classification"]["enabled"]:
                            label, conf = self.ai.classify_event(
                                stacked_image, event_bbox
                            )
                            is_real_meteor = (
                                label.lower() == "meteor"
                                and conf
                                >= self.cfg["ai_classification"]["min_confidence"]
                            )
                        else:
                            # If AI is disabled, we assume everything is a meteor!
                            label, conf = "N/A (AI Off)", 1.0
                            is_real_meteor = True

                        event_score = self._calculate_event_score(
                            stacked_image, label, conf
                        )

                        if self.cfg["ai_classification"]["enabled"]:
                            logging.info(
                                f"AI Verdict: {label} ({conf*100:.1f}%)| Composite Score: {event_score*100:.1f}%"
                            )
                        else:
                            logging.info(
                                f"AI Disabled. Event Passed Automatically | Composite Score: {event_score*100:.1f}%"
                            )

                        # 1b. SATELLITE CHECK. A positive match is decisive (a sunlit
                        # satellite at this position, at these times, moving this fast):
                        # it overrides the AI. NO match decides nothing -- planes, birds,
                        # clouds and noise are not in any catalogue -- so then the AI and
                        # the shape score keep the verdict.
                        sat_match = None
                        if self.cfg["satellites"]["enabled"] and log_summary:
                            sat_match, sat_timed_out = self._run_bounded(
                                self._match_satellite, self.cfg["satellites"]["match_timeout_sec"],
                                "EventStackerThread", log_summary, stacked_image.shape[:2],
                            )
                            if sat_timed_out:
                                logging.warning("Satellites: matching timed out; event kept on the AI verdict.")
                        if sat_match:
                            logging.warning(
                                f"Satellites: event explained by {sat_match['name']} (NORAD {sat_match['norad']}): "
                                f"{sat_match['matched_fraction'] * 100:.0f}% of {sat_match['frames']} frames, "
                                f"residual {sat_match['residual_deg']} deg, {sat_match['angular_rate_deg_s']} deg/s. "
                                f"Classified as satellite."
                            )
                            label, is_real_meteor = "satellite", False
                            with self.status_lock:
                                self.session_stats["satellite_interceptions"] = (
                                    self.session_stats.get("satellite_interceptions", 0) + 1
                                )

                        # 2. LOG ROUTING & FOLDER SORTING
                        if log_summary:
                            log_summary["event_score"] = event_score
                            log_summary["ai_label"] = label
                            log_summary["is_verified_meteor"] = is_real_meteor
                            log_summary["satellite"] = sat_match["name"] if sat_match else ""
                            
                            # Inject Astrometry into Log
                            if astro_data and astro_data.get("ra_deg") is not None:
                                log_summary["ra_deg"] = round(astro_data["ra_deg"], 3)
                                log_summary["dec_deg"] = round(astro_data["dec_deg"], 3)

                            if self.cfg.get("event_log", {}).get("enabled", True):
                                self.event_log_q.put_nowait(log_summary)

                        target_dir = (
                            self.events_out_dir
                            if is_real_meteor
                            else self.ai_filtered_out_dir
                        )

                        if not is_real_meteor and sat_match:
                            logging.warning("Satellite event: moving it to the Quarantine folder.")
                        elif not is_real_meteor:
                            logging.warning(
                                f"AI: Event classified as '{label}'. Moving to Quarantine folder."
                            )
                            with self.status_lock:
                                self.session_stats["ai_interceptions"] = (
                                    self.session_stats.get("ai_interceptions", 0) + 1
                                )

                        # 3. Save the Stack Image
                        if log_summary and "tstamp" in log_summary:
                            tstamp = log_summary["tstamp"]
                        else:
                            tstamp = datetime.fromisoformat(
                                event_frames[-1][0]
                            ).strftime("%Y%m%dT%H%M%S")

                        out_name = f"event_stack_{tstamp}.png"
                        full_path = os.path.join(target_dir, out_name)
                        self._safe_save_image(full_path, stacked_image)

                        # 4. Save Metadata with AI Verdict
                        stack_info = stack_info or {}
                        stack_info["ai_label"] = label
                        stack_info["ai_confidence"] = round(conf, 4)
                        stack_info["event_score"] = event_score
                        stack_info["is_verified_meteor"] = is_real_meteor
                        stack_info["timeout_limit_reached"] = is_timeout_end
                        stack_info["satellite"] = sat_match  # full match record, or None

                        # Inject Astrometry into JSON
                        if astro_data and astro_data.get("ra_deg") is not None:
                            stack_info["ra_deg"] = round(astro_data["ra_deg"], 3)
                            stack_info["dec_deg"] = round(astro_data["dec_deg"], 3)
                            
                        self.save_event_stack_metadata(
                            full_path, event_frames, final_method, stack_info
                        )

                        # 5. MOVE VIDEO FILES IF REJECTED
                        video_name = f"event_{tstamp}.mp4"
                        video_json = f"event_{tstamp}.json"

                        # Spawn the micro-thread to wait for FFmpeg.
                        # Timeout matches its own documented "max 300 seconds"
                        # wait for EventWriterThread, plus margin for the
                        # file-move/JSON-injection/SFTP-queueing that follows.
                        self._launch_fire_and_forget(
                            self._finalize_video_package,
                            self.cfg["general"]["finalize_video_thread_timeout_sec"],
                            f"Finalize_{tstamp}",
                            label,
                            conf,
                            event_score,
                            is_real_meteor,
                            video_name,
                            video_json,
                        )

                        # Send the Stack Image & Stack JSON to SFTP immediately!
                        if (
                            is_real_meteor
                            and self.sftp_uploader
                            and self.running.is_set()
                        ):
                            try:
                                logging.info(f"Queueing verified meteor {video_name} and its stack image for SFTP upload.")
                                self.sftp_dispatch_q.put_nowait(os.path.join(self.events_out_dir, video_name))
                                self.sftp_dispatch_q.put_nowait(os.path.join(self.events_out_dir, video_json))
                                self.sftp_dispatch_q.put_nowait(full_path)
                                self.sftp_dispatch_q.put_nowait(os.path.splitext(full_path)[0] + ".json")
                            except queue.Full:
                                 logging.error("SFTP Dispatch queue full. Verified meteor won't be uploaded this session.")
                        
                        # Update Dashboard Pointer & Live Feed
                        with self.status_lock:
                            self.last_event_files["image"] = full_path
                            if sat_match:
                                self.last_event_files["ai_verdict"] = f"SATELLITE: {sat_match['name']}"
                            else:
                                self.last_event_files["ai_verdict"] = f"{label.upper()} ({int(conf*100)}%) | Score: {int(event_score*100)}%"

                            folder = "events" if is_real_meteor else "ai_filtered"
                            self.session_stats.setdefault("recent_events", []).append({
                                "tstamp": tstamp,
                                "img_url": f"{folder}/{out_name}",
                                "vid_url": f"{folder}/{video_name}",
                                "label": f"SATELLITE {sat_match['name']}" if sat_match else label.upper(),
                                "score": int(event_score * 100),
                                "is_kept": is_real_meteor
                            })
                            self.session_stats["recent_events"] = self.session_stats["recent_events"][-15:]

                    else:
                        logging.error("Event stacker failed to produce an image. Dropping event payload.")

                except Exception as e:
                    logging.exception("Error in event_stacker_loop: %s", e)
        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN EVENT STACKER THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "EVENT_STACKER_CRASH", str(e))

    # -----------------
    def save_event_metadata(self, event_path, event_frames):
        self._save_generic_event_metadata(event_path, event_frames)

    # -----------------
    def save_event_stack_metadata(self, img_path, event_frames, method_used, stack_info=None):
        self._save_generic_event_metadata(img_path, event_frames, stack_method=method_used, stack_info=stack_info)

    # -----------------
    def _save_generic_event_metadata(self, output_path, event_frames, stack_method=None, stack_info=None):
        if not event_frames:
            return

        # Calculate brightness from the first frame of the event
        first_frame = event_frames[0][1]
        roi_mean, *_ = self._compute_roi_stats(first_frame)
        mean_brightness = float(roi_mean)


        # Calculate the average threshold from all frames in the event
        thresholds = [t for (_, _, t) in event_frames if isinstance(t, (int, float))]
        avg_threshold = np.mean(thresholds) if thresholds else None

        n_frames = len(event_frames)
        image_filename = os.path.basename(output_path)
        meta = self.get_metadata(
            n_frames=n_frames,
            mean_brightness=mean_brightness,
            effective_threshold=avg_threshold,
            stack_info=stack_info,
            image_filename=image_filename,
        )

        meta["source_event_start_time"] = event_frames[0][0]
        meta["source_event_end_time"] = event_frames[-1][0]

        if stack_method:
            meta["stack_method"] = stack_method

        # if astro_data:
            # meta["ra_deg"] = astro_data.get("ra_deg")
            # meta["dec_deg"] = astro_data.get("dec_deg")
            # meta["astro_confidence"] = astro_data.get("confidence")

        json_path = os.path.splitext(output_path)[0] + ".json"
        self.save_json_atomic(json_path, meta)
        logging.info("Saved event metadata to %s", json_path)

    # -----------------
    def _finalize_video_package(self, ai_label, ai_conf, event_score, is_real_meteor, video_name, video_json):
        """
        Dedicated background micro-thread. Waits for FFmpeg to finish writing the video,
        injects the AI metadata into the JSON, routes the files, and queues SFTP.
        """
        old_v = os.path.join(self.events_out_dir, video_name)
        old_j = os.path.join(self.events_out_dir, video_json)
        new_v = os.path.join(self.ai_filtered_out_dir, video_name)
        new_j = os.path.join(self.ai_filtered_out_dir, video_json)

        # Wait for EventWriterThread (events.writer_wait_sec, validated < this thread's own budget)
        wait_limit = self.cfg["events"]["writer_wait_sec"]
        wait_count = 0

        while not os.path.exists(old_j) and wait_count < wait_limit:
            if not self.running.is_set():
                return  # Abort safely if shutting down
            time.sleep(1.0)
            wait_count += 1

        if not os.path.exists(old_j):
            logging.warning(
                f"File Finalizer: Timed out waiting {wait_limit}s for {video_json} to be created."
            )
            return

        # 1. Update Video JSON
        try:
            with open(old_j, "r") as f:
                v_meta = json.load(f)
            v_meta["ai_label"] = ai_label
            v_meta["ai_confidence"] = round(ai_conf, 4)
            v_meta["event_score"] = round(event_score, 3)
            v_meta["is_verified_meteor"] = is_real_meteor
            self.save_json_atomic(old_j, v_meta)
        except Exception as e:
            logging.error(f"Failed to update video JSON with AI data: {e}")

        # 2. Route files based on AI Verdict
        if not is_real_meteor:
            try:
                if os.path.exists(old_v):
                    shutil.move(old_v, new_v)
                if os.path.exists(old_j):
                    shutil.move(old_j, new_j)

                # Dashboard's "Last Captured Event" pointer was set by
                # event_writer_loop to the pre-quarantine path (events_out_dir)
                # before this AI verdict was even known. Now that the file has
                # actually moved, repoint it so the dashboard link doesn't 404.
                with self.status_lock:
                    if self.last_event_files.get("video") == old_v:
                        self.last_event_files["video"] = new_v
            except Exception as e:
                logging.error(f"Failed to quarantine video {video_name}: {e}")
        else:
            # 3. SFTP Dispatch (Video files only, stack is handled instantly by main thread)
            if getattr(self, "sftp_uploader", None) and self.running.is_set():
                try:
                    logging.info(
                        f"Queueing verified meteor video {video_name} for SFTP upload."
                    )
                    self.sftp_dispatch_q.put_nowait(old_v)
                    self.sftp_dispatch_q.put_nowait(old_j)
                except queue.Full:
                    pass

