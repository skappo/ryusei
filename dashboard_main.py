"""
dashboard_main.py

The "/" main dashboard page for the Flask dashboard: live status, sky/moon,
system health, threads, recent detections and the tail of pipeline.log.
Split out of dashboard.py -- method body is unchanged, only its physical
location (it is one single large route, kept as one function here rather
than split further since every panel shares the same `status` snapshot).
Registered into the shared Flask app by dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
this is a plain function), kept under that name so every `self.xxx` call
copied from the original nested route(s) needed no rewriting.
"""
import html
import logging
import os
import threading
import time
from datetime import datetime

import psutil

from flask import request

from config_models import SKY_SCORE_VALUES
from dashboard_theme import THEME_ROOT


def register_main_routes(app, self, DASH_TOKEN, require_token, EDITABLE_PARAMS, TIMELAPSE_DIR):
    """The main status/control dashboard page."""

    # Main Dashboard Page
    @app.route("/")
    @require_token
    def dashboard():
        try:
            #            import html
            token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""

            # User Feedback Banners
            resume_status = request.args.get("resume_status")
            resume_error = html.escape(
                request.args.get("error", "Unknown Hardware Error")
            )
            feedback_html = ""
            if resume_status == "success":
                feedback_html = '<div class="banner banner-success">Normal operation resumed successfully.</div>'
            elif resume_status == "ignored":
                feedback_html = '<div class="banner banner-warn">Resume command ignored. The system is currently outside of its scheduled active time. The scheduler will start capture automatically when the time is right.</div>'
            elif resume_status == "fail":
                # Red banner for hardware failure
                feedback_html = (
                    f'<div class="banner banner-danger" style="background-color: #8b0000; color: white;">'
                    f"<strong>CRITICAL HARDWARE ERROR:</strong> {resume_error}</div>"
                )

            # Gather Live Metrics
            cpu_temp_obj = psutil.sensors_temperatures().get("cpu_thermal", [None])[0]
            cpu_temp_raw = cpu_temp_obj.current if cpu_temp_obj else 0.0
            cpu_temp = f"{cpu_temp_raw:.1f}°C" if cpu_temp_obj else "N/A"
            mem = psutil.virtual_memory()
            mem_used = f"{mem.used / (1024**3):.2f} GB"
            # disk = psutil.disk_usage(self.cfg["janitor"].get("monitor_path"))
            # disk_used = f"{disk.used / (1024**3):.2f} GB"
            load1, _, _ = os.getloadavg()
            uptime_sec = time.time() - psutil.boot_time()
            uptime_days = int(uptime_sec // 86400)
            uptime_hours = int((uptime_sec % 86400) // 3600)
            uptime_minutes = int((uptime_sec % 3600) // 60)
            uptime_str = f"{uptime_days}d {uptime_hours}h {uptime_minutes}m"
            now = datetime.now()
            system_date = now.strftime("%Y-%m-%d")
            system_time = now.strftime("%H:%M:%S")

            with self.status_lock:
                status = {
                    "version": self.version,
                    "debug_level": self.cfg["general"].get("debug_level"),
                    "capture_active": self.producer_thread_active(),
                    "is_stable": self.capture_stable.is_set(),
                    "daylight_mode": self.daylight_mode_active.is_set(),
                    "weather_hold": self.weather_hold_active.is_set(),
                    "power_status": self.power_status,
                    "event_count": self.event_counter,
                    "last_illuminance": self.last_illuminance,
                    "last_sky_stddev": self.last_sky_conditions.get("stddev"),
                    "last_sky_stars": self.last_sky_conditions.get("stars"),
                    "last_sky_status": self.last_sky_status,
                    "moon": self.last_moon_status,
                    "sky_score": self.sky_score,
                    "last_heartbeat": self.last_heartbeat_status,
                    "live_threads": {
                        t.name for t in threading.enumerate() if t.is_alive()
                    },
                    "calib_error": self.last_calibration_error,
                    "last_calib_image": self.last_calibration_image_path,
                    "last_event_files": self.last_event_files.copy(),
                    "is_calibrating": self.is_calibrating.is_set(),
                    "emergency_stop": self.emergency_stop_active.is_set(),
                    "storage_level": self.storage.level.name,
                    "maintenance_mode": self.in_maintenance_mode.is_set(),
                    "is_in_schedule": self.within_schedule(),
                    "memory_critical": self.memory_critical.is_set(),
                    "current_exposure": self.current_exposure_us,
                    "current_gain": self.current_gain,
                    "effective_threshold": self.last_effective_threshold,
                    "pre_frames": getattr(self, "pre_event_frames", 0),
                    "cool_frames": getattr(self, "event_cooldown_frames", 0),
                    "start": self.cfg.get("general", {}).get("start_time"),
                    "end": self.cfg.get("general", {}).get("end_time"),
                    "shutdown": self.cfg.get("general", {}).get("shutdown_time"),
                    "loc_name": self.cfg["general"].get("location") or "Unknown",
                    "lat": self.cfg["general"].get("latitude"),
                    "lon": self.cfg["general"].get("longitude"),
                    "cam_az": self.cfg["general"].get("camera_azimuth"),
                    "cam_alt": self.cfg["general"].get("camera_altitude"),
                    "thermal_status": (
                        "OK"
                        if (self.ds18b20_cam_path or self.ds18b20_ext_path)
                        else "SENSOR_FAIL"
                    ),
                    "sensor_temp_cam": self.current_sensor_temp_cam or 0.0,
                    "sensor_temp_ext": self.current_sensor_temp_ext or 0.0,
                    "dark_mode": (
                        "Dynamic"
                        if getattr(self, "use_dynamic_darks", False)
                        else ("Static" if self.master_dark is not None else "Off")
                    ),
                    "obs_condition": self.session_stats.get("obs_condition"),
                    "obs_conf": self.session_stats.get("obs_confidence"),
                    "loop_time_ms": self.session_stats.get("loop_time_ms", 0.0),
                    "recent_events": self.session_stats.get("recent_events", []),
                }

            if status["is_stable"]:
                # Green: Normal Operation
                pipeline_state_str = "Running (Acquiring Data)"
                pipeline_state_class = "ok"
            else:
                # Red: Paused for Maintenance/Tuning
                pipeline_state_str = "ON HOLD (Sky Monitor Check)"
                pipeline_state_class = "err"

                # "last_queue_status": self.last_queue_status # Get queue status
            #                calib_error = self.last_calibration_error
            with self.maintenance_timeout_lock:
                status["maintenance_timeout_until"] = self.maintenance_timeout_until

            system_status_class = "ok" if status["is_in_schedule"] else "warn"

            # 2. Determine the CSS class for illuminance
            illuminance_class = "ok"  # Default to green
            try:
                # Convert last_illuminance to a float for comparison
                last_lux_val = float(status["last_illuminance"])
                if last_lux_val > self.cfg.get("daylight").get("lux_threshold"):
                    illuminance_class = "err"  # It's daytime, show red
            except (ValueError, TypeError):
                illuminance_class = "warn"  # Not a valid number, show yellow

            # 3. Determine the CSS class for star count
            stars_class = "ok"  # Default to green
            try:
                # Convert last_sky_stars to an int for comparison
                # last_stars_val = int(status['last_sky_stars'])
                if int(status["last_sky_stars"]) < self.cfg.get("daylight").get(
                    "min_stars"
                ):
                    stars_class = "err"  # Not enough stars, show red
            except (ValueError, TypeError):
                stars_class = "warn"  # Not a valid number, show yellow

            stddev_class = "ok"  # Default to green
            try:
                # Convert last_sky_stars to an int for comparison
                if int(status["last_sky_stddev"]) > self.cfg.get("daylight").get(
                    "stddev_threshold"
                ):
                    stddev_class = "err"  # Not enough stars, show red
            except (ValueError, TypeError):
                stddev_class = "warn"  # Not a valid number, show yellow

            threshold = self.cfg["janitor"].get("threshold")

            # ==============================================================
            # PRESENTATION
            # Everything below turns the values gathered above into the page.
            # Every dynamic string is html.escape()d, and the page is returned
            # as a plain string: it is NOT passed through Jinja. The old
            # render_template_string() executed any "{{ ... }}" found in log
            # lines, filenames or error messages (crash at best, template
            # injection at worst).
            # ==============================================================
            esc = lambda v: html.escape(str(v))
            cfg_day = self.cfg["daylight"]

            def kv(label, value, cls=""):
                return (f'<div class="kv"><dt>{esc(label)}</dt>'
                        f'<dd class="{cls}">{value}</dd></div>')

            def chip(text, cls):
                return f'<span class="chip {cls}"><i></i>{esc(text)}</span>'

            def meter(label, value, limit, higher_is_better, cls, fmt="{:.0f}"):
                """Value against the threshold the pipeline actually decides on."""
                try:
                    v, lim = float(value), float(limit)
                except (TypeError, ValueError):
                    return kv(label, esc(value), "muted")
                scale = max(lim * 1.5, v, 1e-9)
                fill = min(v / scale, 1.0) * 100
                mark = min(lim / scale, 1.0) * 100
                rule = "needs at least" if higher_is_better else "limit"
                return (f'<div class="meter"><div class="meter-head"><span>{esc(label)}</span>'
                        f'<span class="{cls}"><b>{fmt.format(v)}</b> <small>{rule} {fmt.format(lim)}</small></span></div>'
                        f'<div class="track"><span class="fill {cls}" style="width:{fill:.1f}%"></span>'
                        f'<span class="mark" style="left:{mark:.1f}%"></span></div></div>')

            def ordinal(n):
                n = int(n)
                suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"{n}{suffix}"

            # ---------- state words (also used by the header chips) ----------
            if status["weather_hold"] and status["last_sky_status"] == "ERROR":
                weather_str, weather_class = "Sky sensor error, safe mode", "err"
            elif status["weather_hold"]:
                weather_str, weather_class = "Weather hold (cloudy)", "warn"
            else:
                weather_str, weather_class = "Sky clear", "ok"

            if status["emergency_stop"]:
                pipeline_mode_str, pipeline_mode_class = "Emergency stop", "err"
            elif status["maintenance_mode"]:
                pipeline_mode_str, pipeline_mode_class = "Paused for maintenance", "warn"
            elif not status["capture_active"]:
                pipeline_mode_str, pipeline_mode_class = "Idle, not capturing", "idle"
            elif status["daylight_mode"] or status["weather_hold"]:
                reason = "daylight" if status["daylight_mode"] else "weather hold"
                pipeline_mode_str, pipeline_mode_class = f"Timelapse only ({reason})", "warn"
            else:
                pipeline_mode_str, pipeline_mode_class = "Timelapse and meteor detection", "ok"

            storage_level = status.get("storage_level", "OK")
            storage_class = {"OK": "ok", "SHED": "warn", "HALT": "err"}.get(storage_level, "idle")
            storage_word = {"OK": "Storage OK", "SHED": "Storage tight", "HALT": "Storage full, halted"}.get(storage_level, storage_level)

            # ---------- banners ----------
            banners = [feedback_html] if feedback_html else []
            if status["emergency_stop"]:
                banners.append(
                    '<div class="banner err"><b>Emergency stop.</b> A critical fault left the system in an unknown state, '
                    f'so all acquisition is halted. Check <a href="/logs/pipeline.log{token_param}">pipeline.log</a> '
                    f'and <a href="/logs/faulthandler.log{token_param}">faulthandler.log</a>, then restart the service.</div>')
            if status["memory_critical"]:
                banners.append('<div class="banner warn"><b>Memory almost full.</b> Meteor detection is suspended until RAM '
                               'is freed; timelapse capture continues.</div>')
            if storage_level == "HALT":
                banners.append(f'<div class="banner err"><b>Data drive below its reserve.</b> Capture is paused and resumes by itself '
                               f'once space is freed. Use <a href="/files{token_param}">Files</a> to delete data now.</div>')
            if status["thermal_status"] == "SENSOR_FAIL" and self.cfg["detection"].get("auto_dark_enabled"):
                banners.append('<div class="banner warn"><b>Temperature sensor not found.</b> Dynamic dark frames are off; '
                               'the static dark is used instead.</div>')

            maintenance_panel = ""
            if status["maintenance_mode"] and not status["emergency_stop"]:
                remaining_txt = ""
                with self.maintenance_timeout_lock:
                    if status["maintenance_timeout_until"] > 0:
                        remaining_sec = self.maintenance_timeout_until - time.monotonic()
                        if remaining_sec > 0:
                            remaining_txt = f" Normal operation resumes automatically in about {int(remaining_sec / 60)} min."
                banners.append(f'<div class="banner info"><b>Maintenance mode.</b> Capture is paused and auto-refresh is off.{remaining_txt}</div>')

                form_rows = ""
                for section, params in EDITABLE_PARAMS.items():
                    form_rows += f'<tr><th colspan="2">{esc(section.replace("_", " ").capitalize())}</th></tr>'
                    for param in params:
                        value = self.cfg.get(section, {}).get(param, "N/A")
                        form_rows += (f'<tr><td><label for="f_{section}_{param}">{esc(param)}</label></td>'
                                      f'<td><input id="f_{section}_{param}" type="text" name="{section}_{param}" value="{esc(value)}"></td></tr>')
                calib_side = ""
                if status["calib_error"]:
                    calib_side += (f'<div class="banner err"><b>Test shot failed.</b> <code>{esc(status["calib_error"])}</code><br>'
                                   'Adjust the settings and try again; details are in pipeline.log.</div>')
                if status["is_calibrating"]:
                    calib_side += '<p class="muted">Taking a test shot. Reload the page in a few seconds.</p>'
                elif self.last_calibration_image_path:
                    shot_url = f"/calibration_image{token_param}&t={time.time()}"
                    calib_side += (f'<figure class="shot"><a href="{shot_url}" target="_blank"><img src="{shot_url}" alt="Last test shot"></a>'
                                   '<figcaption>Last test shot. Open for full size.</figcaption></figure>')
                maintenance_panel = f"""
                <section class="panel span-12">
                  <h2>Capture settings and test shot</h2>
                  <div class="maint">
                    <form action="/api/save_and_test{token_param}" method="post">
                      <table class="form">{form_rows}</table>
                      <button type="submit" class="btn primary">Save and take test shot</button>
                    </form>
                    <div>{calib_side}</div>
                  </div>
                </section>"""

            if status["maintenance_mode"]:
                action_button = f'<a class="btn primary" href="/api/resume{token_param}">Resume capture</a>'
            elif status["emergency_stop"]:
                action_button = ""
            else:
                action_button = f'<a class="btn" href="/api/pause{token_param}">Pause capture</a>'

            # ---------- hero: latest sky stack ----------
            hero_img = ""
            if self.effective_stack_n > 0 and status["capture_active"] and not status["maintenance_mode"]:
                latest_tl_path = self.get_latest_file(TIMELAPSE_DIR)
                if latest_tl_path and os.path.exists(latest_tl_path):
                    tl_mtime = int(os.path.getmtime(latest_tl_path))
                    token_qs = f"&token={DASH_TOKEN}" if DASH_TOKEN else ""
                    tl_url = f"/latest_timelapse_image?t={tl_mtime}{token_qs}"
                    tl_when = datetime.fromtimestamp(tl_mtime).strftime("%H:%M:%S")
                    hero_img = (f'<a href="{tl_url}" target="_blank"><img src="{tl_url}" alt="Latest timelapse stack"></a>'
                                f'<figcaption>Latest stack, {tl_when}. Open for full size.</figcaption>')
            if not hero_img:
                why = ("Capture is paused for maintenance." if status["maintenance_mode"]
                       else "Capture is not running." if not status["capture_active"]
                       else "Waiting for the first stack of the session.")
                hero_img = f'<div class="empty tall">{esc(why)}</div>'

            # ---------- tonight ----------
            actual_duration = datetime.now() - self.session_stats["start_time"]
            dur_total_min = max(1, int(actual_duration.total_seconds() / 60))
            measured_min = (self.session_stats["clear_sky_minutes"] + self.session_stats["cloudy_sky_minutes"]) or 1
            clear_pct = 100 - (self.session_stats["cloudy_sky_minutes"] / measured_min) * 100
            avg_score = (sum(self.session_stats["avg_sky_score"]) / len(self.session_stats["avg_sky_score"])
                         if self.session_stats["avg_sky_score"] else 0)
            if avg_score > SKY_SCORE_VALUES["excellent"]:
                sky_grade, grade_class = "Excellent", "ok"
            elif avg_score > SKY_SCORE_VALUES["good"]:
                sky_grade, grade_class = "Good", "ok"
            elif avg_score > SKY_SCORE_VALUES["fair"]:
                sky_grade, grade_class = "Fair", "warn"
            else:
                sky_grade, grade_class = "Poor", "err"
            noise_ratio = self.session_stats["rejected_contours"] / (self.session_stats["total_events"] or 1)

            ctx = self._get_chronicle_context()
            if ctx:
                chronicle = (f"Night {ctx['nights_total']} on record. Event count ranks in the "
                             f"{ordinal(ctx['events_pct'])} percentile, sky quality in the {ordinal(ctx['score_pct'])}.")
            else:
                chronicle = "First night on record."

            fp = self._generate_atmosphere_fingerprint(self.session_stats.get("sky_history", []))
            sparkline = ""
            sl = fp.get("sparkline", [])
            if len(sl) >= 2:
                W, H = 240, 40
                mn, mx_val = min(sl), max(sl)
                rng = mx_val - mn if mx_val > mn else 1.0
                pts = " ".join(f"{int(i / (len(sl) - 1) * (W - 6)) + 3},{int((1 - (v - mn) / rng) * (H - 6)) + 3}"
                               for i, v in enumerate(sl))
                trend_cls = {"improving": "ok", "degrading": "err"}.get(fp["trend"], "idle")
                sparkline = (f'<div class="spark"><svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="{trend_cls}">'
                             f'<polyline points="{pts}"/></svg>'
                             f'<span class="{trend_cls}">Sky quality {esc(fp["trend"])}, {fp.get("slope", 0.0):+.2f} per hour</span></div>')

            stars_cls = "ok" if stars_class == "ok" else ("err" if stars_class == "err" else "warn")
            noise_cls = "ok" if stddev_class == "ok" else ("err" if stddev_class == "err" else "warn")
            tonight_panel = f"""
            <section class="panel span-4 tonight">
              <h2>Tonight</h2>
              <div class="stats">
                <div><b>{self.session_stats['total_events']}</b><span>meteor events</span></div>
                <div><b>{self.session_stats.get('ai_interceptions', 0)}</b><span>filtered by AI</span></div>
                <div><b>{clear_pct:.0f}%</b><span>clear sky</span></div>
                <div><b class="{grade_class}">{sky_grade}</b><span>sky grade</span></div>
              </div>
              {sparkline}
              <h3>Sky now</h3>
              {meter("Stars in view", status['last_sky_stars'], cfg_day['min_stars'], True, stars_cls)}
              {meter("Background noise", status['last_sky_stddev'], cfg_day['stddev_threshold'], False, noise_cls, "{:.1f}")}
              <dl class="kvs compact">
                {kv("Session", f"{dur_total_min // 60} h {dur_total_min % 60:02d} min")}
                {kv("Noise blobs rejected", f"{self.session_stats['rejected_contours']} (ratio {noise_ratio:.2f})")}
                {kv("Moon in view", f"{self.session_stats['moon_visible_minutes']} min, peak impact {int(self.session_stats['max_moon_impact'] * 100)}%")}
              </dl>
              <p class="note">{esc(chronicle)}</p>
            </section>"""

            # ---------- activity strip ----------
            recent_events = status.get("recent_events", []) or []
            if recent_events:
                items = ""
                for ev in reversed(recent_events):
                    try:
                        t_disp = datetime.strptime(ev["tstamp"], "%Y%m%dT%H%M%S").strftime("%H:%M:%S")
                    except Exception:
                        t_disp = str(ev.get("tstamp", ""))
                    kept_cls = "ok" if ev.get("is_kept") else "err"
                    vid_link = f"/{ev['vid_url']}{token_param}"
                    thumb = (f'<img src="/{ev["img_url"]}{token_param}" alt="">' if ev.get("img_url")
                             else '<span class="noimg">No image</span>')
                    items += (f'<a class="det {kept_cls}" href="{vid_link}" target="_blank" title="Watch video">{thumb}'
                              f'<span class="t">{esc(t_disp)}</span>'
                              f'<span class="l {kept_cls}">{esc(ev.get("label", "?"))} {esc(ev.get("score", ""))}%</span></a>')
                activity = f'<div class="strip">{items}</div><p class="note">Green: kept. Red: filtered. Select a detection to watch its video.</p>'
            else:
                if status["weather_hold"] or status["daylight_mode"]:
                    why = "Meteor detection is paused while " + ("it is daylight." if status["daylight_mode"] else "the weather hold is active.")
                else:
                    why = "Detection is running; nothing has triggered yet."
                activity = f'<div class="empty">No detections yet tonight. {esc(why)}</div>'
            activity_panel = f'<section class="panel span-12"><h2>Detections tonight</h2>{activity}</section>'

            # ---------- last event ----------
            last_event = status["last_event_files"]
            event_image_path = last_event.get("image")
            event_video_path = last_event.get("video")
            if event_image_path and os.path.exists(event_image_path):
                ai_verdict = last_event.get("ai_verdict", "")
                verdict_cls = "ok" if "METEOR" in ai_verdict.upper() else "warn"
                folder = "ai_filtered" if "ai_filtered" in event_image_path else "events"
                img_url = f"/{folder}/{os.path.basename(event_image_path)}{token_param}"
                evt_mtime = int(os.path.getmtime(event_image_path))
                video_link = ""
                if event_video_path and os.path.exists(event_video_path):
                    vfolder = "ai_filtered" if "ai_filtered" in event_video_path else "events"
                    video_link = f'<a class="btn" href="/{vfolder}/{os.path.basename(event_video_path)}{token_param}" download>Download video</a>'
                last_event_body = f"""
                  <div class="head-row">{chip(f"AI: {ai_verdict}", verdict_cls)}{video_link}</div>
                  <a href="{img_url}" target="_blank"><img src="{img_url}&t={evt_mtime}" alt="Last event stack"></a>
                  <figcaption>{esc(os.path.basename(event_image_path))}, in {esc(folder.replace('_', ' '))}.</figcaption>"""
            else:
                last_event_body = '<div class="empty tall">No event captured yet this session.</div>'
            last_event_panel = f'<section class="panel span-7 media"><h2>Last event</h2>{last_event_body}</section>'

            # ---------- accumulated trails ----------
            narrative = self.get_cosmic_narrative(status["sky_score"], status["last_sky_stars"], status["moon"]["alt"] > 0)
            nrt_mtime = int(os.path.getmtime(self.mip_path)) if (self.mip_path and os.path.exists(self.mip_path)) else 0
            token_qs = f"&token={DASH_TOKEN}" if DASH_TOKEN else ""
            trails_panel = f"""
            <section class="panel span-5 media">
              <h2>All trails since sunset</h2>
              <img src="/output/masterpiece/masterpiece_live.png?t={nrt_mtime}{token_qs}" alt="Accumulated meteor trails">
              <figcaption>{esc(narrative)} {status['event_count']} paths recorded tonight. <a href="/sessions{token_param}">Earlier nights</a></figcaption>
            </section>"""

            # ---------- detail lists ----------
            thermal_class = "ok" if status["thermal_status"] == "OK" else "err"
            dark_class = {"Dynamic": "ok", "Static": ""}.get(status["dark_mode"], "warn")
            obs = (status["obs_condition"] or "n/a").lower()
            obs_class = "ok" if obs in ("clear_day", "clear_night") else "warn" if obs in ("cloud", "cloudy") else "muted" if "n/a" in obs else "err"
            spd = status.get("loop_time_ms", 0.0)
            time_budget_ms = 33.3 if status.get("daylight_mode") else max(33.3, status["current_exposure"] / 1000.0)
            speed_class = "ok" if spd < time_budget_ms * 0.75 else "warn" if spd < time_budget_ms * 0.95 else "err"
            base_thr = (self.cfg["detection"]["base_threshold"] if self.cfg["detection"]["auto_sensitivity_tuning"]
                        else self.cfg["detection"]["static_threshold"])
            sftp_row = ""
            if self.sftp_uploader:
                sftp_class, sftp_message = self.sftp_uploader.get_status()
                sftp_row = kv("Upload (SFTP)", esc(sftp_message), sftp_class)
            debug_row = kv("Debug mode", f"Level {status['debug_level']}", "warn") if status["debug_level"] > 0 else ""

            acquisition_panel = f"""
            <section class="panel span-4"><h2>Acquisition</h2><dl class="kvs">
              {kv("State", esc(pipeline_state_str), pipeline_state_class)}
              {kv("Mode", esc(pipeline_mode_str), pipeline_mode_class)}
              {kv("Exposure", f"{status['current_exposure'] / 1e6:.2f} s")}
              {kv("Gain", f"{status['current_gain']:.2f}×")}
              {kv("Detection threshold", f"{status['effective_threshold']} <small>base {base_thr}</small>")}
              {kv("Event buffers", f"{status['pre_frames']} before, {status['cool_frames']} after")}
              {kv("Frame processing", f"{spd:.0f} ms <small>of {time_budget_ms / 1000:.2f} s</small>", speed_class)}
              {kv("Dark frames", esc(status['dark_mode']), dark_class)}
              {debug_row}
            </dl></section>"""

            moon = status["moon"]
            sky_panel = f"""
            <section class="panel span-4"><h2>Sky and Moon</h2><dl class="kvs">
              {kv("Conditions", esc(weather_str), weather_class)}
              {kv("Observatory AI", f"{esc(obs.replace('_', ' '))} <small>{int((status['obs_conf'] or 0) * 100)}% confident</small>", obs_class)}
              {kv("Illuminance", f"{esc(status['last_illuminance'])} lux", illuminance_class)}
              {kv("Moon altitude", f"{moon['alt']:.1f}°")}
              {kv("Moon in field of view", "Yes" if moon['in_fov'] else "No", "err" if moon['in_fov'] else "ok")}
              {kv("Moon impact", f"{int(moon['impact'] * 100)}%")}
              {kv("Camera pointing", f"az {status['cam_az']:.1f}°, alt {status['cam_alt']:.1f}°")}
              {kv("Site", f"<a href=\"/locations{token_param}\">{esc(status['loc_name'])}</a> <small>{status['lat']:.4f}, {status['lon']:.4f}</small>")}
            </dl></section>"""

            try:
                du = psutil.disk_usage(self.cfg["janitor"]["monitor_path"])
                disk_txt = f"{du.used / 1024**3:.1f} of {du.total / 1024**3:.0f} GB ({du.percent:.0f}%)"
                disk_pct = du.percent
            except OSError:
                # disk_used_mb is in MB (the old card printed it with a "GB" unit)
                disk_txt = f"{self.disk_used_mb / 1024:.1f} GB ({self.disk_percent}%)"
                disk_pct = self.disk_percent
            disk_class = "err" if disk_pct > 90 else "warn" if disk_pct > threshold else "ok"
            sat_row = ""
            if self.cfg["satellites"]["enabled"]:
                # Unattended download schedule (satellites.SatelliteCatalog): show the state and the
                # next attempt; there is nothing for the operator to do.
                sat_cat, _ = self.satellite_catalog.catalogue()
                sat_age = self.satellite_catalog.age_hours()
                sched = self.satellite_catalog.status()
                failing = {g: r for g, r in sched.items() if r.get("result") is False}
                nxt = min((r["next_attempt"] for r in sched.values()), default=0)
                nxt_txt = datetime.fromtimestamp(nxt).strftime("%d/%m %H:%M") if nxt > time.time() else "soon"
                fail_txt = ""
                if failing:
                    g, r = next(iter(failing.items()))
                    fail_txt = (f"<br><small class=\"warn\">{esc(g)}: last download failed ({esc(r.get('status'))}), "
                                f"{esc(r.get('failures'))} in a row; next try {nxt_txt}</small>")
                if len(sat_cat):
                    sat_row = kv("Satellite catalogue", f"{len(sat_cat)} objects <small>{sat_age:.0f} h old</small>{fail_txt}",
                                 "warn" if failing or sat_age >= self.cfg["satellites"]["refresh_hours"] * 2 else "ok")
                else:
                    sat_row = kv("Satellite catalogue", f"Not available yet <small>next try {nxt_txt}</small>{fail_txt}", "warn")
            system_panel = f"""
            <section class="panel span-4"><h2>System</h2><dl class="kvs">
              {kv("Data drive", disk_txt, disk_class)}
              {kv("Storage governor", esc(storage_word), storage_class)}
              {kv("Memory", f"{mem_used} ({mem.percent:.0f}%)", "warn" if mem.percent > 80 else "ok")}
              {kv("CPU temperature", cpu_temp, "warn" if cpu_temp_raw > 75 else "ok")}
              {kv("Camera / outside", f"{status['sensor_temp_cam']:.1f}°C / {status['sensor_temp_ext']:.1f}°C", thermal_class)}
              {kv("Load (1 min)", f"{load1:.2f}")}
              {kv("Power", esc(status['power_status']), "ok" if status['power_status'] == "OK" else "muted")}
              {sftp_row}
              {sat_row}
              {kv("Heartbeat", esc(status['last_heartbeat']))}
              {kv("Uptime", uptime_str)}
            </dl></section>"""

            # ---------- health ----------
            health_stats = self.get_health_statistics(self.health_log_out_file)
            if health_stats:
                rows = ""
                for e in health_stats:
                    lvl = str(e.get("level", "INFO")).lower()
                    lvl_cls = {"warning": "warn", "error": "err", "critical": "err"}.get(lvl, "muted")
                    rows += (f'<tr><td><span class="dot {lvl_cls}"></span>{esc(e.get("event_type", "N/A"))}</td>'
                             f'<td class="num">{esc(e.get("count", 0))}</td><td>{esc(e.get("last_message", ""))}</td></tr>')
                health_body = (f'<div class="scroll"><table class="grid"><thead><tr><th>Event</th><th class="num">Count</th>'
                               f'<th>Last message</th></tr></thead><tbody>{rows}</tbody></table></div>')
            else:
                health_body = '<div class="empty">No health events recorded yet.</div>'
            health_panel = f'<section class="panel span-7"><h2>Health events</h2>{health_body}</section>'

            # ---------- threads: core threads listed, server helpers summarised ----------
            base_thread_names = {t.name for t in self.control_threads} | {t.name for t in self.worker_threads}
            active_thread_names = set(status["live_threads"])
            core, helpers = [], 0
            for name in sorted(base_thread_names | active_thread_names):
                if name == "MainThread":
                    continue
                running = name in active_thread_names
                if name not in base_thread_names and not name.endswith("Thread"):
                    helpers += running  # web-server / library helper threads
                    continue
                if not running and name not in base_thread_names:
                    continue
                core.append((name, running))
            stopped = [n for n, r in core if not r]
            thread_chips = "".join(
                f'<li class="{"ok" if r else "err"}"><span class="dot {"ok" if r else "err"}"></span>{esc(n.replace("Thread", ""))}</li>'
                for n, r in core)
            thread_summary = (f'<p class="err"><b>{len(stopped)} stopped:</b> {esc(", ".join(stopped))}</p>' if stopped
                              else f'<p class="ok">All {len(core)} pipeline threads running.</p>')
            threads_panel = (f'<section class="panel span-5"><h2>Threads</h2>{thread_summary}'
                             f'<ul class="threads">{thread_chips}</ul>'
                             f'<p class="note">{helpers} web-server helper thread(s) not listed.</p></section>')

            # ---------- log ----------
            log_lines = self.read_last_lines(os.path.join(self.general_log_dir, "pipeline.log"), num_lines=30)
            def _lvl(line):
                return ("err" if ("[ERROR]" in line or "[CRITICAL]" in line)
                        else "warn" if "[WARNING]" in line else "")
            log_html = "\n".join(f'<span class="{_lvl(l)}">{esc(l.rstrip())}</span>' for l in log_lines)
            log_panel = (f'<section class="panel span-12"><h2>Log <small>last {len(log_lines)} lines, newest at the bottom</small></h2>'
                         f'<pre class="log" id="log">{log_html}</pre></section>')

            # ---------- footer ----------
            footer = f"""
            <footer>
              <nav><h3>Analysis</h3><a href="/sessions{token_param}">Session archive</a><a href="/stats{token_param}">Event statistics</a><a href="/night_archive{token_param}">Night journals (PDF)</a></nav>
              <nav><h3>Files</h3><a href="/files{token_param}">Browse events and timelapses</a></nav>
              <nav><h3>Configuration</h3><a href="/locations{token_param}">Locations</a><a href="/settings{token_param}">Full configuration</a><a href="/system_time{token_param}">Set system time</a></nav>
              <nav><h3>Downloads</h3><a href="/logs/pipeline.log{token_param}">pipeline.log</a><a href="/logs/events.csv{token_param}">events.csv</a>
                <a href="/logs/health_stats.csv{token_param}">health_stats.csv</a><a href="/logs/system_monitor.csv{token_param}">system_monitor.csv</a>
                <a href="/logs/camera_k.csv{token_param}">camera_k.csv</a><a href="/logs/faulthandler.log{token_param}">faulthandler.log</a></nav>
            </footer>"""

            # ---------- header ----------
            schedule_txt = f"{'Active' if status['is_in_schedule'] else 'Outside schedule'} {status['start']}–{status['end']}"
            header = f"""
            <header>
              <div class="brand">
                <img src="/assets/logo_dashboard.jpg{token_param}" alt="">
                <div><b>Ryusei</b> <small>v{esc(status['version'])}</small><span>{esc(status['loc_name'])}</span></div>
              </div>
              <div class="chips">
                {chip(schedule_txt, system_status_class)}
                {chip(pipeline_mode_str, pipeline_mode_class)}
                {chip(storage_word, storage_class)}
              </div>
              <div class="right">
                <span class="clock"><b>{system_time}</b><small>{system_date}{', auto-refresh 10 s' if not status['maintenance_mode'] else ''}</small></span>
                {action_button}
              </div>
            </header>"""

            refresh_meta = ("" if status["maintenance_mode"]
                            else f'<meta http-equiv="refresh" content="10; url=/{token_param}">')

            html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
{refresh_meta}
<title>Ryusei · {esc(status['loc_name'])}</title>
<style>
{THEME_ROOT}
* {{ box-sizing:border-box; }}
html {{ background:var(--ink); }}
body {{ margin:0; color:var(--text); background:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-variant-numeric:tabular-nums; }}
a {{ color:var(--star); text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
a:focus-visible, button:focus-visible, input:focus-visible {{ outline:2px solid var(--star); outline-offset:2px; }}
small {{ color:var(--muted); font-size:12px; }}
.ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .err {{ color:var(--err); }}
.idle, .muted {{ color:var(--idle); }}

header {{ position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; align-items:center; gap:12px 24px;
  padding:10px 24px; background:rgba(15,20,28,.94); border-bottom:1px solid var(--rule); backdrop-filter:blur(6px); }}
.brand {{ display:flex; align-items:center; gap:10px; }}
.brand img {{ width:34px; height:34px; border-radius:6px; object-fit:cover; }}
.brand b {{ font-size:17px; letter-spacing:.2px; }}
.brand span {{ display:block; color:var(--muted); font-size:12px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; flex:1; }}
.chip {{ display:inline-flex; align-items:center; gap:7px; padding:4px 11px; border-radius:999px; font-size:13px;
  background:var(--raised); border:1px solid var(--rule); color:var(--text); white-space:nowrap; }}
.chip i {{ width:8px; height:8px; border-radius:50%; background:currentColor; }}
.chip.ok i {{ background:var(--ok); }} .chip.warn i {{ background:var(--warn); }} .chip.err i {{ background:var(--err); }} .chip.idle i {{ background:var(--idle); }}
.chip.ok, .chip.warn, .chip.err, .chip.idle {{ color:var(--text); }}
.right {{ display:flex; align-items:center; gap:16px; }}
.clock {{ text-align:right; line-height:1.2; }} .clock b {{ font-size:17px; display:block; }}
.btn {{ display:inline-block; padding:7px 14px; border-radius:6px; border:1px solid var(--rule); background:var(--raised);
  color:var(--text); font:inherit; cursor:pointer; }}
.btn:hover {{ border-color:var(--muted); text-decoration:none; }}
.btn.primary {{ background:var(--star); border-color:var(--star); color:#0f141c; font-weight:600; }}

main {{ max-width:1440px; margin:0 auto; padding:20px 24px 8px; display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px; }}
.banners {{ max-width:1440px; margin:0 auto; padding:16px 24px 0; display:grid; gap:8px; }}
.banner {{ padding:10px 14px; border-radius:var(--r); border:1px solid; background:var(--panel); }}
.banner.err {{ border-color:var(--err); }} .banner.warn {{ border-color:var(--warn); }}
.banner.info, .banner-success {{ border-color:var(--star); }} .banner-warn {{ border-color:var(--warn); }} .banner-danger {{ border-color:var(--err); }}
.banner b {{ color:var(--text); }}

.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:var(--r); padding:16px 18px; min-width:0; }}
.panel h2 {{ margin:0 0 12px; font-size:16px; font-weight:600; }}
.panel h2 small {{ font-weight:400; margin-left:6px; }}
.panel h3 {{ margin:16px 0 8px; font-size:14px; font-weight:600; color:var(--muted); }}
.span-4 {{ grid-column:span 4; }} .span-5 {{ grid-column:span 5; }} .span-7 {{ grid-column:span 7; }}
.span-8 {{ grid-column:span 8; }} .span-12 {{ grid-column:span 12; }}

.hero {{ padding:0; overflow:hidden; display:flex; flex-direction:column; background:#05070b; }}
.hero img {{ display:block; width:100%; height:auto; }}
.hero figcaption, .media figcaption {{ padding:8px 0 0; color:var(--muted); font-size:12px; }}
.hero figcaption {{ padding:8px 14px 10px; background:var(--panel); border-top:1px solid var(--rule); }}
.media img {{ display:block; width:100%; height:360px; object-fit:contain; border-radius:4px; background:#05070b; }}
.head-row {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:10px; }}

.stats {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:6px; overflow:hidden; }}
.stats div {{ background:var(--raised); padding:10px 12px; }}
.stats b {{ display:block; font-size:26px; font-weight:600; line-height:1.1; }}
.stats span {{ color:var(--muted); font-size:12px; }}
.spark {{ margin-top:10px; }} .spark svg {{ width:100%; height:40px; }}
.spark polyline {{ fill:none; stroke:currentColor; stroke-width:2; stroke-linejoin:round; vector-effect:non-scaling-stroke; }}
.spark span {{ font-size:12px; }}
.meter {{ margin:10px 0; }}
.meter-head {{ display:flex; justify-content:space-between; gap:8px; font-size:13px; margin-bottom:5px; }}
.meter-head small {{ margin-left:4px; }}
.track {{ position:relative; height:6px; border-radius:3px; background:var(--raised); }}
.fill {{ position:absolute; inset:0 auto 0 0; border-radius:3px; background:currentColor; }}
.mark {{ position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--text); }}

dl.kvs {{ margin:0; }}
.kv {{ display:grid; grid-template-columns:minmax(120px,42%) 1fr; gap:12px; padding:6px 0; border-top:1px solid var(--rule); }}
.kv:first-child {{ border-top:0; }}
.kv dt {{ color:var(--muted); }} .kv dd {{ margin:0; overflow-wrap:anywhere; }}
.compact .kv {{ padding:4px 0; font-size:13px; }}
.note {{ margin:10px 0 0; color:var(--muted); font-size:12px; }}

.empty {{ padding:22px; text-align:center; color:var(--muted); border:1px dashed var(--rule); border-radius:6px; }}
.empty.tall {{ min-height:220px; display:grid; place-items:center; }}

.strip {{ display:flex; gap:10px; overflow-x:auto; padding-bottom:4px; }}
.det {{ flex:0 0 132px; display:flex; flex-direction:column; gap:3px; padding:6px; border-radius:6px; background:var(--raised);
  border:1px solid var(--rule); color:var(--text); }}
.det:hover {{ border-color:var(--muted); text-decoration:none; }}
.det img, .det .noimg {{ width:100%; height:74px; object-fit:cover; border-radius:4px; background:#05070b; }}
.det .noimg {{ display:grid; place-items:center; color:var(--muted); font-size:12px; }}
.det.ok {{ border-left:3px solid var(--ok); }} .det.err {{ border-left:3px solid var(--err); }}
.det .t {{ font-size:13px; }} .det .l {{ font-size:12px; }}

.scroll {{ max-height:240px; overflow:auto; }}
table.grid {{ width:100%; border-collapse:collapse; font-size:13px; }}
table.grid th {{ position:sticky; top:0; background:var(--panel); text-align:left; font-weight:600; color:var(--muted); padding:6px 8px; border-bottom:1px solid var(--rule); }}
table.grid td {{ padding:6px 8px; border-bottom:1px solid var(--rule); vertical-align:top; }}
.num {{ text-align:right; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; background:var(--idle); vertical-align:middle; }}
.dot.ok {{ background:var(--ok); }} .dot.warn {{ background:var(--warn); }} .dot.err {{ background:var(--err); }}
ul.threads {{ list-style:none; margin:8px 0 0; padding:0; display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:4px 12px; font-size:13px; }}
ul.threads li {{ color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
ul.threads li.err {{ color:var(--err); }}

pre.log {{ margin:0; max-height:320px; overflow:auto; padding:10px 12px; background:var(--ink); border:1px solid var(--rule);
  border-radius:6px; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }}

.maint {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px; }}
table.form {{ width:100%; border-collapse:collapse; margin-bottom:12px; }}
table.form th {{ text-align:left; color:var(--muted); font-weight:600; padding:10px 0 4px; }}
table.form td {{ padding:4px 8px 4px 0; }}
table.form input {{ width:100%; padding:6px 8px; background:var(--ink); color:var(--text); border:1px solid var(--rule); border-radius:4px; font:inherit; }}
.shot img {{ max-width:100%; border-radius:4px; }}

footer {{ max-width:1440px; margin:8px auto 0; padding:20px 24px 32px; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px;
  border-top:1px solid var(--rule); }}
footer h3 {{ margin:0 0 6px; font-size:13px; color:var(--muted); font-weight:600; }}
footer nav a {{ display:block; padding:2px 0; }}

@media (max-width:1100px) {{
  .span-4, .span-5, .span-7, .span-8 {{ grid-column:span 6; }}
  .hero {{ grid-column:span 12; }} .tonight {{ grid-column:span 12; }}
}}
@media (max-width:720px) {{
  main {{ grid-template-columns:1fr; padding:12px; gap:12px; }}
  .span-4, .span-5, .span-7, .span-8, .span-12, .hero, .tonight {{ grid-column:1 / -1; }}
  header {{ padding:10px 12px; }} .chips {{ order:3; flex-basis:100%; }}
  .maint {{ grid-template-columns:1fr; }} footer {{ grid-template-columns:1fr 1fr; padding:16px 12px 24px; }}
  .media img {{ height:auto; max-height:60vh; }}
}}
</style>
</head>
<body>
{header}
{'<div class="banners">' + "".join(banners) + '</div>' if banners else ''}
<main>
  {maintenance_panel}
  <figure class="panel span-8 hero" style="margin:0">{hero_img}</figure>
  {tonight_panel}
  {activity_panel}
  {last_event_panel}
  {trails_panel}
  {acquisition_panel}
  {sky_panel}
  {system_panel}
  {health_panel}
  {threads_panel}
  {log_panel}
</main>
{footer}
<script>var l=document.getElementById("log");if(l)l.scrollTop=l.scrollHeight;</script>
</body>
</html>"""

            return html_page
        except Exception as e:
            logging.error("Dashboard render error", exc_info=True)
            return f"<pre>Dashboard error: {html.escape(str(e))}</pre>", 500
