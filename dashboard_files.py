"""
dashboard_files.py

Static-file serving and the Files manager page (download/delete/zip) for the
Flask dashboard. Split out of dashboard.py -- method bodies are unchanged,
only their physical location. Registered into the shared Flask app by
dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
this is a plain function), kept under that name so every `self.xxx` call
copied from the original nested route(s) needed no rewriting.
"""
import io
import logging
import os
import html
from zipfile import ZipFile

from flask import Response, redirect, request, send_from_directory, url_for

from dashboard_theme import render_page


def register_file_routes(app, self, DASH_TOKEN, require_token, ALLOWED_DIRS, TIMELAPSE_DIR, EVENTS_DIR):
    """Assets, calibration image, events/timelapse/daylight/masterpiece file serving, and the Files page."""

    @app.route("/assets/<path:filename>")
    @require_token
    def serve_assets(filename):
        """Serves the dashboard logos and UI assets."""
        return send_from_directory(self.assets_dir, filename)

    @app.route("/calibration_image")
    @require_token
    # The image coming from the calibration settings
    def serve_calibration_image():
        with self.status_lock:
            path = self.last_calibration_image_path
        if path and os.path.exists(path):
            return send_from_directory(
                os.path.dirname(path), os.path.basename(path)
            )
        return "No image available.", 404

    @app.route("/files/action", methods=["POST"])
    @require_token
    # Download and delete file from files page
    def file_action():
        selected_files = request.form.getlist("selected_files")
        action = request.form.get("action")

        if not selected_files:
            logging.warning("File action requested but no files were selected.")
            return redirect(url_for("file_manager", token=DASH_TOKEN))

        # SECURITY VALIDATION
        for f_path in selected_files:
            if not self.is_safe_path(ALLOWED_DIRS, f_path):
                logging.error(
                    "SECURITY ALERT: Attempted action on unsafe path: %s", f_path
                )
                return "Forbidden: Path is outside of allowed directories.", 403

        if action == "delete":
            logging.warning(
                "User initiated deletion of %d file(s) from dashboard.",
                len(selected_files),
            )
            deleted_count = 0
            for f_path in selected_files:
                try:
                    os.remove(f_path)
                    # Also try to remove associated .json file
                    json_path = os.path.splitext(f_path)[0] + ".json"
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    deleted_count += 1
                except OSError as e:
                    logging.error("Failed to delete file %s: %s", f_path, e)
            logging.info("Successfully deleted %d file(s).", deleted_count)
            return redirect(url_for("file_manager", token=DASH_TOKEN))

        elif action == "download":
            if len(selected_files) == 1:
                # Download a single file directly
                f_path = selected_files[0]
                try:
                    directory = os.path.dirname(f_path)
                    filename = os.path.basename(f_path)
                    return send_from_directory(
                        directory, filename, as_attachment=True
                    )
                except FileNotFoundError:
                    logging.error(
                        "Attempted to download a file that does not exist: %s",
                        f_path,
                    )
                    return "File not found.", 404
            else:
                # Zip multiple files and download
                memory_file = io.BytesIO()
                with ZipFile(memory_file, "w") as zipf:
                    for f_path in selected_files:
                        try:
                            # This operation is now protected.
                            if os.path.exists(f_path):
                                zipf.write(f_path, arcname=os.path.basename(f_path))
                            else:
                                logging.warning(
                                    "Skipping missing file during zip creation: %s",
                                    f_path,
                                )
                        except Exception as e:
                            # Catch any other file-related errors (e.g., permissions)
                            logging.error(
                                "Error adding file to zip archive %s: %s", f_path, e
                            )
                memory_file.seek(0)

                # If we added at least one file, send the zip. Otherwise, redirect.
                if len(zipf.infolist()) > 0:
                    return Response(
                        memory_file,
                        mimetype="application/zip",
                        headers={
                            "Content-Disposition": "attachment;filename=ryusei_download.zip"
                        },
                    )
                else:
                    logging.warning(
                        "Multi-file download requested, but no valid files could be zipped."
                    )
                    return redirect(url_for("file_manager", token=DASH_TOKEN))

        return "Invalid action.", 400

    @app.route("/files")
    @require_token
    # Files page
    def file_manager():
        event_files = self.get_file_list(EVENTS_DIR)
        timelapse_files = self.get_file_list(TIMELAPSE_DIR)
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""

        def file_table(files, group):
            if not files:
                return '<div class="empty">No files.</div>'
            rows = "".join(
                f'<tr><td><input type="checkbox" name="selected_files" value="{html.escape(f["path"])}" '
                f'data-group="{group}" aria-label="Select {html.escape(f["name"])}"></td>'
                f'<td>{html.escape(f["name"])}</td><td class="num">{html.escape(str(f["size_mb"]))}</td>'
                f'<td>{html.escape(str(f["mtime"]))}</td></tr>'
                for f in files)
            return (f'<div class="scroll"><table class="grid"><thead><tr>'
                    f'<th><input type="checkbox" data-all="{group}" aria-label="Select all"></th>'
                    f'<th>File</th><th class="num">Size</th><th>Modified</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div>')

        body = f"""
        <h1 class="page-title">Files</h1>
        <p class="page-intro">Event and timelapse files on the data drive. Select files to download or delete them.</p>
        <form action="/files/action{token_param}" method="post" style="display:grid;gap:16px">
          <section class="panel"><h2>Events <small>{len(event_files)} files</small></h2>{file_table(event_files, "ev")}</section>
          <section class="panel"><h2>Timelapse <small>{len(timelapse_files)} files</small></h2>{file_table(timelapse_files, "tl")}</section>
          <div class="sticky-actions actions">
            <button type="submit" name="action" value="download" class="btn primary">Download selected</button>
            <button type="submit" name="action" value="delete" class="btn danger"
                    onclick="return confirm('Permanently delete the selected files?');">Delete selected</button>
            <small id="selcount">0 selected</small>
          </div>
        </form>
        <script>
          const boxes = () => document.querySelectorAll('input[name="selected_files"]');
          const count = () => document.getElementById('selcount').textContent =
              [...boxes()].filter(b => b.checked).length + ' selected';
          document.querySelectorAll('input[data-all]').forEach(all => all.addEventListener('change', () => {{
            document.querySelectorAll('input[data-group="' + all.dataset.all + '"]').forEach(b => b.checked = all.checked);
            count();
          }}));
          boxes().forEach(b => b.addEventListener('change', count));
        </script>"""
        return render_page("Files", body, token_param=token_param, current="/files",
                           location=self.cfg["general"].get("location") or "",
                           extra_css="table.grid th:first-child, table.grid td:first-child { width:40px; }")

    @app.route("/latest_timelapse_image")
    @require_token
    # The last timelapse image
    def serve_latest_timelapse_image():
        # This logic is safe because it only ever serves from one specific directory
        latest_image_path = self.get_latest_file(TIMELAPSE_DIR)
        if latest_image_path and os.path.exists(latest_image_path):
            return send_from_directory(
                os.path.dirname(latest_image_path),
                os.path.basename(latest_image_path),
            )
        # Return a 404 if no image is found
        return "No timelapse image available.", 404

    @app.route("/events/<path:filename>")
    @require_token
    def serve_event_file(filename):
        """Serves the captured event images and videos."""
        # A quarantined event's video/json is moved to ai_filtered_out_dir
        # by a background thread shortly after the URL is handed out, so a
        # link built before that move completes must still resolve.
        if not os.path.exists(os.path.join(self.events_out_dir, filename)) and os.path.exists(
            os.path.join(self.ai_filtered_out_dir, filename)
        ):
            return send_from_directory(self.ai_filtered_out_dir, filename)
        return send_from_directory(self.events_out_dir, filename)

    @app.route('/ai_filtered/<path:filename>')
    @require_token
    def serve_ai_filtered_file(filename):
        if not os.path.exists(os.path.join(self.ai_filtered_out_dir, filename)) and os.path.exists(
            os.path.join(self.events_out_dir, filename)
        ):
            return send_from_directory(self.events_out_dir, filename)
        return send_from_directory(self.ai_filtered_out_dir, filename)

    @app.route("/output/daylight/<path:filename>")
    @require_token
    def serve_daylight_file(filename):
        """Serves daylight/sky-condition snapshots."""
        return send_from_directory(self.daylight_out_dir, filename)
        
    @app.route("/output/masterpiece/<path:filename>")
    @require_token
    def serve_masterpiece_file(filename):
        """Serves the nightly masterpiece."""
        return send_from_directory(self.general_log_dir, filename)

    # Log Download Endpoint (unchanged)
    @app.route("/logs/<path:filename>")
    @require_token
    def download_log(filename):
        log_dir = os.path.abspath(self.general_log_dir)
        allowed_files = [
            "pipeline.log",
            "system_monitor.csv",
            "events.csv",
            "health_stats.csv",
            "camera_k.csv",
            "faulthandler.log",  # crash/forced-exit stack dumps (see _enable_fault_forensics)
        ]
        
        is_allowed_name = filename in allowed_files
        is_pdf = filename.lower().endswith(".pdf")
        
        if is_allowed_name or is_pdf:
            if os.path.exists(os.path.join(log_dir, filename)):
                return send_from_directory(log_dir, filename, as_attachment=True)
            else:
                return "File not found.", 404
        else:
            return "Access denied: Unauthorized file type.", 403
