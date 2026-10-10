"""
dashboard_settings.py

The /settings page and its save endpoint for the Flask dashboard. Split out
of dashboard.py -- method bodies are unchanged, only their physical location.
Registered into the shared Flask app by dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
this is a plain function), kept under that name so every `self.xxx` call
copied from the original nested route(s) needed no rewriting.
"""
import logging
import json
import html

from flask import redirect, request, url_for

from config_models import DEFAULTS, MainConfig
from locations import LOCATION_FIELDS
from dashboard_theme import render_page


def register_settings_routes(app, self, DASH_TOKEN, require_token):
    """Full configuration page: render current config, validate and save edits."""

    # Settings page
    @app.route("/api/save_settings", methods=["POST"])
    @require_token
    def save_settings():
        logging.warning(
            "Received request to save full configuration from dashboard."
        )
        try:
            # 1. Build a dictionary from the multi-part form data.
            # This correctly handles checkboxes by taking the last value ('true' if checked).
            form_dict = {}
            for key in request.form.keys():
                parts = key.split(".")
                d = form_dict
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = request.form.getlist(key)[-1]

            # 1b. The site is not on this form (Locations page): keep the configured one,
            # or this save would reset it to the defaults.
            with open(self.config_path, encoding="utf-8") as f:
                on_disk = json.load(f)
            general_disk = on_disk.get("general", {})
            form_dict.setdefault("general", {})
            for k in LOCATION_FIELDS:
                if k in general_disk:
                    form_dict["general"][k] = general_disk[k]
            # 1c. List fields arrive as JSON text.
            for section, fields in DEFAULTS.items():
                if not isinstance(fields, dict):
                    continue
                for k, default in fields.items():
                    v = form_dict.get(section, {}).get(k)
                    if isinstance(default, list) and isinstance(v, str):
                        try:
                            form_dict[section][k] = json.loads(v)
                        except ValueError:
                            pass  # left as text: validation reports it

            # 2. Validate the data from the form. Pydantic will handle type coercion.
            try:
                validated_model = MainConfig.model_validate(form_dict)
                final_cfg = validated_model.model_dump()
            except Exception as e:
                logging.error(f"New configuration failed validation: {e}")
                return f"Configuration validation failed. Check logs. <a href='/settings?token={DASH_TOKEN}'>Go back</a>"

            # 3. Save the perfectly validated configuration directly to disk.
            # This completely bypasses the old, buggy dictionary merging logic.
            self.save_json_atomic(self.config_path, final_cfg)

            # 4. Trigger the smart reloader.
            self.reload_config_signal.set()
            self.config_reloaded_ack.wait(timeout=3.0)
            self.config_reloaded_ack.clear()
            logging.info(
                "Successfully saved updated configuration file and reloaded."
            )
            return redirect(
                url_for("settings", token=DASH_TOKEN, save_success="true")
            )

        except Exception as e:
            logging.exception(
                "A critical error occurred in /api/save_settings: %s", e
            )
            return f"An internal error occurred. <a href='/settings?token={DASH_TOKEN}'>Go back</a>"
    

    # OTA update.
    @app.route('/api/ota_update', methods=['POST'])
    @require_token
    def ota_update():
        logging.warning("User-initiated OTA Update command received from dashboard.")
        success = self.perform_ota_update()
        
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
        refresh_url = url_for('dashboard', _external=True) + token_param
        
        if success:
            body = f"""<section class="panel status-card">
              <h1 class="ok">Update installed</h1>
              <p>The program is restarting to run the new code.</p>
              <p class="muted">You will be taken back to the dashboard automatically.
                <a href="{refresh_url}">Go now</a></p></section>"""
            head = f'<meta http-equiv="refresh" content="8;url={refresh_url}">'
        else:
            body = f"""<section class="panel status-card">
              <h1 class="err">Update failed</h1>
              <p>Nothing was changed and the program was <b>not</b> restarted.</p>
              <p class="muted">Details are in <a href="/logs/pipeline.log{token_param}">pipeline.log</a>.
                <a href="{refresh_url}">Back to the dashboard</a></p></section>"""
            head = ""
        return render_page("OTA update", body, token_param=token_param,
                           location=self.cfg["general"].get("location") or "", head_extra=head)

    token_param_settings = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""

    def generate_form_fields(data, prefix=""):
        html_out = ""
        # EDITABLE_BOOLEANS = ["auto_exposure_tuning", "auto_sensitivity_tuning", "stack_align", "astro_stretch"]
        # Iterate in dictionary order, to match DEFAULTS structure
        for key, value in data.items():
            current_prefix = f"{prefix}.{key}" if prefix else key
            if prefix == "general" and key in LOCATION_FIELDS:
                if key == LOCATION_FIELDS[0]:
                    html_out += (f'<tr><td><label>Site</label></td><td>{html.escape(str(data.get("location") or "not set"))} '
                                 f'<small>managed on the <a href="/locations{token_param_settings}">Locations</a> page</small></td></tr>')
                continue
            if isinstance(value, (list, tuple)):
                # Lists are edited as JSON (e.g. ["visual", "active"]); save_settings parses them back.
                val_str = html.escape(json.dumps(list(value)))
                html_out += (f'<tr><td><label for="{current_prefix}">{key}</label></td>'
                             f'<td><input type="text" id="{current_prefix}" name="{current_prefix}" value="{val_str}"></td></tr>')
                continue
            if isinstance(value, dict):
                html_out += f'<tr class="section"><th colspan="2">{html.escape(key.replace("_", " ").capitalize())}</th></tr>'
                html_out += generate_form_fields(value, prefix=current_prefix)
            else:
                if isinstance(value, bool):
                    is_editable = key in self.editable_booleans

                    if is_editable:
                        # This checkbox is user-editable
                        # label_cell = f'<td><label for="{current_prefix}">{key}</label></td>'
                        label_cell = f'<td><label class="editable-label" for="{current_prefix}">{key}</label></td>'
                        input_cell = f"""
                        <td>
                            <input type="hidden" name="{current_prefix}" value="false">
                            <input type="checkbox" id="{current_prefix}" name="{current_prefix}" value="true" {"checked" if value else ""}>
                        </td>
                        """
                    else:
                        # For booleans (read-only), generate a label WITHOUT the 'for' attribute
                        # This severs the link that allows the label to toggle the checkbox
                        # for future use, actually yhe checkboxes are disabled
                        label_cell = f"<td><label>{key}</label></td>"
                        input_cell = f"""
                        <td class="readonly-cell">
                            <input type="hidden" name="{current_prefix}" value="false">
                            <input type="checkbox" id="{current_prefix}" name="{current_prefix}" value="true" {"checked" if value else ""}>
                        </td>
                        """
                    html_out += f"<tr>{label_cell}{input_cell}</tr>"
                else:
                    # For all other editable types, keep the 'for' attribute for better UX
                    label_cell = (
                        f'<td><label for="{current_prefix}">{key}</label></td>'
                    )
                    val_str = html.escape(str(value if value is not None else ""))
                    input_cell = f'<td><input type="text" id="{current_prefix}" name="{current_prefix}" value="{val_str}" size="40"></td>'
                    html_out += f"<tr>{label_cell}{input_cell}</tr>"
        return html_out

    @app.route("/settings")
    @require_token
    def settings():
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""

        current_merged_config = self.load_and_validate_config(
            self.config_path, merge=False
        )
        form_fields = generate_form_fields(current_merged_config)

        save_success_banner = ""
        if request.args.get("save_success") == "true":
            save_success_banner = '<div class="banner ok"><b>Configuration saved and reloaded.</b></div>'

        body = f"""
        <h1 class="page-title">Configuration</h1>
        <p class="page-intro">The active configuration. Highlighted switches can be changed here; the others are shown for reference.</p>
        {save_success_banner}
        <form action="/api/save_settings{token_param}" method="post" style="display:grid;gap:16px">
          <section class="panel"><table class="grid settings">{form_fields}</table></section>
          <div class="sticky-actions actions">
            <button type="submit" class="btn primary">Save and reload configuration</button>
            <small>Values are validated before they are applied; an invalid file is rejected.</small>
          </div>
        </form>
        <section class="panel">
          <h2>System administration</h2>
          <p class="muted" style="margin-top:0">After significant changes a full restart is recommended.</p>
          <div class="actions">
            <form action="/api/restart_service{token_param}" method="post"
                  onsubmit="return confirm('Restart the Ryusei program? Active captures are interrupted.');">
              <button type="submit" class="btn warn">Restart program</button></form>
            <form action="/api/reboot_system{token_param}" method="post"
                  onsubmit="return confirm('Reboot the whole Raspberry Pi? The connection will be lost for about a minute.');">
              <button type="submit" class="btn danger">Reboot system</button></form>
            <form action="/api/ota_update{token_param}" method="post"
                  onsubmit="return confirm('Download and apply the OTA update? The program restarts.');">
              <button type="submit" class="btn">Apply OTA update</button></form>
            <a class="btn" href="/system_time{token_param}">Set system time</a>
          </div>
          <p class="note">A system reboot takes longer than a program restart.</p>
        </section>"""
        return render_page("Configuration", body, token_param=token_param, current="/settings",
                           location=self.cfg["general"].get("location") or "",
                           extra_css="""
            table.settings td:first-child { width:40%; color:var(--muted); }
            table.settings label.editable-label { color:var(--warn); font-weight:600; }
            table.settings .readonly-cell { pointer-events:none; opacity:.5; }
            """)
