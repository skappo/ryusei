"""
dashboard_locations.py

The /locations page and its add/select/delete endpoints for the Flask
dashboard: observing sites kept in assets/locations.json (see locations.py),
with the ability to switch the station's active site (general.location /
latitude / longitude / camera_azimuth / camera_altitude in config.json).
Registered into the shared Flask app by dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
these are plain functions), kept under that name so every `self.xxx` call
needed no rewriting.
"""
import json
import logging
import html

from flask import redirect, request

from config_models import MainConfig
from locations import LOCATION_FIELDS, same_site
from dashboard_theme import render_page


def register_location_routes(app, self, DASH_TOKEN, require_token):
    """Observing-site list page plus its add/select/delete API endpoints."""

    def _apply_site(rec):
        """
        Make `rec` the active site: its five values go into config.json (general.*),
        the rest of the file is left exactly as it is, the whole configuration is
        validated before writing, and the live reloader applies it.
        """
        with open(self.config_path, encoding="utf-8") as f:
            raw = json.load(f)
        raw.setdefault("general", {})
        for k in LOCATION_FIELDS:
            raw["general"][k] = rec[k]
        MainConfig.model_validate(raw)  # raises: nothing is written
        self.save_json_atomic(self.config_path, raw)
        self.reload_config_signal.set()
        self.config_reloaded_ack.wait(timeout=3.0)
        self.config_reloaded_ack.clear()
        logging.warning(
            f"Locations: active site set to '{rec['location']}' (lat {rec['latitude']}, lon {rec['longitude']}, "
            f"camera az {rec['camera_azimuth']}, alt {rec['camera_altitude']})."
        )

    @app.route("/locations")
    @require_token
    def locations_page():
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
        esc = html.escape
        msg = {"added": ("ok", "Location added."), "selected": ("ok", "Active location changed and applied."),
               "deleted": ("ok", "Location removed.")}.get(request.args.get("done", ""))
        err = request.args.get("error")
        banner = ""
        if msg:
            banner = f'<div class="banner {msg[0]}"><b>{msg[1]}</b></div>'
        if err:
            banner = f'<div class="banner err"><b>Not saved:</b> {esc(err)}</div>'
        try:
            sites = self.location_store.all()
        except (OSError, ValueError) as e:
            sites = []
            banner += f'<div class="banner err"><b>Locations database unreadable:</b> {esc(str(e))}</div>'
        g = self.cfg["general"]
        rows = ""
        for r in sites:
            active = same_site(r, g)
            action = ('<span class="chip-ok">In use</span>' if active else
                      f'<form method="post" action="/api/locations/select{token_param}" '
                      f'onsubmit="return confirm(\'Switch the station to {esc(r["location"])}? Sky and satellite '
                      f'calculations change immediately.\');"><input type="hidden" name="id" value="{esc(r["id"])}">'
                      f'<button class="btn primary" type="submit">Use this location</button></form>'
                      f'<form method="post" action="/api/locations/delete{token_param}" '
                      f'onsubmit="return confirm(\'Remove {esc(r["location"])} from the list?\');">'
                      f'<input type="hidden" name="id" value="{esc(r["id"])}">'
                      f'<button class="btn danger" type="submit">Remove</button></form>')
            rows += (f'<tr{" class=active" if active else ""}><td><b>{esc(r["location"])}</b></td>'
                     f'<td class="num">{r["latitude"]:.6f}</td><td class="num">{r["longitude"]:.6f}</td>'
                     f'<td class="num">{r["camera_azimuth"]:.1f}°</td><td class="num">{r["camera_altitude"]:.1f}°</td>'
                     f'<td><div class="actions">{action}</div></td></tr>')
        table = (f'<div class="scroll"><table class="grid"><thead><tr><th>Location</th><th class="num">Latitude</th>'
                 f'<th class="num">Longitude</th><th class="num">Camera azimuth</th><th class="num">Camera altitude</th>'
                 f'<th></th></tr></thead><tbody>{rows}</tbody></table></div>' if rows else
                 '<div class="empty">No locations saved yet. Add one below.</div>')
        in_use = next((r["location"] for r in sites if same_site(r, g)), None)
        body = f"""
        <h1 class="page-title">Locations</h1>
        <p class="page-intro">Observing sites and camera pointing. The location in use is written to config.json;
          the list is kept in <code>assets/locations.json</code>.</p>
        {banner}
        <section class="panel"><h2>Saved locations <small>in use: {esc(in_use or g.get("location") or "not set")}</small></h2>{table}</section>
        <section class="panel"><h2>Add a location</h2>
          <form method="post" action="/api/locations/add{token_param}" class="loc-form">
            <div class="field"><label for="location">Name</label>
              <input type="text" id="location" name="location" required maxlength="80" placeholder="e.g. Tokyo"></div>
            <div class="field"><label for="latitude">Latitude <small>degrees, north positive (-90 to 90)</small></label>
              <input type="number" id="latitude" name="latitude" required step="any" min="-90" max="90"></div>
            <div class="field"><label for="longitude">Longitude <small>degrees, east positive (-180 to 180)</small></label>
              <input type="number" id="longitude" name="longitude" required step="any" min="-180" max="180"></div>
            <div class="field"><label for="camera_azimuth">Camera azimuth <small>0 = North, 90 = East (0 to 360)</small></label>
              <input type="number" id="camera_azimuth" name="camera_azimuth" required step="any" min="0" max="360" value="0"></div>
            <div class="field"><label for="camera_altitude">Camera altitude <small>90 = zenith, 0 = horizon</small></label>
              <input type="number" id="camera_altitude" name="camera_altitude" required step="any" min="0" max="90" value="90"></div>
            <div class="actions" style="grid-column:1/-1">
              <button type="submit" name="then" value="save" class="btn">Save</button>
              <button type="submit" name="then" value="use" class="btn primary">Save and use now</button>
            </div>
          </form>
        </section>"""
        return render_page("Locations", body, token_param=token_param, current="/locations",
                           location=g.get("location") or "", extra_css="""
            .loc-form { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:4px 18px; align-items:end; }
            .loc-form input { height:36px; max-width:none; }
            tr.active td { background:var(--raised); }
            .chip-ok { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--ok); color:var(--ok); font-size:12px; }
            table.grid td .actions { justify-content:flex-end; }
            """)

    def _back(**params):
        from urllib.parse import urlencode
        if DASH_TOKEN:
            params = {"token": DASH_TOKEN, **params}
        return redirect("/locations?" + urlencode(params))

    @app.route("/api/locations/add", methods=["POST"])
    @require_token
    def locations_add():
        try:
            rec = self.location_store.add({k: request.form.get(k, "") for k in LOCATION_FIELDS})
            if request.form.get("then") == "use":
                _apply_site(rec)
                return _back(done="selected")
            return _back(done="added")
        except Exception as e:
            logging.warning(f"Locations: add rejected: {e}")
            return _back(error=str(e).splitlines()[0][:200] if not hasattr(e, "errors") else
                         "; ".join(f"{x['loc'][0]}: {x['msg']}" for x in e.errors())[:300])

    @app.route("/api/locations/select", methods=["POST"])
    @require_token
    def locations_select():
        rec = self.location_store.get(request.form.get("id", ""))
        if rec is None:
            return _back(error="unknown location")
        try:
            _apply_site(rec)
        except Exception as e:
            logging.error(f"Locations: could not apply '{rec['location']}': {e}")
            return _back(error=f"configuration rejected the location: {str(e).splitlines()[0][:200]}")
        return _back(done="selected")

    @app.route("/api/locations/delete", methods=["POST"])
    @require_token
    def locations_delete():
        rec = self.location_store.get(request.form.get("id", ""))
        if rec is None:
            return _back(error="unknown location")
        if same_site(rec, self.cfg["general"]):
            return _back(error="the location in use cannot be removed; switch to another one first")
        self.location_store.delete(rec["id"])
        return _back(done="deleted")
