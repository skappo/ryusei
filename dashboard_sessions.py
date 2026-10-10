"""
dashboard_sessions.py

Session archive pages (/sessions, /night_archive) for the Flask dashboard --
browsing masterpiece/startrail/keogram images and PDF journals from past
nights. Split out of dashboard.py -- method bodies are unchanged, only
their physical location. Registered into the shared Flask app by
dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
this is a plain function), kept under that name so every `self.xxx` call
copied from the original nested route(s) needed no rewriting.
"""
import html
import logging
import os

from flask import Response, abort, request, send_from_directory

from session_archive import (IMAGE_NAME_RE, collect_sessions, night_of,
                              prune_thumbnails, thumbnail_bytes)
from dashboard_theme import render_page


def register_session_routes(app, self, DASH_TOKEN, require_token):
    """Archived-night image/thumbnail serving plus the /sessions and /night_archive pages."""

    # ---------------- Session archive ----------------
    # Archived sessions live in logs/archive only (moved there when the next
    # session starts); the current session's files are never shown here.
    ARCHIVE_KINDS = ("masterpiece", "startrail", "keogram")
    THUMB_CACHE = os.path.join(self.general_log_dir, ".session_thumbs")

    def _archive_path(kind, name):
        """Strict allow-list: known kind, canonical file name, existing file. Else 404."""
        if kind not in ARCHIVE_KINDS or not IMAGE_NAME_RE.match(name) or not name.startswith(kind + "_"):
            abort(404)
        path = os.path.join(self.logs_archive, name)
        if not os.path.isfile(path):
            abort(404)
        return path

    @app.route("/sessions/image/<kind>/<name>")
    @require_token
    def session_image(kind, name):
        _archive_path(kind, name)
        return send_from_directory(self.logs_archive, name)

    @app.route("/sessions/thumb/<kind>/<name>")
    @require_token
    def session_thumb(kind, name):
        path = _archive_path(kind, name)
        d = self.cfg["dashboard"]
        try:
            data = thumbnail_bytes(path, THUMB_CACHE, d["archive_thumb_width"],
                                   d["archive_thumb_jpeg_quality"], admits=self.storage.admits)
        except (OSError, ValueError) as e:
            logging.warning(f"Session archive: no thumbnail for {name}: {e}")
            abort(404)
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp

    @app.route("/sessions")
    @require_token
    def sessions():
        esc = lambda v: html.escape(str(v))
        tok = f"token={DASH_TOKEN}" if DASH_TOKEN else ""
        q = lambda extra="": "?" + "&".join(p for p in (tok, extra) if p) if (tok or extra) else ""
        per_page = self.cfg["dashboard"]["archive_nights_per_page"]

        nights = collect_sessions(self.logs_archive)
        prune_thumbnails(THUMB_CACHE, {n for night in nights for k in ("masterpiece", "startrail", "keogram")
                                       for n, _ in night[k]})
        pages = max(1, -(-len(nights) // per_page))
        try:
            page = min(max(1, int(request.args.get("page", 1))), pages)
        except ValueError:
            page = 1
        shown = nights[(page - 1) * per_page: page * per_page]

        labels = {"masterpiece": "Meteor trails", "startrail": "Star trail", "keogram": "Keogram"}
        missing = {"masterpiece": "No trails archived", "startrail": "Not generated",
                   "keogram": "Not available"}

        def figure(kind, items):
            if not items:
                return (f'<figure class="shot"><div class="missing">{missing[kind]}</div>'
                        f'<figcaption>{labels[kind]}</figcaption></figure>')
            name, _ = items[0]
            more = ""
            if len(items) > 1:
                links = ", ".join(f'<a href="/sessions/image/{kind}/{esc(n)}{q()}" target="_blank">{i + 2}</a>'
                                  for i, (n, _) in enumerate(items[1:]))
                more = f' <small>(also from another session that night: {links})</small>'
            return (f'<figure class="shot"><a href="/sessions/image/{kind}/{esc(name)}{q()}" target="_blank" '
                    f'title="Open full size"><img loading="lazy" src="/sessions/thumb/{kind}/{esc(name)}{q()}" '
                    f'alt="{labels[kind]}"></a><figcaption>{labels[kind]}{more}</figcaption></figure>')

        rows = ""
        for night in shown:
            d = night["night"]
            journal = ""
            if night["journal"]:
                jname = night["journal"][0][0]
                journal = f'<a class="btn" href="/logs/archive/{esc(jname)}{q()}">Journal (PDF)</a>'
            rows += (f'<section class="night"><header><h2>Night of {d:%A} {d.day} {d:%B %Y}</h2>{journal}</header>'
                     f'<div class="trio">{figure("masterpiece", night["masterpiece"])}'
                     f'{figure("startrail", night["startrail"])}{figure("keogram", night["keogram"])}</div></section>')
        if not rows:
            rows = ('<div class="empty">No archived sessions yet. Each night is archived when the next '
                    'session starts.</div>')

        nav = ""
        if pages > 1:
            prev_l = (f'<a class="btn" href="/sessions{q(f"page={page - 1}")}">Newer nights</a>'
                      if page > 1 else '<span></span>')
            next_l = (f'<a class="btn" href="/sessions{q(f"page={page + 1}")}">Older nights</a>'
                      if page < pages else '<span></span>')
            nav = f'<nav class="pager">{prev_l}<span>Page {page} of {pages}</span>{next_l}</nav>'

        body = f"""
        <h1 class="page-title">Session archive</h1>
        <p class="page-intro">{len(nights)} archived night(s), newest first. Nights are dated by the evening they begin.
          Select an image to open it at full size.</p>
        {nav}
        {rows}
        {nav}"""
        return render_page("Session archive", body, token_param=q(), current="/sessions",
                           location=self.cfg["general"].get("location") or "", extra_css="""
            .night { background:var(--panel); border:1px solid var(--rule); border-radius:var(--r); padding:16px 18px; }
            .night header { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px; }
            .night h2 { margin:0; font-size:16px; }
            .trio { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
            .shot { margin:0; }
            .shot img, .missing { display:block; width:100%; aspect-ratio:16/9; object-fit:contain; border-radius:4px; background:#05070b; }
            .missing { display:grid; place-items:center; color:var(--muted); border:1px dashed var(--rule); background:transparent; }
            .shot figcaption { margin-top:6px; color:var(--muted); font-size:13px; }
            .pager { display:flex; justify-content:space-between; align-items:center; color:var(--muted); }
            @media (max-width:860px) { .trio { grid-template-columns:1fr; } }
            """)

    @app.route('/night_archive')
    @require_token
    def night_archive():
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
        pdfs = []
        try:
            entries = list(os.scandir(self.logs_archive))
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".pdf"):
                try:
                    st = entry.stat()
                except FileNotFoundError:
                    continue
                pdfs.append({"name": entry.name, "night": night_of(st.st_mtime), "size_kb": st.st_size / 1024})
        pdfs.sort(key=lambda x: (x["night"], x["name"]), reverse=True)

        if pdfs:
            rows = "".join(
                f'<tr><td>{p["night"]:%a %d %b %Y}</td><td>{html.escape(p["name"])}</td>'
                f'<td class="num">{p["size_kb"]:.0f} KB</td>'
                f'<td class="num"><a class="btn" href="/logs/archive/{html.escape(p["name"])}{token_param}">Open PDF</a></td></tr>'
                for p in pdfs)
            table = (f'<div class="scroll"><table class="grid"><thead><tr><th>Night of</th><th>File</th>'
                     f'<th class="num">Size</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>')
        else:
            table = '<div class="empty">No journals archived yet. A journal is written at the end of each session.</div>'
        body = f"""
        <h1 class="page-title">Night journals</h1>
        <p class="page-intro">{len(pdfs)} PDF journal(s), newest first. Nights are dated by the evening they begin.</p>
        <section class="panel"><h2>Journals</h2>{table}</section>"""
        return render_page("Night journals", body, token_param=token_param, current="/night_archive",
                           location=self.cfg["general"].get("location") or "")
