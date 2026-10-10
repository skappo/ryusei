"""
dashboard_theme.py

The dashboard's shared presentation layer: design tokens (THEME_ROOT), component
styles (BASE_CSS), the sub-page navigation (NAV_LINKS) and render_page(), the page
shell every sub-page is rendered with.

A leaf module: it imports nothing from the dashboard. dashboard.py and every
dashboard_*.py page module import FROM it, so the dependency graph has no cycle
(before, the page modules imported render_page from dashboard.py, which in turn
imports the page modules).
"""
import html


# Design tokens shared by every dashboard page (night palette; see the main
# dashboard template for the rationale). Plain CSS, inserted into f-strings.
THEME_ROOT = """:root {
  --ink:#0f141c; --panel:#161d28; --raised:#1c2533; --rule:#263142;
  --text:#d5dce6; --muted:#8a96a8; --star:#8fb4ff;
  --ok:#5cb88a; --warn:#e0a84e; --err:#e5646c; --idle:#8a96a8;
  --r:8px;
}"""

# Component styles shared by every page other than the main dashboard (which carries
# the same rules inline). One look: same header, panels, buttons, tables, forms.
BASE_CSS = THEME_ROOT + """
* { box-sizing:border-box; }
html, body { background:var(--ink); }
body { margin:0; color:var(--text);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-variant-numeric:tabular-nums; }
a { color:var(--star); text-decoration:none; } a:hover { text-decoration:underline; }
a:focus-visible, button:focus-visible, input:focus-visible { outline:2px solid var(--star); outline-offset:2px; }
small, .muted { color:var(--muted); } small { font-size:12px; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .err { color:var(--err); }
h1, h2, h3 { font-weight:600; }

.top { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; align-items:center; gap:10px 20px;
  padding:10px 24px; background:rgba(15,20,28,.94); border-bottom:1px solid var(--rule); backdrop-filter:blur(6px); }
.top .brand { display:flex; align-items:center; gap:10px; color:var(--text); }
.top .brand:hover { text-decoration:none; }
.top .brand img { width:34px; height:34px; border-radius:6px; object-fit:cover; }
.top .brand b { font-size:17px; } .top .brand span { display:block; color:var(--muted); font-size:12px; }
.top nav { display:flex; flex-wrap:wrap; gap:4px; margin-left:auto; }
.top nav a { padding:6px 10px; border-radius:6px; color:var(--muted); }
.top nav a:hover { color:var(--text); background:var(--raised); text-decoration:none; }
.top nav a[aria-current="page"] { color:var(--text); background:var(--raised); }

main { max-width:1440px; margin:0 auto; padding:20px 24px 40px; display:grid; gap:16px; }
main.narrow { max-width:960px; }
.page-title { margin:0; font-size:20px; }
.page-intro { margin:-8px 0 0; color:var(--muted); }

.panel { background:var(--panel); border:1px solid var(--rule); border-radius:var(--r); padding:16px 18px; min-width:0; }
.panel h2 { margin:0 0 12px; font-size:16px; }
.panel h2 small { font-weight:400; margin-left:6px; }
.note { margin:10px 0 0; color:var(--muted); font-size:12px; }
.empty { padding:24px; text-align:center; color:var(--muted); border:1px dashed var(--rule); border-radius:6px; }

.btn { display:inline-block; padding:7px 14px; border-radius:6px; border:1px solid var(--rule); background:var(--raised);
  color:var(--text); font:inherit; cursor:pointer; white-space:nowrap; }
.btn:hover { border-color:var(--muted); text-decoration:none; }
.btn.primary { background:var(--star); border-color:var(--star); color:#0f141c; font-weight:600; }
.btn.warn { border-color:var(--warn); color:var(--warn); }
.btn.danger { border-color:var(--err); color:var(--err); }
.btn.warn:hover, .btn.danger:hover { background:var(--ink); }
.actions { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.actions form { margin:0; }
.sticky-actions { position:sticky; bottom:0; z-index:4; padding:12px 18px; background:rgba(22,29,40,.96);
  border:1px solid var(--rule); border-radius:var(--r); backdrop-filter:blur(6px); }

.banner { padding:10px 14px; border-radius:var(--r); border:1px solid var(--star); background:var(--panel); }
.banner.ok { border-color:var(--ok); } .banner.warn { border-color:var(--warn); } .banner.err { border-color:var(--err); }
.banner b { color:var(--text); }

.scroll { max-height:520px; overflow:auto; }
table.grid { width:100%; border-collapse:collapse; font-size:13px; }
table.grid th { position:sticky; top:0; z-index:1; background:var(--panel); text-align:left; font-weight:600; color:var(--muted);
  padding:7px 10px; border-bottom:1px solid var(--rule); }
table.grid td { padding:7px 10px; border-bottom:1px solid var(--rule); vertical-align:middle; }
table.grid tr:hover td { background:var(--raised); }
table.grid .num { text-align:right; }
table.grid tr.section th { position:static; padding-top:18px; color:var(--text); font-size:14px; background:transparent; }

label { color:var(--text); }
input[type=text], input[type=date], input[type=time], input[type=number], select {
  width:100%; max-width:420px; padding:7px 9px; background:var(--ink); color:var(--text);
  border:1px solid var(--rule); border-radius:5px; font:inherit; }
input[type=checkbox] { width:16px; height:16px; accent-color:var(--star); vertical-align:middle; }
.field { display:grid; gap:6px; margin-bottom:14px; }
.field label { font-weight:600; }

.status-card { max-width:560px; margin:12vh auto 0; text-align:center; }
.status-card h1 { margin:0 0 8px; font-size:22px; }

@media (max-width:720px) {
  .top { padding:10px 12px; } .top nav { margin-left:0; }
  main { padding:12px; }
  table.grid { font-size:12px; } table.grid th, table.grid td { padding:6px; }
}
"""

# Sub-page navigation (path, label).
NAV_LINKS = (
    ("/", "Dashboard"), ("/sessions", "Sessions"), ("/stats", "Statistics"),
    ("/night_archive", "Journals"), ("/files", "Files"), ("/locations", "Locations"), ("/settings", "Settings"),
)


def render_page(title, body, *, token_param="", current="", location="", narrow=False,
                head_extra="", extra_css=""):
    """
    Full HTML page in the dashboard's look: shared header (brand + navigation),
    BASE_CSS, then `body` inside <main>. Returned as a plain string -- never passed
    through Jinja, so text from files and logs can never be executed as a template.
    `body`, `head_extra` and `extra_css` are trusted markup built by the caller;
    every dynamic value inside them must already be html.escape()d.
    """
    nav = "".join(
        f'<a href="{path}{token_param}"' + (' aria-current="page"' if path == current else "") + f'>{label}</a>'
        for path, label in NAV_LINKS
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="dark">
<title>{html.escape(title)}, Ryusei</title>
<style>{BASE_CSS}{extra_css}</style>
{head_extra}
</head><body>
<header class="top">
  <a class="brand" href="/{token_param}"><img src="/assets/logo_dashboard.jpg{token_param}" alt="">
    <div><b>Ryusei</b><span>{html.escape(location)}</span></div></a>
  <nav aria-label="Pages">{nav}</nav>
</header>
<main{' class="narrow"' if narrow else ''}>
{body}
</main>
</body></html>"""
