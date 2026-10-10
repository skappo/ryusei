"""
dashboard_stats.py

Event-statistics API endpoints and the /stats page (charts + nightly
history table) for the Flask dashboard. Split out of dashboard.py -- method
bodies are unchanged, only their physical location. Registered into the
shared Flask app by dashboard.py's dashboard_loop().

The `self` parameter below is the Pipeline instance (not a method's self --
this is a plain function), kept under that name so every `self.xxx` call
copied from the original nested route(s) needed no rewriting.
"""
import csv
import json
import logging
import os
from datetime import datetime, timedelta

from flask import jsonify

from config_models import EVENT_LOG_FIELDS
from dashboard_theme import render_page


def register_stats_routes(app, self, DASH_TOKEN, require_token):
    """Chronicle/detailed-events/event-stats JSON APIs and the /stats page."""

    # 1. API for the Chronicle (Historical Nights)
    @app.route('/api/chronicle')
    @require_token
    def get_chronicle():
        path = os.path.join(self.general_log_dir, "night_chronicle.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify([])

    # 2. Updated Event Stats with Filtering
    @app.route('/api/detailed_events')
    @require_token
    def detailed_events():
        events = []
        if not os.path.exists(self.event_log_out_file):
            return jsonify([])

        # events.csv may be headerless (see event_logger_loop): fall back to
        # explicit field names so DictReader doesn't eat the first event row.
        event_log_fields = EVENT_LOG_FIELDS

        try:
            with open(self.event_log_out_file, 'r', encoding="utf-8", newline="") as f:
                first_line = f.readline()
                f.seek(0)
                has_header = first_line.startswith("filename,")
                reader = (
                    csv.DictReader(f)
                    if has_header
                    else csv.DictReader(f, fieldnames=event_log_fields)
                )
                for row in reader:
                    # Convert status to a simple boolean for the frontend
                    row['is_meteor'] = (row.get('status') == "KEPT")
                    events.append(row)
        except Exception as e:
            logging.error(f"Failed to read detailed events: {e}")
            return jsonify({"error": "Failed to process event log"}), 500
        return jsonify(events)

    # Event Statistics
    @app.route("/api/event_stats")
    @require_token
    def event_stats():
        """API endpoint to read and process events.csv for charting."""
        stats = {"by_hour": {str(h): 0 for h in range(24)}, "by_night": {}}
        if not os.path.exists(self.event_log_out_file):
            return jsonify(stats)  # Return empty stats if file doesn't exist

        # events.csv is created in append mode by event_logger_loop and only
        # gets a header row after the janitor rotates it. Without a header,
        # csv.DictReader would consume the first event as field names and every
        # row["timestamp_local"] lookup would KeyError. Supply explicit field
        # names when the header is missing.
        event_log_fields = EVENT_LOG_FIELDS

        try:
            with open(
                self.event_log_out_file, "r", encoding="utf-8", newline=""
            ) as f:
                first_line = f.readline()
                f.seek(0)
                has_header = first_line.startswith("filename,")
                reader = (
                    csv.DictReader(f)
                    if has_header
                    else csv.DictReader(f, fieldnames=event_log_fields)
                )
                for row in reader:
                    try:
                        # Parse the local timestamp
                        ts_str = row.get("timestamp_local") or ""
                        dt_local = datetime.fromisoformat(ts_str)

                        # Aggregate by hour
                        hour_key = str(dt_local.hour)
                        stats["by_hour"][hour_key] += 1

                        # Aggregate by "night" (noon to noon)
                        # If it's before noon, it belongs to the previous day's "night".
                        if dt_local.hour < 12:
                            night_date = (dt_local - timedelta(days=1)).strftime(
                                "%Y-%m-%d"
                            )
                        else:
                            night_date = dt_local.strftime("%Y-%m-%d")

                        stats["by_night"][night_date] = (
                            stats["by_night"].get(night_date, 0) + 1
                        )

                    except (ValueError, KeyError):
                        continue  # Skip malformed rows

            return jsonify(stats)
        except Exception as e:
            logging.error(f"Failed to process event stats: {e}")
            return jsonify({"error": "Failed to process event log"}), 500

    @app.route("/stats")
    @require_token
    def stats_page():
        token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
        body = f"""
        <h1 class="page-title">Event statistics</h1>
        <p class="page-intro">Accepted meteor events across all recorded nights.</p>
        <div class="charts">
          <section class="panel"><h2>Meteors by hour of night <small>local time, noon to noon</small></h2>
            <div class="chart"><canvas id="hourlyChart" aria-label="Meteors per hour"></canvas></div></section>
          <section class="panel"><h2>Accepted events per night</h2>
            <div class="chart"><canvas id="nightlyChart" aria-label="Events per night"></canvas></div></section>
        </div>
        <section class="panel"><h2>Nightly history</h2>
          <div class="scroll"><table class="grid" id="chronicleTable">
            <thead><tr><th>Night</th><th class="num">Watched</th><th class="num">Events</th><th class="num">Clear sky</th>
              <th class="num">Sky score</th><th class="num">Moon</th><th class="num">Data</th><th>Trend</th></tr></thead>
            <tbody id="chronicleBody"></tbody></table></div>
          <p class="note" id="chartNote"></p>
        </section>
        <script>
          const css = getComputedStyle(document.documentElement);
          const C = n => css.getPropertyValue(n).trim();
          function chartsOk() {{
            if (window.Chart) {{
              Chart.defaults.color = C('--muted'); Chart.defaults.borderColor = C('--rule');
              Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
              Chart.defaults.plugins.legend.display = false;
              return true;
            }}
            // Chart.js comes from a CDN: without internet the charts are skipped, the table still works.
            document.getElementById('chartNote').textContent = 'Charts unavailable: the charting library could not be loaded (no internet access?).';
            return false;
          }}
          const bar = (id, labels, data, color) => new Chart(document.getElementById(id), {{
            type: 'bar',
            data: {{ labels, datasets: [{{ data, backgroundColor: color, borderRadius: 3 }}] }},
            options: {{ maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true, ticks: {{ precision: 0 }} }}, x: {{ grid: {{ display: false }} }} }} }}
          }});
          (async () => {{
            try {{
              const stats = await (await fetch('/api/event_stats{token_param}')).json();
              // Noon-to-noon axis, so a night is one contiguous block instead of being split at midnight.
              const hours = Array.from({{length: 24}}, (_, i) => (i + 12) % 24);
              if (chartsOk()) bar('hourlyChart', hours.map(h => String(h).padStart(2, '0') + ':00'),
                                  hours.map(h => Number(stats.by_hour[String(h)]) || 0), C('--star'));
            }} catch (err) {{ console.error('Hourly chart failed:', err); }}
          }})();
          (async () => {{
            try {{
              const chronicle = await (await fetch('/api/chronicle{token_param}')).json();
              const nights = (Array.isArray(chronicle) ? chronicle : []).filter(n => n && n.date)
                               .sort((a, b) => String(a.date).localeCompare(String(b.date)));
              if (chartsOk()) bar('nightlyChart', nights.map(n => n.date), nights.map(n => Number(n.events_accepted) || 0), C('--ok'));
              const body = document.getElementById('chronicleBody');
              const esc = v => String(v ?? '').replace(/[&<>"]/g, c => ({{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}})[c]);
              if (!nights.length) {{
                body.innerHTML = '<tr><td colspan="8" class="muted">No nights recorded yet.</td></tr>';
                return;
              }}
              body.innerHTML = nights.slice().reverse().map(n => {{
                const score = Number(n.sky_score_avg) || 0;
                const scoreCls = score >= 1.0 ? 'ok' : (score >= 0.8 ? 'warn' : 'err');
                const trendCls = n.sky_trend === 'improving' ? 'ok' : (n.sky_trend === 'degrading' ? 'err' : 'muted');
                return `<tr><td>${{esc(n.date)}}</td><td class="num">${{esc(n.hours_watched)}} h</td>
                  <td class="num">${{esc(n.events_accepted)}}</td><td class="num">${{esc(n.clarity_pct)}}%</td>
                  <td class="num ${{scoreCls}}">${{score.toFixed(3)}}</td>
                  <td class="num">${{((Number(n.max_moon_impact) || 0) * 100).toFixed(0)}}%</td>
                  <td class="num">${{((Number(n.data_written_mb) || 0) / 1024).toFixed(2)}} GB</td>
                  <td class="${{trendCls}}">${{esc(n.sky_trend || 'stable')}}</td></tr>`;
              }}).join('');
            }} catch (err) {{ console.error('Chronicle failed:', err); }}
          }})();
        </script>"""
        return render_page("Event statistics", body, token_param=token_param, current="/stats",
                           location=self.cfg["general"].get("location") or "",
                           head_extra='<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.js"></script>',
                           extra_css="""
            .charts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
            .chart { position:relative; height:280px; }
            @media (max-width:900px) { .charts { grid-template-columns:1fr; } }
            """)
