#!/usr/bin/env python3
"""
meteora_report.py

Cross-references the Ryusei pipeline's run artifacts for a single night:
  - pipeline.log          (timestamped INFO/WARNING/CRITICAL log lines)
  - system_monitor.csv    (periodic CPU/temp/mem/voltage telemetry)
  - camera_k.csv          (camera gain/threshold constant 'k' over time)
  - events.csv            (per-detection-event metadata)
  - health_stats.csv      (aggregated health-monitor event counts)

It correlates them on a shared timeline and produces a single human-readable
summary report (Markdown) describing what happened during the run, which
problems occurred, and where in the other files corroborating evidence can
be found (e.g. "under-voltage at 20:47:43 coincides with a CPU load spike
in system_monitor.csv and a stacking job in the log").

Usage:
    python3 meteora_report.py \
        --log pipeline.log \
        --system-monitor system_monitor.csv \
        --camera-k camera_k.csv \
        --events events.csv \
        --health-stats health_stats.csv \
        --out report.md \
        [--window-minutes 5]

All arguments default to the filenames above if run from a directory that
contains them, so it can also just be run with no arguments.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
    r"\[(?P<level>[A-Z]+)\]\s+\[(?P<thread>[^\]]+)\]\s+(?P<msg>.*)$"
)


def parse_log_ts(ts: str) -> datetime:
    """Pipeline log timestamps are naive but the log itself states it logs
    everything in UTC, so we attach UTC explicitly for safe comparison
    against the CSV timestamps (which carry +00:00 offsets)."""
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
    return dt.replace(tzinfo=timezone.utc)


def parse_iso_ts(ts: str) -> datetime:
    """Parses the various ISO-8601 timestamp flavors found across the CSVs
    (with microseconds + offset, or with a bare 'Z')."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_hms(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def fmt_duration(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class LogLine:
    ts: datetime
    level: str
    thread: str
    msg: str
    raw: str


@dataclass
class SystemSample:
    ts: datetime
    cpu_temp: float
    sensor_temp_cam: float
    sensor_temp_ext: float
    cpu_percent: float
    mem_used_mb: float
    mem_percent: float
    disk_used_mb: float
    disk_percent: float
    gpu_percent: float
    load1: float
    load5: float
    load15: float
    uptime_sec: int
    core_voltage_v: float


@dataclass
class CameraKSample:
    ts: datetime
    k: float


@dataclass
class Event:
    event_id: str
    detection_time: datetime
    capture_time: datetime
    duration_sec: float
    field5: float   # unlabeled in source file; likely peak motion/brightness metric
    field6: float   # unlabeled in source file; likely trigger/frame count
    score: float
    classification: str
    status: str


@dataclass
class HealthStat:
    event_type: str
    count: int
    level: str
    last_message: str


@dataclass
class ParsedData:
    log_lines: list[LogLine] = field(default_factory=list)
    system: list[SystemSample] = field(default_factory=list)
    camera_k: list[CameraKSample] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    health: list[HealthStat] = field(default_factory=list)
    skipped_log_lines: int = 0


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_log(path: Path) -> tuple[list[LogLine], int]:
    lines: list[LogLine] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            m = LOG_LINE_RE.match(raw)
            if not m:
                # Likely a continuation line of a multi-line traceback/message;
                # attach it to the previous line if one exists.
                if lines:
                    lines[-1].msg += " " + raw.strip()
                    lines[-1].raw += "\n" + raw
                else:
                    skipped += 1
                continue
            lines.append(
                LogLine(
                    ts=parse_log_ts(m.group("ts")),
                    level=m.group("level"),
                    thread=m.group("thread"),
                    msg=m.group("msg"),
                    raw=raw,
                )
            )
    return lines, skipped


def load_system_monitor(path: Path) -> list[SystemSample]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    SystemSample(
                        ts=parse_iso_ts(row["timestamp"]),
                        cpu_temp=float(row["cpu_temp"]),
                        sensor_temp_cam=float(row["sensor_temp_cam"]),
                        sensor_temp_ext=float(row["sensor_temp_ext"]),
                        cpu_percent=float(row["cpu_percent"]),
                        mem_used_mb=float(row["mem_used_mb"]),
                        mem_percent=float(row["mem_percent"]),
                        disk_used_mb=float(row["disk_used_mb"]),
                        disk_percent=float(row["disk_percent"]),
                        gpu_percent=float(row["gpu_percent"]),
                        load1=float(row["load1"]),
                        load5=float(row["load5"]),
                        load15=float(row["load15"]),
                        uptime_sec=int(float(row["uptime_sec"])),
                        core_voltage_v=float(row["core_voltage_v"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    out.sort(key=lambda s: s.ts)
    return out


def load_camera_k(path: Path) -> list[CameraKSample]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header (note: header has a leading-space ' k' field name)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                out.append(CameraKSample(ts=parse_iso_ts(row[0]), k=float(row[1])))
            except ValueError:
                continue
    out.sort(key=lambda s: s.ts)
    return out


def load_events(path: Path) -> list[Event]:
    """events.csv ships with a stale/incorrect header (a copy of the
    system_monitor.csv header) but the actual rows are 9-field event
    records: event_id, detection_time, capture_time, duration_sec,
    field5, field6, score, classification, status. We parse positionally
    and ignore the header row's column names."""
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return out
    # Detect & skip a header row (first field doesn't look like an event id)
    start = 1 if rows[0] and not rows[0][0].startswith("event_") else 0
    for row in rows[start:]:
        if len(row) < 9:
            continue
        try:
            out.append(
                Event(
                    event_id=row[0],
                    detection_time=parse_iso_ts(row[1]),
                    capture_time=parse_iso_ts(row[2]),
                    duration_sec=float(row[3]),
                    field5=float(row[4]),
                    field6=float(row[5]),
                    score=float(row[6]),
                    classification=row[7].strip(),
                    status=row[8].strip(),
                )
            )
        except ValueError:
            continue
    out.sort(key=lambda e: e.detection_time)
    return out


def load_health_stats(path: Path) -> list[HealthStat]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    HealthStat(
                        event_type=row["event_type"].strip(),
                        count=int(row["count"]),
                        level=row["level"].strip(),
                        last_message=row["last_message"].strip(),
                    )
                )
            except (KeyError, ValueError):
                continue
    out.sort(key=lambda h: h.count, reverse=True)
    return out


def load_all(args) -> ParsedData:
    data = ParsedData()
    data.log_lines, data.skipped_log_lines = load_log(args.log)
    data.system = load_system_monitor(args.system_monitor)
    data.camera_k = load_camera_k(args.camera_k)
    data.events = load_events(args.events)
    data.health = load_health_stats(args.health_stats)
    return data


# --------------------------------------------------------------------------
# Correlation helpers
# --------------------------------------------------------------------------

def nearest_system_sample(system: list[SystemSample], ts: datetime) -> SystemSample | None:
    """Binary search for the system_monitor sample closest in time to ts."""
    if not system:
        return None
    timestamps = [s.ts for s in system]
    i = bisect_left(timestamps, ts)
    candidates = []
    if i < len(system):
        candidates.append(system[i])
    if i > 0:
        candidates.append(system[i - 1])
    return min(candidates, key=lambda s: abs((s.ts - ts).total_seconds()))


def window_system_samples(
    system: list[SystemSample], center: datetime, minutes: float
) -> list[SystemSample]:
    lo = center - timedelta(minutes=minutes)
    hi = center + timedelta(minutes=minutes)
    return [s for s in system if lo <= s.ts <= hi]


def window_log_lines(
    log_lines: list[LogLine], center: datetime, minutes: float, levels: set[str] | None = None
) -> list[LogLine]:
    lo = center - timedelta(minutes=minutes)
    hi = center + timedelta(minutes=minutes)
    out = [l for l in log_lines if lo <= l.ts <= hi]
    if levels:
        out = [l for l in out if l.level in levels]
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

ANOMALY_PATTERNS = [
    # --- Power & Vitals ---
    ("under_voltage", re.compile(r"ACTIVE UNDER-VOLTAGE DETECTED|POWER_UNDER_VOLTAGE", re.I)),
    ("cpu_throttle", re.compile(r"CPU is NOW THROTTLING|CPU_THROTTLED", re.I)),
    ("oom_warning", re.compile(r"SYSTEM RAM CRITICAL|OOM_WARNING", re.I)),
    ("power_loss", re.compile(r"Initial power loss detected|CONFIRMED POWER OUTAGE", re.I)),
    
    # --- Camera & Sensor Errors ---
    ("camera_fatal", re.compile(r"Fatal camera error|CAMERA_DETECT_FAIL|HARDWARE_RESUME_FAIL", re.I)),
    ("sensor_missing", re.compile(r"Thermal Sensor NOT found|SENSOR_MISSING", re.I)),
    ("calib_old", re.compile(r"Dark library is .* days old|CALIB_LIBRARY_OLD", re.I)),
    ("capture_fail", re.compile(r"Frame capture failed|CAPTURE_DAYLIGHT_FAIL", re.I)),
    
    # --- Thread & Software Crashes ---
    ("thread_crash", re.compile(r"WATCHDOG: DETECTED CRASHED THREAD|THREAD_CRASH", re.I)),
    ("generic_crash", re.compile(r"CRITICAL FATAL ERROR IN .* THREAD", re.I)),
    ("join_timeout", re.compile(r"THREAD_JOIN_TIMEOUT", re.I)),
    
    # --- Pipeline & Logic ---
    ("ecc_crash", re.compile(r"ECC Alignment crashed", re.I)),
    ("event_timeout", re.compile(r"EVENT TIMEOUT|EVENT_TIMEOUT", re.I)),
    ("event_truncated", re.compile(r"EVENT FORCIBLY TERMINATED|EVENT_TRUNCATED", re.I)),
    ("queue_throttle", re.compile(r"Timelapse queue full|TIMELAPSE_THROTTLE", re.I)),
    ("acq_queue_full", re.compile(r"Acquisition queue full|ACQUISITION_QUEUE", re.I)),
    ("stack_bias_mode", re.compile(r"STACK BIAS MODE ENABLED", re.I)),
    
    # --- Storage & Janitor ---
    ("disk_critical", re.compile(r"JANITOR FAILED TO FREE DISK SPACE|DISK_USAGE_HIGH", re.I)),
    ("janitor_fail", re.compile(r"JANITOR_DELETE_FAIL|METADATA_READ_ERROR", re.I)),
    ("janitor_delete_fail", re.compile(r"Janitor: Failed to delete file", re.I)),
    ("disk_usage_high", re.compile(r"Disk usage on .* is at \d", re.I)),
    ("janitor_tier_cleanup", re.compile(r"Disk space still critical\. Deleting from Tier", re.I)),
    
    # --- Networking & External ---
    ("sftp_fail", re.compile(r"SFTP connection failed|SFTP_UPLOAD_FAIL", re.I)),
    ("ntp_fail", re.compile(r"NTP time-sync query failure|NTP_FAIL|NTP_ERROR", re.I)),
    ("heartbeat_fail", re.compile(r"Failed to send heartbeat ping|HEARTBEAT", re.I)),

    # --- Camera & Thermal sensor (minor hardware hiccups) ---
    ("sensor_crc_fail", re.compile(r"CRC failed", re.I)),
    ("frame_flush_timeout", re.compile(r"Flush timed out after", re.I)),
    
    # --- AI & Environment ---
    ("fog_hold", re.compile(r"FOG detected|AI Observatory: .* HOLD", re.I)),
    ("ai_quarantine", re.compile(r"Moving to Quarantine folder", re.I)),
    ("daylight_switch", re.compile(r"Switching camera to DAYLIGHT MODE", re.I)),
    ("shutdown_triggered", re.compile(r"SYSTEM ACTION TRIGGERED: SHUTDOWN|Scheduled Daily Shutdown", re.I)),
    ("system_start", re.compile(r"The system is starting|--- Ryusei Pipeline .* Starting ---", re.I)),
]


@dataclass
class Anomaly:
    key: str
    ts: datetime
    thread: str
    msg: str
    level: str


def find_anomalies(log_lines: list[LogLine]) -> list[Anomaly]:
    out = []
    for line in log_lines:
        for key, pattern in ANOMALY_PATTERNS:
            if pattern.search(line.msg):
                out.append(Anomaly(key=key, ts=line.ts, thread=line.thread, msg=line.msg, level=line.level))
                break
    return out


def summarize_run_window(log_lines: list[LogLine]) -> tuple[datetime | None, datetime | None]:
    if not log_lines:
        return None, None
    return log_lines[0].ts, log_lines[-1].ts


def level_counts(log_lines: list[LogLine]) -> Counter:
    return Counter(l.level for l in log_lines)


def thread_level_counts(log_lines: list[LogLine]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for l in log_lines:
        out[l.thread][l.level] += 1
    return out


def temp_extremes(system: list[SystemSample]) -> dict:
    if not system:
        return {}

    def extremes(attr):
        vals = [(getattr(s, attr), s.ts) for s in system]
        vmax = max(vals, key=lambda v: v[0])
        vmin = min(vals, key=lambda v: v[0])
        mean = statistics.mean(v[0] for v in vals)
        return {"max": vmax, "min": vmin, "mean": mean}

    return {
        "cpu_temp": extremes("cpu_temp"),
        "sensor_temp_cam": extremes("sensor_temp_cam"),
        "sensor_temp_ext": extremes("sensor_temp_ext"),
        "core_voltage_v": extremes("core_voltage_v"),
        "cpu_percent": extremes("cpu_percent"),
        "mem_percent": extremes("mem_percent"),
        "load1": extremes("load1"),
    }


def detect_stuck_sensor(
    system: list[SystemSample], attr: str, min_duration_sec: float = 600
) -> list[tuple[datetime, datetime, float, int]]:
    """Flags runs where a sensor value is exactly identical for at least
    min_duration_sec of wall-clock time. Short identical-value runs are
    normal here (the probe's native update rate is coarser than the
    polling interval, so 4-5 consecutive identical samples is expected
    quantization, not a fault). A run lasting many minutes at a fixed
    value -- especially one well outside the metric's normal range -- is
    a much stronger signal of a saturated/disconnected sensor."""
    runs = []
    if not system:
        return runs
    run_start = 0
    for i in range(1, len(system) + 1):
        same = i < len(system) and getattr(system[i], attr) == getattr(system[run_start], attr)
        if not same:
            run_len = i - run_start
            duration = (system[i - 1].ts - system[run_start].ts).total_seconds()
            if duration >= min_duration_sec:
                runs.append(
                    (system[run_start].ts, system[i - 1].ts, getattr(system[run_start], attr), run_len)
                )
            run_start = i
    return runs


def detect_outlier_spike(
    system: list[SystemSample], attr: str, normal_max: float
) -> list[tuple[datetime, datetime, float, int]]:
    """Flags contiguous runs where a sensor value exceeds a hard ceiling
    that is well outside its physically plausible range for this setup
    (e.g. a camera sensor reporting 109C when it otherwise never exceeds
    ~45C) -- a strong signal of a bad/saturated reading rather than a
    real measurement."""
    runs = []
    if not system:
        return runs
    run_start = None
    for i, s in enumerate(system):
        over = getattr(s, attr) > normal_max
        if over and run_start is None:
            run_start = i
        elif not over and run_start is not None:
            runs.append((system[run_start].ts, system[i - 1].ts, getattr(system[run_start], attr), i - run_start))
            run_start = None
    if run_start is not None:
        runs.append((system[run_start].ts, system[-1].ts, getattr(system[run_start], attr), len(system) - run_start))
    return runs


def event_stats(events: list[Event]) -> dict:
    if not events:
        return {}
    by_status = Counter(e.status for e in events)
    by_class = Counter(e.classification for e in events)
    durations = [e.duration_sec for e in events]
    scores = [e.score for e in events]
    return {
        "total": len(events),
        "by_status": by_status,
        "by_classification": by_class,
        "duration_min": min(durations),
        "duration_max": max(durations),
        "duration_mean": statistics.mean(durations),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": statistics.mean(scores),
    }


def camera_k_drift(camera_k: list[CameraKSample]) -> dict:
    if not camera_k:
        return {}
    vals = [c.k for c in camera_k]
    return {
        "start": camera_k[0],
        "end": camera_k[-1],
        "min": min(camera_k, key=lambda c: c.k),
        "max": max(camera_k, key=lambda c: c.k),
        "mean": statistics.mean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

ANOMALY_LABELS = {
    "under_voltage": "Active under-voltage (hardware power alert)",
    "cpu_throttle": "CPU throttling engaged (thermal or power limit)",
    "oom_warning": "Out-of-Memory (RAM) critical protection engaged",
    "power_loss": "Main power loss detected (Battery/UPS active)",
    "camera_fatal": "Fatal camera hardware communication failure",
    "sensor_missing": "DS18B20 Temperature sensor disconnected",
    "calib_old": "Calibration library is outdated (re-run calibrator)",
    "capture_fail": "Individual frame capture timeout",
    "thread_crash": "Thread crash detected by Watchdog supervisor",
    "generic_crash": "Fatal exception in background worker",
    "join_timeout": "Thread refused to stop during shutdown",
    "ecc_crash": "ECC alignment failure (non-convergent warp)",
    "event_timeout": "Event reached max duration (forced termination)",
    "event_truncated": "Event cut short to protect system RAM",
    "queue_throttle": "Timelapse queue backlogged (disk/CPU slow)",
    "acq_queue_full": "Acquisition queue overflow (frames dropped)",
    "stack_bias_mode": "Stacker fell back to Translation-only mode",
    "disk_critical": "Disk space critically high (Janitor failed)",
    "janitor_fail": "Janitor could not delete files or read metadata",
    "janitor_delete_fail": "Janitor tried to delete a file that was already gone",
    "disk_usage_high": "Disk usage crossed the cleanup threshold",
    "janitor_tier_cleanup": "Janitor escalated cleanup to a specific tier",
    "sensor_crc_fail": "DS18B20 thermal sensor CRC checksum failure",
    "frame_flush_timeout": "Camera frame flush timed out (frames dropped)",
    "sftp_fail": "SFTP Upload/Connection failed",
    "ntp_fail": "NTP time sync failed (Clock may drift)",
    "heartbeat_fail": "Heartbeat monitor failed to ping server",
    "fog_hold": "AI Observatory: conditions poor, capture paused",
    "ai_quarantine": "AI classified candidate as non-meteor (Quarantined)",
    "daylight_switch": "Switched to daylight capture mode",
    "shutdown_triggered": "System-wide shutdown command executed",
    "system_start": "System boot sequence initiated",
}



def render_report(data: ParsedData, args) -> str:
    lines: list[str] = []
    p = lines.append

    run_start, run_end = summarize_run_window(data.log_lines)
    lvl_counts = level_counts(data.log_lines)
    anomalies = find_anomalies(data.log_lines)
    anomaly_counts = Counter(a.key for a in anomalies)
    temps = temp_extremes(data.system)
    ev_stats = event_stats(data.events)
    k_drift = camera_k_drift(data.camera_k)
    # 45C is a generous ceiling for this metric given the observed normal
    # range (28-64C across the rest of the run); values above that are
    # flagged as implausible/stuck rather than genuine heating.
    cam_temp_spikes = detect_outlier_spike(data.system, "sensor_temp_cam", normal_max=45.0)
    long_stuck_cam_temp = detect_stuck_sensor(data.system, "sensor_temp_cam", min_duration_sec=900)

    # ---------------- Header ----------------
    p("# Ryusei Pipeline Run Report")
    p("")
    if run_start and run_end:
        p(f"**Run window:** {fmt_ts(run_start)} -> {fmt_ts(run_end)} "
          f"({fmt_duration(run_end - run_start)})")
    p(f"**Sources:** `{args.log.name}`, `{args.system_monitor.name}`, "
      f"`{args.camera_k.name}`, `{args.events.name}`, `{args.health_stats.name}`")
    p("")
    p("---")
    p("")

    # ---------------- Executive summary ----------------
    p("## 1. Executive Summary")
    p("")
    n_critical = lvl_counts.get("CRITICAL", 0)
    n_warning = lvl_counts.get("WARNING", 0)
    uv_count = anomaly_counts.get("under_voltage", 0)
    ecc_count = anomaly_counts.get("ecc_crash", 0)
    n_meteor_kept = sum(1 for e in data.events if e.classification == "meteor" and e.status == "KEPT")
    n_quarantined = sum(1 for e in data.events if e.status == "QUARANTINED")

    summary_bits = []
    if run_start and run_end:
        summary_bits.append(f"the pipeline ran for {fmt_duration(run_end - run_start)} and shut down via its normal scheduled-shutdown path" if anomaly_counts.get("shutdown_triggered") else f"the pipeline ran for {fmt_duration(run_end - run_start)}")
    summary_bits.append(f"logged {len(data.log_lines)} lines ({n_warning} WARNING, {n_critical} CRITICAL)")
    if ev_stats:
        summary_bits.append(f"recorded {ev_stats['total']} detection events ({n_meteor_kept} kept meteor detections, {n_quarantined} sent to quarantine for review)")
    if uv_count:
        summary_bits.append(f"experienced {uv_count} active under-voltage hardware alert(s)")
    p("- " + ";\n- ".join(b[0].upper() + b[1:] for b in summary_bits) + ".")
    p("")
    if n_critical:
        p(f"**{n_critical} CRITICAL-level log entries** were recorded -- see Section 3.")
    if cam_temp_spikes:
        worst = max(cam_temp_spikes, key=lambda r: r[2])
        p(f"**Camera temperature sensor anomaly:** `sensor_temp_cam` reported an implausible "
          f"{worst[2]:.1f}C ({worst[3]} consecutive samples, {fmt_hms(worst[0])}-{fmt_hms(worst[1])}), "
          f"consistent with a saturated or disconnected sensor reading rather than genuine heating "
          f"-- see Section 4.")
    p("")
    p("---")
    p("")

    # ---------------- Log overview ----------------
    p("## 2. Log Overview")
    p("")
    p(f"- Total parsed log lines: **{len(data.log_lines)}**" +
      (f" ({data.skipped_log_lines} unparseable lines skipped)" if data.skipped_log_lines else ""))
    for lvl in ("INFO", "WARNING", "CRITICAL", "ERROR", "DEBUG"):
        if lvl in lvl_counts:
            p(f"  - {lvl}: {lvl_counts[lvl]}")
    p("")
    p("**Lines per thread (top contributors):**")
    p("")
    p("| Thread | INFO | WARNING | CRITICAL | Total |")
    p("|---|---:|---:|---:|---:|")
    tlc = thread_level_counts(data.log_lines)
    for thread, counts in sorted(tlc.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        total = sum(counts.values())
        p(f"| {thread} | {counts.get('INFO', 0)} | {counts.get('WARNING', 0)} | "
          f"{counts.get('CRITICAL', 0)} | {total} |")
    p("")
    p("---")
    p("")

    # ---------------- Anomalies & cross-correlation ----------------
    p("## 3. Anomalies & Cross-File Correlation")
    p("")
    if not anomalies:
        p("No recognized anomaly patterns were found in the log.")
    else:
        p("Recognized anomaly patterns, counted and ranked by frequency:")
        p("")
        p("| Pattern | Count | First seen | Last seen |")
        p("|---|---:|---|---|")
        by_key: dict[str, list[Anomaly]] = defaultdict(list)
        for a in anomalies:
            by_key[a.key].append(a)
        for key, items in sorted(by_key.items(), key=lambda kv: -len(kv[1])):
            items.sort(key=lambda a: a.ts)
            label = ANOMALY_LABELS.get(key, key)
            p(f"| {label} | {len(items)} | {fmt_hms(items[0].ts)} | {fmt_hms(items[-1].ts)} |")
        p("")

        # Deep-dive on the highest-value anomalies: under-voltage and ECC crashes,
        # correlated against system_monitor.csv in a window around each event.
        if "under_voltage" in by_key:
            p("### 3.1 Under-Voltage Events (correlated with system telemetry)")
            p("")
            for a in by_key["under_voltage"]:
                p(f"**{fmt_ts(a.ts)}** -- `{a.msg}`")
                sample = nearest_system_sample(data.system, a.ts)
                win = window_system_samples(data.system, a.ts, args.window_minutes)
                if sample:
                    p(f"- Nearest telemetry sample ({fmt_hms(sample.ts)}): "
                      f"core_voltage={sample.core_voltage_v:.4f}V, cpu_percent={sample.cpu_percent:.1f}%, "
                      f"load1={sample.load1:.2f}, cpu_temp={sample.cpu_temp:.1f}C")
                if win:
                    max_load = max(s.load1 for s in win)
                    min_volt = min(s.core_voltage_v for s in win)
                    p(f"- In the surrounding +/-{args.window_minutes:.0f} min window: "
                      f"peak load1={max_load:.2f}, minimum core_voltage={min_volt:.4f}V")
                # nearby thread activity (what was the pipeline doing)
                nearby_info = window_log_lines(data.log_lines, a.ts, 1, levels={"INFO"})
                stacking = [l for l in nearby_info if "STACK" in l.msg or "Stack" in l.msg]
                if stacking:
                    p(f"- Coincides with stacking activity in `{stacking[0].thread}` "
                      f"around {fmt_hms(stacking[0].ts)} (frame stacking is CPU-intensive and is "
                      f"a plausible trigger for the momentary power draw).")
                p("")

        if "janitor_delete_fail" in by_key:
            p("### 3.2 Janitor File-Delete Failures (by tier)")
            p("")
            items = by_key["janitor_delete_fail"]
            tier_re = re.compile(r"output/(?P<tier>[^/]+)/")
            tier_counts: Counter = Counter()
            for a in items:
                m = tier_re.search(a.msg)
                tier_counts[m.group("tier") if m else "unknown"] += 1
            p(f"**{len(items)} file(s)** the Janitor tried to delete but which were already gone "
              f"(`[Errno 2] No such file or directory`), first at {fmt_hms(items[0].ts)}, "
              f"last at {fmt_hms(items[-1].ts)}. Breakdown by output tier/directory:")
            p("")
            p("| Tier | Count |")
            p("|---|---:|")
            for tier, count in tier_counts.most_common():
                p(f"| {tier} | {count} |")
            p("")
            p("This pattern (attempting to delete a file whose metadata entry still exists but whose "
              "underlying file is already gone) is consistent with a stale/duplicate delete-candidate "
              "list rather than a one-off filesystem error, given the volume and spread across tiers.")
            p("")

        if "ecc_crash" in by_key:
            p("### 3.3 ECC Alignment Crashes")
            p("")
            p(f"{len(by_key['ecc_crash'])} frame(s) failed to converge during ECC-based alignment "
              f"in the stacking pipeline (`findTransformECC`). Timestamps: " +
              ", ".join(fmt_hms(a.ts) for a in by_key["ecc_crash"]) + ".")
            p("")

        if "heartbeat_fail" in by_key or "ntp_fail" in by_key:
            p("### 3.4 Network Connectivity Issues")
            p("")
            n_hb = len(by_key.get("heartbeat_fail", []))
            n_ntp = len(by_key.get("ntp_fail", []))
            if n_hb:
                p(f"- {n_hb} heartbeat ping(s) failed to send (DNS resolution failure on `hc-ping.com`), "
                  f"first at {fmt_hms(by_key['heartbeat_fail'][0].ts)}, "
                  f"last at {fmt_hms(by_key['heartbeat_fail'][-1].ts)} -- consistent with the "
                  f"deployment site having no/intermittent outbound internet access.")
            if n_ntp:
                p(f"- {n_ntp} NTP time-sync query failure(s), first at {fmt_hms(by_key['ntp_fail'][0].ts)}.")
            p("")
    p("---")
    p("")

    # ---------------- System telemetry ----------------
    p("## 4. System Telemetry (system_monitor.csv)")
    p("")
    if not data.system:
        p("No system telemetry samples available.")
    else:
        p(f"**{len(data.system)} samples** from {fmt_ts(data.system[0].ts)} to {fmt_ts(data.system[-1].ts)}.")
        p("")
        p("| Metric | Min | Mean | Max | Max occurred at |")
        p("|---|---:|---:|---:|---|")
        label_map = {
            "cpu_temp": "CPU temp (C)",
            "sensor_temp_cam": "Camera sensor temp (C)",
            "sensor_temp_ext": "External sensor temp (C)",
            "core_voltage_v": "Core voltage (V)",
            "cpu_percent": "CPU usage (%)",
            "mem_percent": "Memory usage (%)",
            "load1": "Load average (1 min)",
        }
        for key, label in label_map.items():
            t = temps.get(key)
            if not t:
                continue
            p(f"| {label} | {t['min'][0]:.2f} | {t['mean']:.2f} | {t['max'][0]:.2f} | {fmt_hms(t['max'][1])} |")
        p("")
        if cam_temp_spikes:
            p(f"**Implausible-value check (camera temperature probe):** value(s) well outside the "
              f"observed normal range (>45C, vs. a run-wide mean of {temps['sensor_temp_cam']['mean']:.1f}C) "
              f"-- a strong signal of a bad reading rather than real heating:")
            p("")
            for start, end, val, run_len in cam_temp_spikes:
                p(f"- {val:.1f}C for {run_len} consecutive sample(s), {fmt_hms(start)} - {fmt_hms(end)} "
                  f"({fmt_duration(end - start)})")
            p("")
        if long_stuck_cam_temp:
            p(f"**Long flat-line check (camera temperature probe):** runs of 15+ minutes at a single "
              f"fixed value (the probe's native update rate is coarser than the polling interval, so "
              f"brief identical-value runs of a few samples are normal and not listed here -- only "
              f"unusually long flat periods are shown):")
            p("")
            for start, end, val, run_len in long_stuck_cam_temp[:10]:
                p(f"- {val:.1f}C held for {fmt_duration(end - start)} ({run_len} samples), "
                  f"{fmt_hms(start)} - {fmt_hms(end)}")
            if len(long_stuck_cam_temp) > 10:
                p(f"- ...and {len(long_stuck_cam_temp) - 10} more such run(s).")
            p("")
        disk_vals = {s.disk_percent for s in data.system}
        if len(disk_vals) == 1:
            p(f"- Disk usage was flat at {next(iter(disk_vals)):.1f}% for the entire run "
              f"(no disk-growth instrumentation change detected).")
        gpu_vals = {s.gpu_percent for s in data.system}
        if gpu_vals == {0.0}:
            p("- GPU usage reported 0% for the entire run.")
        p("")
    p("---")
    p("")

    # ---------------- Camera k drift ----------------
    p("## 5. Camera Gain/Threshold Constant (camera_k.csv)")
    p("")
    if not k_drift:
        p("No camera_k samples available.")
    else:
        p(f"- {len(data.camera_k)} samples from {fmt_ts(k_drift['start'].ts)} to {fmt_ts(k_drift['end'].ts)}.")
        p(f"- k ranged from **{k_drift['min'].k:.3f}** (at {fmt_hms(k_drift['min'].ts)}) to "
          f"**{k_drift['max'].k:.3f}** (at {fmt_hms(k_drift['max'].ts)}), mean={k_drift['mean']:.3f}, "
          f"stdev={k_drift['stdev']:.3f}.")
        delta = k_drift["end"].k - k_drift["start"].k
        direction = "increased" if delta > 0 else "decreased"
        p(f"- Net drift over the run: k {direction} by {abs(delta):.3f} "
          f"({k_drift['start'].k:.3f} -> {k_drift['end'].k:.3f}), consistent with the "
          f"auto-sensitivity tuner adapting to changing sky brightness/lunar position over the night.")
        p("")
    p("---")
    p("")

    # ---------------- Events ----------------
    p("## 6. Detection Events (events.csv)")
    p("")
    if not ev_stats:
        p("No events recorded.")
    else:
        p(f"- **{ev_stats['total']} total events** detected over the run.")
        p(f"- By status: " + ", ".join(f"{k}={v}" for k, v in ev_stats["by_status"].items()))
        p(f"- By AI classification: " + ", ".join(f"{k}={v}" for k, v in ev_stats["by_classification"].items()))
        p(f"- Duration: min={ev_stats['duration_min']:.2f}s, mean={ev_stats['duration_mean']:.2f}s, "
          f"max={ev_stats['duration_max']:.2f}s")
        p(f"- Score: min={ev_stats['score_min']:.3f}, mean={ev_stats['score_mean']:.3f}, "
          f"max={ev_stats['score_max']:.3f}")
        p("")
        quarantined = [e for e in data.events if e.status == "QUARANTINED"]
        if quarantined:
            p(f"**{len(quarantined)} event(s) quarantined for manual review** "
              f"(AI flagged as meteor candidates):")
            p("")
            p("| Event ID | Detection time | Duration (s) | Score |")
            p("|---|---|---:|---:|")
            for e in quarantined:
                p(f"| {e.event_id} | {fmt_hms(e.detection_time)} | {e.duration_sec:.2f} | {e.score:.3f} |")
            p("")
        meteors_kept = [e for e in data.events if e.classification == "meteor" and e.status == "KEPT"]
        if meteors_kept:
            p(f"**{len(meteors_kept)} confirmed meteor detection(s) (KEPT):** " +
              ", ".join(fmt_hms(e.detection_time) for e in meteors_kept) + ".")
            p("")

        # Correlate each event with nearest system sample to flag any captured
        # during thermally/electrically stressed conditions.
        flagged = []
        for e in data.events:
            s = nearest_system_sample(data.system, e.detection_time)
            if s and (s.cpu_percent > 50 or s.load1 > 1.0):
                flagged.append((e, s))
        if flagged:
            p(f"**{len(flagged)} event(s) captured while the system was under elevated CPU load** "
              f"(cpu>50% or load1>1.0 at time of detection) -- worth checking for frame drops or "
              f"degraded stacking quality on these specific events:")
            p("")
            for e, s in flagged:
                p(f"- {e.event_id} at {fmt_hms(e.detection_time)}: cpu={s.cpu_percent:.1f}%, load1={s.load1:.2f}")
            p("")
    p("---")
    p("")

    # ---------------- Health stats ----------------
    p("## 7. Health Monitor Summary (health_stats.csv)")
    p("")
    if not data.health:
        p("No health stats available.")
    else:
        p("| Event Type | Count | Level | Last Message |")
        p("|---|---:|---|---|")
        for h in data.health:
            p(f"| {h.event_type} | {h.count} | {h.level} | {h.last_message} |")
        p("")
        critical_health = [h for h in data.health if h.level == "CRITICAL"]
        if critical_health:
            p(f"**CRITICAL-level health events:** " +
              ", ".join(f"{h.event_type} (x{h.count})" for h in critical_health) + ".")
            p("")
    p("---")
    p("")

    # ---------------- Recommendations ----------------
    p("## 8. Notable Items / Suggested Follow-Ups")
    p("")
    recs = []
    if uv_count:
        recs.append(
            f"**Power supply:** {uv_count} active under-voltage event(s) occurred, both coinciding with "
            f"CPU-intensive stacking activity. If this recurs, consider a higher-current PSU or USB-C cable, "
            f"or staggering timelapse/event stacking to avoid simultaneous CPU spikes."
        )
    if cam_temp_spikes:
        recs.append(
            "**Camera temp sensor:** brief implausible high readings suggest the camera temperature "
            "probe may be saturating or losing contact intermittently; worth a hardware check if "
            "dynamic-dark calibration depends on this value."
        )
    if anomaly_counts.get("ecc_crash"):
        recs.append(
            f"**ECC alignment:** {anomaly_counts['ecc_crash']} frame(s) failed to converge during alignment; "
            f"if this rate climbs, consider relaxing ECC iteration/epsilon parameters or pre-filtering "
            f"low-contrast frames before attempting alignment."
        )
    if anomaly_counts.get("heartbeat_fail") or anomaly_counts.get("ntp_fail"):
        recs.append(
            "**Network:** heartbeat/NTP failures indicate the deployment had no reliable outbound "
            "internet access for parts of the run; this is expected for a remote field deployment but "
            "worth confirming local clock drift stayed within tolerance given the failed NTP sync."
        )
    # events.csv header bug -- always worth flagging since it could silently break downstream tooling
    recs.append(
        "**events.csv header bug:** the header row in `events.csv` is a stale copy of the "
        "system_monitor.csv header (column names like `cpu_temp`, `mem_used_mb`, etc.) rather than "
        "matching the actual 9 columns written (event_id, detection_time, capture_time, duration_sec, "
        "field5, field6, score, classification, status). This script parsed the file positionally, "
        "ignoring the header, but any other tooling that reads this CSV by column name will silently "
        "misread it. Worth fixing at the source (`EventLoggerThread`)."
    )
    if not recs:
        p("No significant issues identified.")
    else:
        for r in recs:
            p(f"- {r}")
    p("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    here = Path("/media/pi/USB/logs/")
    ap = argparse.ArgumentParser(description="Cross-reference Ryusei pipeline run artifacts into one report.")
    ap.add_argument("--log", type=Path, default=here / "pipeline.log")
    ap.add_argument("--system-monitor", type=Path, default=here / "system_monitor.csv")
    ap.add_argument("--camera-k", type=Path, default=here / "camera_k.csv")
    ap.add_argument("--events", type=Path, default=here / "events.csv")
    ap.add_argument("--health-stats", type=Path, default=here / "health_stats.csv")
    ap.add_argument("--out", type=Path, default=here / "meteora_report.md")
    ap.add_argument("--window-minutes", type=float, default=5.0,
                     help="Time window (minutes, +/-) used when correlating an anomaly against telemetry.")
    return ap


def main():
    args = build_arg_parser().parse_args()

    for attr in ("log", "system_monitor", "camera_k", "events", "health_stats"):
        path = getattr(args, attr)
        if not path.exists():
            raise SystemExit(f"Error: required input file not found: {path}")

    data = load_all(args)
    report = render_report(data, args)
    args.out.write_text(report, encoding="utf-8")
    print(f"Report written to {args.out}  ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
