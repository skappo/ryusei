"""
satellites.py

Was a detected event a satellite? Answered locally, on board, for the event's
own frame times -- no per-event network call.

DATA SOURCE AND CELESTRAK USAGE POLICY (https://celestrak.org/usage-policy.php)
    * Format: OMM in CSV (FORMAT=CSV). Since 2026-07-11 new objects get 6-digit
      catalogue numbers, which the TLE format cannot carry: a TLE download silently
      misses every recent launch (e.g. new Starlink shells).
    * At most one download per group per GP update. CelesTrak updates GP data every
      2 hours and, for large groups such as GROUP=active, answers a second download
      within that window with HTTP 403 ("GP data has not updated since your last
      successful download"). Attempts are persisted to disk BEFORE each request, so
      restarts can never cause repeat downloads.
    * Unattended schedule with a guaranteed gap and back-off after failures (see
      SatelliteCatalog): at most one request per group per 2 h 30 min, never on
      start-up alone, fewer and fewer while errors persist. Redirects are not
      followed (a 301 is recorded as a failure, not chased).
    * Each group is cached in its own file, so one group's failure never discards
      another group's successful (and already counted) download.

PROPAGATION
    SGP4 (python-sgp4, vectorised SatrecArray) -> TEME position -> Earth-fixed via
    GMST -> topocentric alt/az for the station. Sunlit test: cylindrical Earth
    shadow (umbra and penumbra merged at the shadow edge, < 0.3 deg of orbital arc
    for LEO), Sun direction from PyEphem. Observer elevation is ignored: a few hundred
    metres against a >= 160 km slant range shift a satellite by < 0.1 deg.

PHYSICS OF A MATCH (all must hold): above the horizon; sunlit; at the event's frames,
at the same angular distance from the camera boresight as the motion; and sweeping,
during one exposure, an arc as long as the streak. The last two tests compare only
angles from the boresight and arc lengths, never image directions, so the camera's
roll about its own axis (not in the configuration) is not needed. A meteor fails the
arc test: it crosses 5-30 deg/s against <= ~2.8 deg/s for any satellite.
"""
import csv
import io
import json
import logging
import math
import os
import threading
import time
from datetime import timezone

import ephem
import numpy as np
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray, jday

from constants import CELESTRAK_MIN_REQUEST_INTERVAL_H, LEO_MAX_ANGULAR_RATE_DEG_S

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
EARTH_RADIUS_KM = WGS84_A_KM
AU_KM = 149_597_870.7

# ------------------------------------------------------------------ catalogue data


def parse_omm_csv(text):
    """-> list of (name, norad, Satrec). Invalid rows are skipped (counted in the log)."""
    out, bad = [], 0
    for fields in omm.parse_csv(io.StringIO(text)):
        try:
            sat = Satrec()
            omm.initialize(sat, fields)
            err, _, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
            if err != 0:
                raise ValueError(f"sgp4 error {err} at epoch")
            out.append((fields.get("OBJECT_NAME", "").strip() or f"NORAD {fields['NORAD_CAT_ID']}",
                        str(fields["NORAD_CAT_ID"]).strip(), sat))
        except (KeyError, ValueError) as e:
            bad += 1
            logging.debug(f"Satellites: skipped OMM row: {e}")
    if bad:
        logging.info(f"Satellites: {bad} OMM row(s) skipped as invalid.")
    return out


class Catalogue:
    """Immutable, vectorised view of the merged catalogue."""

    def __init__(self, entries):
        self.names = [n for n, _, _ in entries]
        self.norads = [k for _, k, _ in entries]
        self.array = SatrecArray([s for _, _, s in entries]) if entries else None

    def __len__(self):
        return len(self.names)


class SatelliteCatalog:
    """
    Disk-cached CelesTrak GP catalogue (one CSV file per group) with an unattended,
    restart-proof download schedule.

    SCHEDULE (per group, persisted in state.json -- never in memory only)
        Every request is recorded BEFORE it is sent (time + "in flight"), then its
        outcome: result true/false, HTTP status or error, server message. Whether a
        group may be requested again is decided ONLY from that record, re-read from
        disk each time, so restarts change nothing:

          after a success  -> next download when the data is refresh_hours old
          after a failure  -> retry after SAFE_GAP x 2^(n-1), capped at refresh_hours
                              (n = consecutive failures: 2.5 h, 5 h, 10 h, 20 h, 24 h)
          never attempted  -> now
          interrupted      -> (crash/restart while in flight) treated as a failure:
                              CelesTrak may have counted it

        SAFE_GAP = CelesTrak's 2-hour update interval + satellites.retry_margin_min.
        No request is EVER sent less than SAFE_GAP after the previous one for that
        group, whatever happened.

    WHY THIS IS SAFE WITHOUT A HUMAN (CelesTrak usage policy)
        The policy asks clients to stop on errors because repeated error responses --
        about 50 within 2 hours -- get the IP address firewalled. This schedule sends
        at most one request per group per SAFE_GAP, fewer and fewer while errors
        persist, and never on start-up alone. Failures are still reported (log,
        health event, dashboard), just not as a stop that nobody can lift on a remote
        station.

    CLOCK
        Times are stored as UTC epoch seconds (+ ISO text for reading): daylight-saving
        changes cannot shorten a wait. If the clock reads EARLIER than the recorded
        attempt (a Pi without RTC booting before NTP sync), the wait is treated as not
        elapsed -- the safe direction.
    """

    def __init__(self, cache_dir, cfg_getter, http_get, user_agent, admits=None, on_failure=None):
        self._dir = cache_dir
        self._cfg = cfg_getter              # -> current cfg["satellites"]
        self._get = http_get                # requests.get-compatible
        self._ua = user_agent
        self._admits = admits               # storage governor (path, nbytes) -> bool
        self._on_failure = on_failure       # callback(group, record): report (health event)
        self._lock = threading.Lock()
        self._state_path = os.path.join(cache_dir, "state.json")
        self._loaded_key = None
        self._entries = []                  # [(name, norad, Satrec)]
        self._catalogue = Catalogue([])
        self.refresh_in_flight = threading.Event()

    # ---------------- persistent state ----------------
    def _state(self):
        try:
            with open(self._state_path, encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            return {"schema": 2, "groups": {}}
        if st.get("schema") != 2:          # first version: {"groups": {g: {last_attempt, last_success}}, "halt": {...}}
            groups = {}
            for g, rec in (st.get("groups") or {}).items():
                ok = rec.get("last_success", 0) >= rec.get("last_attempt", 0) > 0
                groups[g] = {"last_attempt": rec.get("last_attempt", 0), "last_success": rec.get("last_success"),
                             "result": ok, "status": 200 if ok else None, "message": "", "failures": 0 if ok else 1}
            halt = st.get("halt")
            if halt:
                g = next((g for g in groups if f"GROUP={g}&" in halt.get("url", "")), None)
                if g:
                    groups[g].update(result=False, status=halt.get("status"), message=halt.get("message", ""),
                                     failures=max(1, groups[g].get("failures", 0)))
            st = {"schema": 2, "groups": groups}
        return st

    def _save_state(self, st):
        os.makedirs(self._dir, exist_ok=True)
        for rec in st["groups"].values():            # human-readable mirrors of the epochs
            for k in ("last_attempt", "last_success", "next_attempt"):
                if rec.get(k):
                    rec[k + "_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec[k]))
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, self._state_path)

    # ---------------- schedule ----------------
    def _group_path(self, group):
        return os.path.join(self._dir, f"{group}.csv")

    def _age_h(self, path):
        try:
            return (time.time() - os.path.getmtime(path)) / 3600.0
        except OSError:
            return math.inf

    def age_hours(self):
        """Age of the OLDEST group file (inf if any group was never downloaded)."""
        return max((self._age_h(self._group_path(g)) for g in self._cfg()["groups"]), default=math.inf)

    def safe_gap_s(self):
        return CELESTRAK_MIN_REQUEST_INTERVAL_H * 3600.0 + self._cfg()["retry_margin_min"] * 60.0

    def next_attempt(self, rec, group=None):
        """Epoch time from which this group may be requested again (see class docstring)."""
        if not rec or not rec.get("last_attempt"):
            return 0.0
        cfg, gap = self._cfg(), self.safe_gap_s()
        last = rec["last_attempt"]
        if rec.get("result") is True:
            if group is not None and not os.path.exists(self._group_path(group)):
                return last + gap          # cached file deleted: fetch again, still after the safe gap
            return last + max(gap, cfg["refresh_hours"] * 3600.0)
        n = max(1, int(rec.get("failures") or 1))     # result False, or None = interrupted
        return last + min(gap * 2 ** (n - 1), max(gap, cfg["refresh_hours"] * 3600.0))

    def due_groups(self):
        st, now = self._state(), time.time()
        due = []
        for g in self._cfg()["groups"]:
            rec = st["groups"].get(g)
            if rec and rec.get("last_attempt", 0) > now:
                continue   # clock behind the recorded attempt (boot before NTP): wait
            if now >= self.next_attempt(rec, g):
                due.append(g)
        return due

    def is_due(self):
        return bool(self.due_groups())

    def status(self):
        """Per-group view for the dashboard: {group: record + next_attempt}."""
        st = self._state()
        return {g: {**st["groups"].get(g, {}), "next_attempt": self.next_attempt(st["groups"].get(g), g)}
                for g in self._cfg()["groups"]}

    # ---------------- download ----------------
    def refresh(self):
        """Request every due group once. Never raises for a refused download: the
        outcome is recorded and reported. Returns the number of objects downloaded."""
        cfg = self._cfg()
        total = 0
        for group in self.due_groups():
            url = cfg["url_template"].format(group=group)
            st = self._state()
            rec = st["groups"].setdefault(group, {})
            rec.update(last_attempt=time.time(), result=None, status=None, message="request in flight")
            rec["next_attempt"] = self.next_attempt(rec)
            self._save_state(st)  # BEFORE the request: a crash/restart cannot cause an early repeat
            try:
                resp = self._get(url, timeout=cfg["request_timeout_sec"],
                                 headers={"User-Agent": self._ua}, allow_redirects=False)
                status, text = resp.status_code, getattr(resp, "text", "") or ""
            except Exception as e:                      # network down, DNS, timeout, firewall
                status, text = type(e).__name__, str(e)
            entries = parse_omm_csv(text) if status == 200 else []
            if status == 200 and entries:
                path = self._group_path(group)
                if self._admits is not None and not self._admits(path, len(text)):
                    self._record(group, False, "storage", "storage governor refused the write (drive full)")
                    continue
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, path)
                self._record(group, True, 200, f"{len(entries)} objects")
                logging.info(f"Satellites: GROUP={group} downloaded, {len(entries)} objects.")
                total += len(entries)
            else:
                msg = " ".join(text.split())[:400] if status != 200 else "response contained no valid OMM records"
                self._record(group, False, status, msg)
        return total

    def _record(self, group, ok, status, message):
        st = self._state()
        rec = st["groups"].setdefault(group, {})
        rec.update(result=ok, status=status, message=message)
        if ok:
            rec["last_success"], rec["failures"] = rec["last_attempt"], 0
        else:
            # An interrupted attempt (result None) was already counted by next_attempt;
            # count it here so the back-off keeps growing across restarts too.
            rec["failures"] = int(rec.get("failures") or 0) + 1
        rec["next_attempt"] = self.next_attempt(rec)
        self._save_state(st)
        if not ok:
            wait_h = (rec["next_attempt"] - time.time()) / 3600.0
            logging.warning(
                f"Satellites: GROUP={group} download failed ({status}: \"{message[:200]}\"); "
                f"failure {rec['failures']} in a row, next attempt in {wait_h:.1f} h "
                f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(rec['next_attempt']))}). "
                f"The cached catalogue stays in use while younger than {self._cfg()['max_catalog_age_days']:g} days."
            )
            if self._on_failure:
                self._on_failure(group, rec)

    # ---------------- load ----------------
    def catalogue(self):
        """Merged catalogue of all usable group files (reloaded only when a file changed)."""
        cfg = self._cfg()
        files = []
        for g in cfg["groups"]:
            p = self._group_path(g)
            if self._age_h(p) <= cfg["max_catalog_age_days"] * 24.0:
                files.append((p, os.path.getmtime(p)))
        key = tuple(files)
        with self._lock:
            if key != self._loaded_key:
                merged, seen = [], set()
                for p, _ in files:
                    with open(p, encoding="utf-8") as f:
                        for name, norad, sat in parse_omm_csv(f.read()):
                            if norad not in seen:
                                seen.add(norad)
                                merged.append((name, norad, sat))
                self._entries = merged
                self._catalogue = Catalogue(merged)
                self._loaded_key = key
            return self._catalogue, self._entries


# --------------------------------------------------------------- geometry helpers

def _unit_altaz(alt_rad, az_rad):
    """East-North-Up unit vector."""
    ca = math.cos(alt_rad)
    return np.array([ca * math.sin(az_rad), ca * math.cos(az_rad), math.sin(alt_rad)])


def _angle_deg(u, v):
    """Angle between unit vectors (broadcasts over leading axes)."""
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


class CameraModel:
    """Pixel -> ray in the camera frame; only angles between rays are ever used."""

    def __init__(self, width, height, hfov_deg, projection):
        self.cx, self.cy = width / 2.0, height / 2.0
        self.projection = projection
        half = math.radians(hfov_deg) / 2.0
        self.f = self.cx / half if projection == "equidistant" else self.cx / math.tan(half)

    def ray(self, x, y):
        dx, dy = x - self.cx, y - self.cy
        r = math.hypot(dx, dy)
        theta = r / self.f if self.projection == "equidistant" else math.atan2(r, self.f)
        phi = math.atan2(dy, dx)
        s = math.sin(theta)
        return np.array([s * math.cos(phi), s * math.sin(phi), math.cos(theta)])

    def off_axis_deg(self, x, y):
        return float(_angle_deg(self.ray(x, y), np.array([0.0, 0.0, 1.0])))

    def bbox_streak_deg(self, bbox):
        """Angular length of a streak spanning bbox (mean of both diagonals)."""
        x0, y0, x1, y1 = bbox
        return 0.5 * float(_angle_deg(self.ray(x0, y0), self.ray(x1, y1)) + _angle_deg(self.ray(x0, y1), self.ray(x1, y0)))

    def half_diagonal_deg(self):
        return self.off_axis_deg(0.0, 0.0)


def to_utc(local_naive):
    """Frame timestamps are naive LOCAL times (datetime.now())."""
    return local_naive.astimezone(timezone.utc)


def _jd(dt_utc_list):
    jd, fr = zip(*(jday(d.year, d.month, d.day, d.hour, d.minute, d.second + d.microsecond * 1e-6) for d in dt_utc_list))
    return np.array(jd), np.array(fr)


def _gmst_rad(jd, fr):
    """IAU-82 GMST (the rotation SGP4's TEME frame is defined against); UT1 ~ UTC."""
    t = ((jd - 2451545.0) + fr) / 36525.0
    sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t ** 2 - 6.2e-6 * t ** 3
    return np.radians((sec % 86400.0) / 240.0)


def _observer_ecef(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A_KM / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return np.array([n * math.cos(lat) * math.cos(lon), n * math.cos(lat) * math.sin(lon), n * (1 - e2) * math.sin(lat)])


def _sun_unit_teme(dt_utc):
    """Geocentric apparent Sun direction (equinox of date ~ TEME to arc-minutes)."""
    s = ephem.Sun(ephem.Date(dt_utc.replace(tzinfo=None)))
    ra, dec = float(s.g_ra), float(s.g_dec)
    return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


class SatelliteMatcher:
    def __init__(self, lat_deg, lon_deg):
        self.lat, self.lon = lat_deg, lon_deg
        self._obs_ecef = _observer_ecef(lat_deg, lon_deg)
        la, lo = math.radians(lat_deg), math.radians(lon_deg)
        # ECEF -> ENU rotation rows
        self._enu = np.array([
            [-math.sin(lo), math.cos(lo), 0.0],
            [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
            [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
        ])

    def topocentric(self, sat_array, times_utc):
        """
        -> (dirs ENU unit vectors [N, T, 3], alt_deg [N, T], sunlit [N, T], ok [N, T])
        for every satellite of sat_array at every time.
        """
        jd, fr = _jd(times_utc)
        err, r, _ = sat_array.sgp4(jd, fr)                      # TEME km, [N, T, 3]
        g = _gmst_rad(jd, fr)                                    # [T]
        cg, sg = np.cos(g), np.sin(g)
        x = cg * r[..., 0] + sg * r[..., 1]
        y = -sg * r[..., 0] + cg * r[..., 1]
        ecef = np.stack([x, y, r[..., 2]], axis=-1)
        d = (ecef - self._obs_ecef) @ self._enu.T                # ENU km
        rng = np.linalg.norm(d, axis=-1)
        dirs = d / rng[..., None]
        alt = np.degrees(np.arcsin(np.clip(dirs[..., 2], -1, 1)))
        sun = np.stack([_sun_unit_teme(t) for t in times_utc])   # [T, 3] (TEME ~ equinox of date)
        proj = np.sum(r * sun[None], axis=-1)
        perp = np.linalg.norm(r - proj[..., None] * sun[None], axis=-1)
        sunlit = ~((proj < 0) & (perp < EARTH_RADIUS_KM))
        return dirs, alt, sunlit, err == 0

    def boresight(self, cam_az_deg, cam_alt_deg, astro_data, when_utc, camera, tol_deg):
        """Boresight ENU unit vector, and whether it comes from a plate solve (centred crop only)."""
        if astro_data and astro_data.get("ra_deg") is not None:
            ox, oy = astro_data.get("roi_origin_px") or (None, None)
            sh = astro_data.get("roi_shape_px")
            if ox is not None and sh and camera.off_axis_deg(ox + sh[1] / 2.0, oy + sh[0] / 2.0) <= tol_deg:
                star = ephem.FixedBody()
                star._ra, star._dec, star._epoch = math.radians(astro_data["ra_deg"]), math.radians(astro_data["dec_deg"]), ephem.J2000
                obs = ephem.Observer()
                obs.lat, obs.lon, obs.pressure = str(self.lat), str(self.lon), 0
                obs.date = ephem.Date(when_utc.replace(tzinfo=None))
                star.compute(obs)
                return _unit_altaz(float(star.alt), float(star.az)), True
        return _unit_altaz(math.radians(cam_alt_deg), math.radians(cam_az_deg)), False

    def match(self, catalogue, entries, track, exposure_s, camera, bore, solved, cfg):
        """
        track: [(utc datetime of frame end, [x0, y0, x1, y1]), ...] (>= 1 entry)
        -> best match dict, or None.
        """
        from datetime import timedelta
        t_start = time.monotonic()
        if not len(catalogue):
            return None
        tol = cfg["position_tolerance_solved_deg" if solved else "position_tolerance_deg"]
        pad = cfg["time_tolerance_sec"]

        frames = []
        for t_end, bbox in track:
            cxp, cyp = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            frames.append((t_end, camera.off_axis_deg(cxp, cyp), camera.bbox_streak_deg(bbox)))
        t0, t1 = min(f[0] for f in frames), max(f[0] for f in frames)

        # Pre-filter on a time grid fine enough that no satellite can cross the field
        # between two samples unseen: step = half the FOV radius at the fastest LEO rate.
        fov_r = camera.half_diagonal_deg()
        step = (fov_r / 2.0) / LEO_MAX_ANGULAR_RATE_DEG_S
        first = t0 - timedelta(seconds=exposure_s + pad)
        span = (t1 + timedelta(seconds=pad) - first).total_seconds()
        n = max(2, int(math.ceil(span / step)) + 1)
        grid = [first + timedelta(seconds=k * span / (n - 1)) for k in range(n)]
        dirs, alt, sunlit, ok = self.topocentric(catalogue.array, grid)
        near = (alt > 0) & sunlit & ok & (_angle_deg(dirs, bore) <= fov_r + tol)
        cand = np.nonzero(near.any(axis=1))[0]

        best = None
        if len(cand):
            sub = SatrecArray([entries[i][2] for i in cand])
            times = []
            for t_end, _, _ in frames:
                times += [t_end - timedelta(seconds=exposure_s), t_end]
            d2, a2, s2, ok2 = self.topocentric(sub, times)           # [C, 2F, ...]
            off = _angle_deg(d2, bore)                               # [C, 2F]
            for ci, idx in enumerate(cand):
                matched, residuals, rates = 0, [], []
                for fi, (_, off_evt, streak_evt) in enumerate(frames):
                    j0, j1 = 2 * fi, 2 * fi + 1
                    if not (ok2[ci, j0] and ok2[ci, j1]):
                        continue
                    visible = (a2[ci, j0] > 0 and s2[ci, j0]) or (a2[ci, j1] > 0 and s2[ci, j1])
                    if not visible:
                        continue                                     # in shadow or below horizon
                    arc = float(_angle_deg(d2[ci, j0], d2[ci, j1]))  # swept in one exposure
                    rate = arc / exposure_s if exposure_s > 0 else 0.0
                    lo = min(off[ci, j0], off[ci, j1]) - rate * pad  # timestamp uncertainty
                    hi = max(off[ci, j0], off[ci, j1]) + rate * pad
                    pos_res = 0.0 if lo - tol <= off_evt <= hi + tol else min(abs(off_evt - lo), abs(off_evt - hi))
                    len_res = abs(streak_evt - arc)
                    if pos_res <= tol and len_res <= tol:
                        matched += 1
                        residuals.append(max(pos_res, len_res))
                        rates.append(rate)
                frac = matched / len(frames)
                if frac >= cfg["min_matching_fraction"]:
                    score = float(np.mean(residuals))
                    if best is None or (frac, -score) > (best["matched_fraction"], -best["residual_deg"]):
                        best = {
                            "name": catalogue.names[idx], "norad": catalogue.norads[idx],
                            "matched_fraction": round(frac, 3), "frames": len(frames),
                            "residual_deg": round(score, 3), "angular_rate_deg_s": round(float(np.mean(rates)), 3),
                            "solved_pointing": solved, "tolerance_deg": tol,
                        }
        logging.info(
            f"Satellites: {len(catalogue)} in catalogue, {len(cand)} sunlit near the field during the event, "
            f"{'match: ' + best['name'] if best else 'no match'} ({time.monotonic() - t_start:.2f}s)."
        )
        return best
