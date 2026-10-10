"""
locations.py

Observing sites, kept in assets/locations.json (a small JSON database):

    {"schema": 1,
     "locations": [{"id": "a1b2c3d4", "location": "Roma", "latitude": 41.780143,
                    "longitude": 12.526866, "camera_azimuth": 280.0, "camera_altitude": 90.0}, ...]}

The ACTIVE site is not stored here: it is whatever general.location / latitude /
longitude / camera_azimuth / camera_altitude say in config.json, which the pipeline
reads as before. Selecting a site copies its five values into config.json; the
database only lists the sites one can choose from. One source of truth each:
config.json for "where am I now", this file for "where can I be".

Every record is validated with the same bounds as the configuration (a site that
could not be applied is never stored); writes are atomic (temp file + rename) and
serialised by a lock (dashboard requests run in parallel threads).
"""
import json
import os
import threading
import uuid
from typing import Optional

from pydantic import BaseModel, confloat, constr

LOCATION_FIELDS = ("location", "latitude", "longitude", "camera_azimuth", "camera_altitude")
SCHEMA_VERSION = 1


class LocationRecord(BaseModel):
    location: constr(strip_whitespace=True, min_length=1, max_length=80)
    latitude: confloat(ge=-90.0, le=90.0)
    longitude: confloat(ge=-180.0, le=180.0)
    camera_azimuth: confloat(ge=0.0, le=360.0)
    camera_altitude: confloat(ge=0.0, le=90.0)


def same_site(a, b):
    """True if two records describe the same site and pointing (name case-insensitive)."""
    try:
        return (str(a["location"]).strip().casefold() == str(b["location"]).strip().casefold()
                and all(abs(float(a[k]) - float(b[k])) < 1e-6 for k in LOCATION_FIELDS[1:]))
    except (KeyError, TypeError, ValueError):
        return False


class LocationStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()

    # ---------------- persistence ----------------
    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"schema": SCHEMA_VERSION, "locations": []}
        if not isinstance(data, dict) or not isinstance(data.get("locations"), list):
            raise ValueError(f"{self.path} is not a locations database")
        return data

    def _write(self, data):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, self.path)

    # ---------------- queries ----------------
    def all(self):
        """Valid records, sorted by name. Invalid hand-edited entries are skipped, not fatal."""
        with self._lock:
            out = []
            for rec in self._read()["locations"]:
                try:
                    clean = LocationRecord.model_validate(rec).model_dump()
                except Exception:
                    continue
                clean["id"] = str(rec.get("id") or "")
                out.append(clean)
            return sorted(out, key=lambda r: r["location"].casefold())

    def get(self, loc_id) -> Optional[dict]:
        return next((r for r in self.all() if r["id"] == loc_id), None)

    # ---------------- changes ----------------
    def add(self, fields):
        """Validate and store a new site; returns its record. Names must be unique."""
        rec = LocationRecord.model_validate(fields).model_dump()
        with self._lock:
            data = self._read()
            if any(str(r.get("location", "")).strip().casefold() == rec["location"].casefold()
                   for r in data["locations"]):
                raise ValueError(f"a location named '{rec['location']}' already exists")
            rec = {"id": uuid.uuid4().hex[:8], **rec}
            data["locations"].append(rec)
            data["schema"] = SCHEMA_VERSION
            self._write(data)
            return rec

    def delete(self, loc_id):
        with self._lock:
            data = self._read()
            before = len(data["locations"])
            data["locations"] = [r for r in data["locations"] if r.get("id") != loc_id]
            if len(data["locations"]) == before:
                raise KeyError(loc_id)
            self._write(data)

    def ensure_contains(self, general):
        """
        Make sure the site currently in config.json is in the database (first run, or a
        site set by hand in config.json). No-op if the config has no valid site.
        """
        try:
            current = LocationRecord.model_validate({k: general.get(k) for k in LOCATION_FIELDS}).model_dump()
        except Exception:
            return None
        if any(same_site(r, current) for r in self.all()):
            return None
        try:
            return self.add(current)
        except ValueError:
            # Same name, different coordinates/pointing: keep the database entry, store
            # the config's variant under a distinguishing name.
            current["location"] = f"{current['location']} (from config)"
            try:
                return self.add(current)
            except ValueError:
                return None
