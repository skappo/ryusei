"""
session_archive.py

Index of archived observing sessions (one entry per night) and a thumbnail
cache for the dashboard's session archive page.

NIGHT DATING. A session runs across midnight, so every image is assigned to
the night it belongs to with the astronomical convention: a night is labelled
by the date of the evening it begins, and anything recorded before local noon
belongs to the previous evening. The time used is the file's modification
time, i.e. the moment the image was last written during that night (shutil
.move preserves it when files are archived). This is what makes the dating
correct even for files whose NAME carries the wrong date: until this fix the
masterpiece/star-trail archive names used the process start time, so after a
daytime restart the previous night was filed under the next day's date.

Source: logs/archive only -- the archive of COMPLETED sessions (files are moved there
at the start of the next session by the scheduler and the janitor):
    masterpiece_<date>[_n].png   accumulated meteor trails
    startrail_<date>[_n].png     star trail composite
    keogram_<date>.png           keogram
    journal_<date>*.pdf          nightly journal (linked)
"""
import logging
import os
import re
from datetime import datetime, timedelta

import cv2

from constants import NIGHT_SPLIT_HOUR

KINDS = ("masterpiece", "startrail", "keogram")
IMAGE_NAME_RE = re.compile(r"^(masterpiece|startrail|keogram)_\d{8}(?:T\d{6})?(?:_\d+)?\.png$")
JOURNAL_NAME_RE = re.compile(r"^journal_\d{8}.*\.pdf$")


def night_of(timestamp):
    """Observing night (the evening's date) for an epoch timestamp."""
    return (datetime.fromtimestamp(timestamp) - timedelta(hours=NIGHT_SPLIT_HOUR)).date()


def archive_name(prefix, src_path, dest_dir):
    """
    Collision-free archive path '<prefix>_<YYYYMMDD>[_n].png' for src_path, dated by
    the night its content belongs to (see module docstring).
    """
    stem = f"{prefix}_{night_of(os.path.getmtime(src_path)):%Y%m%d}"
    ext = os.path.splitext(src_path)[1]
    path = os.path.join(dest_dir, stem + ext)
    n = 1
    while os.path.exists(path):
        path = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    return path


def _scan(directory, pattern, kinds=None):
    out = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return out
    for e in entries:
        if not (e.is_file() and pattern.match(e.name)):
            continue
        kind = e.name.split("_", 1)[0]
        if kinds and kind not in kinds:
            continue
        try:
            mtime = e.stat().st_mtime
        except FileNotFoundError:
            continue  # removed meanwhile (janitor / upload handling)
        out.append((kind, e.name, mtime))
    return out


def collect_sessions(archive_dir):
    """
    -> list of {"night": date, "masterpiece": [...], "startrail": [...], "keogram": [...],
                "journal": [...]} newest first; each list holds (name, mtime) newest first.
    A night with several sessions (a restart mid-night) simply lists several images.
    """
    nights = {}
    found = (_scan(archive_dir, IMAGE_NAME_RE, set(KINDS))
             + [("journal", n, m) for _, n, m in _scan(archive_dir, JOURNAL_NAME_RE)])
    for kind, name, mtime in found:
        entry = nights.setdefault(night_of(mtime), {k: [] for k in KINDS + ("journal",)})
        entry[kind].append((name, mtime))
    result = []
    for night in sorted(nights, reverse=True):
        entry = nights[night]
        for k in entry:
            entry[k].sort(key=lambda nm: nm[1], reverse=True)
        entry["night"] = night
        result.append(entry)
    return result


def thumbnail_bytes(src_path, cache_dir, width, jpeg_quality, admits=None):
    """
    JPEG thumbnail of src_path, cached as <cache_dir>/<name>.<mtime>.<width>.jpg so a
    replaced source or a changed width regenerates it. Older variants of the same
    source are removed. admits(path, nbytes) is the storage governor's check: if it
    refuses, the thumbnail is still returned, just not cached.
    """
    st = os.stat(src_path)
    base = os.path.basename(src_path)
    key = f"{base}.{int(st.st_mtime)}.{width}.jpg"
    cached = os.path.join(cache_dir, key)
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            return f.read()
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {base}")
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise ValueError(f"JPEG encoding failed: {base}")
    data = buf.tobytes()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        for old in os.listdir(cache_dir):
            if old.startswith(base + ".") and old != key:
                os.remove(os.path.join(cache_dir, old))
        if admits is None or admits(cached, len(data)):
            tmp = cached + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, cached)  # atomic: concurrent requests never read half a file
    except OSError as e:
        logging.warning(f"Session archive: thumbnail cache write failed for {base}: {e}")
    return data


def prune_thumbnails(cache_dir, live_names):
    """Remove cached thumbnails whose source image no longer exists."""
    try:
        for f in os.listdir(cache_dir):
            if f.split(".png.", 1)[0] + ".png" not in live_names:
                os.remove(os.path.join(cache_dir, f))
    except OSError:
        pass
