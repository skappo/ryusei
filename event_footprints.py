"""
event_footprints.py

Thread-safe registry of where and when the detector saw a transient, so the
timelapse stacker can protect exactly those pixels from temporal sigma-clipping.

WHY: a meteor, plane or satellite crossing during a routine timelapse stack is,
at each pixel it touches, a single-frame bright outlier -- precisely what a
median/MAD clip rejects. Without protection the timelapse shows plain stars at
the moment the event stack shows the object. With protection, clipping still
removes cosmic rays, hot pixels and noise spikes everywhere else.

PRODUCER (DetectionThread, one writer):
    open() at the trigger, extend() on every frame of the event, close() at its
    end, and mark_evaluated() once per frame the detector has fully decided on.

CONSUMER (TimelapseThread):
    wait_covered(last_ts, ...) before combining a stack, then footprints().

THE COMPLETENESS GUARANTEE (why wait_covered exists): an event's time span
starts at the oldest frame of the detector's pre-event buffer, i.e. up to
`lookback_frames` frames BEFORE the trigger frame. So a frame is known to be
event-free only once the detector has evaluated that frame AND lookback_frames
frames after it. wait_covered() blocks until exactly that holds for the last
frame of a stack; without it, a stack finishing just before a trigger would
miss the event's pre-trigger frames.

Timestamps are the pipeline's per-frame `ts` (ISO strings or datetimes, as
produced by the capture loop); they are the join key between the two threads.
"""
import threading
from collections import deque
from datetime import datetime


def as_datetime(ts):
    """Pipeline frame ts (ISO string or datetime) -> datetime."""
    return ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))


class EventFootprintRegistry:
    def __init__(self):
        self._cond = threading.Condition()
        self._events = {}  # id -> {"t_start", "t_end", "bbox": [x0,y0,x1,y1] | None, "open": bool}
        self._next_id = 0
        self._lookback = 0
        self._evaluated = deque(maxlen=1)  # ts of the most recent (lookback+1) evaluated frames

    # ---------------- producer side ----------------
    def configure(self, lookback_frames):
        """Called by the detector at loop start with its pre-event buffer length."""
        with self._cond:
            self._lookback = max(0, int(lookback_frames))
            self._evaluated = deque(self._evaluated, maxlen=self._lookback + 1)
            self._cond.notify_all()

    def open(self, t_start, t_now, bbox):
        with self._cond:
            eid = self._next_id
            self._next_id += 1
            self._events[eid] = {
                "t_start": as_datetime(t_start),
                "t_end": as_datetime(t_now),
                "bbox": list(bbox) if bbox else None,
                "open": True,
            }
            return eid

    def extend(self, eid, t_now, bbox=None):
        """Extend the event's time span to t_now; grow its bbox if this frame had motion."""
        with self._cond:
            ev = self._events.get(eid)
            if ev is None:
                return
            ev["t_end"] = max(ev["t_end"], as_datetime(t_now))
            if bbox:
                if ev["bbox"] is None:
                    ev["bbox"] = list(bbox)
                else:
                    b = ev["bbox"]
                    b[0], b[1] = min(b[0], bbox[0]), min(b[1], bbox[1])
                    b[2], b[3] = max(b[2], bbox[2]), max(b[3], bbox[3])

    def close(self, eid):
        with self._cond:
            ev = self._events.get(eid)
            if ev is not None:
                ev["open"] = False

    def mark_evaluated(self, ts):
        """The detector has finished deciding about the frame with this ts."""
        with self._cond:
            self._evaluated.append(as_datetime(ts))
            self._cond.notify_all()

    # ---------------- consumer side ----------------
    def _covered(self, ts):
        # The (lookback+1)-th most recent evaluated frame is at or after ts:
        # ts itself and `lookback` successors have all been decided.
        return len(self._evaluated) == self._evaluated.maxlen and self._evaluated[0] >= ts

    def wait_covered(self, ts, timeout_sec):
        """Block up to timeout_sec until no future trigger can still claim frame ts.
        Contract of a heartbeat waiter: returns True as soon as covered."""
        ts = as_datetime(ts)
        with self._cond:
            return self._cond.wait_for(lambda: self._covered(ts), timeout=timeout_sec)

    @property
    def lookback_frames(self):
        with self._cond:
            return self._lookback

    def footprints(self, t_first, t_last):
        """Snapshot of events (t_start, t_end, bbox) whose span overlaps [t_first, t_last]."""
        t_first, t_last = as_datetime(t_first), as_datetime(t_last)
        with self._cond:
            return [
                (ev["t_start"], ev["t_end"], tuple(ev["bbox"]))
                for ev in self._events.values()
                if ev["bbox"] is not None and ev["t_start"] <= t_last and ev["t_end"] >= t_first
            ]

    def prune(self, before_ts):
        """Drop CLOSED events that ended before before_ts (no future stack can overlap them)."""
        before_ts = as_datetime(before_ts)
        with self._cond:
            for eid in [k for k, ev in self._events.items() if not ev["open"] and ev["t_end"] < before_ts]:
                del self._events[eid]
