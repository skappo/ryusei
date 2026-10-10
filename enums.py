"""
enums.py

Small, shared, standalone enum + associated state-machine dict, extracted
from the main pipeline file. Needed by both sky_monitor_mixin.py and the
image-stacking methods still in Pipeline's own body (a future
stacking_mixin.py candidate) -- lives here as a neutral shared location
both can import from without circularity.
"""
import logging
from enum import Enum, auto


class SkyConditionStatus(Enum):
    # Represents the possible sky condition check result.
    CLEAR = auto()
    CLOUDY = auto()
    ERROR = auto()

SKY_STATE_MACHINE = {
    SkyConditionStatus.CLEAR: {
        "hold_active": False,
        "timelapse_speed_key": "nightmode_interval_sec",
        "failsafe_threshold": False,
        "log_level": logging.INFO,
        "log_msg": "Sky Monitor: Sky is CLEAR. Releasing WEATHER HOLD.",
    },
    SkyConditionStatus.CLOUDY: {
        "hold_active": True,
        "timelapse_speed_key": "cloudy_interval_sec",
        "failsafe_threshold": False,
        "log_level": logging.WARNING,
        "log_msg": "Sky Monitor: CLOUDY/POOR conditions. Engaging WEATHER HOLD.",
    },
    SkyConditionStatus.ERROR: {
        "hold_active": True,
        "timelapse_speed_key": "cloudy_interval_sec",
        "failsafe_threshold": True,
        "log_level": logging.ERROR,
        "log_msg": "Sky Monitor: SENSOR ERROR. Engaging SAFE MODE.",
    },
}

DEBUG_LEVELS = {
    1: "Stats",
    2: "Images",
    3: "Full/Stream",
    4: "Variables dump",
}