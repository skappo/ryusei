"""
constants.py

Mathematical and physical constants: values fixed by mathematics, physics or a
standard, which are NOT tunables and therefore do not belong in the config.
Named here once instead of being re-typed (and potentially mistyped) across
modules. Everything tunable lives in config_models.py with validation.

Plain unit conversions (1_000_000 us/s, 60 s/min, 86400 s/day), dtype limits
(255 for uint8) and array geometry (3 colour channels, 2x3 affine matrices)
stay inline: they are self-describing and naming them adds nothing.
"""
import math

# Robust scale estimators -> Gaussian sigma (consistency constants).
# sigma = MAD / Phi^-1(3/4) = 1.4826 * MAD
MAD_TO_SIGMA = 1.0 / 0.6744897501960817          # 1.482602218505602
# sigma = IQR / (2 * Phi^-1(3/4)) = 0.7413 * IQR
IQR_TO_SIGMA = 1.0 / (2.0 * 0.6744897501960817)  # 0.741301109252801

# ITU-R BT.601 luma coefficients, in OpenCV's B, G, R channel order.
REC601_LUMA_BGR = (0.114, 0.587, 0.299)

# Standard deviation of uniform quantisation noise for a 1-DN step: 1/sqrt(12).
# The smallest stddev an integer-valued (uint8) measurement can meaningfully
# report; used as the floor for noise-ratio denominators instead of an
# arbitrary epsilon.
QUANTIZATION_SIGMA_DN = 1.0 / math.sqrt(12.0)    # 0.2887

# Observing-night convention: a night is labelled by the date of the evening it
# begins; anything recorded before local noon belongs to the previous evening.
# A definition (as in astronomical night logs), not a tunable.
NIGHT_SPLIT_HOUR = 12

# Fastest apparent angular rate of an Earth satellite: a circular orbit at the
# lowest altitude where one survives more than a few revolutions (~160 km), seen
# at the zenith: omega = v / h with v = sqrt(GM / (R + h)).
#   v = sqrt(398600.4418 / (6371 + 160)) = 7.81 km/s ; 7.81 / 160 = 0.0488 rad/s
LEO_MAX_ANGULAR_RATE_DEG_S = 2.8

# CelesTrak updates GP data once every 2 hours and enforces one download per group
# per update (https://celestrak.org/usage-policy.php). A policy, not a tunable.
CELESTRAK_MIN_REQUEST_INTERVAL_H = 2.0
