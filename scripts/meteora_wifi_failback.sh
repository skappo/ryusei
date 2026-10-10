#!/bin/bash
# ==============================================================================
# Ryusei Pipeline - Wi-Fi Failback Watchdog
#
# Companion to setup_hotspot.sh. That script makes the Pi fall FORWARD into
# its own AP-mode hotspot when no known Wi-Fi is in range, by setting
# connection.autoconnect-priority (home networks = 10, hotspot = 0). But
# NetworkManager only picks the best-priority connection when it has to
# choose one from scratch (boot, or an explicit disconnect) -- once wlan0 is
# actively HOSTING the hotspot, NM considers that a fully successful,
# connected state and never re-scans to preempt it with a higher-priority
# client connection on its own. A single Wi-Fi radio can't be in AP mode and
# client mode at the same time, so nothing ever tells NM to reconsider.
#
# This script is the missing return path: run periodically (via the paired
# systemd timer), it checks whether the hotspot is the currently active
# connection on wlan0, and if a known Wi-Fi network has come back into range,
# switches back to it.
#
# Installed and enabled automatically by setup_hotspot.sh.
# ==============================================================================

HOTSPOT_PROFILE="Meteora_Hotspot"
LOG_TAG="ryusei-wifi-failback"
WIFI_IFACE="wlan0"

# Is the hotspot currently the active connection on wlan0? If not, we're
# either already on a real network or the interface is down/unmanaged --
# either way, there's nothing for this script to do.
ACTIVE_CONN=$(nmcli -t -f NAME,DEVICE connection show --active | grep ":${WIFI_IFACE}$" | cut -d: -f1)

if [ "$ACTIVE_CONN" != "$HOTSPOT_PROFILE" ]; then
    exit 0
fi

logger -t "$LOG_TAG" "Currently in hotspot mode. Scanning for known Wi-Fi..."

# Ask the radio for a fresh scan. Best-effort: a device that's busy hosting
# an AP can sometimes refuse a rescan request, so don't treat failure here
# as fatal -- fall through and work with whatever nmcli already has cached.
nmcli device wifi rescan ifname "$WIFI_IFACE" >/dev/null 2>&1
sleep 5

# Every known Wi-Fi profile (i.e. anything that isn't our own hotspot).
KNOWN_WIFI=$(nmcli -t -f NAME,TYPE connection show | grep 802-11-wireless | cut -d: -f1 | grep -v "^${HOTSPOT_PROFILE}$")

if [ -z "$KNOWN_WIFI" ]; then
    logger -t "$LOG_TAG" "No known Wi-Fi profiles configured. Staying in hotspot mode."
    exit 0
fi

# SSIDs currently visible over the air.
VISIBLE_SSIDS=$(nmcli -t -f SSID device wifi list ifname "$WIFI_IFACE" 2>/dev/null)

while IFS= read -r PROFILE; do
    # The connection's stored SSID (profile name and SSID can differ if the
    # user renamed the connection, so don't assume they're the same string).
    PROFILE_SSID=$(nmcli -g 802-11-wireless.ssid connection show "$PROFILE" 2>/dev/null)
    [ -z "$PROFILE_SSID" ] && PROFILE_SSID="$PROFILE"

    if echo "$VISIBLE_SSIDS" | grep -Fxq "$PROFILE_SSID"; then
        logger -t "$LOG_TAG" "Known network '$PROFILE_SSID' is back in range. Switching back from hotspot."
        # Bringing this connection up on wlan0 tears down the AP connection
        # automatically -- the radio can only run one at a time.
        nmcli connection up "$PROFILE" ifname "$WIFI_IFACE" >/dev/null 2>&1
        exit 0
    fi
done <<< "$KNOWN_WIFI"

logger -t "$LOG_TAG" "No known networks in range yet. Staying in hotspot mode."
