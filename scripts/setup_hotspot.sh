#!/bin/bash

# ==============================================================================
# Ryusei Pipeline - Auto-Fallback Hotspot Setup
# Designed for Raspberry Pi OS (Bookworm) using NetworkManager
# ==============================================================================

# Ensure the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run this script with sudo:"
  echo "   sudo ./setup_hotspot.sh"
  exit 1
fi

# Check if NetworkManager is installed and running
if ! command -v nmcli &> /dev/null; then
    echo "❌ NetworkManager (nmcli) is not installed."
    echo "This script requires Debian 12 (Bookworm) or newer."
    exit 1
fi

echo "============================================================"
echo "🛰️  Ryusei Field Hotspot Setup"
echo "============================================================"
echo "This script will configure your Raspberry Pi to automatically"
echo "broadcast its own Wi-Fi network when you are out in the field"
echo "and your home Wi-Fi is out of range."
echo ""

# Get user input for Hotspot Name and Password
read -p "Enter Hotspot Name (SSID) [Default: Ryusei_AP]: " HOTSPOT_SSID
HOTSPOT_SSID=${HOTSPOT_SSID:-Ryusei_AP}

read -p "Enter Hotspot Password (min 8 chars)[Default: Ryusei2026!]: " HOTSPOT_PASS
HOTSPOT_PASS=${HOTSPOT_PASS:-Ryusei2026!}

if [ ${#HOTSPOT_PASS} -lt 8 ]; then
    echo "❌ Password must be at least 8 characters long. Aborting."
    exit 1
fi

HOTSPOT_PROFILE="Meteora_Hotspot"

# ---------------------------------------------------------
# STEP 1: Prioritize existing Wi-Fi connections
# ---------------------------------------------------------
echo ""
echo "🔍 Scanning for existing Wi-Fi profiles..."

# Get all wifi connections except the hotspot we are about to create
EXISTING_WIFI=$(nmcli -t -f NAME,TYPE connection show | grep 802-11-wireless | cut -d: -f1 | grep -v "^$HOTSPOT_PROFILE$")

if [ -z "$EXISTING_WIFI" ]; then
    echo "⚠️  No existing Wi-Fi profiles found. (Are you connected via Ethernet?)"
else
    # Loop through existing networks and set their priority to 10
    while IFS= read -r WIFI_NAME; do
        echo "   -> Boosting priority of Home Network: '$WIFI_NAME' to 10"
        nmcli connection modify "$WIFI_NAME" connection.autoconnect-priority 10
    done <<< "$EXISTING_WIFI"
fi

# ---------------------------------------------------------
# STEP 2: Clean up old hotspot profiles
# ---------------------------------------------------------
if nmcli connection show "$HOTSPOT_PROFILE" &> /dev/null; then
    echo "🧹 Removing old hotspot profile '$HOTSPOT_PROFILE'..."
    nmcli connection delete "$HOTSPOT_PROFILE" > /dev/null
fi

# ---------------------------------------------------------
# STEP 3: Create the Fallback Hotspot
# ---------------------------------------------------------
echo "📡 Creating new Fallback Hotspot: '$HOTSPOT_SSID'..."

# Add the Wi-Fi connection
nmcli connection add type wifi ifname wlan0 con-name "$HOTSPOT_PROFILE" autoconnect yes ssid "$HOTSPOT_SSID" > /dev/null

# Configure AP mode, shared internet, and WPA2 security
nmcli connection modify "$HOTSPOT_PROFILE" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$HOTSPOT_PASS"

# Set priority to 0 (Lowest) so it only triggers if Home Wi-Fi is missing
nmcli connection modify "$HOTSPOT_PROFILE" connection.autoconnect-priority 0

# ---------------------------------------------------------
# STEP 4: Restart NetworkManager to apply changes
# ---------------------------------------------------------
echo "🔄 Restarting NetworkManager..."
systemctl restart NetworkManager
sleep 3

# ---------------------------------------------------------
# STEP 5: Install the Wi-Fi failback watcher
# ---------------------------------------------------------
# The priorities set above only take effect when NetworkManager picks a
# connection from scratch (boot, or an explicit disconnect). Once wlan0 is
# actively HOSTING the hotspot, NM won't automatically re-scan and switch
# back to home Wi-Fi on its own -- a single radio can't be in AP mode and
# client mode at once, so nothing ever tells NM to reconsider. Without this
# watcher, the Pi only returns to known Wi-Fi on its next reboot even after
# you're back in range. See meteora_wifi_failback.sh for the full mechanism.
echo ""
echo "🔁 Installing Wi-Fi failback watcher..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILBACK_SRC="$SCRIPT_DIR/meteora_wifi_failback.sh"
FAILBACK_DEST="/usr/local/bin/meteora_wifi_failback.sh"

if [ -f "$FAILBACK_SRC" ]; then
    cp "$FAILBACK_SRC" "$FAILBACK_DEST"
    chmod +x "$FAILBACK_DEST"

    cat > /etc/systemd/system/meteora-wifi-failback.service << 'EOF'
[Unit]
Description=Ryusei Wi-Fi failback check (return from fallback hotspot to known Wi-Fi)
After=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/meteora_wifi_failback.sh
EOF

    cat > /etc/systemd/system/meteora-wifi-failback.timer << 'EOF'
[Unit]
Description=Run Ryusei Wi-Fi failback check periodically

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
Unit=meteora-wifi-failback.service

[Install]
WantedBy=timers.target
EOF

    systemctl daemon-reload
    systemctl enable --now meteora-wifi-failback.timer
    echo "   -> Installed. The Pi will now check every 2 minutes while in"
    echo "      hotspot mode and switch back automatically once home Wi-Fi"
    echo "      is back in range."
else
    echo "⚠️  meteora_wifi_failback.sh not found next to this script -- skipping"
    echo "   watcher installation. Copy it alongside setup_hotspot.sh and re-run"
    echo "   to enable automatic failback."
fi

echo ""
echo "✅ SUCCESS! Your field hotspot is configured."
echo "============================================================"
echo "When your home Wi-Fi is NOT found, connect to:"
echo "   Network Name : $HOTSPOT_SSID"
echo "   Password     : $HOTSPOT_PASS"
echo ""
echo "Once connected in the field, access the dashboard at:"
echo "   http://10.42.0.1:5000"
echo "   (Don't forget to append ?token=YOUR_TOKEN if enabled!)"
echo "============================================================"