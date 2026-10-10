#!/usr/bin/env python3
"""
setup_service.py

A helper script to create, enable, and start the systemd service for the
Ryusei Pipeline. This script is designed to be run with sudo.

Usage:
    sudo python3 setup_service.py
"""

import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# These variables define the service. Adjust them if you rename files or change user.

SERVICE_NAME = "ryusei.service"
SERVICE_DESCRIPTION = "Ryusei Pipeline Service"
PYTHON_EXECUTABLE = sys.executable  # Uses the same python that runs this script
USER = os.environ.get("SUDO_USER", "pi") # Safely get the user who called sudo

# Get the absolute path to the directory containing this setup script
SCRIPT_DIR = Path(__file__).parent.resolve()
MAIN_SCRIPT_PATH = SCRIPT_DIR / "ryusei.py"
CONFIG_FILE_PATH = SCRIPT_DIR / "config.json"

# --- Main Logic ---

def run_command(command):
    """Runs a shell command and prints its output, exiting on failure."""
    print(f"--> Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        print("-------------")
        sys.exit(1)
    print("--> Success.")

def main():
    """Main function to set up the systemd service."""
    # Check if running as root, which is required for systemd operations
    if os.geteuid() != 0:
        print("This script must be run with sudo to interact with systemd.")
        print("Please run: sudo python3 setup_service.py")
        sys.exit(1)
        
    print(f"--- Ryusei Pipeline Service Setup ---")
    print(f"Service Name: {SERVICE_NAME}")
    print(f"User:         {USER}")
    print(f"Script Path:  {MAIN_SCRIPT_PATH}")
    print(f"Config Path:  {CONFIG_FILE_PATH}")
    print("--------------------------------------\n")
    
    # Define the full content of the .service file
    # This uses the definitive configuration with watchdog support
    service_content = f"""
[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target

[Service]
# The 'notify' type tells systemd to wait for the READY=1 signal from our script.
Type=notify

Environment="RYUSEI_DASH_TOKEN=your_long_secret_token"
Environment="SFTP_PASS=your_sftp_password"
Environment="OPENWEATHER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# ExecStart={PYTHON_EXECUTABLE} {MAIN_SCRIPT_PATH} --config {CONFIG_FILE_PATH}
ExecStart=/bin/bash {SCRIPT_DIR}/launcher.sh
WorkingDirectory={SCRIPT_DIR}
StandardOutput=journal
StandardError=journal

# Only restart if the script exits with a non-zero (failure) code.
# A clean exit from the dashboard's "Stop Service" button will not trigger a restart.
Restart=on-failure
RestartSec=3
User={USER}

# Needed for rebooting
AmbientCapabilities=CAP_SYS_ADMIN

# If systemd doesn't receive a watchdog ping for 3 minutes, it will hard-reboot the Pi.
WatchdogSec=30s

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path("/etc/systemd/system") / SERVICE_NAME
    
    # --- Idempotent Check and Creation ---
    # Only write the file if it doesn't exist or if the content has changed
    create_or_update_service_file = True
    if service_path.exists():
        print(f"Service file '{service_path}' already exists.")
        with open(service_path, "r") as f:
            if f.read().strip() == service_content.strip():
                print("Content is identical. No changes needed.")
                create_or_update_service_file = False
            else:
                print("Content has changed. The service file will be overwritten.")
    
    if create_or_update_service_file:
        print(f"Writing service file to '{service_path}'...")
        with open(service_path, "w") as f:
            f.write(service_content)
        print("Service file created/updated successfully.")
    
    # --- Systemd Operations ---
    print("\n--- Configuring systemd ---")
    
    # Reload the systemd manager configuration
    run_command(["systemctl", "daemon-reload"])
    
    # Enable the service to start on boot
    run_command(["systemctl", "enable", SERVICE_NAME])
    
    # Start (or restart) the service immediately
    run_command(["systemctl", "restart", SERVICE_NAME])
    
    print("\n--- Setup Complete! ---")
    print("The Ryusei Pipeline service is now running and enabled on boot.")
    print("You can check its status with:")
    print(f"  sudo systemctl status {SERVICE_NAME}")
    print("You can view its live logs with:")
    print(f"  journalctl -fu {SERVICE_NAME}")

if __name__ == "__main__":
    main()