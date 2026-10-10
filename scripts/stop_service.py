#!/usr/bin/env python3
"""
stop_service.py

A helper script to cleanly stop and disable the systemd service for the
Ryusei Pipeline. This script is designed to be run with sudo.

By using 'systemctl stop', this script triggers the pipeline's own graceful
shutdown procedure.

Usage:
    sudo python3 stop_service.py
"""

import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# This should match the service name in setup_service.py
SERVICE_NAME = "ryusei.service"

# --- Main Logic ---

def run_command_gracefully(command):
    """
    Runs a shell command and prints its output.
    It does not exit on failure, as a non-zero exit code for 'stop' or
    'disable' often just means the service was already in that state.
    """
    print(f"--> Running: {' '.join(command)}")
    try:
        # We use check=False and handle the return code manually.
        result = subprocess.run(
            command,
            check=False,  # Do not raise an exception on non-zero exit codes
            text=True,
            capture_output=True
        )
        if result.returncode == 0:
            print("--> Success.")
        else:
            # Print stderr as a warning, but don't treat it as a fatal error.
            print(f"--> Warning: Command returned exit code {result.returncode}.")
            if result.stderr:
                print(f"--> Message: {result.stderr.strip()}")
    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"Command '{command[0]}' not found. Is systemd installed and in your PATH?")
        sys.exit(1)
    except Exception as e:
        print(f"--- UNEXPECTED ERROR ---")
        print(e)
        sys.exit(1)


def main():
    """Main function to stop and disable the systemd service."""
    # Check if running as root, which is required for systemd operations
    if os.geteuid() != 0:
        print("This script must be run with sudo to interact with systemd.")
        print("Please run: sudo python3 stop_service.py")
        sys.exit(1)

    print(f"--- Ryusei Pipeline Service Shutdown ---")
    print(f"Service to be stopped: {SERVICE_NAME}")
    print("-----------------------------------------\n")

    # --- Systemd Operations ---

    # 1. Stop the currently running service.
    #    systemd sends a SIGTERM signal, which our app catches for a clean shutdown.
    print("--- Step 1: Stopping the service ---")
    run_command_gracefully(["systemctl", "stop", SERVICE_NAME])

    # 2. Disable the service from starting automatically on the next boot.
    print("\n--- Step 2: Disabling the service from starting on boot ---")
    run_command_gracefully(["systemctl", "disable", SERVICE_NAME])

    # 3. Reload the systemd manager configuration to ensure it knows the service is disabled.
    print("\n--- Step 3: Reloading systemd daemon ---")
    run_command_gracefully(["systemctl", "daemon-reload"])

    print("\n--- Shutdown Complete! ---")
    print("The Ryusei Pipeline service has been stopped and disabled.")
    print("You can verify its status with:")
    print(f"  systemctl status {SERVICE_NAME}")
    print("\nTo re-enable and start the service in the future, run:")
    print("  sudo python3 setup_service.py")

if __name__ == "__main__":
    main()