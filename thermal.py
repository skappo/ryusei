"""
thermal_mixin.py

Thermal sensor reading and throttling/voltage monitoring methods for the
Ryusei pipeline, extracted from the main Pipeline class as a pure move
-- method bodies are unchanged, only their physical location. Composed
into Pipeline via multiple inheritance.

Three non-contiguous chunks in the original file: thermal_sensor_loop
(far from the others), check_throttling_and_voltage + _init_thermal_sensors
+ get_ds18b20_temp (contiguous with each other), and get_core_voltage
(separate again). Confirmed via an explicit interleaving check before
extraction -- many unrelated methods (camera/capture helpers, file
utilities) sit between these groups in the original file.

CROSS-MIXIN DEPENDENCIES (verified by grep, not assumed) -- calls
self.ping_watchdog (ConcurrencyClass) and self.log_health_event (not yet
extracted, still in Pipeline's own body). Must be provided somewhere in
the composed Pipeline class.

Imports here are exactly what this extracted block actually uses --
verified by grep and confirmed clean with pyflakes before finalizing.
"""
import os
import glob
import subprocess
import logging
import time


class ThermalClass:
    """Thermal sensor reading and throttling/voltage monitoring, composed into Pipeline."""

    def thermal_sensor_loop(self):
        """
        Dedicated thread to continuously read the DS18B20 sensor.
        Prevents the 750ms 1-wire I/O block from lagging the Dashboard, Metadata, or Detection loops.
        """
        if not self.ds18b20_cam_path and not self.ds18b20_ext_path:
            logging.info("Thermal Sensor Thread: No sensor found. Thread exiting.")
            return

        logging.info("Thermal Sensor Thread started. Reading every 15 seconds.")
        try:
            while self.running.is_set():
                try:
                    # Read both sensors
                    t_cam = (
                        self.get_ds18b20_temp(self.ds18b20_cam_path)
                        if self.ds18b20_cam_path
                        else None
                    )
                    time.sleep(0.1)
                    t_ext = (
                        self.get_ds18b20_temp(self.ds18b20_ext_path)
                        if self.ds18b20_ext_path
                        else None
                    )

                    with self.status_lock:
                        if t_cam is not None:
                            self.current_sensor_temp_cam = t_cam
                        if t_ext is not None:
                            self.current_sensor_temp_ext = t_ext
                except Exception as e:
                    logging.error(f"Thermal Sensor Thread: Failed to read temp: {e}")

                for _ in range(self.cfg["general"]["thermal_read_interval_sec"]):
                    self.ping_watchdog()
                    if not self.running.is_set():
                        break
                    time.sleep(1)

        except Exception as e:
            logging.critical(
                "CRITICAL FATAL ERROR IN THERMAL SENSOR THREAD!", exc_info=True
            )
            self.log_health_event("CRITICAL", "THERMAL_SENSOR_CRASH", str(e))


    def check_throttling_and_voltage(self):
        """
        Uses vcgencmd to check for active or historical power/thermal issues.
        Returns: (is_undervoltage_now, has_undervoltage_occurred, is_throttled_now, hex_string)
        """
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.cfg["general"]["hw_query_timeout_sec"],
            )
            # Output format is usually: throttled=0x50005
            val_str = result.stdout.split("=")[1].strip()
            val = int(val_str, 16)

            # Bit 0: Under-voltage detected NOW
            is_uv_now = bool(val & 0x1)
            # Bit 16: Under-voltage has occurred since last boot
            has_uv_occurred = bool(val & 0x10000)
            # Bit 2: Currently throttled (due to heat or power)
            is_throttled_now = bool(val & 0x4)

            return is_uv_now, has_uv_occurred, is_throttled_now, val_str
        except Exception as e:
            return False, False, False, "0x0"

    # -----------------
    def _init_thermal_sensors(self):
        """Locates and assigns DS18B20 sensors STRICTLY based on config IDs."""
        self.ds18b20_cam_path = None
        self.ds18b20_ext_path = None

        try:
            base_dir = "/sys/bus/w1/devices/"
            device_folders = glob.glob(base_dir + "28*")

            if not device_folders:
                logging.warning("Hardware: No DS18B20 Thermal Sensors found on the 1-Wire bus.")
                return
            else:
                found_ids = [os.path.basename(f) for f in device_folders]
                logging.info(f"Hardware: 1-Wire bus contains {len(found_ids)} active sensors: {found_ids}")

            cam_id = self.cfg["general"].get("sensor_camera_id")
            ext_id = self.cfg["general"].get("sensor_ext_id")

            # --- CAMERA SENSOR (Critical for Darks) ---
            if cam_id is None:
                logging.warning("Hardware: 'sensor_camera_id' is set to NULL in config. Dynamic Dark Scaling will be DISABLED.")
            elif cam_id == "":
                logging.warning("Hardware: 'sensor_camera_id' is EMPTY in config. Dynamic Dark Scaling will be DISABLED.")
            else:
                # Attempt to bind
                path = os.path.join(base_dir, cam_id.strip(), "w1_slave")
                if os.path.exists(path):
                    self.ds18b20_cam_path = path
                    logging.info(f"Hardware: Camera Sensor bound to {cam_id}")
                else:
                    logging.error(f"Hardware: Camera Sensor ID '{cam_id}' specified but folder NOT found in /sys/bus/w1/devices/")

            # --- EXTERNAL SENSOR (Telemetry only) ---
            if ext_id is None or ext_id == "":
                logging.info("Hardware: 'sensor_ext_id' is null/empty. External telemetry disabled.")
            else:
                path = os.path.join(base_dir, ext_id.strip(), "w1_slave")
                if os.path.exists(path):
                    self.ds18b20_ext_path = path
                    logging.info(f"Hardware: External Sensor bound to {ext_id}")
                else:
                    logging.warning(f"Hardware: External Sensor ID '{ext_id}' specified but folder NOT found.")

        except Exception as e:
            logging.error(f"Error during thermal sensor initialization: {e}")

    # -----------------
    def get_ds18b20_temp(self, sensor_path):
        """
        Reads the raw temperature from the DS18B20 sensor.
        Returns float (Celsius) or None if read fails.
        """
        if not sensor_path:
            return None

        try:
            # The sensor file typically has two lines.
            # Line 1 contains the CRC check, Line 2 contains the 't=XXXXX' value.
            with open(sensor_path, "r") as f:
                lines = f.readlines()

            # Ensure the reading is valid (CRC check)
            if len(lines) < 2 or "YES" not in lines[0]:
                logging.warning(
                    f"DS18B20 ({os.path.basename(os.path.dirname(sensor_path))}): CRC failed."
                )
                return None

            # Extract the temperature string
            parts = lines[1].split("t=")
            if len(parts) < 2:
                return None

            temp_raw = float(parts[1])
            temp_c = temp_raw / 1000.0

            # Sanity check (DS18B20 range is -55 to +125)
            if -55 <= temp_c <= 125:
                return round(temp_c, 2)
            else:
                return None

        except Exception as e:
            logging.error(f"DS18B20: Failed to read sensor: {e}")
            return None

    # -----------------

    def get_core_voltage(self):
        """
        Gets the core voltage for the current Raspberry Pi model, attempting multiple methods for robustness.
        Supports Raspberry Pi 5 (sysfs) and older models (vcgencmd).
        Returns the voltage in Volts, or 0.0 on error.
        """
        # Method 1: Try the modern sysfs interface first (for Pi 5)
        try:
            # Find the hwmon device for the DA9091 PMIC
            for path in glob.glob("/sys/class/hwmon/hwmon*/name"):
                with open(path, "r") as f:
                    if f.read().strip() == "da9091":
                        hwmon_path = os.path.dirname(path)
                        # Find the specific voltage input file labeled "VDD_CORE"
                        for label_path in glob.glob(
                            os.path.join(hwmon_path, "in*_label")
                        ):
                            with open(label_path, "r") as f_label:
                                if f_label.read().strip() == "VDD_CORE":
                                    input_path = label_path.replace("_label", "_input")
                                    with open(input_path, "r") as f_in:
                                        # Read value (in millivolts) and convert to Volts
                                        return int(f_in.read().strip()) / 1000.0
        except (IOError, ValueError, FileNotFoundError):
            # This will catch permission errors or cases where files don't exist.
            # We don't log an error here, just silently fall back to the next method.
            pass

        # Method 2: Fallback to vcgencmd (for Pi 4 and as a backup for Pi 5)
        try:
            # result = subprocess.run(['vcgencmd', 'measure_volts', 'core'], capture_output=True, text=True, check=True)
            result = subprocess.run(
                ["vcgencmd", "measure_volts", "core"],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.cfg["general"]["hw_query_timeout_sec"],
            )

            # Output is typically "volt=1.2345V"
            voltage_str = result.stdout.split("=")[1].replace("V", "").strip()
            return float(voltage_str)
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            IndexError,
            ValueError,
            subprocess.TimeoutExpired,
        ):
            # This will catch errors if vcgencmd isn't found, fails, or returns unexpected output.
            pass

        # Final Fallback
        # If both methods fail, return 0.0
        return 0.0
