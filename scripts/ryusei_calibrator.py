#!/usr/bin/env python3
"""
Ryusei Thermal Master v7.0
1. Captures physical Bias frames.
2. Captures physical Dark frames at STEP_SIZE intervals.
3. Interpolates/Extrapolates a full library from INTERP_MIN to INTERP_MAX.

Usage: 
python ryusei_calibrator.py           # Runs without heating (passive)
python ryusei_calibrator.py --stress  # Enables CPU stress to reach targets
"""

import os
import time
import glob
import logging
import argparse
import numpy as np
import cv2
import multiprocessing
from picamera2 import Picamera2

# --- CONFIGURATION ---
SENSOR_ID = "28-174504304b00" 
# Physical Capture Settings
CAPTURE_RANGE = (15, 38)      # Range to actually heat the sensor
STEP_SIZE = 3                 # Capture every 3 degrees
GAINS = [1.0, 2.0, 3.0, 4.0] 
CAPTURE_COUNT = 15            # Frames to stack per dark
BIAS_COUNT = 50               # Frames to stack per bias

# Interpolation Settings (The final library range)
INTERP_MIN = 10               # Extrapolate down to this
INTERP_MAX = 50               # Extrapolate up to this

# Camera Settings
SOAK_TIME = 30                # Seconds to stabilize temp before capture
EXPOSURE_REF = 5000000        # 5 Seconds reference for Darks
EXPOSURE_BIAS = 100           # 0.1ms reference for Bias
RES = (1920, 1080)
OUTPUT_DIR = "./assets/darks"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Ryusei Thermal Master")
    # CHANGED: Swapped --no-stress for --stress
    parser.add_argument("--stress", action="store_true", help="Enable CPU heating to reach target temperatures")
    return parser.parse_args()

class RyuseiMaster:
    # CHANGED: Defaulted stress_enabled to False
    def __init__(self, stress_enabled: bool = False):
        self.stress_enabled = stress_enabled
        self.sensor_id = SENSOR_ID
        self.temp_sensor_path = self._verify_ds18b20()
        self.picam2 = Picamera2()
        self.stress_processes = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self.target_steps = list(range(CAPTURE_RANGE[0], CAPTURE_RANGE[1] + 1, STEP_SIZE))

    def _verify_ds18b20(self):
        path = f'/sys/bus/w1/devices/{self.sensor_id}/w1_slave'
        if not os.path.exists(path):
            available = [os.path.basename(d) for d in glob.glob('/sys/bus/w1/devices/28*')]
            raise RuntimeError(f"Sensor {self.sensor_id} NOT FOUND.\nAvailable sensors: {available}")
        logging.info(f"Using sensor: {self.sensor_id}")
        return path

    def get_sensor_temp(self):
        try:
            with open(self.temp_sensor_path, 'r') as f:
                lines = f.readlines()
            if "YES" not in lines[0]: return None
            return round(float(lines[1].split("t=")[1]) / 1000.0, 1)
        except: return None

    def start_stress(self):
        # Only proceed if stress is actually enabled in settings
        if not self.stress_enabled: 
            return
        
        if self.stress_processes: 
            return # Already running

        logging.info("🔥 Heating CPU...")
        for _ in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=self._stress_task)
            p.daemon = True
            p.start()
            self.stress_processes.append(p)

    def stop_stress(self):
        # Clean up processes if they exist, regardless of the flag
        if not self.stress_processes:
            return
            
        for p in self.stress_processes: 
            p.terminate()
        self.stress_processes = []
        
        if self.stress_enabled: 
            logging.info("❄️ Heat stopped. Stabilizing...")

    def _stress_task(self):
        while True: _ = [x**2 for x in range(10000)]

    def capture_stack(self, exposure, gain, count, filename):
        logging.info(f"  -> Capturing {count} frames: G{gain}, {exposure/1e6}s")
        config = self.picam2.create_still_configuration(main={"format": 'RGB888', "size": RES})
        self.picam2.configure(config)
        self.picam2.set_controls({"ExposureTime": exposure, "AnalogueGain": gain, "AwbEnable": False})

        frames = []
        self.picam2.start()
        for _ in range(2): self.picam2.capture_array() # Purge
        for _ in range(count):
            # NOTE: Despite the "RGB888" label, libcamera/picamera2 delivers this buffer
            # in physical BGR byte order (a known libcamera naming quirk, not a bug).
            # capture_array() output is ALREADY BGR — safe to feed directly into cv2.*
            # functions (imwrite, mean, etc.) with NO cv2.cvtColor() conversion.
            # Verified empirically against a known-color reference; do not "fix" this
            # by adding a COLOR_RGB2BGR call, it will re-invert the channels.
            raw = self.picam2.capture_array()
            # frames.append(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
            frames.append(raw)
            print(".", end="", flush=True)
        self.picam2.stop()
        print("")

        master = np.median(np.array(frames), axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), master)

    def run_biases(self):
        logging.info("--- PHASE 1: MASTER BIAS ---")
        for g in GAINS:
            fname = f"bias_G{int(g)}.png"
            if not os.path.exists(os.path.join(OUTPUT_DIR, fname)):
                self.capture_stack(EXPOSURE_BIAS, g, BIAS_COUNT, fname)

    def run_physical_darks(self):
        logging.info(f"--- PHASE 2: PHYSICAL CAPTURE ---")
        
        ambient_temp = self.get_sensor_temp()
        if ambient_temp is None:
            logging.error("Could not read sensor. Aborting.")
            return

        # Filter targets that are at or above current ambient (with 0.2 leeway)
        reachable_steps = [s for s in self.target_steps if s >= (ambient_temp - 0.2)]
        
        if not reachable_steps:
            logging.warning(f"Ambient ({ambient_temp}°C) is already above all targets.")
            return

        logging.info(f"Ambient: {ambient_temp}°C. Targets: {reachable_steps}")

        captured_temps = set()
        # Resume logic: see what we already have
        for f in glob.glob(os.path.join(OUTPUT_DIR, "dark_*C_G1_5s.png")):
            try:
                t = int(os.path.basename(f).split('_')[1].replace('C',''))
                captured_temps.add(t)
            except: continue

        try:
            for target in reachable_steps:
                if target in captured_temps:
                    continue
                
                trigger_temp = target - 0.2
                
                # --- FIX: STRESS CHECK ---
                temp = self.get_sensor_temp()
                if not self.stress_enabled and temp < trigger_temp:
                    logging.info(f"⏭️  Skipping target {target}°C (Current {temp}°C < Trigger {trigger_temp}°C and --stress is OFF)")
                    continue

                logging.info(f"⏳ Waiting for {target}°C (Trigger at {trigger_temp}°C)")
                self.start_stress() # Only starts if stress_enabled is True
                
                while True:
                    temp = self.get_sensor_temp()
                    if temp is None: continue

                    print(f"🌡️  Current: {temp}°C | Goal: {target}°C ", end="\r")

                    if temp >= trigger_temp:
                        print(f"\n🎯 Target reached ({temp}°C).")
                        self.stop_stress() 
                        
                        # Only soak/wait if we were actively heating
                        if self.stress_enabled:
                            logging.info(f"Soaking for {SOAK_TIME}s to stabilize...")
                            time.sleep(SOAK_TIME)
                        
                        actual_start_temp = self.get_sensor_temp()
                        logging.info(f"Capturing Master Darks at {actual_start_temp}°C")
                        
                        for g in GAINS:
                            fname = f"dark_{target}C_G{int(g)}_5s.png"
                            self.capture_stack(EXPOSURE_REF, g, CAPTURE_COUNT, fname)
                        
                        captured_temps.add(target)
                        break 
                    
                    # Safety exit for the while loop if stress is disabled
                    if not self.stress_enabled:
                        break
                        
                    time.sleep(1)
        finally:
            self.stop_stress()

    def run_interpolation(self):
        logging.info(f"--- PHASE 3: INTERPOLATION ({INTERP_MIN} to {INTERP_MAX}°C) ---")
        for gain in GAINS:
            bias_path = os.path.join(OUTPUT_DIR, f"bias_G{int(gain)}.png")
            if not os.path.exists(bias_path): continue
            
            bias = cv2.imread(bias_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            files = glob.glob(os.path.join(OUTPUT_DIR, f"dark_*C_G{int(gain)}_5s.png"))
            anchors = {}
            for f in files:
                t = int(os.path.basename(f).split('_')[1].replace('C',''))
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32)
                anchors[t] = cv2.subtract(img, bias)
            
            avail_t = sorted(anchors.keys())
            if not avail_t: continue

            for target_t in range(INTERP_MIN, INTERP_MAX + 1):
                out_name = f"dark_{target_t}C_G{int(gain)}_5s.png"
                if target_t in avail_t: continue

                if target_t < avail_t[0]: 
                    scale = 2.0 ** ((target_t - avail_t[0]) / 6.0)
                    thermal = anchors[avail_t[0]] * scale
                elif target_t > avail_t[-1]: 
                    scale = 2.0 ** ((target_t - avail_t[-1]) / 6.0)
                    thermal = anchors[avail_t[-1]] * scale
                else: 
                    t1 = max([t for t in avail_t if t < target_t])
                    t2 = min([t for t in avail_t if t > target_t])
                    w2 = (target_t - t1) / (t2 - t1)
                    log_t1 = np.log2(np.clip(anchors[t1], 1e-3, None))
                    log_t2 = np.log2(np.clip(anchors[t2], 1e-3, None))
                    thermal = 2.0 ** (log_t1 * (1 - w2) + log_t2 * w2)
                
                synth = cv2.add(bias, np.clip(thermal, 0, None))
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), np.clip(synth, 0, 255).astype(np.uint8))
            logging.info(f"  ✅ Gain {gain} complete.")

if __name__ == "__main__":
    args = parse_args()
    # CHANGED: Logic passed directly from args.stress
    master = RyuseiMaster(stress_enabled=args.stress)
    master.run_biases()
    master.run_physical_darks()
    master.run_interpolation()
    print("\n🎉 ALL DONE. Full library ready in ./assets/darks")