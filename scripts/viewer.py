#!/usr/bin/env python3
import zmq
import json
import cv2
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
PI_IP = "192.168.1.111"  # <--- Change this to your Pi's IP address
PORT = "5555"

def run_viewer():
    # ZMQ Setup
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{PI_IP}:{PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "debug")
    
    print(f"[*] Ryusei Viewer started. Connecting to {PI_IP}:{PORT}...")
    print("[*] Press 'q' on the image window to quit.")

    # State variables for the UI
    last_stats = {}
    last_status = "Waiting for data..."
    status_color = (0, 255, 0)

    cv2.namedWindow("Ryusei Remote Debug", cv2.WINDOW_NORMAL)

    while True:
        try:
            # Receive multipart: [Topic, JSON_Header, Optional_Image]
            parts = socket.recv_multipart()
            topic = parts[0].decode()
            header = json.loads(parts[1].decode())
            
            msg_type = header.get("type", "unknown")
            ts = datetime.now().strftime("%H:%M:%S")

            # --- Handle Stats-only messages ---
            if msg_type == "detection_stats":
                last_stats = header
                if header.get("event_active"):
                    last_status = "!!! METEOR EVENT IN PROGRESS !!!"
                    status_color = (0, 0, 255) # Red
                else:
                    last_status = "Monitoring Sky..."
                    status_color = (0, 255, 0) # Green

            elif msg_type == "stack_status":
                last_status = f"STACK: {header.get('mode')} - {header.get('action')}"
                if header.get("mode") == "BIAS_MODE":
                    status_color = (0, 165, 255) # Orange
                else:
                    status_color = (255, 255, 0) # Cyan

            # --- Console fallback for stats-only messages (no image attached) ---
            # At debug level 1 ("Stats"), the pipeline sends detection_stats /
            # stack_status headers with no image (images only ride along at
            # level >= 3). Without this, those messages update last_stats /
            # last_status above but are never drawn anywhere, since the image
            # overlay code below only runs when len(parts) > 2 -- so level 1
            # looked completely silent even though data was arriving.
            if len(parts) <= 2 and msg_type in ("detection_stats", "stack_status"):
                if msg_type == "detection_stats":
                    print(f"[{ts}] {last_status} | thr={header.get('threshold')} "
                          f"noise={header.get('noise_sigma')} "
                          f"cnts={header.get('contours_accepted')}/{header.get('contours_total')}")
                else:
                    print(f"[{ts}] {last_status}")

            if header.get("type") == "state_telemetry":
                vars_dict = header.get("variables", {})
                
                print("\n" + "="*60)
                print(f" RYUSEI INTERNAL STATE @ {header['ts_local']}")
                print("-"*60)
                
                # Print alphabetically sorted keys
                for k in sorted(vars_dict.keys()):
                    val = vars_dict[k]
                    # Color coding for terminal (Optional)
                    if val is True: val = f"\033[92m{val}\033[0m"   # Green
                    if val is False: val = f"\033[91m{val}\033[0m"  # Red
                    
                    print(f"{k:35} : {val}")
                print("="*60 + "\n")

            # --- Handle Messages with Images (Mosaic or Detection View) ---
            if len(parts) > 2:
                img_bytes = parts[2]
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Create a Header/Overlay area
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    
                    # Draw a dark semi-transparent bar at the top for stats
                    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                    # 1. System Status
                    cv2.putText(frame, f"[{ts}] {last_status}", (10, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    # 2. Detection Telemetry
                    if msg_type == "detection_stats":
                        stats_str = (f"Thr: {last_stats.get('threshold')} | "
                                     f"Noise: {last_stats.get('noise_sigma')} | "
                                     f"Cnts: {last_stats.get('contours_accepted')}/{last_stats.get('contours_total')}")
                        cv2.putText(frame, stats_str, (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 3. Stack Info (if it's a mosaic)
                    if msg_type == "alignment_mosaic":
                        cc = header.get("correlation", 0)
                        idx = header.get("frame_index", "?")
                        cv2.putText(frame, f"ALIGNING FRAME {idx} | CC: {cc:.3f}", (w-300, 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow("Ryusei Remote Debug", frame)

            # Keep window responsive
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[!] Error: {e}")
            continue

    print("[*] Closing Viewer.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_viewer()