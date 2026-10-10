#!/usr/bin/env python3
"""
meteora_profiler.py

A diagnostic wrapper for py-spy to safely monitor the multi-threaded
Ryusei pipeline in real-time.
"""

import os
import sys
import psutil
import shutil
import argparse
import subprocess

def find_ryusei_pid():
    """Scans system processes to find the running Ryusei Python script."""
    for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
        try:
            cmdline = proc.info.get('cmdline')
            if not cmdline:
                continue
            
            # Check if it's a python process
            if 'python' in proc.info['name'].lower() or 'python' in cmdline[0].lower():
                # Look for our script name in the arguments
                for arg in cmdline:
                    if 'meteora' in arg.lower() and arg.endswith('.py'):
                        # Exclude this profiler script itself
                        if 'meteora_profiler.py' not in arg:
                            return proc.info['pid'], arg
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Ryusei Performance Profiler (py-spy wrapper)")
    parser.add_argument('mode', choices=['top', 'record', 'dump'], 
                        help="Mode: 'top' (Live CPU view), 'record' (SVG Flamegraph), 'dump' (Current thread states).")
    parser.add_argument('--duration', type=int, default=60, 
                        help="Duration in seconds for 'record' mode (default: 60).")
    parser.add_argument('--output', type=str, default="meteora_flamegraph.svg",
                        help="Output filename for the SVG flamegraph (default: meteora_flamegraph.svg).")
    parser.add_argument('--rate', type=int, default=20,
                        help="Samples per second (py-spy --rate, applies to 'top'/'record' only; "
                             "default: 20). py-spy's own default is 100Hz, which on a Pi is enough "
                             "CPU/scheduling load to disturb timing-sensitive kernel drivers -- "
                             "confirmed live: it corrupted every DS18B20 1-wire read (w1-gpio bit-"
                             "banging) for the whole profiling window. Raise only if you need finer "
                             "resolution for a specific investigation and can tolerate that noise.")

    args = parser.parse_args()

    # 1. Privilege Check
    if os.geteuid() != 0:
        print("\n[ERROR] py-spy requires root privileges to read the memory of another process.")
        print("Please run this script with sudo:")
        print(f"sudo python3 {sys.argv[0]} {args.mode}\n")
        sys.exit(1)

    # 2. Py-Spy Installation Check
    # py_spy_path = shutil.which("py-spy")
    # if not py_spy_path:
        # print("\n[ERROR] 'py-spy' is not installed or not in PATH.")
        # print("Install it by running: sudo pip3 install py-spy --break-system-packages\n")
        # sys.exit(1)
        
    py_spy_path="/home/pi/env/bin/py-spy"

    # 3. Find Ryusei PID
    pid, script_name = find_ryusei_pid()
    if not pid:
        print("\n[ERROR] Could not find an active Ryusei process running.")
        print("Make sure the pipeline is currently running before starting the profiler.\n")
        sys.exit(1)

    print("="*60)
    print(f"Ryusei Profiler - Target Acquired: {script_name} (PID: {pid})")
    print("="*60)

    # 4. Build and Execute py-spy command
    cmd = []
    
    if args.mode == 'top':
        print(f"Launching Live Thread Monitor (rate: {args.rate}Hz). Press CTRL+C to exit.\n")
        # --subprocesses includes worker threads
        # OLD: no --rate, so py-spy used its own default (100Hz) -- enough
        # CPU/scheduling load on a Pi to disturb timing-sensitive kernel
        # drivers (confirmed: corrupted every DS18B20 1-wire read for the
        # whole profiling window). See --rate's help text.
        # cmd = [py_spy_path, 'top', '--pid', str(pid), '--subprocesses']
        cmd = [py_spy_path, 'top', '--pid', str(pid), '--subprocesses', '--rate', str(args.rate)]

    elif args.mode == 'record':
        print(f"Recording CPU execution for {args.duration} seconds (rate: {args.rate}Hz)...")
        print(f"Please wait. The result will be saved to: {args.output}\n")
        # Format: Flamegraph
        # cmd = [py_spy_path, 'record', '-o', args.output, '--duration', str(args.duration), '--pid', str(pid), '--subprocesses']
        cmd = [py_spy_path, 'record', '-o', args.output, '--duration', str(args.duration), '--pid', str(pid), '--subprocesses', '--rate', str(args.rate)]
        
    elif args.mode == 'dump':
        print("Dumping current thread call stacks (Useful for finding deadlocks)...\n")
        cmd = [py_spy_path, 'dump', '--pid', str(pid)]

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        if args.mode == 'record':
            print(f"\n[SUCCESS] Flamegraph saved to {os.path.abspath(args.output)}")
            print("Download this file to your PC and open it in any Web Browser (Chrome/Edge) to view it interactively!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] py-spy exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Profiler stopped by user.")

if __name__ == "__main__":
    main()
