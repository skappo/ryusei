#!/usr/bin/env python3
import subprocess
import sys
import os

scripts = ["meteora_report.py", "log_to_excel.py"]

path = "/media/pi/USB/meteora_new/scripts/"


processes = [
    subprocess.Popen([sys.executable, os.path.join(path,script)])
    for script in scripts
]

for process in processes:
    process.wait()