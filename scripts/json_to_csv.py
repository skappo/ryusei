#!/usr/bin/env python3
"""
Extract selected fields from a directory of Ryusei timelapse JSON files
into a single CSV file.

Usage:
    python json_to_csv.py /path/to/json_dir [-o output.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDS = [
    "image_filename",
    "exposure_us",
    "gain",
    "lux",
    "lux_source",
    "sensor_temp_cam_c",
    "sensor_temp_ext_c",
    "mean_brightness",
    "bias_reasons",
]

NA = "n/a"


def extract_row(data: dict, source_file: str) -> dict:
    row = {"source_file": source_file}
    for field in FIELDS:
        if field not in data or data[field] is None:
            row[field] = NA
            continue

        value = data[field]

        # bias_reasons is typically a list -> join into a single string
        if field == "bias_reasons":
            if isinstance(value, list):
                row[field] = "; ".join(str(v) for v in value) if value else NA
            else:
                row[field] = str(value)
        else:
            row[field] = value

    return row


def main():
    parser = argparse.ArgumentParser(description="Extract fields from Ryusei JSON files into a CSV.")
    parser.add_argument("input_dir", help="Directory containing the .json files")
    parser.add_argument("-o", "--output", default="timelapse_summary.csv", help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    rows = []
    errors = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(extract_row(data, jf.name))
        except (json.JSONDecodeError, OSError) as e:
            errors.append((jf.name, str(e)))

    fieldnames = ["source_file"] + FIELDS

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to '{args.output}'")
    if errors:
        print(f"\n{len(errors)} file(s) failed to parse and were skipped:", file=sys.stderr)
        for name, err in errors:
            print(f"  - {name}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
