#!/usr/bin/env python3
"""
sky_score_calibrator.py — Ryusei sky score threshold calibrator
================================================================
Reads all daylight JSON files produced by Ryusei's sky monitor,
reconstructs the raw_score / norm_score / star_score components,
lets you label each frame as CLEAR or CLOUDY from your ground truth,
then recommends MIN_NOISE_SCORE and MIN_STAR_SCORE values for the
AND gate fix in compute_sky_score().

Usage
-----
1. Interactive (label frames one by one):
       python sky_score_calibrator.py --daylight-dir output/daylight

2. Batch (supply a pre-made labels file and skip the prompts):
       python sky_score_calibrator.py --daylight-dir output/daylight \
                                      --labels labels.csv

3. Dry-run against existing CLEAR/CLOUDY decisions (no manual labels):
       python sky_score_calibrator.py --daylight-dir output/daylight \
                                      --use-stored-scores

Labels CSV format  (produced automatically by --save-labels, or hand-written)
--------------
filename,label
20240801_221530.json,CLEAR
20240801_223200.json,CLOUDY
...

Outputs
-------
- Console: recommended MIN_NOISE_SCORE and MIN_STAR_SCORE
- sky_calibration_scatter.png   : noise_score vs star_score scatter
- sky_calibration_histograms.png: per-component score distributions
- sky_calibration_roc.png       : precision/recall vs threshold sweep
- labels.csv                    : saved labels for future reuse
"""

import argparse
import json
import os
import sys
import csv
import math
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation if missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — plots will be skipped. Install with: pip install matplotlib")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("[warn] numpy not found — some stats will be approximate. Install with: pip install numpy")


# ---------------------------------------------------------------------------
# Score reconstruction — mirrors compute_sky_score() exactly
# ---------------------------------------------------------------------------
TEXTURE_LIMIT = 0.12   # hardcoded in Ryusei V201
W_STD  = 0.6
W_STAR = 0.4


def reconstruct_scores(rec: dict) -> Optional[dict]:
    """
    Given one parsed JSON record, return the internal score components.
    Returns None if the record is missing required fields.
    """
    required = ("stddev", "norm_stddev", "mean_brightness", "stars",
                "effective_std_limit", "effective_star_limit")
    for f in required:
        if f not in rec:
            return None

    raw_stddev      = float(rec["stddev"])
    norm_stddev     = float(rec["norm_stddev"])
    mean_brightness = float(rec["mean_brightness"])
    stars           = int(rec["stars"])
    std_limit       = float(rec["effective_std_limit"])
    star_limit      = max(1, int(rec["effective_star_limit"]))

    # Saturation guard
    if mean_brightness > 245.0:
        return {
            "raw_score": 0.0, "norm_score": None, "star_score": 0.0,
            "final_noise_score": 0.0, "total_score": 0.0,
            "saturation_guard": True,
        }

    star_score = stars / star_limit
    raw_score  = std_limit / max(raw_stddev, 0.1)

    if raw_score >= 1.0:
        final_noise_score = raw_score
        norm_score = None
        path = "quiet"
    elif mean_brightness < 5.0:
        final_noise_score = raw_score
        norm_score = None
        path = "pitch_black"
    else:
        norm_score = TEXTURE_LIMIT / max(norm_stddev, 0.001)
        final_noise_score = max(raw_score, norm_score)
        path = "texture_forgiveness" if norm_score > raw_score else "texture_failed"

    total_score = W_STD * final_noise_score + W_STAR * star_score

    return {
        "raw_score":         round(raw_score, 4),
        "norm_score":        round(norm_score, 4) if norm_score is not None else None,
        "star_score":        round(star_score, 4),
        "final_noise_score": round(final_noise_score, 4),
        "total_score":       round(total_score, 4),
        "score_path":        path,
        "saturation_guard":  False,
    }


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_daylight_jsons(daylight_dir: str) -> list[dict]:
    """Load all JSON files from the daylight output directory."""
    p = Path(daylight_dir)
    if not p.exists():
        sys.exit(f"[error] Directory not found: {daylight_dir}")

    records = []
    skipped = 0
    for jf in sorted(p.glob("*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
            # mean_brightness is stored as "mean" inside analyze_sky_conditions
            # result dict but gets merged — try both field names
            if "mean_brightness" not in data and "mean" in data:
                data["mean_brightness"] = data["mean"]
            scores = reconstruct_scores(data)
            if scores is None:
                skipped += 1
                continue
            records.append({
                "filename": jf.name,
                "path": str(jf),
                "stored_score": data.get("sky_score"),
                "moon_in_fov":  data.get("moon_in_fov", False),
                "moon_impact":  data.get("moon_impact", 0.0),
                "timestamp":    data.get("timestamp", ""),
                "raw_stddev":   data.get("stddev"),
                "norm_stddev":  data.get("norm_stddev"),
                "mean_brightness": data.get("mean_brightness"),
                "stars":        data.get("stars"),
                **scores,
            })
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            skipped += 1
            print(f"[warn] Skipping {jf.name}: {e}")

    print(f"[info] Loaded {len(records)} records, skipped {skipped}")
    return records


# ---------------------------------------------------------------------------
# Label loading / interactive labelling
# ---------------------------------------------------------------------------

def load_labels(labels_path: str) -> dict[str, str]:
    """Load filename→label mapping from a CSV."""
    labels = {}
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"].strip()] = row["label"].strip().upper()
    return labels


def save_labels(records: list[dict], out_path: str = "labels.csv"):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for r in records:
            writer.writerow([r["filename"], r.get("label", "")])
    print(f"[info] Labels saved to {out_path}")


def interactive_label(records: list[dict]):
    """
    Walk through records without a label and prompt the user.
    Shows key metrics so you can decide without opening the image.
    """
    print("\n=== Interactive labelling ===")
    print("For each frame enter: C = CLEAR, L = CLOUDY, S = skip, Q = quit\n")

    labelled = 0
    for r in records:
        if r.get("label"):
            continue

        print(f"  File      : {r['filename']}")
        print(f"  Timestamp : {r['timestamp']}")
        print(f"  Stars     : {r['stars']}   StdDev: {r['raw_stddev']:.2f}   "
              f"NormStd: {r['norm_stddev']:.4f}   Mean: {r['mean_brightness']:.1f}")
        print(f"  Scores    : noise={r['final_noise_score']:.3f}  "
              f"star={r['star_score']:.3f}  total={r['total_score']:.3f}  "
              f"path={r['score_path']}")
        print(f"  Stored decision: {'CLEAR' if (r['stored_score'] or 0) >= 0.8 else 'CLOUDY'} "
              f"(score={r['stored_score']})")
        if r["moon_in_fov"]:
            print(f"  Moon: IN FOV  impact={r['moon_impact']:.2f}")

        while True:
            ans = input("  Label [C/L/S/Q]: ").strip().upper()
            if ans == "C":
                r["label"] = "CLEAR"; labelled += 1; break
            elif ans == "L":
                r["label"] = "CLOUDY"; labelled += 1; break
            elif ans == "S":
                break
            elif ans == "Q":
                print(f"[info] Labelled {labelled} frames, stopping.")
                return
            else:
                print("  Please enter C, L, S, or Q")

        print()

    print(f"[info] Labelling complete. {labelled} frames labelled.")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(records: list[dict]) -> dict:
    """
    Separate labelled records into CLEAR and CLOUDY groups,
    compute per-component statistics, find the boundary values,
    and return a results dict.
    """
    clear  = [r for r in records if r.get("label") == "CLEAR"]
    cloudy = [r for r in records if r.get("label") == "CLOUDY"]

    if not clear or not cloudy:
        print(f"[warn] Need both CLEAR and CLOUDY labels. "
              f"Got {len(clear)} CLEAR, {len(cloudy)} CLOUDY.")
        return {}

    def stats(vals, name):
        if not vals:
            return {}
        vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(variance)
        return {
            "n": n, "min": vals[0], "max": vals[-1],
            "mean": round(mean, 4), "std": round(std, 4),
            "p10": vals[max(0, int(n * 0.10))],
            "p25": vals[max(0, int(n * 0.25))],
            "p50": vals[n // 2],
            "p75": vals[min(n - 1, int(n * 0.75))],
            "p90": vals[min(n - 1, int(n * 0.90))],
        }

    # Worst (highest) noise_score seen on a truly cloudy frame → upper bound for gate
    cloudy_noise_max = max(r["final_noise_score"] for r in cloudy)
    # Best (lowest) noise_score seen on a truly clear frame → lower bound for gate
    clear_noise_min  = min(r["final_noise_score"] for r in clear)

    cloudy_star_max  = max(r["star_score"] for r in cloudy)
    clear_star_min   = min(r["star_score"] for r in clear)

    # False positives: Ryusei said CLEAR but ground truth is CLOUDY
    false_positives = [r for r in cloudy if (r["stored_score"] or 0) >= 0.8]
    # False negatives: Ryusei said CLOUDY but ground truth is CLEAR
    false_negatives = [r for r in clear  if (r["stored_score"] or 0) <  0.8]

    return {
        "clear":  clear,
        "cloudy": cloudy,
        "clear_noise_stats":  stats([r["final_noise_score"] for r in clear],  "clear_noise"),
        "cloudy_noise_stats": stats([r["final_noise_score"] for r in cloudy], "cloudy_noise"),
        "clear_star_stats":   stats([r["star_score"] for r in clear],  "clear_star"),
        "cloudy_star_stats":  stats([r["star_score"] for r in cloudy], "cloudy_star"),
        "cloudy_noise_max":   round(cloudy_noise_max, 4),
        "clear_noise_min":    round(clear_noise_min,  4),
        "cloudy_star_max":    round(cloudy_star_max,  4),
        "clear_star_min":     round(clear_star_min,   4),
        "false_positives":    false_positives,
        "false_negatives":    false_negatives,
        # Texture-forgiveness false positives (the main bug path)
        "forgiveness_fp": [r for r in false_positives
                           if r.get("score_path") == "texture_forgiveness"],
    }


def recommend(results: dict) -> tuple[float, float]:
    """
    Derive MIN_NOISE_SCORE and MIN_STAR_SCORE from the analysis.

    Strategy:
      - MIN_NOISE_SCORE = cloudy_noise_max + small safety margin (5%)
        capped so it stays below clear_noise_min (avoid blocking real CLEAR)
      - MIN_STAR_SCORE  = cloudy_star_max  + small safety margin (5%)
        capped similarly
    If the ranges overlap, we note the ambiguity and recommend the midpoint.
    """
    MARGIN = 0.05

    cn_max = results["cloudy_noise_max"]
    cn_min = results["clear_noise_min"]
    cs_max = results["cloudy_star_max"]
    cs_min = results["clear_star_min"]

    # Noise gate
    mn_candidate = round(cn_max + MARGIN, 3)
    if mn_candidate >= cn_min:
        # Overlap — ranges not separable, use midpoint and warn
        mn = round((cn_max + cn_min) / 2, 3)
        noise_overlap = True
    else:
        mn = mn_candidate
        noise_overlap = False

    # Star gate
    ms_candidate = round(cs_max + MARGIN, 3)
    if ms_candidate >= cs_min:
        ms = round((cs_max + cs_min) / 2, 3)
        star_overlap = True
    else:
        ms = ms_candidate
        star_overlap = False

    return mn, ms, noise_overlap, star_overlap


# ---------------------------------------------------------------------------
# Threshold sweep — find optimal cutoffs by F1
# ---------------------------------------------------------------------------

def threshold_sweep(records: list[dict], component: str) -> list[dict]:
    """
    Sweep candidate threshold values for a component (final_noise_score or star_score)
    and compute precision, recall, F1 at each step.
    A frame is classified CLOUDY when the component is below threshold.
    """
    labelled = [r for r in records if r.get("label") in ("CLEAR", "CLOUDY")]
    vals = sorted(set(round(r[component], 3) for r in labelled))
    results = []

    for t in vals:
        tp = sum(1 for r in labelled if r[component] <  t and r["label"] == "CLOUDY")
        fp = sum(1 for r in labelled if r[component] <  t and r["label"] == "CLEAR")
        fn = sum(1 for r in labelled if r[component] >= t and r["label"] == "CLOUDY")
        tn = sum(1 for r in labelled if r[component] >= t and r["label"] == "CLEAR")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter(results: dict, out_path: str = "sky_calibration_scatter.png"):
    if not HAS_MPL:
        return
    clear  = results["clear"]
    cloudy = results["cloudy"]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.scatter([r["final_noise_score"] for r in cloudy],
               [r["star_score"]        for r in cloudy],
               c="#e94560", alpha=0.7, s=40, label="CLOUDY (ground truth)", zorder=3)
    ax.scatter([r["final_noise_score"] for r in clear],
               [r["star_score"]        for r in clear],
               c="#0f3460", alpha=0.7, s=40, label="CLEAR (ground truth)",
               edgecolors="#53d8fb", linewidths=0.5, zorder=3)

    # Mark false positives (Ryusei said CLEAR, truth is CLOUDY)
    fps = results["false_positives"]
    if fps:
        ax.scatter([r["final_noise_score"] for r in fps],
                   [r["star_score"]        for r in fps],
                   c="none", edgecolors="yellow", linewidths=1.5, s=80,
                   label=f"False positive ({len(fps)})", zorder=4)

    ax.axvline(x=0.8, color="#aaaaaa", linestyle="--", linewidth=0.8, label="0.8 CLEAR gate (current)")
    ax.set_xlabel("final_noise_score", color="white")
    ax.set_ylabel("star_score", color="white")
    ax.set_title("Noise score vs star score — CLEAR/CLOUDY ground truth", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Scatter saved to {out_path}")


def plot_histograms(results: dict, out_path: str = "sky_calibration_histograms.png"):
    if not HAS_MPL:
        return
    clear  = results["clear"]
    cloudy = results["cloudy"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#1a1a2e")
    fig.suptitle("Score component distributions — CLEAR vs CLOUDY", color="white")

    components = [
        ("final_noise_score", "Noise score (final)"),
        ("star_score",        "Star score"),
        ("total_score",       "Total score"),
    ]

    for ax, (field, title) in zip(axes, components):
        ax.set_facecolor("#16213e")
        c_vals = [r[field] for r in clear]
        l_vals = [r[field] for r in cloudy]
        bins = 30

        if HAS_NP:
            all_vals = c_vals + l_vals
            b = np.linspace(min(all_vals), max(all_vals), bins) if all_vals else bins
        else:
            b = bins

        ax.hist(l_vals, bins=b, color="#e94560", alpha=0.6, label="CLOUDY")
        ax.hist(c_vals, bins=b, color="#53d8fb", alpha=0.5, label="CLEAR")
        ax.axvline(x=0.8, color="yellow", linestyle="--", linewidth=0.8, label="0.8 gate")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Histograms saved to {out_path}")


def plot_roc(records: list[dict], mn: float, ms: float,
             out_path: str = "sky_calibration_roc.png"):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    fig.suptitle("Threshold sweep — precision, recall, F1 (CLOUDY detection)", color="white")

    for ax, (component, label, recommended) in zip(axes, [
        ("final_noise_score", "noise threshold",   mn),
        ("star_score",        "star threshold",    ms),
    ]):
        ax.set_facecolor("#16213e")
        sweep = threshold_sweep(records, component)
        ts    = [s["threshold"]  for s in sweep]
        precs = [s["precision"]  for s in sweep]
        recs  = [s["recall"]     for s in sweep]
        f1s   = [s["f1"]         for s in sweep]

        ax.plot(ts, precs, color="#53d8fb", linewidth=1.5, label="precision")
        ax.plot(ts, recs,  color="#e94560", linewidth=1.5, label="recall")
        ax.plot(ts, f1s,   color="#f5a623", linewidth=2,   label="F1")
        ax.axvline(x=recommended, color="white", linestyle="--",
                   linewidth=1, label=f"recommended={recommended}")
        ax.set_xlabel(label, color="white")
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=9)
        ax.set_title(label, color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] ROC curves saved to {out_path}")


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

SEPARATOR = "─" * 60


def print_report(results: dict, mn: float, ms: float,
                 noise_overlap: bool, star_overlap: bool):
    clear  = results["clear"]
    cloudy = results["cloudy"]
    fps    = results["false_positives"]
    fns    = results["false_negatives"]
    tex_fp = results["forgiveness_fp"]

    print(f"\n{'═' * 60}")
    print("  SKY SCORE CALIBRATION REPORT")
    print(f"{'═' * 60}")
    print(f"  Ground truth frames : {len(clear) + len(cloudy)}")
    print(f"  CLEAR               : {len(clear)}")
    print(f"  CLOUDY              : {len(cloudy)}")
    print(f"{SEPARATOR}")
    print("  CURRENT SYSTEM ACCURACY")
    print(f"  False positives     : {len(fps)}  (reported CLEAR, truth=CLOUDY)")
    print(f"  False negatives     : {len(fns)}  (reported CLOUDY, truth=CLEAR)")
    print(f"  Texture-forgiveness FP: {len(tex_fp)}  (main bug path)")
    print(f"{SEPARATOR}")
    print("  NOISE SCORE COMPONENT  (final_noise_score)")
    ns = results["clear_noise_stats"]
    cs = results["cloudy_noise_stats"]
    print(f"  CLEAR  : min={ns['min']:.3f}  p25={ns['p25']:.3f}  "
          f"median={ns['p50']:.3f}  p75={ns['p75']:.3f}  max={ns['max']:.3f}")
    print(f"  CLOUDY : min={cs['min']:.3f}  p25={cs['p25']:.3f}  "
          f"median={cs['p50']:.3f}  p75={cs['p75']:.3f}  max={cs['max']:.3f}")
    print(f"  Highest noise_score on a CLOUDY frame : {results['cloudy_noise_max']:.4f}")
    print(f"  Lowest  noise_score on a CLEAR  frame : {results['clear_noise_min']:.4f}")
    if noise_overlap:
        print(f"  [!] Ranges OVERLAP — the noise signal alone cannot fully separate "
              f"these classes.")
        print(f"      Consider lowering stddev_threshold in config first.")

    print(f"{SEPARATOR}")
    print("  STAR SCORE COMPONENT  (star_score)")
    ss_c = results["clear_star_stats"]
    ss_l = results["cloudy_star_stats"]
    print(f"  CLEAR  : min={ss_c['min']:.3f}  p25={ss_c['p25']:.3f}  "
          f"median={ss_c['p50']:.3f}  p75={ss_c['p75']:.3f}  max={ss_c['max']:.3f}")
    print(f"  CLOUDY : min={ss_l['min']:.3f}  p25={ss_l['p25']:.3f}  "
          f"median={ss_l['p50']:.3f}  p75={ss_l['p75']:.3f}  max={ss_l['max']:.3f}")
    print(f"  Highest star_score on a CLOUDY frame  : {results['cloudy_star_max']:.4f}")
    print(f"  Lowest  star_score on a CLEAR  frame  : {results['clear_star_min']:.4f}")
    if star_overlap:
        print(f"  [!] Ranges OVERLAP — star count alone ambiguous on these frames.")

    print(f"{SEPARATOR}")
    print("  RECOMMENDATION")
    print(f"\n  Add to compute_sky_score() in ryusei_V201_3.py:\n")
    print(f"    MIN_NOISE_SCORE = {mn}   # calibrated from your data")
    print(f"    MIN_STAR_SCORE  = {ms}   # calibrated from your data\n")
    print(f"    if final_noise_score < MIN_NOISE_SCORE or star_score < MIN_STAR_SCORE:")
    print(f"        return min(final_noise_score, star_score)\n")
    print(f"    score = (W_STD * final_noise_score) + (W_STAR * star_score)")
    print(f"    return score")
    print(f"\n  These values block any sky with noise_score < {mn} OR star_score < {ms}")
    print(f"  from reaching the CLEAR threshold regardless of the other component.")
    print(f"\n  If ranges overlapped, also lower stddev_threshold in config")
    print(f"  (currently recommended: try 6.0–7.0) to widen the separation.")
    print(f"{'═' * 60}\n")

    if fps:
        print(f"  FALSE POSITIVE DETAILS  (Ryusei=CLEAR, truth=CLOUDY)")
        for r in fps[:10]:
            print(f"    {r['filename']}  noise={r['final_noise_score']:.3f}  "
                  f"star={r['star_score']:.3f}  path={r.get('score_path', '?')}")
        if len(fps) > 10:
            print(f"    ... and {len(fps) - 10} more")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--daylight-dir", default="output/daylight",
                   help="Path to Ryusei's output/daylight directory (default: output/daylight)")
    p.add_argument("--labels",
                   help="Path to existing labels CSV; skips interactive labelling")
    p.add_argument("--save-labels", default="labels.csv",
                   help="Where to save labels after interactive session (default: labels.csv)")
    p.add_argument("--use-stored-scores", action="store_true",
                   help="Use Ryusei's stored sky_score as label (CLEAR if >=0.8, "
                        "CLOUDY otherwise). No manual labelling needed, but accuracy "
                        "depends on the existing (possibly buggy) decisions.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating PNG plots")
    p.add_argument("--plot-dir", default=".",
                   help="Directory to write PNG plots (default: current dir)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load JSON records
    records = load_daylight_jsons(args.daylight_dir)
    if not records:
        sys.exit("[error] No usable JSON files found.")

    # 2. Apply labels
    if args.labels:
        labels = load_labels(args.labels)
        for r in records:
            if r["filename"] in labels:
                r["label"] = labels[r["filename"]]
        print(f"[info] Applied {len(labels)} labels from {args.labels}")

    elif args.use_stored_scores:
        for r in records:
            stored = r.get("stored_score")
            if stored is not None:
                r["label"] = "CLEAR" if float(stored) >= 0.8 else "CLOUDY"
        labelled = sum(1 for r in records if "label" in r)
        print(f"[info] Auto-labelled {labelled} records from stored sky_score "
              f"(treating Ryusei's existing decisions as ground truth).")
        print(f"[warn] This mode will undercount false positives since the "
              f"labels come from the same (buggy) logic. Manual labelling "
              f"gives more accurate calibration.")

    else:
        interactive_label(records)
        save_labels(records, args.save_labels)

    # 3. Analyse
    results = analyse(records)
    if not results:
        sys.exit("[error] Not enough labelled data to proceed.")

    # 4. Recommend
    mn, ms, noise_overlap, star_overlap = recommend(results)

    # 5. Print report
    print_report(results, mn, ms, noise_overlap, star_overlap)

    # 6. Plots
    if not args.no_plots and HAS_MPL:
        pd = args.plot_dir
        os.makedirs(pd, exist_ok=True)
        plot_scatter(results,    os.path.join(pd, "sky_calibration_scatter.png"))
        plot_histograms(results, os.path.join(pd, "sky_calibration_histograms.png"))
        plot_roc(records, mn, ms, os.path.join(pd, "sky_calibration_roc.png"))
    elif not HAS_MPL:
        print("[info] Install matplotlib to get plots: pip install matplotlib")


if __name__ == "__main__":
    main()
