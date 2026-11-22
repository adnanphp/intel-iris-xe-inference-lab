#!/usr/bin/env python3
"""
Compare benchmark results across models.

Reads multiple result CSVs, deduplicates to the latest row per
(model, device, batch_size), and prints a comparison table with
GPU/CPU speedup ratios. Also writes a combined throughput plot.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    """Return list of dicts with typed fields, or [] if file missing."""
    if not os.path.isfile(path):
        print(f"Warning: missing {path}", file=sys.stderr)
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "model": r["model"],
                    "device": r["device"],
                    "batch": int(r["batch_size"]),
                    "mean_ms": float(r["mean_ms"]),
                    "throughput": float(r["throughput_img_s"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def dedupe_latest(rows):
    """Keep the latest entry per (model, device, batch)."""
    latest = {}
    for r in rows:
        latest[(r["model"], r["device"], r["batch"])] = r
    return list(latest.values())


def short_name(model_filename):
    """Turn 'face-detection-retail-0004.xml' -> 'face-detection'."""
    name = model_filename.rsplit(".", 1)[0]
    for suffix in ("-retail-0004", "-1.0-224-tf"):
        name = name.replace(suffix, "")
    return name


def build_table(rows):
    """Structure: {model: {batch: {device: row}}}"""
    table = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        table[short_name(r["model"])][r["batch"]][r["device"]] = r
    return table


def print_table(table):
    for model in sorted(table):
        print()
        print("=" * 88)
        print(f"MODEL: {model}")
        print("=" * 88)
        header = (
            f"{'Batch':>5} | "
            f"{'CPU ms':>9} {'CPU thr':>9} | "
            f"{'GPU ms':>9} {'GPU thr':>9} | "
            f"{'Speedup':>8} {'Winner':>7}"
        )
        print(header)
        print("-" * len(header))

        for batch in sorted(table[model]):
            cell = table[model][batch]
            cpu = cell.get("CPU")
            gpu = cell.get("GPU")
            if not cpu or not gpu:
                continue
            speedup = cpu["mean_ms"] / gpu["mean_ms"]
            winner = "GPU" if speedup > 1.0 else "CPU"
            print(
                f"{batch:>5} | "
                f"{cpu['mean_ms']:>9.3f} {cpu['throughput']:>9.1f} | "
                f"{gpu['mean_ms']:>9.3f} {gpu['throughput']:>9.1f} | "
                f"{speedup:>7.2f}x {winner:>7}"
            )


def plot(table, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(9, 6))

    styles = {
        ("CPU", "face-detection"): ("tab:blue", "o", "-"),
        ("GPU", "face-detection"): ("tab:blue", "s", "--"),
        ("CPU", "mobilenet-v3-large"): ("tab:orange", "o", "-"),
        ("GPU", "mobilenet-v3-large"): ("tab:orange", "s", "--"),
    }

    for model in sorted(table):
        for device in ("CPU", "GPU"):
            pts = sorted(
                (b, table[model][b][device]["throughput"])
                for b in table[model]
                if device in table[model][b]
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color, marker, linestyle = styles.get(
                (device, model), ("gray", "o", "-"))
            plt.plot(xs, ys, marker=marker, linestyle=linestyle,
                     color=color, label=f"{model} / {device}")

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (images/sec)")
    plt.title("Throughput vs batch size: CPU vs GPU across two models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(outdir, "model_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPlot written: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv", nargs="+",
        default=["results/sweep_python.csv", "results/mobilenet_results.csv"],
        help="One or more result CSVs to combine")
    p.add_argument("--outdir", default="results/plots")
    args = p.parse_args()

    rows = []
    for path in args.csv:
        rows.extend(load_csv(path))

    if not rows:
        sys.exit("No data loaded — check that the CSV paths exist.")

    rows = dedupe_latest(rows)
    table = build_table(rows)

    print_table(table)
    plot(table, args.outdir)


if __name__ == "__main__":
    main()
