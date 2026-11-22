#!/usr/bin/env python3
"""Plot throughput vs workers for sync / async / multiprocess."""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "device": r["device"],
                "mode": r["mode"],
                "batch": int(r["batch_size"]),
                "workers": int(r["workers"]),
                "thr": float(r["throughput_img_s"]),
                "mean_ms": float(r["mean_ms"]),
            })
    return rows


def latest_per(rows, key):
    d = {}
    for r in rows:
        d[key(r)] = r
    return list(d.values())


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/parallel_results.csv"
    outdir = "results/plots"
    os.makedirs(outdir, exist_ok=True)

    rows = load(csv_path)
    # latest per (device, mode, batch, workers)
    rows = latest_per(rows, lambda r: (r["device"], r["mode"], r["batch"], r["workers"]))

    # Group by device, then by mode
    data = defaultdict(list)   # (device, mode) -> [(workers, thr, mean_ms)]
    for r in rows:
        data[(r["device"], r["mode"])].append((r["workers"], r["thr"], r["mean_ms"]))

    devices = sorted({r["device"] for r in rows})

    # -------- Plot 1: throughput vs workers --------
    plt.figure()
    for dev in devices:
        for mode in ("sync", "async", "multiprocess"):
            key = (dev, mode)
            if key not in data:
                continue
            pts = sorted(data[key])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label = f"{dev} {mode}"
            if mode == "sync":
                # single point, plot as a marker with an hline
                plt.scatter(xs, ys, s=80, marker="*", label=label)
                plt.axhline(ys[0], linestyle=":", alpha=0.4)
            else:
                plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Workers (concurrency)")
    plt.ylabel("Aggregate throughput (images/sec)")
    plt.title("Throughput vs concurrency: sync / async / multiprocess")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/parallel_throughput.png", dpi=120, bbox_inches="tight")
    plt.close()

    # -------- Plot 2: latency vs workers --------
    plt.figure()
    for dev in devices:
        for mode in ("sync", "async", "multiprocess"):
            key = (dev, mode)
            if key not in data:
                continue
            pts = sorted(data[key])
            xs = [p[0] for p in pts]
            ys = [p[2] for p in pts]
            label = f"{dev} {mode}"
            if mode == "sync":
                plt.scatter(xs, ys, s=80, marker="*", label=label)
            else:
                plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Workers (concurrency)")
    plt.ylabel("Amortized latency (ms)")
    plt.title("Latency vs concurrency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/parallel_latency.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Plots written to {outdir}/")
    print(f"  {outdir}/parallel_throughput.png")
    print(f"  {outdir}/parallel_latency.png")


if __name__ == "__main__":
    main()
