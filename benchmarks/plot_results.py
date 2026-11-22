#!/usr/bin/env python3
"""Plot benchmark results from CSV."""

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
                "batch": int(r["batch_size"]),
                "mean_ms": float(r["mean_ms"]),
                "throughput": float(r["throughput_img_s"]),
            })
    return rows


def series(rows, device, key):
    pts = sorted([(r["batch"], r[key]) for r in rows if r["device"] == device])
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.csv"
    outdir = "results/plots"
    os.makedirs(outdir, exist_ok=True)

    rows = load(csv_path)
    # Deduplicate: keep latest row per (device, batch)
    latest = {}
    for r in rows:
        latest[(r["device"], r["batch"])] = r
    rows = list(latest.values())

    # Latency
    plt.figure()
    for dev in ["CPU", "GPU"]:
        x, y = series(rows, dev, "mean_ms")
        if x:
            plt.plot(x, y, marker="o", label=dev)
    plt.xlabel("Batch size")
    plt.ylabel("Mean latency (ms)")
    plt.title("Inference latency vs batch size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/latency.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Throughput
    plt.figure()
    for dev in ["CPU", "GPU"]:
        x, y = series(rows, dev, "throughput")
        if x:
            plt.plot(x, y, marker="o", label=dev)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (images/sec)")
    plt.title("Inference throughput vs batch size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/throughput.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Speedup
    cpu = {r["batch"]: r["mean_ms"] for r in rows if r["device"] == "CPU"}
    gpu = {r["batch"]: r["mean_ms"] for r in rows if r["device"] == "GPU"}
    batches = sorted(set(cpu) & set(gpu))
    speedups = [cpu[b] / gpu[b] for b in batches]

    plt.figure()
    plt.plot(batches, speedups, marker="o", color="purple")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Batch size")
    plt.ylabel("GPU speedup (CPU ms / GPU ms)")
    plt.title("GPU speedup vs batch size")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/speedup.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Plots written to {outdir}/")
    for f in ("latency.png", "throughput.png", "speedup.png"):
        print(f"  {outdir}/{f}")


if __name__ == "__main__":
    main()
