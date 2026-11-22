#!/usr/bin/env python3
"""Reusable OpenVINO benchmark engine for Intel Iris Xe lab."""

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import openvino as ov


def _percentile(sorted_data, pct):
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = k - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def run_single(model_path, device, batch_size, iterations, warmup,
               input_shape=(3, 300, 300), core=None, verbose=True):
    """Run one benchmark and return a dict of metrics (no CSV write)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    full_shape = [batch_size] + list(input_shape)

    if core is None:
        core = ov.Core()

    available = core.available_devices
    if device not in available:
        raise RuntimeError(
            f"Device '{device}' not available. Available: {available}")

    model = core.read_model(model_path)
    model.reshape({0: full_shape})
    compiled = core.compile_model(model, device)

    input_data = np.random.rand(*full_shape).astype(np.float32)

    for _ in range(warmup):
        compiled([input_data])

    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        compiled([input_data])
        times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    s = sorted(times_ms)

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.path.basename(model_path),
        "device": device,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup": warmup,
        "mean_ms": f"{mean_ms:.4f}",
        "median_ms": f"{median_ms:.4f}",
        "std_ms": f"{std_ms:.4f}",
        "min_ms": f"{s[0]:.4f}",
        "p95_ms": f"{_percentile(s, 95):.4f}",
        "p99_ms": f"{_percentile(s, 99):.4f}",
        "max_ms": f"{s[-1]:.4f}",
        "throughput_img_s": f"{batch_size / (mean_ms / 1000.0):.2f}",
    }

    if verbose:
        print(f"{device:5} | batch={batch_size:<3} | "
              f"mean={mean_ms:8.3f} ms | "
              f"p95={_percentile(s, 95):8.3f} ms | "
              f"thr={batch_size / (mean_ms / 1000.0):8.2f} img/s")

    return result


CSV_FIELDS = [
    "timestamp", "model", "device", "batch_size", "iterations", "warmup",
    "mean_ms", "median_ms", "std_ms", "min_ms", "p95_ms", "p99_ms",
    "max_ms", "throughput_img_s",
]


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})


def parse_args():
    p = argparse.ArgumentParser(description="OpenVINO inference benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True, choices=["CPU", "GPU"])
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--input-shape", default="3,300,300")
    p.add_argument("--csv", default="results/benchmark_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        spatial = tuple(int(x) for x in args.input_shape.split(","))
    except ValueError:
        sys.exit(f"Invalid --input-shape: {args.input_shape}")

    try:
        row = run_single(
            args.model, args.device, args.batch_size,
            args.iterations, args.warmup, spatial)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"Error: {e}")

    write_csv([row], args.csv)
    print(f"Results appended to: {args.csv}")


if __name__ == "__main__":
    main()
