#!/usr/bin/env python3
"""Run the full benchmark matrix from a YAML config."""

import argparse
import os
import sys

import yaml
import openvino as ov

# Make sibling import work regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import run_single, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/benchmark_config.yaml")
    p.add_argument("--csv", default=None,
                   help="Override output_csv from config")
    args = p.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = cfg["model"]
    shape = tuple(cfg.get("input_shape", [3, 300, 300]))
    devices = cfg["devices"]
    batches = cfg["batch_sizes"]
    iterations = cfg.get("iterations", 50)
    warmup = cfg.get("warmup", 10)
    csv_path = args.csv or cfg.get("output_csv",
                                   "results/benchmark_results.csv")

    if not os.path.isfile(model):
        sys.exit(f"Model not found: {model}")

    print(f"Model:      {model}")
    print(f"Input:      {shape}")
    print(f"Devices:    {devices}")
    print(f"Batches:    {batches}")
    print(f"Iterations: {iterations}  Warmup: {warmup}")
    print(f"Output CSV: {csv_path}")
    print("-" * 70)

    core = ov.Core()
    print(f"Available:  {core.available_devices}")
    print("-" * 70)

    rows = []
    for device in devices:
        for bs in batches:
            try:
                row = run_single(model, device, bs, iterations, warmup,
                                 shape, core=core, verbose=True)
                rows.append(row)
            except Exception as e:
                print(f"{device} batch={bs} FAILED: {e}", file=sys.stderr)

    write_csv(rows, csv_path)
    print("-" * 70)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
