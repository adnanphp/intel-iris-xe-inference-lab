#!/usr/bin/env python3
"""
Parallel inference benchmark for OpenVINO.

Modes:
  sync         - one synchronous request at a time (baseline)
  async        - AsyncInferQueue with N parallel requests
  multiprocess - N worker processes, each with its own compiled model

Writes results to parallel_results.csv.
"""

import argparse
import csv
import multiprocessing as mp
import os
import statistics
import sys
import time

import numpy as np
import openvino as ov

CSV_FIELDS = [
    "timestamp", "model", "device", "mode", "batch_size",
    "workers", "iterations",
    "mean_ms", "median_ms", "std_ms", "p95_ms", "p99_ms",
    "throughput_img_s",
    "per_worker_ms",   # only for multiprocess; blank otherwise
]


def _percentile(sorted_data, pct):
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = k - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def _stats(times_ms, batch_size):
    if not times_ms:
        raise RuntimeError("No timing samples collected")
    s = sorted(times_ms)
    mean = statistics.mean(s)
    return {
        "mean_ms": f"{mean:.4f}",
        "median_ms": f"{statistics.median(s):.4f}",
        "std_ms": f"{(statistics.stdev(s) if len(s) > 1 else 0.0):.4f}",
        "p95_ms": f"{_percentile(s, 95):.4f}",
        "p99_ms": f"{_percentile(s, 99):.4f}",
        "throughput_img_s": f"{batch_size / (mean / 1000.0):.2f}",
    }


def _load(model_path, device, batch_size, input_shape):
    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(
            f"Device '{device}' not available. Got: {core.available_devices}")
    model = core.read_model(model_path)
    full_shape = [batch_size] + list(input_shape)
    model.reshape({0: full_shape})
    compiled = core.compile_model(model, device)
    data = np.random.rand(*full_shape).astype(np.float32)
    return core, compiled, data


# ---------------- sync ----------------

def run_sync(model_path, device, batch_size, iterations, warmup, input_shape):
    _, compiled, data = _load(model_path, device, batch_size, input_shape)
    for _ in range(warmup):
        compiled([data])
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        compiled([data])
        times.append((time.perf_counter() - t0) * 1000.0)
    return times, None, None


# ---------------- async ----------------

def run_async(model_path, device, batch_size, iterations, warmup,
              input_shape, queue_depth):
    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(
            f"Device '{device}' not available. Got: {core.available_devices}")

    model = core.read_model(model_path)
    full_shape = [batch_size] + list(input_shape)
    model.reshape({0: full_shape})
    compiled = core.compile_model(model, device)

    requests = [compiled.create_infer_request() for _ in range(queue_depth)]
    datas = [np.random.rand(*full_shape).astype(np.float32)
             for _ in range(queue_depth)]

    for req, d in zip(requests, datas):
        req.infer({0: d})

    times = []
    t_start = time.perf_counter()
    in_flight = []
    idx = 0
    for _ in range(iterations):
        req = requests[idx % queue_depth]
        data = datas[idx % queue_depth]
        submit_time = time.perf_counter()
        req.start_async({0: data})
        in_flight.append((req, submit_time))

        if len(in_flight) >= queue_depth:
            oldest_req, oldest_submit = in_flight.pop(0)
            oldest_req.wait()
            times.append((time.perf_counter() - oldest_submit) * 1000.0)

        idx += 1

    for req, submit_time in in_flight:
        req.wait()
        times.append((time.perf_counter() - submit_time) * 1000.0)

    wall_ms = (time.perf_counter() - t_start) * 1000.0
    wall_throughput = (iterations * batch_size) / (wall_ms / 1000.0)
    return times, wall_throughput, None


# ---------------- multiprocess ----------------

def _worker(model_path, device, batch_size, iterations, warmup, input_shape,
            result_queue, start_barrier):
    try:
        _, compiled, data = _load(model_path, device, batch_size, input_shape)
        for _ in range(warmup):
            compiled([data])
        # Wait for all workers to reach this point, so wall clock
        # measures concurrent execution, not staggered startup.
        start_barrier.wait()
        t0 = time.perf_counter()
        for _ in range(iterations):
            compiled([data])
        elapsed = time.perf_counter() - t0
        result_queue.put({
            "ok": True,
            "elapsed_s": elapsed,
            "images": iterations * batch_size,
        })
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


def run_multiprocess(model_path, device, batch_size, iterations, warmup,
                     input_shape, workers):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    barrier = ctx.Barrier(workers)
    procs = []
    for _ in range(workers):
        p = ctx.Process(
            target=_worker,
            args=(model_path, device, batch_size, iterations, warmup,
                  input_shape, q, barrier),
        )
        p.start()
        procs.append(p)

    results = [q.get() for _ in range(workers)]
    for p in procs:
        p.join(timeout=180)
        if p.is_alive():
            p.terminate()

    errors = [r for r in results if not r["ok"]]
    if errors:
        raise RuntimeError(f"Worker errors: {errors}")

    total_images = sum(r["images"] for r in results)
    wall_s = max(r["elapsed_s"] for r in results)
    aggregate_throughput = total_images / wall_s

    # Per-worker compute time per iteration (ms)
    per_worker_ms = [(r["elapsed_s"] * 1000.0) / iterations for r in results]
    per_worker_avg = statistics.mean(per_worker_ms)

    # For CSV comparability: report the *effective end-to-end latency*
    # per request = wall_time / (iterations) — i.e., how long the
    # aggregate run took per iteration of the whole workload.
    # This is the honest "amortized latency" under concurrency.
    end_to_end_ms_per_iter = (wall_s * 1000.0) / iterations

    # We hand _stats a synthetic single-element list = amortized latency,
    # then override throughput from wall time.
    times = [end_to_end_ms_per_iter]
    return times, aggregate_throughput, f"{per_worker_avg:.4f}"


# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Parallel inference benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True, choices=["CPU", "GPU"])
    p.add_argument("--mode", required=True,
                   choices=["sync", "async", "multiprocess"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--input-shape", default="3,300,300")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--csv", default="results/parallel_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    spatial = tuple(int(x) for x in args.input_shape.split(","))

    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    wall_throughput = None
    per_worker_str = None

    if args.mode == "sync":
        times, wall_throughput, per_worker_str = run_sync(
            args.model, args.device, args.batch_size,
            args.iterations, args.warmup, spatial)
    elif args.mode == "async":
        times, wall_throughput, per_worker_str = run_async(
            args.model, args.device, args.batch_size,
            args.iterations, args.warmup, spatial, args.workers)
    else:
        times, wall_throughput, per_worker_str = run_multiprocess(
            args.model, args.device, args.batch_size,
            args.iterations, args.warmup, spatial, args.workers)

    stats = _stats(times, args.batch_size)
    if wall_throughput is not None:
        stats["throughput_img_s"] = f"{wall_throughput:.2f}"

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.path.basename(args.model),
        "device": args.device,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "workers": args.workers if args.mode != "sync" else 1,
        "iterations": args.iterations,
        **stats,
        "per_worker_ms": per_worker_str if per_worker_str else "",
    }

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    new_file = not os.path.isfile(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in CSV_FIELDS})

    print(f"{args.mode:12} | device={args.device:3} | "
          f"batch={args.batch_size:<2} | workers={row['workers']:<2} | "
          f"mean={stats['mean_ms']:>10} ms | "
          f"thr={stats['throughput_img_s']:>8} img/s")
    print(f"Appended to: {args.csv}")


if __name__ == "__main__":
    main()
