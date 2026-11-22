# Lessons learned

A short writeup of the non-obvious findings from this project, for readers
who don't want to scroll the full README. Every result here was measured on
the hardware and software listed in the main README (i5-1135G7 + Iris Xe +
OpenVINO 2024.6 on Ubuntu).

---

## 1. "Is the GPU faster?" has no universal answer

For the same hardware, the same framework, and the same benchmark engine:

- **Face detection (SSD, 300×300)** — GPU wins at batch ≥ 2, peaking at
  **2.32× faster** than CPU at batch=8.
- **MobileNetV3-Large (224×224)** — **CPU wins at every batch size**,
  by a margin of 1.1×–1.35×.

The reason is architectural. MobileNetV3 is built on depthwise-separable
convolutions: very few FLOPs per byte moved, memory-bound rather than
compute-bound. The CPU's oneDNN kernels and AVX-512 paths handle this
efficiently, while the integrated GPU gains little from its wider SIMD
throughput when the bottleneck is memory access.

**Practical takeaway:** benchmark the *specific* model on the *specific*
hardware. Marketing claims about "GPU acceleration" are not a substitute
for a measurement.

---

## 2. CPU collapses under large batches; GPU stays stable

On the face-detection model:

| Batch | CPU throughput | CPU std dev | GPU throughput |
|---:|---:|---:|---:|
| 1 | 301 img/s | 0.09 ms | 293 img/s |
| 8 | 139 img/s | **13.8 ms** | 323 img/s |
| 16 | 130 img/s | 4.5 ms | 300 img/s |

The CPU's *standard deviation* spikes at batch=8 — it doesn't just get
slower, it becomes unstable. This is the signature of memory-bandwidth
saturation: larger batches don't fit in cache, and the CPU spends its time
waiting on RAM. The GPU, sharing the same RAM but with a different access
pattern, doesn't hit the same wall.

The practical consequence: for offline / batch workloads, the GPU is not
just faster — it's *more predictable*.

---

## 3. Multiprocess beats async for aggregate throughput

Three concurrency strategies on the same model at batch=1:

| Strategy | Best depth | Throughput | Latency per request |
|---|---:|---:|---:|
| sync (1 at a time) | 1 | 290 img/s | 3.4 ms |
| async pipelining | 2 | 332 img/s | 6.0 ms |
| **multiprocess** | 4 | **375 img/s** | 10.7 ms |

Async pipelining — the "obvious" way to keep a GPU busy — yields only
~12% over sync. **Separate processes with independent OpenVINO contexts
scale further**, at the cost of higher per-request latency.

The scaling curve also plateaus: going from 4 to 8 workers adds almost no
throughput (375 → 364 img/s) while latency nearly doubles. The GPU's
practical ceiling for this model is ~400 img/s regardless of method.

**Takeaway:** concurrency in inference is not a free lunch. Different
techniques optimize different metrics; pick based on whether you care
about throughput (multiprocess) or per-request latency (sync).

---

## 4. INT8 quantization widens CPU's lead rather than closing it

Post-training INT8 quantization of MobileNetV3 (NNCF, 100 random
calibration samples):

| Batch | FP16 CPU | INT8 CPU | FP16 GPU | INT8 GPU |
|---:|---:|---:|---:|---:|
| 1 | 342 img/s | **667 img/s** | 281 img/s | 293 img/s |
| 8 | 330 img/s | **1077 img/s** | 289 img/s | 719 img/s |

The CPU gains **2.6×** from INT8; the GPU gains ~2×. Because the CPU's
gain is larger, the CPU/GPU speedup ratio shifts from ~0.8× (GPU slightly
behind) to 0.44–0.67× (CPU 1.5×–2.3× ahead).

The most likely explanation is VNNI (Vector Neural Network Instructions):
the 11th-gen CPU has native INT8 matrix-operation acceleration. The Iris
Xe's INT8 path exists and works, but is less mature on this hardware.

**Takeaway:** on this laptop, the fastest configuration measured in this
project is **INT8 + CPU at batch=8** — 1077 img/s, roughly 2.6× the best
FP16 result and nearly 4× the best GPU result.

**Caveat:** the INT8 calibration used random tensors, so accuracy was not
measured. This is a throughput result only. A real deployment would
require real calibration data and an accuracy comparison.

---

## 5. Preprocessing contracts matter more than hardware

The first version of the batch-inference app produced **zero detections**
on a clear portrait of Barack Obama, with the top raw confidence at 0.05.
The cause was not the GPU, not the model, not OpenVINO — it was that the
app was converting the image to RGB and normalizing to `[0, 1]` before
inference.

The `face-detection-retail-0004` model expects **BGR, raw 0–255, NCHW**.
Feeding it RGB `[0, 1]` shifts the activation distribution enough to
collapse the detector's confidence. After switching to BGR `[0, 255]` with
no normalization, the same image gave `conf=1.0`.

**Takeaway:** always check a model's declared input format before
debugging the hardware or the framework. Preprocessing mistakes can look
exactly like "the model doesn't work" — and are far more common.

---

## 6. Reproducibility notes

- All measurements use 50 timed iterations after 10 warm-up iterations.
- Timing is per-iteration (`time.perf_counter()` around each call), so
  standard deviation and P95/P99 are meaningful.
- Inputs are synthetic random tensors of the model's declared shape — no
  data loading or preprocessing is measured.
- The GPU shares system RAM with the CPU (integrated graphics). Results
  will differ on hardware with dedicated VRAM.
- Two models were benchmarked: `face-detection-retail-0004` (FP16) and
  `mobilenet-v3-large-1.0-224-tf` (FP16 and INT8).

Full CSVs and plots are regenerable from the repository with `make sweep`,
`make parallel`, `make quantize`, `make plots`, and `make compare`.
