# Intel Iris Xe Inference Lab

Benchmarking and optimizing deep learning inference on Intel Iris Xe
integrated graphics with OpenVINO — CPU vs GPU, batch sizes, async
pipelining, multiprocess scaling, and cross-model comparison.

**Clone URL:** `https://github.com/adnanphp/intel-iris-xe-inference-lab.git`

---

## Hardware

| Component | Specification |
|---|---|
| CPU | Intel Core i5-1135G7 (11th Gen, 4 cores / 8 threads) |
| GPU | Intel Iris Xe Graphics (integrated, shares system RAM) |
| RAM | ~7 GB available |
| OS | Ubuntu 26.04 LTS (64-bit) |
| Desktop | XFCE 4.20 |
| Framework | OpenVINO 2024.6.0 |
| Python | 3.12 |

The key constraint: the GPU is **integrated**, so it shares the same
~7 GB of system memory as the CPU. There is no dedicated VRAM.

---

## What this project investigates

The central question is not "is the GPU faster?" — it's **"under what
conditions is the GPU faster, and by how much?"** The answer turns out
to be highly model-dependent.

Concretely, the project measures:

1. **CPU vs GPU latency and throughput** across batch sizes 1–16.
2. **Async pipelining** — does keeping multiple requests in flight
   improve GPU throughput?
3. **Multiprocess scaling** — does running N worker processes beat a
   single process?
4. **Cross-model comparison** — is the CPU/GPU winner the same for
   different network architectures?

---

## Project structure
intel-iris-xe-inference-lab/
├── benchmarks/
│ ├── run_benchmark.py # single (device, batch) benchmark
│ ├── sweep.py # YAML-driven benchmark matrix
│ ├── parallel_benchmark.py # sync / async / multiprocess modes
│ ├── plot_results.py # latency/throughput/speedup plots
│ ├── plot_parallel.py # parallel scaling plots
│ └── compare_models.py # cross-model comparison table + plot
├── config/
│ └── benchmark_config.yaml
├── model/ # OpenVINO IR models (not committed)
├── results/ # CSV outputs and plots (not committed)
├── docs/
├── requirements.txt
└── README.md

text

---

## Installation

```bash
git clone https://github.com/adnanphp/intel-iris-xe-inference-lab.git
cd intel-iris-xe-inference-lab

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
requirements.txt:

text
openvino>=2024.6.0
numpy>=1.24
matplotlib>=3.5
pyyaml>=6.0
Models
The OpenVINO IR model files (.xml + .bin) are not committed to the
repository. Download them separately:

Face detection (face-detection-retail-0004, input [N,3,300,300]):

bash
cd model
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0004/FP16/face-detection-retail-0004.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0004/FP16/face-detection-retail-0004.bin
MobileNetV3-Large (mobilenet-v3-large-1.0-224-tf, input [N,3,224,224]):

bash
cd model
wget https://huggingface.co/OpenVINO/mobilenet_v3_large-fp16-ov/resolve/main/mobilenet_v3_large.xml -O mobilenet-v3-large-1.0-224-tf.xml
wget https://huggingface.co/OpenVINO/mobilenet_v3_large-fp16-ov/resolve/main/mobilenet_v3_large.bin -O mobilenet-v3-large-1.0-224-tf.bin
Usage
Verify the GPU is available
bash
python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# -> ['CPU', 'GPU']
Single benchmark
bash
python3 benchmarks/run_benchmark.py \
  --model model/face-detection-retail-0004.xml \
  --device GPU --batch-size 8 --iterations 50 \
  --input-shape 3,300,300 \
  --csv results/benchmark_results.csv
Full batch-size sweep (YAML-driven)
bash
python3 benchmarks/sweep.py --csv results/sweep_python.csv
Parallel benchmarks
bash
# Async pipelining, queue depth 2
python3 benchmarks/parallel_benchmark.py \
  --model model/face-detection-retail-0004.xml \
  --device GPU --mode async --batch-size 1 \
  --iterations 100 --workers 2

# Multiprocess, 4 workers
python3 benchmarks/parallel_benchmark.py \
  --model model/face-detection-retail-0004.xml \
  --device GPU --mode multiprocess --batch-size 1 \
  --iterations 50 --workers 4
Plots and comparison
bash
python3 benchmarks/plot_results.py results/sweep_python.csv
python3 benchmarks/plot_parallel.py results/parallel_results.csv
python3 benchmarks/compare_models.py
Results
All measurements are on the hardware listed above, with 50 timed
iterations after 10 warm-up iterations. Latency values are means
over those iterations.

Finding 1 — Batching: the CPU collapses, the GPU holds
Face-detection model ([N,3,300,300]):

Batch	CPU latency	CPU thr	GPU latency	GPU thr	GPU speedup
1	3.33 ms	301 img/s	3.42 ms	293 img/s	0.97x
2	6.98 ms	287 img/s	6.49 ms	308 img/s	1.07x
4	20.21 ms	198 img/s	11.49 ms	348 img/s	1.76x
8	57.50 ms	139 img/s	24.76 ms	323 img/s	2.32x
16	122.75 ms	130 img/s	53.28 ms	300 img/s	2.30x
Key observation: CPU throughput peaks at batch=1 and declines
as batch grows (301 → 130 img/s), while GPU throughput peaks at
batch=4 (348 img/s) and stays roughly flat. The CPU's standard
deviation also spikes at batch=8 (13.8 ms vs ~1 ms at smaller
batches), indicating memory-bandwidth saturation, not just slowness.
The GPU is both faster and more stable at large batches.

Finding 2 — Parallelism: multiprocess wins on throughput
On the face-detection model at batch=1, three concurrency strategies:

Strategy	Workers	Throughput	Notes
sync	1	269–290 img/s	baseline, lowest latency (~3.5 ms)
async	2	332 img/s	best async depth; plateaus beyond 2
async	4	317 img/s	no gain over depth 2
async	8	318 img/s	no gain
multiprocess	2	339 img/s	
multiprocess	4	375 img/s	best overall throughput
multiprocess	6	366 img/s	saturated
multiprocess	8	364 img/s	saturated; no crash on 7 GB RAM
Key observation: For aggregate throughput, multiprocess with
4 workers wins (~375 img/s), beating both sync (~290) and async
(~330). But per-request latency rises under concurrency — a
trade-off to be aware of. Async pipelining yields only a modest
~12% improvement over sync; the GPU is already reasonably
saturated at batch=1 for this small model.

Finding 3 — The CPU/GPU winner depends on the model
MobileNetV3-Large ([N,3,224,224]), same benchmark, same engine:

Batch	CPU latency	CPU thr	GPU latency	GPU thr	GPU speedup
1	2.93 ms	342 img/s	3.56 ms	281 img/s	0.82x
2	4.90 ms	408 img/s	6.58 ms	304 img/s	0.74x
4	10.00 ms	400 img/s	11.21 ms	357 img/s	0.89x
8	24.21 ms	330 img/s	27.69 ms	289 img/s	0.87x
16	66.38 ms	241 img/s	84.90 ms	188 img/s	0.78x
Key observation: For MobileNetV3, the CPU wins at every batch
size — the opposite of the face-detection result. The most likely
explanation is architectural: MobileNetV3 is built on depthwise
separable convolutions, which are memory-bound rather than
compute-bound. The CPU's oneDNN kernels and AVX-512 paths handle
these efficiently, while the integrated GPU gains little from its
wider SIMD throughput when the bottleneck is memory access.
Additionally, the model is shipped in FP16, and GPU-side FP16/FP32
conversion overhead may also play a role.

Practical takeaway: "Is the GPU faster?" has no universal
answer. Benchmark the specific model on the specific hardware
before deciding. For lightweight, memory-bound networks on this
laptop, CPU can be the better choice outright.

Plots
Generated plots (in results/plots/, not committed):

latency.png, throughput.png, speedup.png — face-detection
batch-size sweep.

parallel_throughput.png, parallel_latency.png — sync / async /
multiprocess scaling.

model_comparison.png — four-line throughput comparison
(face-detection × CPU/GPU, MobileNet × CPU/GPU).

Methodology notes
Warm-up: 10 iterations before timing, to exclude first-call
compilation and cache effects.

Timing: time.perf_counter() around each individual iteration,
not around the whole loop, so per-iteration variance (std dev,
P95, P99) is measurable.

Model input: random float32 tensors of the model's declared
input shape. No real image preprocessing — this is a pure
inference-throughput benchmark.

Parallel modes: async uses a pool of InferRequest objects
with explicit round-robin submission and wait() on the oldest
in-flight request; multiprocess uses multiprocessing with the
spawn start method and a barrier so all workers begin their
timed section simultaneously.

CSV schema: timestamp, model, device, mode, batch_size, workers, iterations, mean_ms, median_ms, std_ms, p95_ms, p99_ms, throughput_img_s, per_worker_ms.

Limitations
Single machine, single user. Background load, thermal
throttling, and desktop compositing can all affect numbers by a
few percent between runs.

Synthetic inputs. No data loading or preprocessing is
included; real pipelines will have lower effective throughput.

FP16 model. MobileNetV3 was downloaded as FP16. A direct
FP32 comparison, or INT8 quantization, has not been run yet.

Shared memory. The GPU and CPU share the same 7 GB of RAM.
Results will differ significantly on hardware with dedicated
VRAM.

Integrated GPU driver version. OpenVINO results depend on the
installed Intel GPU driver stack; numbers may shift across driver
versions.

Roadmap
☑ OpenVINO + Iris Xe validation
☑ CPU vs GPU batch-size sweep (face-detection)
☑ Reusable YAML-driven benchmark engine
☑ Async and multiprocess parallel benchmarks
☑ Cross-model comparison (face-detection vs MobileNetV3)
□ Precision experiments (FP32 vs FP16 vs INT8)
□ End-to-end batch image inference application
□ Automated test suite (pytest)
License
MIT (add a LICENSE file before publishing).

---

### Finding 4 — INT8 quantization: CPU wins by a wider margin

MobileNetV3 was post-training quantized to INT8 with NNCF (100 random
calibration samples — **throughput-only result, accuracy not measured**).
The original model is FP16.

| Batch | FP16 CPU | INT8 CPU | CPU gain | FP16 GPU | INT8 GPU | GPU gain | INT8 speedup (CPU/GPU) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 342 img/s | **667 img/s** | 1.95x | 281 img/s | 293 img/s | 1.04x | 0.44x |
| 2 | 409 img/s | **805 img/s** | 1.97x | 304 img/s | 407 img/s | 1.34x | 0.51x |
| 4 | 400 img/s | **992 img/s** | 2.48x | 357 img/s | 513 img/s | 1.44x | 0.52x |
| 8 | 330 img/s | **1077 img/s** | 3.26x | 289 img/s | 719 img/s | 2.49x | 0.67x |

**Key observations:**

1. **INT8 is a dramatic win on CPU.** Peak throughput rises from
   ~409 img/s (FP16) to **~1077 img/s** (INT8) at batch=8 — a **2.6x**
   improvement. This is consistent with the CPU's VNNI instruction
   support (AVX-512 VNNI), which accelerates INT8 matrix operations
   natively.

2. **The GPU also benefits from INT8, but less.** Peak GPU throughput
   rises from ~357 img/s to ~719 img/s (2x). But because the CPU's
   gain is larger, **the CPU's lead over the GPU widens with INT8** —
   from ~0.8x in FP16 (GPU slightly behind) to 0.44–0.67x (CPU 1.5–2.3x
   faster).

3. **The optimal batch size shifts with precision.** FP16 peaked at
   batch=2; INT8 peaks at batch=8. Lower memory pressure allows larger
   batches to be productive.

4. **The INT8 model size is only modestly smaller** in this build
   (10.95 MB → 9.56 MB), because the original model was already FP16
   and many weights remained in 16-bit. The speedup comes from
   activation quantization and VNNI execution, not solely from reduced
   model size.

**Practical takeaway:** for this MobileNet-style workload on this
laptop, **INT8 + CPU at batch=8 is by far the fastest configuration
measured** — 1077 img/s, roughly 2.6x the best FP16 result and nearly
4x the best GPU result. The Iris Xe's INT8 path is real but less
mature than the CPU's on this hardware.

**Caveat:** calibration used random tensors, so this measures
*throughput only*. A real deployment would require real calibration
data and an accuracy comparison; those are not included here.


---

## Application — batch image inference

The benchmark engine is also usable as a real inference tool. `app/batch_inference.py`
takes a folder of images, runs the face-detection model in batches on CPU or
GPU, draws bounding boxes, and writes a JSON report.

### Usage

```bash
python3 app/batch_inference.py \
  --input test_images \
  --model model/face-detection-retail-0004.xml \
  --device GPU \
  --batch-size 4 \
  --output output
Arguments:

--input — folder of .jpg / .jpeg / .png / .bmp images

--model — OpenVINO IR .xml

--device — CPU or GPU

--batch-size — how many images per forward pass (last batch is padded)

--conf — confidence threshold (default 0.5)

--output — output directory

Outputs
text
output/
├── annotated_images/       # one annotated copy per input image
└── report.json             # timing + per-image detections
Example report.json for a single-image run on GPU:

json
{
  "model": "face-detection-retail-0004.xml",
  "device": "GPU",
  "batch_size": 1,
  "conf_threshold": 0.5,
  "images_processed": 1,
  "total_detections": 1,
  "batches_run": 1,
  "padded_batches": 0,
  "total_inference_time_seconds": 0.004,
  "mean_batch_latency_ms": 3.967,
  "throughput_images_per_second": 252.08,
  "per_image": [
    {
      "image": "face1.jpg",
      "detections": 1,
      "boxes": [
        {"conf": 1.0, "xmin": 0.4198, "ymin": 0.1491,
         "xmax": 0.7932, "ymax": 0.5658}
      ]
    }
  ]
}
Box coordinates are normalized to [0, 1] relative to the original
image size.

A preprocessing gotcha worth recording
The first version of the app produced zero detections (top confidence
~0.05) on a clear portrait. The cause was preprocessing: the app was
converting to RGB and dividing by 255 before inference. This model —
face-detection-retail-0004 — expects BGR, raw 0–255, NCHW. Feeding
it RGB [0,1] shifts the activation distribution enough to collapse the
detector's confidence. After switching to BGR [0,255] with no
normalization, the same image gave conf=1.0.

This is a useful reminder that OpenVINO models have per-model
preprocessing contracts. Always check the model's expected input format
before blaming the hardware, the GPU driver, or OpenVINO itself.

