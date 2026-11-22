# Precision study: FP16 vs INT8 on MobileNetV3-Large

A complete speed-and-accuracy comparison of FP16 and INT8 inference on
the Intel Iris Xe lab hardware. This document is the detailed companion
to the INT8 section of the main README.

## Setup

- **Model**: `mobilenet-v3-large-1.0-224-tf` (OpenVINO IR, Hugging Face
  conversion)
- **Calibration data**: 200 images from the Imagenette *training* set,
  resized to 224×224
- **Validation data**: 500 images from the Imagenette *validation* set,
  labeled with their ImageNet-1k indices
- **Preprocessing**: RGB, `(x − 127.5) / 127.5` → `[-1, 1]`, NCHW
- **Device**: CPU for accuracy; both CPU and GPU for throughput
- **Tooling**: OpenVINO 2024.6, NNCF 2.14.1, PyTorch-free pipeline

## Accuracy

| Model | Top-1 | Top-5 |
|---|---:|---:|
| FP16 | 90.20% | 98.00% |
| INT8 (real calibration) | 88.20% | 96.40% |
| INT8 (random calibration) | 4.60% | 9.40% |

### Interpretation

- Real-calibration INT8 loses **2.0 percentage points** of Top-1 — a
  normal, acceptable quantization cost.
- Random-calibration INT8 collapses to **4.6% Top-1** — effectively
  random guessing among Imagenette's 10 classes (chance = 10%).
- The gap between the two INT8 models (88.2% vs 4.6%) is the clearest
  possible demonstration that **calibration data determines whether
  post-training quantization produces a usable model**.

## Throughput (real-calibration INT8)

| Batch | FP16 CPU | INT8 CPU | CPU speedup | FP16 GPU | INT8 GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 342 | 462 | 1.35× | 281 | 179 |
| 2 | 409 | 575 | 1.41× | 304 | 269 |
| 4 | 400 | 690 | 1.73× | 357 | 410 |
| 8 | 330 | **754** | **2.28×** | 289 | 541 |

*Images/sec.*

### Interpretation

- INT8 delivers a **2.28× CPU speedup** at batch=8 with a **2-point
  accuracy cost** — a strong, production-ready trade-off.
- The GPU also benefits from INT8 (1.87× at batch=8), but the CPU's gain
  is larger, so **the CPU's lead over the GPU widens** under INT8
  (speedup ratio moves from 0.87× to 0.72×).
- **Optimal batch size shifts with precision**: FP16 peaks at batch=2 on
  CPU; INT8 keeps improving through batch=8.

## The random-calibration trap

The random-calibration model ran **faster** (1,077 img/s CPU at batch=8)
than the real-calibration one (754 img/s). A naive reading would
conclude that random calibration is better for throughput. It isn't —
the faster model was **functionally broken** (4.6% Top-1).

**Lesson:** throughput numbers are only meaningful when the model they
were measured on has been validated for accuracy. An inference-
optimization pipeline that does not evaluate accuracy is not a
quantization study — it is a computation benchmark.

## Reproducing

```bash
# 1. Download Imagenette (once)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar xzf imagenette2-160.tgz -C data/

# 2. Prepare calibration + validation data
python3 data/prepare_imagenette.py

# 3. Quantize with real calibration
python3 benchmarks/quantize_model_real.py \
  --model model/mobilenet-v3-large-1.0-224-tf.xml \
  --output model/mobilenet-v3-large-int8-real.xml \
  --calib-dir data/calibration_images --calib-count 200

# 4. Evaluate accuracy
python3 benchmarks/evaluate_accuracy.py \
  --model model/mobilenet-v3-large-int8-real.xml \
  --labels data/val_labels.txt --device CPU --limit 500

# 5. Benchmark throughput
python3 benchmarks/run_benchmark.py \
  --model model/mobilenet-v3-large-int8-real.xml \
  --device CPU --batch-size 8 --iterations 50 \
  --input-shape 3,224,224 \
  --csv results/mobilenet_int8_real_results.csv
Caveats
Accuracy measured on Imagenette's 10 classes, not full ImageNet-1k.
The FP16→INT8 delta is the meaningful result, not the absolute Top-1.

Calibration used 200 images; a larger calibration set might narrow the
2-point accuracy gap further.

Validation used 500 of 3,924 available Imagenette images. Full-set
evaluation would tighten the confidence intervals but is unlikely to
change the conclusions.
