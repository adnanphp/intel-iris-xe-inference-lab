#!/usr/bin/env python3
"""
Batch image inference for the face-detection-retail-0004 OpenVINO model.

Reads images from a folder, runs batched inference on CPU or GPU,
decodes SSD boxes, draws annotations, saves results and a JSON report.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

INPUT_W, INPUT_H = 300, 300
CONF_THRESHOLD = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder of input images")
    p.add_argument("--model", required=True, help="IR .xml model path")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", default="output")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    return p.parse_args()


def load_image(path):
    """
    Read and preprocess an image for face-detection-retail-0004.

    This model expects BGR, raw 0-255 values, NCHW layout.
    No RGB swap and no /255 normalization — that collapses confidences.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None, None
    original = img.copy()
    resized = cv2.resize(img, (INPUT_W, INPUT_H))
    tensor = resized.astype(np.float32)          # BGR, 0-255
    tensor = np.transpose(tensor, (2, 0, 1))     # HWC -> CHW
    return tensor, original


def decode_ssd(output, conf_thresh):
    """
    Decode face-detection-retail-0004 output.

    Output shape: [1, 1, N, 7]
    Each row: [image_id, label, confidence, xmin, ymin, xmax, ymax]
    Coordinates are normalized [0,1] relative to input size.
    """
    dets = output[0][0]
    boxes = []
    for d in dets:
        conf = float(d[2])
        if conf < conf_thresh:
            continue
        xmin = max(0.0, min(1.0, float(d[3])))
        ymin = max(0.0, min(1.0, float(d[4])))
        xmax = max(0.0, min(1.0, float(d[5])))
        ymax = max(0.0, min(1.0, float(d[6])))
        boxes.append((xmin, ymin, xmax, ymax, conf))
    return boxes


def draw_boxes(img, boxes):
    h, w = img.shape[:2]
    for (xmin, ymin, xmax, ymax, conf) in boxes:
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Input folder not found: {args.input}")
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in exts
    )
    if not image_paths:
        sys.exit(f"No images found in {args.input}")

    print(f"Images:     {len(image_paths)}")
    print(f"Model:      {args.model}")
    print(f"Device:     {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)

    core = ov.Core()
    if args.device not in core.available_devices:
        sys.exit(f"Device {args.device} not available. Got: {core.available_devices}")
    model = core.read_model(args.model)
    model.reshape({0: [args.batch_size, 3, INPUT_H, INPUT_W]})
    compiled = core.compile_model(model, args.device)

    loaded = []
    for p in image_paths:
        tensor, original = load_image(p)
        if tensor is None:
            print(f"  SKIP (unreadable): {p.name}")
            continue
        loaded.append((p, tensor, original))

    if not loaded:
        sys.exit("No readable images.")

    os.makedirs(os.path.join(args.output, "annotated_images"), exist_ok=True)

    # Warm-up
    dummy = np.zeros((args.batch_size, 3, INPUT_H, INPUT_W), dtype=np.float32)
    for _ in range(2):
        compiled([dummy])

    per_batch_times = []
    per_image_results = []
    total_detections = 0
    padded_batches = 0

    for start in range(0, len(loaded), args.batch_size):
        chunk = loaded[start:start + args.batch_size]
        actual = len(chunk)
        tensors = [c[1] for c in chunk]
        if actual < args.batch_size:
            padded_batches += 1
            while len(tensors) < args.batch_size:
                tensors.append(tensors[-1])
        batch = np.stack(tensors, axis=0)

        t0 = time.perf_counter()
        output = compiled([batch])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_batch_times.append(elapsed_ms)

        raw = output[compiled.output(0)]
        for i in range(actual):
            path, _, original = chunk[i]
            boxes = decode_ssd(raw[i:i+1], args.conf)
            total_detections += len(boxes)
            annotated = draw_boxes(original, boxes)
            out_path = os.path.join(args.output, "annotated_images", path.name)
            cv2.imwrite(out_path, annotated)
            per_image_results.append({
                "image": path.name,
                "detections": len(boxes),
                "boxes": [
                    {"conf": round(b[4], 3),
                     "xmin": round(b[0], 4), "ymin": round(b[1], 4),
                     "xmax": round(b[2], 4), "ymax": round(b[3], 4)}
                    for b in boxes
                ],
            })

    total_time_s = sum(per_batch_times) / 1000.0
    throughput = len(loaded) / total_time_s

    report = {
        "model": os.path.basename(args.model),
        "device": args.device,
        "batch_size": args.batch_size,
        "conf_threshold": args.conf,
        "images_processed": len(loaded),
        "total_detections": total_detections,
        "batches_run": len(per_batch_times),
        "padded_batches": padded_batches,
        "total_inference_time_seconds": round(total_time_s, 4),
        "mean_batch_latency_ms": round(float(np.mean(per_batch_times)), 3),
        "throughput_images_per_second": round(throughput, 2),
        "per_image": per_image_results,
    }

    report_path = os.path.join(args.output, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nProcessed:  {len(loaded)} images")
    print(f"Detections: {total_detections}")
    print(f"Time:       {total_time_s:.3f} s")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Report:     {report_path}")
    print(f"Annotated:  {args.output}/annotated_images/")


if __name__ == "__main__":
    main()
