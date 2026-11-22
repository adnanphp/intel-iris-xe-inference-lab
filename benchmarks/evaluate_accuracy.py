#!/usr/bin/env python3
"""
Evaluate Top-1 / Top-5 accuracy of a MobileNet IR model on Imagenette.

The model has 1000 outputs; Imagenette covers only 10 of those classes.
We take the model's top-5 predicted *ImageNet indices* and check whether
the ground-truth ImageNet index (from val_labels.txt) is among them.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import openvino as ov


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--labels", required=True,
                   help="TSV: image_path<TAB>imagenet_index")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of images (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")
    if not os.path.isfile(args.labels):
        sys.exit(f"Labels file not found: {args.labels}")

    entries = []
    with open(args.labels) as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            path, idx = line.rsplit("\t", 1)
            entries.append((path, int(idx)))
    if args.limit > 0:
        entries = entries[:args.limit]

    print(f"Model:      {args.model}")
    print(f"Device:     {args.device}")
    print(f"Images:     {len(entries)}")
    print(f"Batch size: {args.batch_size}")

    core = ov.Core()
    model = core.read_model(args.model)
    model.reshape({0: [args.batch_size, 3, 224, 224]})
    compiled = core.compile_model(model, args.device)

    top1 = 0
    top5 = 0
    total = 0

    # Preload tensors
    print("Loading images...")
    tensors = []
    labels = []
    for path, idx in entries:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        # Hugging Face IR conversion did NOT fuse normalization into the
        # graph, so we must apply it here: RGB + (x-127.5)/127.5 -> [-1,1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        t = np.transpose(rgb, (2, 0, 1))
        tensors.append(t)
        labels.append(idx)

    print(f"Loaded {len(tensors)} readable images. Running inference...")

    for start in range(0, len(tensors), args.batch_size):
        chunk = tensors[start:start + args.batch_size]
        actual = len(chunk)
        while len(chunk) < args.batch_size:
            chunk.append(chunk[-1])
        batch = np.stack(chunk, axis=0)

        out = compiled([batch])[compiled.output(0)]  # [B, 1000]
        for i in range(actual):
            logits = out[i]
            e = np.exp(logits - np.max(logits))
            probs = e / e.sum()
            preds_top5 = np.argsort(-probs)[:5]
            true_idx = labels[start + i]
            total += 1
            if preds_top5[0] == true_idx:
                top1 += 1
            if true_idx in preds_top5:
                top5 += 1

        if (start // args.batch_size) % 20 == 0:
            print(f"  {total}/{len(tensors)} done")

    if total == 0:
        sys.exit("No images processed.")

    print()
    print(f"Top-1: {top1}/{total} = {100.0*top1/total:.2f}%")
    print(f"Top-5: {top5}/{total} = {100.0*top5/total:.2f}%")


if __name__ == "__main__":
    main()
