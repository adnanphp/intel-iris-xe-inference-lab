#!/usr/bin/env python3
"""INT8 PTQ of an OpenVINO IR model using real calibration images."""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import openvino as ov
import nncf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--calib-dir", required=True,
                   help="Folder of calibration images (resized 224x224 PNGs)")
    p.add_argument("--calib-count", type=int, default=200)
    p.add_argument("--input-shape", default="1,3,224,224")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")
    if not os.path.isdir(args.calib_dir):
        sys.exit(f"Calibration dir not found: {args.calib_dir}")

    shape = tuple(int(x) for x in args.input_shape.split(","))
    H, W = shape[2], shape[3]

    images = sorted(glob.glob(os.path.join(args.calib_dir, "*.png")))[:args.calib_count]
    if not images:
        sys.exit(f"No PNG images in {args.calib_dir}")
    print(f"Model:     {args.model}")
    print(f"Calib:     {len(images)} images from {args.calib_dir}")

    core = ov.Core()
    model = core.read_model(args.model)
    model.reshape({0: list(shape)})

    def load(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (W, H))
        # Match the Hugging Face IR model's expected preprocessing:
        # RGB + (x-127.5)/127.5 -> [-1, 1]. Must match evaluate_accuracy.py.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        tensor = np.transpose(rgb, (2, 0, 1))
        return tensor[None, ...]

    calib_data = [load(p) for p in images]

    def transform_fn(item):
        return item

    dataset = nncf.Dataset(calib_data, transform_fn)

    print("Quantizing with real calibration images...")
    quantized = nncf.quantize(model, dataset, subset_size=len(calib_data))
    ov.save_model(quantized, args.output)
    print(f"Saved: {args.output}")

    q_bin = args.output.replace(".xml", ".bin")
    print(f"INT8 .bin size: {os.path.getsize(q_bin)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
