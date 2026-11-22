#!/usr/bin/env python3
"""INT8 post-training quantization of an OpenVINO IR model using NNCF."""

import argparse
import os
import sys

import numpy as np
import openvino as ov
import nncf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Input IR .xml")
    p.add_argument("--output", required=True, help="Output INT8 IR .xml")
    p.add_argument("--input-shape", default="1,3,224,224",
                   help="Shape for calibration samples, e.g. 1,3,224,224")
    p.add_argument("--calib-samples", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    shape = tuple(int(x) for x in args.input_shape.split(","))
    print(f"Model:    {args.model}")
    print(f"Shape:    {shape}")
    print(f"Samples:  {args.calib_samples}")

    core = ov.Core()
    model = core.read_model(args.model)

    # Reshape to a fixed shape so calibration samples match
    model.reshape({0: list(shape)})

    # Synthetic calibration dataset: random float32 tensors.
    # NOTE: for speed benchmarks only. Accuracy requires real images.
    def transform_fn(data_item):
        return data_item

    calibration_data = [
        np.random.rand(*shape).astype(np.float32)
        for _ in range(args.calib_samples)
    ]
    dataset = nncf.Dataset(calibration_data, transform_fn)

    print("Quantizing (this may take a minute)...")
    quantized = nncf.quantize(
        model,
        dataset,
        subset_size=args.calib_samples,
    )

    # Save
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    ov.save_model(quantized, args.output)
    print(f"Saved INT8 model: {args.output}")

    # Report file sizes
    base_xml = args.model
    base_bin = base_xml.replace(".xml", ".bin")
    q_xml = args.output
    q_bin = q_xml.replace(".xml", ".bin")
    for label, x, b in [("original", base_xml, base_bin),
                        ("INT8", q_xml, q_bin)]:
        if os.path.isfile(b):
            print(f"  {label:8} .bin size: {os.path.getsize(b)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
