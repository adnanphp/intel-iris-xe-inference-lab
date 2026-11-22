"""Prepare Imagenette for OpenVINO calibration and accuracy evaluation."""

import os
import random
from pathlib import Path
from PIL import Image

from imagenette_to_imagenet import wordnet_to_imagenet_index

DATA_DIR = Path("data/imagenette2-160")
VAL_DIR = DATA_DIR / "val"
TRAIN_DIR = DATA_DIR / "train"

# Output
CALIB_DIR = Path("data/calibration_images")
VAL_LIST = Path("data/val_labels.txt")

CALIB_COUNT = 200

def main():
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all training images
    train_images = []
    for wn_dir in TRAIN_DIR.iterdir():
        if not wn_dir.is_dir():
            continue
        for img_path in wn_dir.glob("*.JPEG"):
            train_images.append(img_path)
        for img_path in wn_dir.glob("*.jpg"):
            train_images.append(img_path)

    print(f"Found {len(train_images)} training images")
    random.seed(42)
    random.shuffle(train_images)

    # Copy calibration images
    for i, img_path in enumerate(train_images[:CALIB_COUNT]):
        # Resize to 224x224 and save as PNG for NNCF
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img.save(CALIB_DIR / f"calib_{i:04d}.png")

    print(f"Saved {CALIB_COUNT} calibration images to {CALIB_DIR}")

    # Build validation label list
    val_entries = []
    for wn_dir in VAL_DIR.iterdir():
        if not wn_dir.is_dir():
            continue
        wn_id = wn_dir.name
        imagenet_idx = wordnet_to_imagenet_index(wn_id)
        if imagenet_idx is None:
            continue
        for img_path in wn_dir.glob("*.JPEG"):
            val_entries.append(f"{img_path}\t{imagenet_idx}")
        for img_path in wn_dir.glob("*.jpg"):
            val_entries.append(f"{img_path}\t{imagenet_idx}")

    with open(VAL_LIST, "w") as f:
        f.write("\n".join(val_entries))

    print(f"Saved {len(val_entries)} validation entries to {VAL_LIST}")

if __name__ == "__main__":
    main()
