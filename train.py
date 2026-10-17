"""
Assignment 2: training pipeline.

Stage 1: download the dataset from Roboflow Universe.
Stage 2: describe the dataset (counts per class and split, object size).
Stage 3: finetune yolo11n on it.

Quick timing test:  QUICK=1 uv run python main.py train
"""

# Make Python trust the same certificates as the Mac. This must come first.
import truststore

truststore.inject_into_ssl()

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

# The exact dataset version we use (also written in REPORT.md).
WORKSPACE = "defect-5hwax"
PROJECT = "fabric-defect-ieswu"
VERSION = 16
DATA_DIR = "data/fabric-defect"
OUT_DIR = Path("outputs")
WEIGHTS_DIR = Path("weights")

# Training settings. QUICK=1 is only a short test to measure speed.
QUICK = os.environ.get("QUICK") == "1"
MODEL = "yolo11n.pt"
EPOCHS = 1 if QUICK else 30
FRACTION = 0.05 if QUICK else 1.0   # share of the training images used
IMGSZ = 512                          # the dataset images are 512x512; defects are small, so no shrinking
BATCH = 16

# Use the fastest hardware available: NVIDIA GPU, then the Mac's GPU, then the CPU.
if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def download_dataset():
    """Download the dataset in YOLO format and return the folder it was saved in."""
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    dataset = version.download("yolov11", location=DATA_DIR)
    return dataset.location


def describe_dataset(folder):
    """Count images and defects per split and class, and measure defect size.

    Each label line is: class x_center y_center width height (all 0 to 1),
    so width * height is the share of the photo the defect covers.
    """
    folder = Path(folder)
    with open(folder / "data.yaml") as f:
        config = yaml.safe_load(f)
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    stats = {"class_names": names, "splits": {}}
    for split in ["train", "valid", "test"]:
        images = [p for p in (folder / split / "images").iterdir() if p.is_file()]
        counts = {name: 0 for name in names}
        areas = {name: [] for name in names}
        images_with_boxes = 0
        for label_file in (folder / split / "labels").glob("*.txt"):
            rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
            if rows:
                images_with_boxes += 1
            for row in rows:
                name = names[int(row[0])]
                counts[name] += 1
                areas[name].append(float(row[3]) * float(row[4]))
        stats["splits"][split] = {
            "images": len(images),
            "images_with_defects": images_with_boxes,
            "images_without_defects": len(images) - images_with_boxes,
            "instances_per_class": counts,
            "median_size_pct_of_image": {
                name: round(100 * float(np.median(a)), 2) if a else None
                for name, a in areas.items()
            },
        }
    return stats


def print_stats(stats):
    for split, s in stats["splits"].items():
        print(f"\n{split.upper()}: {s['images']} images "
              f"({s['images_with_defects']} with defects, "
              f"{s['images_without_defects']} without)")
        for name in stats["class_names"]:
            print(f"  {name:15s} {s['instances_per_class'][name]:6d} instances, "
                  f"median size {s['median_size_pct_of_image'][name]}% of image")


def write_data_yaml(folder, class_names):
    """Write our own data file with absolute paths, so YOLO always finds the images."""
    folder = Path(folder).resolve()
    config = {
        "path": str(folder),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = folder / "data_abs.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return yaml_path


def train_model(data_yaml):
    """Finetune the pretrained YOLO model and return the path of the best weights."""
    model = YOLO(MODEL)
    start = time.time()
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        fraction=FRACTION,
        seed=0,
        project=str(Path("runs").resolve()),
        name="quick" if QUICK else "train",
        exist_ok=True,
        plots=not QUICK,
        workers=2,
    )
    minutes = (time.time() - start) / 60
    print(f"\nTraining took {minutes:.1f} minutes for {EPOCHS} epoch(s) on {FRACTION:.0%} of the data.")
    return Path(model.trainer.best)


def main():
    print("Training device:", DEVICE)
    folder = download_dataset()
    print("Dataset saved in:", folder)

    stats = describe_dataset(folder)
    print_stats(stats)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    data_yaml = write_data_yaml(folder, stats["class_names"])
    best = train_model(data_yaml)

    if QUICK:
        print("Quick test only: weights were NOT copied to weights/.")
    else:
        WEIGHTS_DIR.mkdir(exist_ok=True)
        shutil.copy(best, WEIGHTS_DIR / "best.pt")
        print("Saved weights/best.pt")


if __name__ == "__main__":
    main()