"""
src/data_processing.py
Parse BDD100K JSON labels and organize images into 5 anomaly classes.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "car":           "obstacle",
    "truck":         "obstacle",
    "bus":           "obstacle",
    "motor":         "obstacle",
    "bike":          "obstacle",
    "trailer":       "obstacle",
    "pedestrian":    "pedestrian",
    "rider":         "pedestrian",
    "traffic light": "traffic_sign",
    "traffic sign":  "traffic_sign",
    "lane":          "lane_violation",
    "other":         "normal",
}

CLASSES = ["normal", "accident", "obstacle",
           "pedestrian", "traffic_sign", "lane_violation"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _dominant_class(frame_labels):
    priority = ["accident", "pedestrian", "lane_violation",
                "traffic_sign", "obstacle", "normal"]
    found = set()
    for obj in frame_labels:
        cat = obj.get("category", "other").lower()
        mapped = CATEGORY_MAP.get(cat, "normal")
        found.add(mapped)
    for cls in priority:
        if cls in found:
            return cls
    return "normal"


def parse_bdd100k_labels(label_json_path):
    logger.info(f"Parsing labels from {label_json_path}")
    with open(label_json_path, "r") as f:
        data = json.load(f)
    image_labels = {}
    for frame in data:
        img_name = frame["name"]
        labels = frame.get("labels", [])
        image_labels[img_name] = _dominant_class(labels)
    dist = Counter(image_labels.values())
    logger.info(f"Class distribution: {dict(dist)}")
    return image_labels


def organize_dataset(images_dir, label_json_path, output_dir,
                     splits=None):
    if splits is None:
        splits = {"train": 0.70, "val": 0.15, "test": 0.15}

    image_labels = parse_bdd100k_labels(label_json_path)
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    by_class = defaultdict(list)
    for img_name, label in image_labels.items():
        for sub in ["train", "val", ""]:
            candidate = images_path / sub / img_name if sub else images_path / img_name
            if candidate.exists():
                by_class[label].append(candidate)
                break

    for cls, files in by_class.items():
        np.random.shuffle(files)
        n = len(files)
        boundaries = {
            "train": int(n * splits["train"]),
            "val":   int(n * (splits["train"] + splits["val"])),
            "test":  n,
        }
        prev = 0
        for split, end in boundaries.items():
            dest = output_path / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in files[prev:end]:
                shutil.copy2(src, dest / src.name)
            prev = end
        logger.info(f"{cls}: {n} images → train/val/test")

    logger.info(f"Dataset ready at {output_dir}")


def extract_frames(video_path, output_dir, target_fps=30,
                   max_frames=None):
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(src_fps / target_fps))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths, frame_idx, saved = [], 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            path = out_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(path), frame)
            frame_paths.append(str(path))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1
    cap.release()
    logger.info(f"Extracted {saved} frames → {output_dir}")
    return frame_paths


if __name__ == "__main__":
    organize_dataset(
        images_dir="data/raw/images",
        label_json_path="data/raw/labels/bdd100k_labels_images_train.json",
        output_dir="data/processed",
    )