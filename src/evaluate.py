"""
src/evaluate.py
Evaluate all 3 models on test set. Outputs per-class metrics,
confusion matrix, and inference latency comparison.
"""

import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score
)
import mlflow
import pandas as pd

from train import build_model, DEVICE, NUM_CLASSES
from data_processing import get_val_transforms, CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlbDataset(datasets.ImageFolder):
    def __init__(self, root, transform_alb):
        super().__init__(root)
        self.transform_alb = transform_alb

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = np.array(__import__("PIL").Image.open(path).convert("RGB"))
        return self.transform_alb(image=img)["image"], label


def evaluate_model(arch, weights_path, test_dir, batch_size=32):
    model = build_model(arch)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    test_ds = AlbDataset(test_dir, get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    all_preds, all_labels, all_probs = [], [], []
    latencies = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            t0 = time.perf_counter()
            outs = model(imgs)
            latencies.append(
                (time.perf_counter() - t0) / imgs.size(0) * 1000)
            probs = torch.softmax(outs, dim=1).cpu().numpy()
            preds = outs.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except Exception:
        auc = 0.0

    avg_latency = float(np.mean(latencies))
    p99_latency = float(np.percentile(latencies, 99))
    param_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024**2

    results = {
        "architecture":   arch,
        "accuracy":       round(acc, 4),
        "auc":            round(auc, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
        "param_mb":       round(param_mb, 1),
    }
    logger.info(
        f"{arch}: acc={acc:.4f} auc={auc:.4f} "
        f"latency={avg_latency:.1f}ms size={param_mb:.0f}MB"
    )

    with mlflow.start_run(run_name=f"eval_{arch}"):
        mlflow.log_metrics({
            "test_accuracy":  acc,
            "test_auc":       auc,
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "param_mb":       param_mb,
        })
    return results


def compare_all_models(weights, test_dir):
    rows = []
    for arch, path in weights.items():
        r = evaluate_model(arch, path, test_dir)
        rows.append({
            "Model":        arch,
            "Accuracy":     r["accuracy"],
            "AUC":          r["auc"],
            "Latency (ms)": r["avg_latency_ms"],
            "P99 (ms)":     r["p99_latency_ms"],
            "Size (MB)":    r["param_mb"],
        })
    df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    logger.info("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    weights = {
        "resnet50":        "best_resnet50.pth",
        "efficientnet_b0": "best_efficientnet_b0.pth",
        "mobilenet_v3":    "best_mobilenet_v3.pth",
    }
    df = compare_all_models(weights, "data/processed/test")
    df.to_csv("model_comparison.csv", index=False)