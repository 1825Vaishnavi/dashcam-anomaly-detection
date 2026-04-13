"""
src/train.py
Fine-tune ResNet50, EfficientNet-B0, MobileNetV3 on BDD100K 5-class dataset.
Tracks all runs with MLflow. Registers best model in MLflow Model Registry.
"""

import os
import time
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch

from data_processing import get_train_transforms, get_val_transforms, CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(CLASSES)


def build_model(arch: str) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES)
    elif arch == "mobilenet_v3":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model.to(DEVICE)


class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform_alb):
        super().__init__(root)
        self.transform_alb = transform_alb

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = np.array(__import__("PIL").Image.open(path).convert("RGB"))
        augmented = self.transform_alb(image=img)
        return augmented["image"], label


def get_loaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    train_ds = AlbumentationsDataset(
        os.path.join(data_dir, "train"), get_train_transforms(img_size))
    val_ds = AlbumentationsDataset(
        os.path.join(data_dir, "val"), get_val_transforms(img_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except Exception:
        auc = 0.0
    return total_loss / total, correct / total, auc


def train_model(arch, data_dir, epochs=15, batch_size=32,
                lr=1e-4, img_size=224):
    mlflow.set_experiment("dashcam_anomaly_detection")
    with mlflow.start_run(run_name=arch):
        mlflow.log_params({
            "architecture": arch,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "img_size": img_size,
            "num_classes": NUM_CLASSES,
            "device": str(DEVICE),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        })
        model = build_model(arch)
        train_loader, val_loader = get_loaders(
            data_dir, batch_size, img_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        best_auc, best_acc = 0.0, 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_auc = eval_epoch(
                model, val_loader, criterion)
            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"[{arch}] Epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} "
                f"auc={val_auc:.4f} {elapsed:.1f}s"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "epoch_time_s": elapsed,
            }, step=epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                best_acc = val_acc
                torch.save(model.state_dict(), f"best_{arch}.pth")

        mlflow.log_metrics({
            "best_val_acc": best_acc,
            "best_val_auc": best_auc,
        })
        mlflow.pytorch.log_model(model, artifact_path=f"model_{arch}")
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model_{arch}"
        mv = mlflow.register_model(model_uri, f"DashcamAnomaly_{arch}")
        logger.info(
            f"Registered DashcamAnomaly_{arch} v{mv.version} "
            f"acc={best_acc:.4f} auc={best_auc:.4f}"
        )
    return best_acc, best_auc


def run_all(data_dir):
    archs = ["resnet50", "efficientnet_b0", "mobilenet_v3"]
    results = {}
    for arch in archs:
        logger.info(f"\n{'='*50}\nTraining {arch}\n{'='*50}")
        acc, auc = train_model(arch, data_dir)
        results[arch] = {"accuracy": acc, "auc": auc}
    best_arch = max(results, key=lambda a: results[a]["auc"])
    logger.info(f"\n Best model: {best_arch}")
    for arch, m in results.items():
        logger.info(f"  {arch}: acc={m['accuracy']:.4f} auc={m['auc']:.4f}")


if __name__ == "__main__":
    run_all("data/processed")