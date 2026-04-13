"""
src/video_inference.py
Real-time dashcam video inference at 30fps using PyTorch + OpenCV.
Draws bounding boxes + class labels on each frame.
"""

import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from train import build_model, DEVICE
from data_processing import CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "normal":         (0,   200,   0),
    "accident":       (0,     0, 255),
    "obstacle":       (0,   165, 255),
    "pedestrian":     (255, 255,   0),
    "traffic_sign":   (255,   0, 255),
    "lane_violation": (0,   255, 255),
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def load_model(arch: str, weights_path: str) -> nn.Module:
    model = build_model(arch)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE))
    model.eval()
    logger.info(f"Loaded {arch} from {weights_path}")
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return INFER_TRANSFORM(pil).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict_frame(model, frame, confidence_threshold=0.45):
    t0 = time.perf_counter()
    tensor = preprocess_frame(frame)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    latency_ms = (time.perf_counter() - t0) * 1000
    predicted_class = IDX_TO_CLASS[idx.item()]
    confidence = conf.item()
    if confidence < confidence_threshold:
        predicted_class = "normal"
    return {
        "class":      predicted_class,
        "confidence": round(confidence, 3),
        "latency_ms": round(latency_ms, 2),
        "all_probs":  {IDX_TO_CLASS[i]: round(p.item(), 3)
                       for i, p in enumerate(probs)},
    }


def draw_overlay(frame, prediction):
    h, w = frame.shape[:2]
    cls = prediction["class"]
    conf = prediction["confidence"]
    latency = prediction["latency_ms"]
    color = CLASS_COLORS.get(cls, (255, 255, 255))

    thickness = 3 if cls != "normal" else 1
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    label = f"{cls.upper()}  {conf*100:.1f}%"
    (lw, lh), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (10, 10),
                  (10 + lw + 12, 10 + lh + 12), color, -1)
    cv2.putText(frame, label, (16, 10 + lh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    lat_label = f"{latency:.1f} ms"
    (llw, llh), _ = cv2.getTextSize(
        lat_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame,
                  (w - llw - 18, h - llh - 18),
                  (w - 4, h - 4), (0, 0, 0), -1)
    cv2.putText(frame, lat_label, (w - llw - 12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if cls in ("accident", "pedestrian", "lane_violation"):
        cv2.putText(frame, f"ALERT: {cls.upper()} DETECTED",
                    (w // 2 - 200, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def process_video(model, input_path, output_path,
                  target_fps=30, confidence_threshold=0.45,
                  display=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(src_fps / target_fps))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc,
                          target_fps, (width, height))

    frame_idx = 0
    latencies = []
    class_counts = {c: 0 for c in CLASSES}

    logger.info(f"Processing {input_path} → {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            prediction = predict_frame(
                model, frame, confidence_threshold)
            annotated = draw_overlay(frame.copy(), prediction)
            out.write(annotated)
            latencies.append(prediction["latency_ms"])
            class_counts[prediction["class"]] += 1
            if display:
                cv2.imshow("Dashcam Anomaly Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        frame_idx += 1

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    stats = {
        "total_frames":   len(latencies),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "class_counts":   class_counts,
        "anomaly_rate":   round(
            1 - class_counts["normal"] / max(len(latencies), 1), 3),
    }
    logger.info(
        f"Done — frames={stats['total_frames']} "
        f"avg={stats['avg_latency_ms']}ms "
        f"p99={stats['p99_latency_ms']}ms"
    )
    return stats


@torch.no_grad()
def batch_predict(model, frames, confidence_threshold=0.45):
    tensors = torch.stack(
        [preprocess_frame(f).squeeze(0) for f in frames]).to(DEVICE)
    t0 = time.perf_counter()
    logits = model(tensors)
    probs = torch.softmax(logits, dim=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    results = []
    per_frame_ms = elapsed_ms / len(frames)
    for i, p in enumerate(probs):
        conf, idx = p.max(0)
        cls = IDX_TO_CLASS[idx.item()]
        if conf.item() < confidence_threshold:
            cls = "normal"
        results.append({
            "frame_index": i,
            "class":       cls,
            "confidence":  round(conf.item(), 3),
            "latency_ms":  round(per_frame_ms, 2),
        })
    return results


if __name__ == "__main__":
    model = load_model("resnet50", "best_resnet50.pth")
    stats = process_video(
        model,
        input_path="data/raw/sample_dashcam.mp4",
        output_path="data/output/annotated_output.mp4",
        target_fps=30,
    )
    print(stats)