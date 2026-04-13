"""
src/predict.py
Single-image prediction utility used by FastAPI endpoints.
"""

import io
import base64
import time
from typing import Union

import numpy as np
import torch
from PIL import Image

from train import build_model, DEVICE
from video_inference import preprocess_frame, IDX_TO_CLASS


def predict_image(model, image_input, confidence_threshold=0.45):
    if isinstance(image_input, str):
        image_bytes = base64.b64decode(image_input)
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(pil)[:, :, ::-1]
    elif isinstance(image_input, bytes):
        pil = Image.open(io.BytesIO(image_input)).convert("RGB")
        frame = np.array(pil)[:, :, ::-1]
    else:
        frame = image_input

    t0 = time.perf_counter()
    tensor = preprocess_frame(frame)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    conf, idx = probs.max(0)
    predicted = IDX_TO_CLASS[idx.item()]
    confidence = conf.item()
    if confidence < confidence_threshold:
        predicted = "normal"

    return {
        "predicted_class": predicted,
        "confidence":      round(confidence, 4),
        "latency_ms":      round(latency_ms, 2),
        "class_probabilities": {
            IDX_TO_CLASS[i]: round(p.item(), 4)
            for i, p in enumerate(probs)
        },
        "is_anomaly": predicted != "normal",
        "severity":   _severity(predicted),
    }


def _severity(cls):
    if cls in {"accident", "pedestrian"}:      return "HIGH"
    if cls in {"obstacle", "lane_violation"}:  return "MEDIUM"
    if cls == "traffic_sign":                  return "LOW"
    return "NONE"