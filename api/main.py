"""
api/main.py
FastAPI app — /predict and /batch_predict endpoints.
Sub-100ms per-frame latency.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from train import build_model, DEVICE
from predict import predict_image, _severity
from video_inference import batch_predict, IDX_TO_CLASS, CLASSES
from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse, BatchPredictItem,
    HealthResponse, ModelInfoResponse,
)

_batch_predict = batch_predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS: Dict[str, torch.nn.Module] = {}
WEIGHT_PATHS = {
    "resnet50":        os.getenv("RESNET50_WEIGHTS",  "best_resnet50.pth"),
    "efficientnet_b0": os.getenv("EFFNET_WEIGHTS",    "best_efficientnet_b0.pth"),
    "mobilenet_v3":    os.getenv("MOBILENET_WEIGHTS", "best_mobilenet_v3.pth"),
}


def _load_models():
    for arch, path in WEIGHT_PATHS.items():
        if os.path.exists(path):
            try:
                m = build_model(arch)
                m.load_state_dict(torch.load(path, map_location=DEVICE))
                m.eval()
                MODELS[arch] = m
                logger.info(f"Loaded {arch}")
            except Exception as e:
                logger.warning(f"Could not load {arch}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    MODELS.clear()


app = FastAPI(
    title="Dashcam Road Anomaly Detection API",
    description="Real-time vehicle safety system — ResNet50, EfficientNet-B0, MobileNetV3 on BDD100K.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(MODELS.keys()),
        device=str(DEVICE),
    )


@app.get("/model-info/{architecture}",
         response_model=ModelInfoResponse, tags=["System"])
async def model_info(architecture: str):
    if architecture not in MODELS:
        raise HTTPException(404, f"Model '{architecture}' not loaded.")
    model = MODELS[architecture]
    params = sum(p.numel() for p in model.parameters())
    return ModelInfoResponse(
        architecture=architecture,
        num_classes=len(CLASSES),
        classes=CLASSES,
        param_count=params,
        param_mb=round(params * 4 / 1024**2, 1),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    if req.architecture not in MODELS:
        raise HTTPException(400, f"Architecture '{req.architecture}' not available.")
    try:
        result = predict_image(MODELS[req.architecture],
                               req.image_base64,
                               req.confidence_threshold)
    except Exception as e:
        raise HTTPException(500, str(e))
    return PredictResponse(**result)


@app.post("/predict/upload",
          response_model=PredictResponse, tags=["Inference"])
async def predict_upload(
    file: UploadFile = File(...),
    architecture: str = "resnet50",
    confidence_threshold: float = 0.45,
):
    if architecture not in MODELS:
        raise HTTPException(400, f"Architecture '{architecture}' not available.")
    image_bytes = await file.read()
    result = predict_image(MODELS[architecture],
                           image_bytes, confidence_threshold)
    return PredictResponse(**result)


@app.post("/batch_predict",
          response_model=BatchPredictResponse, tags=["Inference"])
async def batch_predict_endpoint(req: BatchPredictRequest):
    import base64, io
    from PIL import Image
    if req.architecture not in MODELS:
        raise HTTPException(400, f"Architecture '{req.architecture}' not available.")
    frames = []
    for b64 in req.images_base64:
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        frames.append(np.array(pil)[:, :, ::-1])
    try:
        results = _batch_predict(
            MODELS[req.architecture],
            frames,
            req.confidence_threshold)
    except Exception as e:
        raise HTTPException(500, str(e))
    items = [
        BatchPredictItem(
            frame_index=r["frame_index"],
            predicted_class=r["class"],
            confidence=r["confidence"],
            latency_ms=r["latency_ms"],
            is_anomaly=r["class"] != "normal",
            severity=_severity(r["class"]),
        ) for r in results
    ]
    return BatchPredictResponse(
        results=items,
        total_frames=len(items),
        anomaly_count=sum(1 for i in items if i.is_anomaly),
        avg_latency_ms=round(
            sum(i.latency_ms for i in items) / max(len(items), 1), 2),
        architecture=req.architecture,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)