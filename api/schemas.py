"""
api/schemas.py
Pydantic request/response schemas for FastAPI endpoints.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    architecture: str = Field(default="resnet50")


class PredictResponse(BaseModel):
    predicted_class:     str
    confidence:          float
    latency_ms:          float
    class_probabilities: Dict[str, float]
    is_anomaly:          bool
    severity:            str


class BatchPredictRequest(BaseModel):
    images_base64: List[str]
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    architecture: str = Field(default="resnet50")


class BatchPredictItem(BaseModel):
    frame_index:    int
    predicted_class: str
    confidence:     float
    latency_ms:     float
    is_anomaly:     bool
    severity:       str


class BatchPredictResponse(BaseModel):
    results:        List[BatchPredictItem]
    total_frames:   int
    anomaly_count:  int
    avg_latency_ms: float
    architecture:   str


class HealthResponse(BaseModel):
    status:        str
    models_loaded: List[str]
    device:        str


class ModelInfoResponse(BaseModel):
    architecture: str
    num_classes:  int
    classes:      List[str]
    param_count:  int
    param_mb:     float