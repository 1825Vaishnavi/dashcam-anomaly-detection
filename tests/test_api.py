"""tests/test_api.py"""

import base64, io, sys, os
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))

from data_processing import CLASSES, CLASS_TO_IDX
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def make_b64(w=224, h=224):
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def make_predict_result():
    return {
        "predicted_class":     "normal",
        "confidence":          0.91,
        "latency_ms":          38.4,
        "class_probabilities": {c: round(1/len(CLASSES), 4)
                                 for c in CLASSES},
        "is_anomaly":          False,
        "severity":            "NONE",
    }


def make_batch_results(n):
    return [
        {"frame_index": i, "class": "normal",
         "confidence": 0.9, "latency_ms": 30.0}
        for i in range(n)
    ]


@pytest.fixture
def client():
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor(
        [[3.0, 0.5, 1.0, 0.3, 0.2, 0.1]])
    mock_model.parameters = lambda: iter([torch.zeros(1)])

    def dynamic_batch_predict(model, frames, confidence_threshold=0.45):
        return make_batch_results(len(frames))

    with patch("predict.predict_image",
               return_value=make_predict_result()), \
         patch("main._batch_predict",
               side_effect=dynamic_batch_predict):

        from main import app, MODELS
        MODELS.clear()
        MODELS["resnet50"] = mock_model

        from fastapi.testclient import TestClient
        client = TestClient(app)
        yield client
        MODELS.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "resnet50" in r.json()["models_loaded"]


def test_predict_200(client):
    r = client.post("/predict", json={
        "image_base64": make_b64(),
        "architecture": "resnet50"})
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] in CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] > 0
    assert data["severity"] in ["NONE", "LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["is_anomaly"], bool)


def test_predict_bad_arch(client):
    r = client.post("/predict", json={
        "image_base64": make_b64(),
        "architecture": "unknown_arch"})
    assert r.status_code == 400


def test_predict_confidence_range(client):
    r = client.post("/predict", json={
        "image_base64": make_b64(),
        "architecture": "resnet50"})
    assert 0.0 <= r.json()["confidence"] <= 1.0


def test_predict_class_probabilities(client):
    r = client.post("/predict", json={
        "image_base64": make_b64(),
        "architecture": "resnet50"})
    probs = r.json()["class_probabilities"]
    assert set(probs.keys()) == set(CLASSES)


def test_batch_predict_count(client):
    n = 4
    r = client.post("/batch_predict", json={
        "images_base64": [make_b64() for _ in range(n)],
        "architecture":  "resnet50"})
    assert r.status_code == 200
    assert r.json()["total_frames"] == n


def test_batch_predict_anomaly_count(client):
    n = 5
    r = client.post("/batch_predict", json={
        "images_base64": [make_b64() for _ in range(n)],
        "architecture":  "resnet50"})
    assert r.status_code == 200
    data = r.json()
    assert "anomaly_count" in data
    assert 0 <= data["anomaly_count"] <= data["total_frames"]


def test_batch_predict_returns_all_fields(client):
    r = client.post("/batch_predict", json={
        "images_base64": [make_b64()],
        "architecture":  "resnet50"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "total_frames" in data
    assert "anomaly_count" in data
    assert "avg_latency_ms" in data
    assert "architecture" in data


def test_health_device_present(client):
    r = client.get("/health")
    assert "device" in r.json()
    assert r.json()["device"] in ["cpu", "cuda"]