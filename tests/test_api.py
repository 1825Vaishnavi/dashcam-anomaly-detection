"""tests/test_api.py"""

import base64, io, sys, os
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))


def make_b64(w=224, h=224):
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def client():
    import torch
    mock = MagicMock()
    mock.return_value = torch.tensor([[3.0, 0.5, 1.0, 0.3, 0.2, 0.1]])
    mock.parameters = lambda: iter([torch.zeros(1)])
    with patch.dict("sys.modules", {"train": MagicMock(),
                                     "video_inference": MagicMock()}):
        from main import app, MODELS
        MODELS["resnet50"] = mock
        from fastapi.testclient import TestClient
        yield TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_200(client):
    r = client.post("/predict", json={
        "image_base64": make_b64(), "architecture": "resnet50"})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert "severity" in data


def test_predict_bad_arch(client):
    r = client.post("/predict", json={
        "image_base64": make_b64(), "architecture": "unknown"})
    assert r.status_code == 400


def test_batch_predict_count(client):
    n = 4
    r = client.post("/batch_predict", json={
        "images_base64": [make_b64() for _ in range(n)],
        "architecture": "resnet50"})
    assert r.status_code == 200
    assert r.json()["total_frames"] == n