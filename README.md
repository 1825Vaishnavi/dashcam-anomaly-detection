#  Vehicle Dashcam Road Anomaly Detection

![CI/CD](https://github.com/1825Vaishnavi/dashcam-anomaly-detection/actions/workflows/ci_cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)
![Tests](https://img.shields.io/badge/Tests-63%20passing-green)

Real-time computer vision pipeline for classifying road anomalies from dashcam footage + vehicle health monitoring system built for production vehicle safety systems.

---

##  Project Highlights

- **3 CNN architectures** fine-tuned on BDD100K - ResNet50 achieving **84% accuracy, 0.91 AUC**
- **Real-time 30fps** video inference pipeline with OpenCV
- **Sub-100ms** per-frame latency suitable for vehicle safety systems
- **Vehicle diagnostics** - fuel leak detection, brake failure alerts, battery monitoring
- **63 tests passing** - unit, integration, performance, and robustness tests
- **Full MLOps** - MLflow tracking, Evidently AI drift detection, CI/CD pipeline

---

!<img width="1919" height="1016" alt="Screenshot 2026-04-13 022230" src="https://github.com/user-attachments/assets/c707e24a-c2be-4db6-9b78-c3aef3de7328" />
!<img width="1919" height="1022" alt="Screenshot 2026-04-13 022426" src="https://github.com/user-attachments/assets/ab2d7137-7d64-4e2c-bc1e-b7e45de5c9ae" />
!<img width="1875" height="1020" alt="Screenshot 2026-04-13 023240" src="https://github.com/user-attachments/assets/ac3f2814-18ff-4b5d-8b85-12c5e9ac366b" />




##  Architecture
BDD100K Dataset (100K dashcam images)
│
▼
data_processing.py    ← parse JSON labels → 5 anomaly classes
│
▼
train.py              ← fine-tune ResNet50 / EfficientNet-B0 / MobileNetV3
← MLflow experiment tracking + Model Registry
│
▼
video_inference.py    ← OpenCV 30fps frame pipeline + bounding box overlay
│
▼
api/main.py           ← FastAPI /predict + /batch_predict
│
▼
monitoring/           ← Evidently AI drift detection + Streamlit dashboard

---

## 📊 Model Results

| Model | Accuracy | AUC | Avg Latency | Size |
|-------|----------|-----|-------------|------|
| **ResNet50** ⭐ | **84%** | **0.91** | 42ms | 98MB |
| EfficientNet-B0 | 81% | 0.88 | 32ms | 21MB |
| MobileNetV3 | 77% | 0.84 | 18ms | 6MB |

All models achieve **sub-100ms per-frame latency** ✅

---

## 🚨 Vehicle Diagnostics System

Beyond dashcam detection, this project includes a real-time vehicle health monitoring system:

| Component | Monitored Metrics | Alert Levels |
|-----------|------------------|--------------|
| Engine | Temperature, RPM | WARNING → CRITICAL → EMERGENCY |
| Fuel | Level %, Pressure | Detects fuel leaks |
| Brakes | Pad wear, Temperature | Safety-critical alerts |
| Battery | Voltage, Charge % | Prevents unexpected stalls |
| Tires | PSI pressure | Blowout prevention |
| Oil | Pressure, Level | Engine protection |

```python
# Example: Fuel leak detection
diag = VehicleDiagnostics()
diag.process_reading(SensorReading(SystemComponent.FUEL, 1.5, "bar"))
# → [EMERGENCY] FUEL pressure_bar = 1.5bar
# → ACTION: STOP VEHICLE — possible fuel leak detected
```

---

## 🧪 Test Suite - 63 Tests Passing
tests/
├── test_api.py          # FastAPI endpoint tests
├── test_model.py        # CNN model shape/output tests
├── test_performance.py  # Real-time SLA, robustness, memory tests
└── test_diagnostics.py  # Vehicle health monitoring tests

Key tests:
-  Sub-100ms SLA per frame (real-time safety requirement)
-  Night driving (dark frame robustness)
-  Glare handling (overexposed frames)
-  Multi-resolution support (480p, 720p, 1080p)
-  Fuel leak detection
-  Brake failure alerts
-  Battery failure detection

---

##  Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train all 3 models
```bash
python src/train.py
```

### Run video inference
```python
from src.train import build_model
from src.video_inference import load_model, process_video

model = load_model("resnet50", "best_resnet50.pth")
stats = process_video(model, "dashcam.mp4", "output.mp4", target_fps=30)
```

### Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### Docker
```bash
docker compose up -d
```

### Run tests
```bash
pytest tests/ -v
```

---

##  API Endpoints

### POST /predict
```json
{
  "image_base64": "<base64-encoded JPEG>",
  "confidence_threshold": 0.45,
  "architecture": "resnet50"
}
```
Response:
```json
{
  "predicted_class": "pedestrian",
  "confidence": 0.872,
  "latency_ms": 38.4,
  "is_anomaly": true,
  "severity": "HIGH"
}
```

### POST /batch_predict
Classify multiple frames in one GPU forward pass.

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Models | ResNet50, EfficientNet-B0, MobileNetV3 |
| Training | PyTorch + torchvision |
| Video | OpenCV (30fps) |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI + Streamlit |
| Dataset | BDD100K (100K dashcam images) |

---

##  Project Structure
dashcam-anomaly-detection/
├── src/
│   ├── data_processing.py     # BDD100K parsing → 5 classes
│   ├── train.py               # Fine-tune 3 CNNs with MLflow
│   ├── evaluate.py            # Per-class metrics, AUC
│   ├── predict.py             # Single image prediction
│   ├── video_inference.py     # 30fps OpenCV pipeline
│   └── vehicle_diagnostics.py # Vehicle health monitoring
├── api/
│   ├── main.py                # FastAPI endpoints
│   └── schemas.py             # Pydantic schemas
├── monitoring/
│   ├── drift_detection.py     # Evidently AI
│   └── dashboard.py           # Streamlit dashboard
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_performance.py
│   └── test_diagnostics.py
├── .github/workflows/ci_cd.yml
├── Dockerfile
└── docker-compose.yml

---

##  Future Work
- Deploy on GCP Vertex AI for production scale
- Add real-time GPS integration for location-based alerts
- Implement model quantization for edge deployment
- Add night vision enhancement preprocessing
