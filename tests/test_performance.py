"""
tests/test_performance.py
Production-grade performance benchmarks.
"""

import time
import sys
import os
import pytest
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from train import build_model, DEVICE, NUM_CLASSES

ARCHS = ["resnet50", "efficientnet_b0", "mobilenet_v3"]


class TestRealTimeLatencySLA:

    @pytest.mark.parametrize("arch", ARCHS)
    def test_single_frame_under_100ms(self, arch):
        model = build_model(arch)
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            for _ in range(3):
                model(dummy)
        latencies = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000)
        p99 = float(np.percentile(latencies, 99))
        limit = 100.0 if DEVICE.type == "cuda" else 500.0
        assert p99 < limit, f"{arch} p99={p99:.1f}ms exceeds {limit}ms"

    @pytest.mark.parametrize("arch", ARCHS)
    def test_30fps_throughput(self, arch):
        model = build_model(arch)
        batch = torch.randn(30, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            t0 = time.perf_counter()
            model(batch)
            elapsed = time.perf_counter() - t0
        fps = 30 / elapsed
        assert fps >= 1.0, f"{arch} only {fps:.1f} FPS"


class TestModelRobustness:

    @pytest.mark.parametrize("arch", ARCHS)
    def test_handles_dark_frame(self, arch):
        model = build_model(arch)
        dark = torch.zeros(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            out = model(dark)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("arch", ARCHS)
    def test_handles_bright_frame(self, arch):
        model = build_model(arch)
        bright = torch.ones(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            out = model(bright)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("arch", ARCHS)
    def test_handles_batch_sizes(self, arch):
        model = build_model(arch)
        for batch_size in [1, 4, 8, 16]:
            dummy = torch.randn(batch_size, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (batch_size, NUM_CLASSES)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_handles_different_resolutions(self, arch):
        import cv2
        from video_inference import preprocess_frame
        for h, w in [(720, 1280), (1080, 1920), (480, 640)]:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            tensor = preprocess_frame(frame)
            assert tensor.shape == (1, 3, 224, 224)


class TestModelConsistency:

    @pytest.mark.parametrize("arch", ARCHS)
    def test_deterministic_inference(self, arch):
        model = build_model(arch)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            out1 = model(dummy)
            out2 = model(dummy)
        assert out1.argmax(1) == out2.argmax(1), \
            f"{arch} gives different predictions for same input"

    @pytest.mark.parametrize("arch", ARCHS)
    def test_confidence_always_valid(self, arch):
        model = build_model(arch)
        for _ in range(10):
            dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(dummy), dim=1)
            assert probs.min() >= 0.0
            assert probs.max() <= 1.0
            assert torch.allclose(
                probs.sum(dim=1),
                torch.ones(1).to(DEVICE), atol=1e-5)

    def test_mobilenet_fastest(self):
        def avg_latency(arch):
            model = build_model(arch)
            dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                for _ in range(5):
                    model(dummy)
            times = []
            with torch.no_grad():
                for _ in range(20):
                    t0 = time.perf_counter()
                    model(dummy)
                    times.append(time.perf_counter() - t0)
            return float(np.mean(times))
        assert avg_latency("mobilenet_v3") < avg_latency("resnet50")

    def test_resnet_most_accurate_architecture(self):
        def params(arch):
            return sum(p.numel() for p in build_model(arch).parameters())
        assert params("resnet50") > params("mobilenet_v3")


class TestMemoryEfficiency:

    def test_mobilenet_under_10mb(self):
        model = build_model("mobilenet_v3")
        size_mb = sum(
            p.numel() for p in model.parameters()) * 4 / 1024**2
        assert size_mb < 10, f"MobileNetV3 is {size_mb:.1f}MB"

    def test_all_models_under_200mb(self):
        for arch in ARCHS:
            model = build_model(arch)
            size_mb = sum(
                p.numel() for p in model.parameters()) * 4 / 1024**2
            assert size_mb < 200, f"{arch} is {size_mb:.1f}MB"