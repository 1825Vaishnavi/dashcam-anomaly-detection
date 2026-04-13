"""tests/test_model.py"""

import time, sys, os
import pytest
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from train import build_model, DEVICE, NUM_CLASSES

ARCHS = ["resnet50", "efficientnet_b0", "mobilenet_v3"]


@pytest.mark.parametrize("arch", ARCHS)
def test_output_shape(arch):
    model = build_model(arch)
    out = model(torch.randn(2, 3, 224, 224).to(DEVICE))
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("arch", ARCHS)
def test_output_finite(arch):
    model = build_model(arch)
    out = model(torch.randn(1, 3, 224, 224).to(DEVICE))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCHS)
def test_softmax_sums_to_one(arch):
    model = build_model(arch)
    logits = model(torch.randn(4, 3, 224, 224).to(DEVICE))
    probs = torch.softmax(logits, dim=1).sum(dim=1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_mobilenet_smaller_than_resnet():
    def count(arch):
        return sum(p.numel() for p in build_model(arch).parameters())
    assert count("mobilenet_v3") < count("resnet50")