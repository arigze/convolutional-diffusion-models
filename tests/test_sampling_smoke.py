from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from config import load_config
from models.ddim import DDIMDiffusion


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class DummyBackbone(nn.Module):
    """
    Simple epsilon-predictor stub used only to validate wrapper behavior.
    Returns zeros with the same shape as x_t.
    """

    def __init__(self):
        super().__init__()
        self.last_x_shape = None
        self.last_t_shape = None
        self.last_labels_shape = None
        self.call_count = 0

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, labels=None) -> torch.Tensor:
        self.call_count += 1
        self.last_x_shape = tuple(x_t.shape)
        self.last_t_shape = tuple(t.shape)
        self.last_labels_shape = None if labels is None else tuple(labels.shape)
        return torch.zeros_like(x_t)


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_mnist_sample_shape_smoke(device: str):
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    x = diffusion.sample(batch_size=4, seed=123, device=device)

    assert x.shape == (4, 1, 28, 28)
    assert x.device.type == device
    assert backbone.call_count == cfg.diffusion.sampler.steps
    assert backbone.last_labels_shape is None


@pytest.mark.parametrize("device", _available_devices())
def test_cifar10_sample_shape_smoke(device: str):
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    x = diffusion.sample(batch_size=4, seed=123, labels=labels, device=device)

    assert x.shape == (4, 3, 32, 32)
    assert x.device.type == device
    assert backbone.call_count == cfg.diffusion.sampler.steps
    assert backbone.last_labels_shape == (4,)


@pytest.mark.parametrize("device", _available_devices())
def test_conditional_labels_required(device: str):
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    with pytest.raises(ValueError, match="requires labels"):
        diffusion.sample(batch_size=4, seed=123, device=device)


@pytest.mark.parametrize("device", _available_devices())
def test_unconditional_rejects_labels(device: str):
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="must not receive labels"):
        diffusion.sample(batch_size=4, seed=123, labels=labels, device=device)


@pytest.mark.parametrize("device", _available_devices())
def test_conditional_accepts_label_shape_b1(device: str):
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    labels = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
    x = diffusion.sample(batch_size=4, seed=123, labels=labels, device=device)

    assert x.shape == (4, 3, 32, 32)
    assert backbone.last_labels_shape == (4,)


@pytest.mark.parametrize("device", _available_devices())
def test_deterministic_sampling_same_seed(device: str):
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    x1 = diffusion.sample(batch_size=4, seed=999, device=device)
    x2 = diffusion.sample(batch_size=4, seed=999, device=device)

    assert torch.allclose(x1, x2)


@pytest.mark.parametrize("device", _available_devices())
def test_different_seeds_produce_different_outputs(device: str):
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    x1 = diffusion.sample(batch_size=4, seed=111, device=device)
    x2 = diffusion.sample(batch_size=4, seed=222, device=device)

    assert not torch.allclose(x1, x2)


@pytest.mark.parametrize("device", _available_devices())
def test_q_sample_shape(device: str):
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    x0 = torch.randn(4, 1, 28, 28, device=device)
    t = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    xt = diffusion.q_sample(x0, t)

    assert xt.shape == x0.shape
    assert xt.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_forward_delegates_to_backbone(device: str):
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    backbone = DummyBackbone().to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    x_t = torch.randn(4, 3, 32, 32, device=device)
    t = torch.tensor([5, 6, 7, 8], device=device, dtype=torch.long)
    labels = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)

    eps = diffusion(x_t, t, labels=labels)

    assert eps.shape == x_t.shape
    assert backbone.last_x_shape == (4, 3, 32, 32)
    assert backbone.last_t_shape == (4,)
    assert backbone.last_labels_shape == (4,)