from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import load_config
from models.ddim import DDIMDiffusion
from models.resnet import MinimalResNet


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("device", _available_devices())
def test_mnist_resnet_forward_shape(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    model = MinimalResNet.from_config(cfg).to(device)

    x = torch.randn(4, 1, 28, 28, device=device)
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    out = model(x, t)

    assert out.shape == x.shape
    assert out.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_cifar10_resnet_forward_shape(device: str) -> None:
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    model = MinimalResNet.from_config(cfg).to(device)

    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.tensor([10, 20, 30, 40], dtype=torch.long, device=device)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)

    out = model(x, t, labels=y)

    assert out.shape == x.shape
    assert out.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_resnet_accepts_b1_timestep_shape(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    model = MinimalResNet.from_config(cfg).to(device)

    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.tensor([[5], [6]], dtype=torch.long, device=device)

    out = model(x, t)

    assert out.shape == x.shape


@pytest.mark.parametrize("device", _available_devices())
def test_resnet_backward_pass(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    model = MinimalResNet.from_config(cfg).to(device)

    x = torch.randn(4, 1, 28, 28, device=device)
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    out = model(x, t)
    loss = out.square().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)


def test_mnist_resnet_parameter_count_sanity() -> None:
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    model = MinimalResNet.from_config(cfg)

    nparams = _count_params(model)

    # Wide enough to avoid brittleness, narrow enough to catch wiring mistakes.
    assert 3_500_000 <= nparams <= 4_500_000


def test_cifar10_resnet_parameter_count_sanity() -> None:
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    model = MinimalResNet.from_config(cfg)

    nparams = _count_params(model)

    assert 3_500_000 <= nparams <= 4_500_000


@pytest.mark.parametrize("device", _available_devices())
def test_resnet_sampling_smoke_mnist(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_resnet.yaml")
    backbone = MinimalResNet.from_config(cfg).to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    x = diffusion.sample(batch_size=2, seed=123, device=device)

    assert x.shape == (2, 1, 28, 28)
    assert x.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_resnet_sampling_smoke_cifar10(device: str) -> None:
    cfg = load_config(CONFIGS / "cifar10_resnet.yaml")
    backbone = MinimalResNet.from_config(cfg).to(device)
    diffusion = DDIMDiffusion.from_config(backbone, cfg).to(device)

    labels = torch.tensor([1, 7], dtype=torch.long, device=device)
    x = diffusion.sample(batch_size=2, seed=123, labels=labels, device=device)

    assert x.shape == (2, 3, 32, 32)
    assert x.device.type == device