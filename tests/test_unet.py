from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import load_config
from models.ddim import DDIMDiffusion
from models.unet import MinimalUNet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_mnist_unet_forward_shape(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)

    x = torch.randn(4, 1, 28, 28, device=device)
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    out = model(x, t)

    assert out.shape == x.shape
    assert out.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_cifar10_unet_forward_shape(device: str) -> None:
    cfg = load_config(CONFIGS / "cifar10_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)

    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    out = model(x, t, labels=y)

    assert out.shape == x.shape
    assert out.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_unet_accepts_t_shape_b1(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)

    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.tensor([[7], [8]], dtype=torch.long, device=device)
    out = model(x, t)

    assert out.shape == x.shape


@pytest.mark.parametrize("device", _available_devices())
def test_skip_shape_alignment(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)

    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.tensor([5, 6], dtype=torch.long, device=device)

    captures = {}

    def save_shape(name):
        def _hook(_module, inputs, output):
            captures[name] = {
                "input_shapes": [tuple(inp.shape) for inp in inputs if isinstance(inp, torch.Tensor)],
                "output_shape": tuple(output.shape),
            }
        return _hook

    h1 = model.enc2.register_forward_hook(save_shape("enc2"))
    h2 = model.upconv2.register_forward_hook(save_shape("upconv2"))
    h3 = model.up2.register_forward_hook(save_shape("up2"))
    h4 = model.enc1.register_forward_hook(save_shape("enc1"))
    h5 = model.upconv1.register_forward_hook(save_shape("upconv1"))
    h6 = model.up1.register_forward_hook(save_shape("up1"))

    try:
        out = model(x, t)
    finally:
        for h in [h1, h2, h3, h4, h5, h6]:
            h.remove()

    assert out.shape == x.shape
    assert captures["enc2"]["output_shape"][2:] == captures["upconv2"]["output_shape"][2:]
    assert captures["enc1"]["output_shape"][2:] == captures["upconv1"]["output_shape"][2:]

    up2_inputs = captures["up2"]["input_shapes"]
    up1_inputs = captures["up1"]["input_shapes"]
    assert up2_inputs[0][2:] == up2_inputs[1][2:]
    assert up1_inputs[0][2:] == up1_inputs[1][2:]


@pytest.mark.parametrize("device", _available_devices())
def test_unet_backward_pass(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)

    x = torch.randn(4, 1, 28, 28, device=device)
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    target = torch.randn_like(x)

    out = model(x, t)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


@pytest.mark.parametrize("device", _available_devices())
def test_mnist_unet_sampling_smoke(device: str) -> None:
    cfg = load_config(CONFIGS / "mnist_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)
    diffusion = DDIMDiffusion.from_config(model, cfg).to(device)

    x = diffusion.sample(batch_size=2, seed=123, device=device)
    assert x.shape == (2, 1, 28, 28)
    assert x.device.type == device


@pytest.mark.parametrize("device", _available_devices())
def test_cifar10_unet_sampling_smoke(device: str) -> None:
    cfg = load_config(CONFIGS / "cifar10_unet.yaml")
    model = MinimalUNet.from_config(cfg).to(device)
    diffusion = DDIMDiffusion.from_config(model, cfg).to(device)

    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    x = diffusion.sample(batch_size=2, seed=123, labels=labels, device=device)
    assert x.shape == (2, 3, 32, 32)
    assert x.device.type == device