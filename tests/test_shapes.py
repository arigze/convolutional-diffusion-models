from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.embeddings import TimestepLabelEmbedding, sinusoidal_timestep_embedding


def _make_cfg(*, embedding_dim: int, conditional: bool, num_classes: int | None):
    return SimpleNamespace(
        model=SimpleNamespace(embedding_dim=embedding_dim),
        dataset=SimpleNamespace(
            conditional=conditional,
            num_classes=num_classes,
        ),
    )


def test_sinusoidal_timestep_embedding_shape_cpu() -> None:
    t = torch.tensor([0, 1, 2, 999], dtype=torch.long)
    emb = sinusoidal_timestep_embedding(t, embedding_dim=256)
    assert emb.shape == (4, 256)
    assert emb.device == t.device


def test_sinusoidal_timestep_embedding_accepts_b1_shape() -> None:
    t = torch.tensor([[0], [1], [2]], dtype=torch.long)
    emb = sinusoidal_timestep_embedding(t, embedding_dim=256)
    assert emb.shape == (3, 256)


def test_sinusoidal_timestep_embedding_odd_dimension() -> None:
    t = torch.tensor([0, 1], dtype=torch.long)
    emb = sinusoidal_timestep_embedding(t, embedding_dim=255)
    assert emb.shape == (2, 255)


def test_unconditional_forward_shape() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=False,
        num_classes=None,
    )
    t = torch.tensor([0, 10, 25, 999], dtype=torch.long)
    out = module(timesteps=t)
    assert out.shape == (4, 256)


def test_conditional_forward_shape() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=True,
        num_classes=10,
    )
    t = torch.tensor([0, 10, 25, 999], dtype=torch.long)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    out = module(timesteps=t, labels=y)
    assert out.shape == (4, 256)


def test_conditional_forward_accepts_b1_labels() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=True,
        num_classes=10,
    )
    t = torch.tensor([0, 1, 2], dtype=torch.long)
    y = torch.tensor([[1], [2], [3]], dtype=torch.long)
    out = module(timesteps=t, labels=y)
    assert out.shape == (3, 256)


def test_unconditional_rejects_labels() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=False,
        num_classes=None,
    )
    t = torch.tensor([0, 1], dtype=torch.long)
    y = torch.tensor([1, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="received labels"):
        _ = module(timesteps=t, labels=y)


def test_conditional_requires_labels() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=True,
        num_classes=10,
    )
    t = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="requires labels"):
        _ = module(timesteps=t, labels=None)


def test_same_input_same_output_unconditional() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=False,
        num_classes=None,
    )
    t = torch.tensor([5, 10, 15], dtype=torch.long)

    out1 = module(timesteps=t)
    out2 = module(timesteps=t)

    assert torch.allclose(out1, out2)


def test_same_input_same_output_conditional() -> None:
    torch.manual_seed(0)
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=True,
        num_classes=10,
    )
    t = torch.tensor([5, 10, 15], dtype=torch.long)
    y = torch.tensor([1, 2, 3], dtype=torch.long)

    out1 = module(timesteps=t, labels=y)
    out2 = module(timesteps=t, labels=y)

    assert torch.allclose(out1, out2)


def test_from_config_unconditional() -> None:
    cfg = _make_cfg(embedding_dim=256, conditional=False, num_classes=None)
    module = TimestepLabelEmbedding.from_config(cfg)
    t = torch.tensor([0, 1], dtype=torch.long)
    out = module(timesteps=t)
    assert out.shape == (2, 256)


def test_from_config_conditional() -> None:
    cfg = _make_cfg(embedding_dim=256, conditional=True, num_classes=10)
    module = TimestepLabelEmbedding.from_config(cfg)
    t = torch.tensor([0, 1], dtype=torch.long)
    y = torch.tensor([3, 7], dtype=torch.long)
    out = module(timesteps=t, labels=y)
    assert out.shape == (2, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unconditional_forward_gpu_device_match() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=False,
        num_classes=None,
    ).cuda()
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
    out = module(timesteps=t)
    assert out.shape == (4, 256)
    assert out.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_conditional_forward_gpu_device_match() -> None:
    module = TimestepLabelEmbedding(
        embedding_dim=256,
        conditional=True,
        num_classes=10,
    ).cuda()
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long, device="cuda")
    out = module(timesteps=t, labels=y)
    assert out.shape == (4, 256)
    assert out.device.type == "cuda"