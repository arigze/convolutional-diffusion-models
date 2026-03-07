from __future__ import annotations

import math
from dataclasses import is_dataclass

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.ndim == 2:
        if timesteps.shape[1] != 1:
            raise ValueError(
                f"Expected timesteps with shape [B] or [B, 1], got {tuple(timesteps.shape)}."
            )
        timesteps = timesteps[:, 0]
    elif timesteps.ndim != 1:
        raise ValueError(
            f"Expected timesteps with shape [B] or [B, 1], got {tuple(timesteps.shape)}."
        )

    batch_size = timesteps.shape[0]
    device = timesteps.device

    half_dim = embedding_dim // 2
    if half_dim == 0:
        raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}.")

    timesteps = timesteps.to(torch.float32)

    freq_exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=device,
    ) / half_dim
    freqs = torch.exp(freq_exponent)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if embedding_dim % 2 == 1:
        emb = torch.cat(
            [emb, torch.zeros(batch_size, 1, device=device, dtype=emb.dtype)],
            dim=1,
        )

    return emb


class TimestepLabelEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditional: bool,
        num_classes: int | None = None,
        max_period: int = 10000,
    ) -> None:
        super().__init__()

        if embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}.")
        if conditional and num_classes is None:
            raise ValueError("Conditional embedding requires num_classes.")
        if not conditional and num_classes is not None:
            raise ValueError("Unconditional embedding must not specify num_classes.")

        self.embedding_dim = embedding_dim
        self.conditional = conditional
        self.num_classes = num_classes
        self.max_period = max_period

        self.class_embedding: nn.Embedding | None
        if self.conditional:
            self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        else:
            self.class_embedding = None

    @classmethod
    def from_config(cls, cfg) -> "TimestepLabelEmbedding":
        try:
            embedding_dim = cfg.model.embedding_dim
            conditional = cfg.dataset.conditional
            num_classes = cfg.dataset.num_classes
        except AttributeError as exc:
            raise TypeError(
                "Config object must define cfg.model.embedding_dim, "
                "cfg.dataset.conditional, and cfg.dataset.num_classes."
            ) from exc

        return cls(
            embedding_dim=embedding_dim,
            conditional=conditional,
            num_classes=num_classes,
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_emb = sinusoidal_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.embedding_dim,
            max_period=self.max_period,
        )

        if not self.conditional:
            if labels is not None:
                raise ValueError("Unconditional embedding received labels, but conditional=False.")
            return time_emb

        if labels is None:
            raise ValueError("Conditional embedding requires labels, but labels=None was provided.")

        if labels.ndim == 2:
            if labels.shape[1] != 1:
                raise ValueError(
                    f"Expected labels with shape [B] or [B, 1], got {tuple(labels.shape)}."
                )
            labels = labels[:, 0]
        elif labels.ndim != 1:
            raise ValueError(
                f"Expected labels with shape [B] or [B, 1], got {tuple(labels.shape)}."
            )

        labels = labels.to(device=time_emb.device, dtype=torch.long)
        class_emb = self.class_embedding(labels)
        return time_emb + class_emb