from __future__ import annotations

from dataclasses import is_dataclass

import torch
import torch.nn as nn

from config import FullConfig
from models.embeddings import TimestepLabelEmbedding


def _normalize_timesteps(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if t.ndim == 2:
        if t.shape[1] != 1:
            raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")
        t = t[:, 0]
    elif t.ndim != 1:
        raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")

    if t.shape[0] != batch_size:
        raise ValueError(
            f"t batch size must match x batch size: got {t.shape[0]} vs {batch_size}"
        )

    return t.to(device=device)


def _padding_mode_to_torch(mode: str) -> str:
    if mode == "zeros":
        return "zeros"
    if mode == "circular":
        return "circular"
    raise ValueError(f"Unsupported padding mode: {mode!r}")


class ConvWithEmbedding(nn.Module):
    """
    Single conv layer with additive timestep/class embedding injection.

    Given:
        x:   [B, C_in, H, W]
        emb: [B, D]

    Computes:
        conv(x) + proj(emb)[:, :, None, None]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        kernel_size: int,
        padding_mode: str,
    ) -> None:
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=_padding_mode_to_torch(padding_mode),
            bias=True,
        )
        self.embedding_proj = nn.Linear(embedding_dim, out_channels, bias=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim != 2:
            raise ValueError(f"emb must have shape [B, D], got {tuple(emb.shape)}")
        if x.shape[0] != emb.shape[0]:
            raise ValueError(
                f"x and emb batch sizes must match: got {x.shape[0]} vs {emb.shape[0]}"
            )

        h = self.conv(x)
        emb_bias = self.embedding_proj(emb).to(dtype=h.dtype)
        return h + emb_bias[:, :, None, None]


class MinimalResNet(nn.Module):
    """
    Minimal ResNet backbone for the paper's convolution-only diffusion experiments.

    Structure (App. C.1):
        input projection conv   (in_channels -> hidden_channels)  + residual via 1x1 projection
        6 intermediate residual conv layers                        (hidden_channels -> hidden_channels)
        output projection conv  (hidden_channels -> out_channels)  no residual, no activation

    Design:
        - hidden width = 256
        - kernel size = 3
        - embedding injected into every layer
        - no normalization
        - zero or circular padding
        - output has same shape as input
        - residual connections on ALL layers (input projection uses a 1x1 conv
          shortcut when in_channels != hidden_channels, matching He et al. 2016)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_mid_layers: int,
        embedding_dim: int,
        conditional: bool,
        num_classes: int | None,
        padding: str,
    ) -> None:
        super().__init__()

        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if num_mid_layers <= 0:
            raise ValueError(f"num_mid_layers must be positive, got {num_mid_layers}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_mid_layers = num_mid_layers
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.conditional = conditional
        self.num_classes = num_classes

        self.embedding = TimestepLabelEmbedding(
            embedding_dim=embedding_dim,
            conditional=conditional,
            num_classes=num_classes,
        )

        self.input_layer = ConvWithEmbedding(
            in_channels=in_channels,
            out_channels=hidden_channels,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            padding_mode=padding,
        )

        # FIX: 1x1 projection shortcut for the input layer residual connection.
        # The input layer changes channel depth (e.g. 1->256 for MNIST, 3->256
        # for CIFAR10), so a direct identity shortcut is impossible. A 1x1 conv
        # projects the input to hidden_channels so the residual can be added,
        # exactly as in He et al. (2016). Previously the input layer had NO
        # residual connection at all, inconsistent with the paper's description
        # of residual connections between layers.
        self.input_skip: nn.Module
        if in_channels != hidden_channels:
            self.input_skip = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        else:
            self.input_skip = nn.Identity()

        self.mid_layers = nn.ModuleList(
            [
                ConvWithEmbedding(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    padding_mode=padding,
                )
                for _ in range(num_mid_layers)
            ]
        )

        # Output layer: projects back to out_channels. No residual or activation —
        # the raw predicted noise should be unconstrained.
        self.output_layer = ConvWithEmbedding(
            in_channels=hidden_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            padding_mode=padding,
        )

        self.activation = nn.ReLU()

    @classmethod
    def from_config(cls, cfg: FullConfig) -> "MinimalResNet":
        if not is_dataclass(cfg):
            raise TypeError("cfg must be a FullConfig dataclass instance")

        if cfg.model.architecture != "resnet":
            raise ValueError(
                f"MinimalResNet.from_config expected architecture='resnet', "
                f"got {cfg.model.architecture!r}"
            )

        if cfg.model.hidden_channels is None:
            raise ValueError("ResNet config missing hidden_channels")
        if cfg.model.kernel_size is None:
            raise ValueError("ResNet config missing kernel_size")
        if cfg.model.num_mid_layers is None:
            raise ValueError("ResNet config missing num_mid_layers")

        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            hidden_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            num_mid_layers=cfg.model.num_mid_layers,
            embedding_dim=cfg.model.embedding_dim,
            conditional=cfg.dataset.conditional,
            num_classes=cfg.dataset.num_classes,
            padding=cfg.model.padding,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"x_t must have shape [B, C, H, W], got {tuple(x_t.shape)}")
        if x_t.shape[1] != self.in_channels:
            raise ValueError(
                f"x_t channel count must match in_channels={self.in_channels}, "
                f"got {x_t.shape[1]}"
            )

        batch_size = x_t.shape[0]
        device = x_t.device

        t = _normalize_timesteps(t, batch_size=batch_size, device=device)
        emb = self.embedding(timesteps=t, labels=labels)

        # Input layer with residual via 1x1 projection shortcut
        # FIX: previously there was no residual here at all.
        delta = self.input_layer(x_t, emb)
        delta = self.activation(delta)
        h = delta + self.input_skip(x_t)

        # 6 intermediate residual layers (channels constant, identity shortcut)
        for layer in self.mid_layers:
            delta = layer(h, emb)
            delta = self.activation(delta)
            h = h + delta

        # Output projection — no residual, no activation
        out = self.output_layer(h, emb)

        if out.shape != x_t.shape[:1] + (self.out_channels,) + x_t.shape[2:]:
            raise RuntimeError(
                f"Unexpected output shape {tuple(out.shape)} for input shape {tuple(x_t.shape)}"
            )

        return out