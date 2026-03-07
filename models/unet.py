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
    Single conv layer with embedding injection.

    Matches the author's UBlock design: the embedding is projected and added
    to the input feature map BEFORE the convolution runs, so the conv sees
    the time-conditioned activations directly.

    Author's pattern:
        return self.model(x + self.emb(embedding)[:,:,None,None])

    Previous (wrong) pattern:
        h = conv(x)
        return h + proj(emb)   # embedding added AFTER conv — incorrect
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        kernel_size: int = 3,
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
        # FIX: ReLU before the Linear matches the author's
        #   self.emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, infeatures))
        # The previous bare nn.Linear had no nonlinearity, which severely
        # limits how expressively the network can condition on time.
        self.embedding_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim, in_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim != 2:
            raise ValueError(f"emb must have shape [B, D], got {tuple(emb.shape)}")
        if x.shape[0] != emb.shape[0]:
            raise ValueError(
                f"x and emb batch sizes must match: got {x.shape[0]} vs {emb.shape[0]}"
            )

        # FIX: inject embedding into x BEFORE the conv, not after.
        # The author does: self.model(x + self.emb(embedding)[:,:,None,None])
        emb_bias = self.embedding_proj(emb).to(dtype=x.dtype)
        h = self.conv(x + emb_bias[:, :, None, None])
        return h


class ResidualUBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        padding: str,
        residual: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvWithEmbedding(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            kernel_size=3,
            padding_mode=padding,
        )
        self.conv2 = ConvWithEmbedding(
            in_channels=out_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            kernel_size=3,
            padding_mode=padding,
        )
        self.activation = nn.ReLU()
        self.skip = None
        if residual and in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.conv1(x, emb))
        h = self.activation(self.conv2(h, emb))

        if self.residual:
            residual = x if self.skip is None else self.skip(x)
            h = h + residual
        return h


class UpBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        embedding_dim: int,
        padding: str,
        residual: bool,
    ) -> None:
        super().__init__()
        self.block = ResidualUBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != skip.shape[0]:
            raise ValueError(
                f"x and skip batch sizes must match: {x.shape[0]} vs {skip.shape[0]}"
            )
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Upsampled tensor spatial shape must match skip shape, got {tuple(x.shape[2:])} "
                f"vs {tuple(skip.shape[2:])}"
            )
        h = torch.cat([x, skip], dim=1)
        return self.block(h, emb)


class MinimalUNet(nn.Module):
    """
    Minimal three-scale UNet for the paper's CNN diffusion experiments.

    Fixes applied vs original:
      1. enc3 skip connection: enc3 output is now saved and used in up3
         (previously it was computed but silently discarded).
      2. Embedding injection order: embedding is now added to x BEFORE
         the conv (matching the author), not added to the conv output after.
      3. Embedding projection: now uses ReLU before the Linear
         (matching the author's nn.Sequential(nn.ReLU(), nn.Linear(...))),
         giving the network nonlinear time conditioning.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        channel_mults: list[int],
        embedding_dim: int,
        conditional: bool,
        num_classes: int | None,
        padding: str,
        residual: bool,
    ) -> None:
        super().__init__()

        if len(channel_mults) != 3:
            raise ValueError(
                f"MinimalUNet expects exactly 3 channel sizes, got {channel_mults}"
            )
        if any(c <= 0 for c in channel_mults):
            raise ValueError(f"All UNet channel sizes must be positive, got {channel_mults}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        c1, c2, c3 = channel_mults
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_mults = list(channel_mults)
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.residual = residual
        self.conditional = conditional
        self.num_classes = num_classes

        self.embedding = TimestepLabelEmbedding(
            embedding_dim=embedding_dim,
            conditional=conditional,
            num_classes=num_classes,
        )

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ResidualUBlock(
            in_channels=in_channels,
            out_channels=c1,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )
        self.enc2 = ResidualUBlock(
            in_channels=c1,
            out_channels=c2,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )
        self.enc3 = ResidualUBlock(
            in_channels=c2,
            out_channels=c3,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )
        self.bottleneck = ResidualUBlock(
            in_channels=c3,
            out_channels=c3,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )

        self.up3 = UpBlock(
            in_channels=c3,
            skip_channels=c3,
            out_channels=c3,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )

        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up2 = UpBlock(
            in_channels=c2,
            skip_channels=c2,
            out_channels=c2,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )

        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.up1 = UpBlock(
            in_channels=c1,
            skip_channels=c1,
            out_channels=c1,
            embedding_dim=embedding_dim,
            padding=padding,
            residual=residual,
        )

        self.final_conv = nn.Conv2d(
            in_channels=c1,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=_padding_mode_to_torch(padding),
            bias=True,
        )

    @classmethod
    def from_config(cls, cfg: FullConfig) -> "MinimalUNet":
        if not is_dataclass(cfg):
            raise TypeError("cfg must be a FullConfig dataclass instance")

        if cfg.model.architecture != "unet":
            raise ValueError(
                f"MinimalUNet.from_config expected architecture='unet', "
                f"got {cfg.model.architecture!r}"
            )
        if cfg.model.channel_mults is None:
            raise ValueError("UNet config missing channel_mults")
        if cfg.model.residual is None:
            raise ValueError("UNet config missing residual")
        if cfg.model.downsample != "maxpool":
            raise ValueError(f"Only downsample='maxpool' is supported, got {cfg.model.downsample!r}")
        if cfg.model.upsample != "transpose_conv":
            raise ValueError(f"Only upsample='transpose_conv' is supported, got {cfg.model.upsample!r}")
        if cfg.model.skip_connection != "concat":
            raise ValueError(
                f"Only skip_connection='concat' is supported, got {cfg.model.skip_connection!r}"
            )
        if cfg.model.embedding_injection != "every_block":
            raise ValueError(
                f"Only embedding_injection='every_block' is supported, got {cfg.model.embedding_injection!r}"
            )

        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            channel_mults=cfg.model.channel_mults,
            embedding_dim=cfg.model.embedding_dim,
            conditional=cfg.dataset.conditional,
            num_classes=cfg.dataset.num_classes,
            padding=cfg.model.padding,
            residual=cfg.model.residual,
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
                f"x_t channel count must match in_channels={self.in_channels}, got {x_t.shape[1]}"
            )

        batch_size = x_t.shape[0]
        device = x_t.device
        t = _normalize_timesteps(t, batch_size=batch_size, device=device)
        emb = self.embedding(timesteps=t, labels=labels)

        skip1 = self.enc1(x_t, emb)
        h = self.downsample(skip1)

        skip2 = self.enc2(h, emb)
        h = self.downsample(skip2)

        skip3 = self.enc3(h, emb)
        h = self.bottleneck(skip3, emb)
        h = self.up3(h, skip3, emb)

        h = self.upconv2(h)
        h = self.up2(h, skip2, emb)

        h = self.upconv1(h)
        h = self.up1(h, skip1, emb)

        out = self.final_conv(h)
        expected_shape = x_t.shape[:1] + (self.out_channels,) + x_t.shape[2:]
        if out.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected output shape {tuple(out.shape)} for input shape {tuple(x_t.shape)}"
            )
        return out