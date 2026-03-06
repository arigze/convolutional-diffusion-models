from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn

from config import FullConfig


def _normalize_labels(labels: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [B] or [B,1], got {tuple(labels.shape)}")
    if labels.shape[0] != batch_size:
        raise ValueError(
            f"labels batch size must match requested batch size: "
            f"got {labels.shape[0]} vs {batch_size}"
        )
    return labels.long()


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal style formulation.

    Returns:
        betas: shape [timesteps]
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(min=1e-8, max=max_beta)
    return betas.to(torch.float32)


@dataclass(frozen=True)
class DDIMShape:
    channels: int
    image_size: int


class DDIMDiffusion(nn.Module):
    """
    Config-driven diffusion wrapper for:
    - cosine noise schedule
    - epsilon prediction
    - deterministic DDIM sampling (eta=0)
    - optional class-conditional labels

    Backbone contract for Step 3:
        backbone(x_t, t, labels=None) -> eps_pred
    """

    def __init__(self, backbone: nn.Module, cfg: FullConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        if cfg.diffusion.prediction_type != "eps":
            raise ValueError(
                f"Only prediction_type='eps' is supported in Step 3, "
                f"got {cfg.diffusion.prediction_type!r}"
            )
        if cfg.diffusion.noise_schedule != "cosine":
            raise ValueError(
                f"Only noise_schedule='cosine' is supported in Step 3, "
                f"got {cfg.diffusion.noise_schedule!r}"
            )
        if cfg.diffusion.sampler.name != "ddim":
            raise ValueError(
                f"Only sampler='ddim' is supported in Step 3, "
                f"got {cfg.diffusion.sampler.name!r}"
            )
        if cfg.diffusion.sampler.eta != 0.0:
            raise ValueError(
                f"Step 3 currently supports deterministic DDIM only (eta=0.0), "
                f"got {cfg.diffusion.sampler.eta}"
            )

        self.shape = DDIMShape(
            channels=cfg.dataset.channels,
            image_size=cfg.dataset.image_size,
        )
        self.is_conditional = cfg.dataset.conditional
        self.num_classes = cfg.dataset.num_classes
        self.timesteps = cfg.diffusion.horizon
        self.default_sampling_steps = cfg.diffusion.sampler.steps

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
        )

    @classmethod
    def from_config(cls, backbone: nn.Module, cfg: FullConfig) -> "DDIMDiffusion":
        return cls(backbone=backbone, cfg=cfg)

    def _device(self) -> torch.device:
        return self.betas.device

    def _extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Gather values[t] and reshape to [B,1,1,1,...] for broadcasting.
        """
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")
        out = values.gather(0, t.long())
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def _validate_labels_for_batch(
        self,
        labels: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        labels = _normalize_labels(labels, batch_size=batch_size)

        if self.is_conditional:
            if labels is None:
                raise ValueError("Conditional diffusion model requires labels.")
            labels = labels.to(device=device, dtype=torch.long)
            if self.num_classes is not None:
                if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
                    raise ValueError(
                        f"labels must be in [0, {self.num_classes - 1}] for this config."
                    )
            return labels

        if labels is not None:
            raise ValueError("Unconditional diffusion model must not receive labels.")
        return None

    def make_timesteps(self, nsteps: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create descending DDIM timestep indices, e.g. [999, ..., 0].
        """
        if nsteps is None:
            nsteps = self.default_sampling_steps
        if nsteps <= 0:
            raise ValueError(f"nsteps must be positive, got {nsteps}")
        if nsteps > self.timesteps:
            raise ValueError(f"nsteps={nsteps} cannot exceed horizon={self.timesteps}")

        device = self._device() if device is None else torch.device(device)
        # Integer linspace over [0, T-1], reversed for denoising.
        steps = torch.linspace(
            0,
            self.timesteps - 1,
            steps=nsteps,
            device=device,
            dtype=torch.float32,
        ).round().long()
        steps = torch.unique_consecutive(steps)
        return torch.flip(steps, dims=[0])

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError(f"noise shape must match x0 shape, got {noise.shape} vs {x0.shape}")

        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        labels = self._validate_labels_for_batch(
            labels=labels,
            batch_size=x_t.shape[0],
            device=x_t.device,
        )
        eps = self.backbone(x_t, t, labels=labels)
        if eps.shape != x_t.shape:
            raise ValueError(
                f"Backbone must return same shape as x_t for eps prediction; "
                f"got {tuple(eps.shape)} vs {tuple(x_t.shape)}"
            )
        return eps

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_recip_ab = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_ab = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_ab * x_t - sqrt_recipm1_ab * eps

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_eps(x_t, t, labels=labels)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        nsteps: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        device: Optional[str | torch.device] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        device = self._device() if device is None else torch.device(device)
        labels = self._validate_labels_for_batch(labels, batch_size=batch_size, device=device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device.type)
            generator.manual_seed(seed)

        x = torch.randn(
            batch_size,
            self.shape.channels,
            self.shape.image_size,
            self.shape.image_size,
            device=device,
            generator=generator,
        )

        ddim_ts = self.make_timesteps(nsteps=nsteps, device=device)
        # Previous timestep for each sampled timestep; final previous is -1 => alpha_bar_prev = 1
        prev_ddim_ts = torch.cat(
            [ddim_ts[1:], torch.tensor([-1], device=device, dtype=torch.long)], dim=0
        )

        for t_scalar, prev_t_scalar in zip(ddim_ts, prev_ddim_ts):
            t = torch.full((batch_size,), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = self.predict_eps(x, t, labels=labels)

            alpha_bar_t = self.alphas_cumprod[t_scalar]
            if prev_t_scalar >= 0:
                alpha_bar_prev = self.alphas_cumprod[prev_t_scalar]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device, dtype=self.alphas_cumprod.dtype)

            x0_pred = self.predict_x0_from_eps(x, t, eps)

            # Deterministic DDIM update (eta = 0):
            # x_{t_prev} = sqrt(alpha_bar_prev) * x0_pred
            #            + sqrt(1 - alpha_bar_prev) * eps
            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + torch.sqrt(1.0 - alpha_bar_prev) * eps
            )

        return x