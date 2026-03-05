import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

class IdealScoreMachine(nn.Module):
    def __init__(self, noise_schedule, dataset, batch_size, image_size, max_samples, timesteps, score_backbone=True):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.dataset = dataset
        self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_samples = max_samples
        self.timesteps = timesteps
        self.score_backbone = score_backbone

        betas = self.noise_schedule(self.timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_at_bt(self, t):
        # t is an integer timestep in [0, timesteps-1]
        alpha_bar = self.alphas_cumprod[t]
        at = alpha_bar ** 0.5         # signal scale
        bt = (1 - alpha_bar) ** 0.5   # noise std
        return at, bt

    def forward(self, x, t, device=None):
        """
        Ideal score function :
        Nabla_x log p_t(x) = - Sigma_i w_i(x)(x - a_t * x_0^i) / (b_t^2 * Sigma_i w_i(x))
        Where w_i(x) = exp(-||x - a_t * x_0^i||^2 / (2 * b_t^2)) and x_0^i are the training samples.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        at, bt = self.get_at_bt(t)

        # first pass: determine a running maximum of exp_args per input
        global_max = torch.full((x.shape[0],), -float('inf'), device=device)
        seen = 0
        for images, _ in self.trainloader:
            images = images.to(device)
            if seen + images.size(0) > self.max_samples:
                images = images[: self.max_samples - seen]
            if images.numel() == 0:
                break

            diff = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            exp_args = -torch.sum(diff**2, dim=(2,3,4)) / (2 * bt**2)
            batch_max = torch.max(exp_args, dim=1)[0]
            global_max = torch.max(global_max, batch_max)

            seen += images.size(0)
            if seen >= self.max_samples:
                break

        # second pass: compute weighted sums using the fixed global maximum
        numerator = torch.zeros(x.shape, device=device)
        denominator = torch.zeros(x.shape[0], device=device)
        seen = 0
        for images, _ in self.trainloader:
            images = images.to(device)
            if seen + images.size(0) > self.max_samples:
                images = images[: self.max_samples - seen]
            if images.numel() == 0:
                break

            diff = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            exp_args = -torch.sum(diff**2, dim=(2,3,4)) / (2 * bt**2)

            stable = exp_args - global_max[:, None]
            exp_vals = torch.exp(stable)

            numerator += torch.mean(exp_vals[:, :, None, None, None] * diff, dim=1)
            denominator += torch.mean(exp_vals, dim=1)

            seen += images.size(0)
            if seen >= self.max_samples:
                break

        return - numerator / (denominator[:, None, None, None] * bt**2)

    def sample(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        x = torch.randn(1, self.in_channels, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.timesteps)):
            at, bt = self.get_at_bt(t)
            at_prev, bt_prev = self.get_at_bt(max(t - 1, 0))
            score = self.forward(x, t, device=device)

            if not self.score_backbone:
                # backbone predicts ε, convert to score: ∇_x log p_t = -ε / β_t^0.5
                score = -score / bt

            x = (at_prev / at) * (x + (bt**2 - bt_prev**2 * (at**2 / at_prev**2)) * score)

        return x


class LocalScoreMachine(IdealScoreMachine):
    def __init__(self, noise_schedule, dataset, batch_size, image_size, max_samples, timesteps, score_backbone=True, kernel_size=3):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.dataset = dataset
        self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_samples = max_samples
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.score_backbone = score_backbone

        betas = self.noise_schedule(self.timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_at_bt(self, t):
        # t is an integer timestep in [0, timesteps-1]
        alpha_bar = self.alphas_cumprod[t]
        at = alpha_bar ** 0.5         # signal scale
        bt = (1 - alpha_bar) ** 0.5   # noise std
        return at, bt

    def forward(self, x, t, device=None):
        """
        Ideal score function :
        Nabla_x log p_t(x) = - Sigma_i w_i(x)(x - a_t * x_0^i) / (b_t^2 * Sigma_i w_i(x))
        Where w_i(x) = exp(-||x - a_t * x_0^i||^2 / (2 * b_t^2)) and x_0^i are the training samples.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        at, bt = self.get_at_bt(t)

        # first pass: determine a running maximum of exp_args per input
        global_max = torch.full((x.shape[0],), -float('inf'), device=device)
        seen = 0
        for images, _ in self.trainloader:
            images = images.to(device)
            if seen + images.size(0) > self.max_samples:
                images = images[: self.max_samples - seen]
            if images.numel() == 0:
                break

            diff = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            pwise_normsquares = torch.sum(diff**2, dim=2) # [b, NP, h, w]
            patches = F.unfold(pwise_normsquares, self.kernel_size, stride=1, padding=self.kernel_size//2) # [b, NP*k^2, h*w]
            patches = patches.view(b, bsize, self.kernel_size**2, h, w) # [b, NP, k^2, h, w]
            exp_args = -torch.sum(patches, dim=2)/(2*bt**2) # [b, NP, h, w]
            batch_max = torch.max(exp_args, dim=1)[0]
            global_max = torch.max(global_max, batch_max)

            seen += images.size(0)
            if seen >= self.max_samples:
                break

        # second pass: compute weighted sums using the fixed global maximum
        numerator = torch.zeros(x.shape, device=device)
        denominator = torch.zeros(x.shape[0], device=device)
        seen = 0
        for images, _ in self.trainloader:
            images = images.to(device)
            if seen + images.size(0) > self.max_samples:
                images = images[: self.max_samples - seen]
            if images.numel() == 0:
                break

            diff = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            pwise_normsquares = torch.sum(diff**2, dim=2) # [b, NP, h, w]
            patches = F.unfold(pwise_normsquares, self.kernel_size, stride=1, padding=self.kernel_size//2) # [b, NP*k^2, h*w]
            patches = patches.view(b, bsize, self.kernel_size**2, h, w) # [b, NP, k^2, h, w]
            exp_args = -torch.sum(patches, dim=2)/(2*bt**2) # [b, NP, h, w]

            stable = exp_args - global_max[:, None]
            exp_vals = torch.exp(stable)

            numerator += torch.mean(exp_vals[:, :, None, None, None] * diff, dim=1)
            denominator += torch.mean(exp_vals, dim=1)

            seen += images.size(0)
            if seen >= self.max_samples:
                break

        return - numerator / (denominator[:, None, None, None] * bt**2)

    def sample(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        x = torch.randn(1, self.in_channels, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.timesteps)):
            at, bt = self.get_at_bt(t)
            at_prev, bt_prev = self.get_at_bt(max(t - 1, 0))
            score = self.forward(x, t, device=device)

            if not self.score_backbone:
                # backbone predicts ε, convert to score: ∇_x log p_t = -ε / β_t^0.5
                score = -score / bt

            x = (at_prev / at) * (x + (bt**2 - bt_prev**2 * (at**2 / at_prev**2)) * score)

        return x


class LocalEquivariantScoreMachine(LocalScoreMachine):
    def __init__(self, noise_schedule, dataset, batch_size, image_size, max_samples, timesteps, score_backbone=True, kernel_size=3):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.dataset = dataset
        self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_samples = max_samples
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.score_backbone = score_backbone

        betas = self.noise_schedule(self.timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_at_bt(self, t):
        # t is an integer timestep in [0, timesteps-1]
        alpha_bar = self.alphas_cumprod[t]
        at = alpha_bar ** 0.5         # signal scale
        bt = (1 - alpha_bar) ** 0.5   # noise std
        return at, bt

    def forward(self, x, t, device=None): # TODO
        """
        Ideal score function :
        Nabla_x log p_t(x) = - Sigma_i w_i(x)(x - a_t * x_0^i) / (b_t^2 * Sigma_i w_i(x))
        Where w_i(x) = exp(-||x - a_t * x_0^i||^2 / (2 * b_t^2)) and x_0^i are the training samples.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        at, bt = self.get_at_bt(t)

        d = self.kernel_size//2

        xpadded = F.pad(x, (d, d, d, d), mode='circular')

        xpatches = F.unfold(xpadded, k, stride=1, padding=0) 
        xnorms = torch.norm(xpatches, dim=1)**2
        xnorms = xnorms.reshape(b, h, w) # [b, h, w]

        # first pass: determine a running maximum of exp_args per input
        global_max = torch.full((x.shape[0],), -float('inf'), device=device)
        seen = 0
        for images, _ in self.trainloader:
            images = images.to(device)
            if seen + images.size(0) > self.max_samples:
                images = images[: self.max_samples - seen]
            if images.numel() == 0:
                break

            diff = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            exp_args = -torch.sum(diff**2, dim=(2,3,4)) / (2 * bt**2)
            batch_max = torch.max(exp_args, dim=1)[0]
            global_max = torch.max(global_max, batch_max)

            seen += images.size(0)
            if seen >= self.max_samples:
                break

        # second pass: compute weighted sums using the fixed global maximum
        numerator = torch.zeros(x.shape, device=device)
        denominator = torch.zeros(x.shape[0], device=device)
        seen = 0
        for images, _ in self.trainloader:
            images = images.to(device)
            if seen + images.size(0) > self.max_samples:
                images = images[: self.max_samples - seen]
            if images.numel() == 0:
                break

            diff = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            exp_args = -torch.sum(diff**2, dim=(2,3,4)) / (2 * bt**2)

            stable = exp_args - global_max[:, None]
            exp_vals = torch.exp(stable)

            numerator += torch.mean(exp_vals[:, :, None, None, None] * diff, dim=1)
            denominator += torch.mean(exp_vals, dim=1)

            seen += images.size(0)
            if seen >= self.max_samples:
                break

        return - numerator / (denominator[:, None, None, None] * bt**2)

    def sample(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        x = torch.randn(1, self.in_channels, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.timesteps)):
            at, bt = self.get_at_bt(t)
            at_prev, bt_prev = self.get_at_bt(max(t - 1, 0))
            score = self.forward(x, t, device=device)

            if not self.score_backbone:
                # backbone predicts ε, convert to score: ∇_x log p_t = -ε / β_t^0.5
                score = -score / bt

            x = (at_prev / at) * (x + (bt**2 - bt_prev**2 * (at**2 / at_prev**2)) * score)

        return x