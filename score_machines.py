import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

class IdealScoreMachine(nn.Module):
    def __init__(self, noise_schedule, dataset, batch_size, timesteps):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.dataset = dataset
        self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.timesteps = timesteps

        # Noise schedule and scale and noise std
        betas = self.noise_schedule(self.timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_bar = alphas_cumprod
        self.mu = alpha_bar ** 0.5            # signal scale
        self.sigma = (1 - alpha_bar) ** 0.5   # noise std

    def forward(self, x, t, device=None):
        """
        Ideal score function :
        ∇_x log p_t(x) = - ∑_i w_i(x)(x - mu_t * x_0^i) / (sigma_t^2 * ∑_i w_i(x))
        Where w_i(x) = exp(-||x - mu_t * x_0^i||^2 / (2 * sigma_t^2)) and x_0^i are the training samples.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = x.to(device)
        mu = self.mu[t].to(device)
        sigma = self.sigma[t].to(device)

        batch = x.shape[0]
        sum_weights = torch.zeros(batch, device=device)
        sum_weighted_diffs = torch.zeros_like(x)

        for images in self.trainloader:
            images = images.to(device)  # [m, channels, height, width]

            diffs = x[:, None] - mu * images[None] # [batch, m, channels, height, width]
            w = torch.exp(-torch.sum(diffs ** 2, dim=[2, 3, 4]) / (2 * sigma**2)) # [batch, m]

            # accumulate weighted differences and weights for each input sample
            sum_weighted_diffs += torch.sum(w[:, :, None, None, None] * diffs, dim=1) # [batch, channels, height, width]
            sum_weights += torch.sum(w, dim=1) # [batch]

        score = -sum_weighted_diffs / (sigma**2 * sum_weights[:, None, None, None]) # [batch, channels, height, width]

        return score

    def sample(self, x, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = x.unsqueeze(0).to(device)  # Add batch dimension

        for t in reversed(range(self.timesteps)):
            mu_t, sigma_t = self.mu[t].to(device), self.sigma[t].to(device)
            mu_prev, sigma_prev = self.mu[max(t - 1, 0)].to(device), self.sigma[max(t - 1, 0)].to(device)
            score = self.forward(x, t, device=device)

            x = (mu_prev / mu_t) * (x + (sigma_prev**2 - sigma_t**2 * (mu_t**2 / mu_prev**2)) * score)

        return x.squeeze(0)  # Remove batch dimension


class LocalScoreMachine(nn.Module):
    def __init__(self, noise_schedule, dataset, batch_size, timesteps):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.dataset = dataset
        self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.timesteps = timesteps

        # Noise schedule and scale and noise std
        betas = self.noise_schedule(self.timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_bar = alphas_cumprod
        self.mu = alpha_bar ** 0.5            # signal scale
        self.sigma = (1 - alpha_bar) ** 0.5   # noise std

    def forward(self, x, t, device=None):
        """
        Ideal score function :
        ∇_x log p_t(x) = - ∑_i w_i(x)(x - mu_t * x_0^i) / (sigma_t^2 * ∑_i w_i(x))
        Where w_i(x) = exp(-||x - mu_t * x_0^i||^2 / (2 * sigma_t^2)) and x_0^i are the training samples.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = x.to(device)
        mu = self.mu[t].to(device)
        sigma = self.sigma[t].to(device)

        batch = x.shape[0]
        sum_weights = torch.zeros(batch, x.shape[2], x.shape[3], device=device)
        sum_weighted_diffs = torch.zeros_like(x)

        for images in self.trainloader:
            images = images.to(device)  # [m, channels, height, width]

            diffs = x[:, None] - mu * images[None] # [batch, m, channels, height, width]
            norm_diffs = torch.sum(diffs ** 2, dim=2) # [batch, m, height, width]
            patches = F.unfold(norm_diffs, kernel_size=3, padding=1) # [batch, m*9, num_patches]
            patches = patches.view(batch, images.shape[0], -1, images.shape[2], images.shape[3]) # [batch, m, 9, height, width]
            w = torch.exp(-torch.sum(patches, dim=2) / (2 * sigma**2)) # [batch, m, height, width]

            # accumulate weighted differences and weights for each input sample
            sum_weighted_diffs += torch.sum(w[:, :, None, :, :] * diffs, dim=1) # [batch, channels, height, width]
            sum_weights += torch.sum(w, dim=1) # [batch, height, width]

        score = -sum_weighted_diffs / (sigma**2 * sum_weights[:, None, :, :]) # [batch, channels, height, width]

        return score

    def sample(self, x, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = x.unsqueeze(0).to(device)  # Add batch dimension

        for t in reversed(range(self.timesteps)):
            mu_t, sigma_t = self.mu[t].to(device), self.sigma[t].to(device)
            mu_prev, sigma_prev = self.mu[max(t - 1, 0)].to(device), self.sigma[max(t - 1, 0)].to(device)
            score = self.forward(x, t, device=device)

            x = (mu_prev / mu_t) * (x + (sigma_prev**2 - sigma_t**2 * (mu_t**2 / mu_prev**2)) * score)

        return x.squeeze(0)  # Remove batch dimension