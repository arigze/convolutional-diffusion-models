import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        x = x.to(device)
        mu = self.mu[t].to(device)
        sigma = self.sigma[t].to(device)

        batch = x.shape[0]
        sum_weights = torch.zeros(batch, device=device)
        sum_weighted_diffs = torch.zeros_like(x)

        # Help stabilize the weights
        subtraction = None

        for images, _ in self.trainloader:
            images = images.to(device)  # [m, channels, height, width]

            diffs = x[:, None] - mu * images[None] # [batch, m, channels, height, width]
            w_for_exp = -torch.sum(diffs ** 2, dim=[2, 3, 4]) / (2 * sigma**2) # [batch, m]
            
            if subtraction is None:
                subtraction = torch.amax(w_for_exp, dim=(0, 1), keepdim=False) # scalar
            else:
                new_subtraction = torch.amax(w_for_exp, dim=(0, 1), keepdim=False) # scalar
                delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
                sum_weights /= torch.exp(delta_subtraction-subtraction)
                sum_weighted_diffs /= torch.exp(delta_subtraction-subtraction)
                subtraction = delta_subtraction
            
            w = torch.exp(w_for_exp - subtraction) # [batch, m]

            # accumulate weighted differences and weights for each input sample
            sum_weighted_diffs += torch.mean(w[:, :, None, None, None] * diffs, dim=1) # [batch, channels, height, width]
            sum_weights += torch.mean(w, dim=1) # [batch]

        score = -sum_weighted_diffs / (sigma**2 * sum_weights[:, None, None, None]) # [batch, channels, height, width]

        return score

    def sample(self, x, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        x = x.unsqueeze(0).to(device)  # Add batch dimension

        for t in tqdm(reversed(range(1, self.timesteps)), total=self.timesteps-1, desc="Sampling"):
            score = self.forward(x, t, device=device)

            mu_t, sigma_t = self.mu[t].to(device), self.sigma[t].to(device)
            mu_prev, sigma_prev = self.mu[max(t - 1, 0)].to(device), self.sigma[max(t - 1, 0)].to(device)

            x_0_pred = (x + sigma_t**2 * score) / mu_t
            x = mu_prev * x_0_pred + sigma_prev * (x - mu_t * x_0_pred) / sigma_t

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
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        x = x.to(device)
        mu = self.mu[t].to(device)
        sigma = self.sigma[t].to(device)

        batch = x.shape[0]
        sum_weights = torch.zeros(batch, x.shape[2], x.shape[3], device=device)
        sum_weighted_diffs = torch.zeros_like(x)

        # Help stabilize the weights
        subtraction = None

        for images, _ in self.trainloader:
            images = images.to(device)  # [m, channels, height, width]

            diffs = x[:, None] - mu * images[None] # [batch, m, channels, height, width]
            # Get the norm of diffs accross channel dimensions
            norm_diffs = torch.sum(diffs ** 2, dim=2) # [batch, m, height, width]
            # Use unfold to get 3x3 patches of the norm_diffs for each pixel
            patches = F.unfold(norm_diffs, kernel_size=3, padding=1) # [batch, m*9, num_patches]
            # Reshape patches to [batch, m, 9, height, width]
            patches = patches.view(batch, images.shape[0], -1, images.shape[2], images.shape[3]) # [batch, m, 9, height, width]
            # Compute weights using the local patches
            w_for_exp = -torch.sum(patches, dim=2) / (2 * sigma**2) # [batch, m, height, width]

            if subtraction is None:
                subtraction = torch.amax(w_for_exp, dim=(0, 1), keepdim=False) # scalar
            else:
                new_subtraction = torch.amax(w_for_exp, dim=(0, 1), keepdim=False) # scalar
                delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
                sum_weights /= torch.exp(delta_subtraction-subtraction)
                sum_weighted_diffs /= torch.exp(delta_subtraction-subtraction)
                subtraction = delta_subtraction

            w = torch.exp(w_for_exp - subtraction) # [batch, m, height, width]

            # accumulate weighted differences and weights for each input sample
            sum_weighted_diffs += torch.sum(w[:, :, None, :, :] * diffs, dim=1) # [batch, channels, height, width]
            sum_weights += torch.sum(w, dim=1) # [batch, height, width]

        score = -sum_weighted_diffs / (sigma**2 * sum_weights[:, None, :, :]) # [batch, channels, height, width]

        return score

    def sample(self, x, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        x = x.unsqueeze(0).to(device)  # Add batch dimension

        for t in tqdm(reversed(range(1, self.timesteps)), total=self.timesteps-1, desc="Sampling"):
            score = self.forward(x, t, device=device)

            mu_t, sigma_t = self.mu[t].to(device), self.sigma[t].to(device)
            mu_prev, sigma_prev = self.mu[max(t - 1, 0)].to(device), self.sigma[max(t - 1, 0)].to(device)

            x_0_pred = (x + sigma_t**2 * score) / mu_t
            x = mu_prev * x_0_pred + sigma_prev * (x - mu_t * x_0_pred) / sigma_t

        return x.squeeze(0)  # Remove batch dimension


class EquivariantLocalScoreMachine(nn.Module):
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
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        x = x.to(device)
        mu = self.mu[t].to(device)
        sigma = self.sigma[t].to(device)

        batch = x.shape[0]
        sum_weights = torch.zeros(batch, x.shape[2], x.shape[3], device=device)
        sum_weighted_diffs = torch.zeros_like(x)

        # Help stabilize the weights
        subtraction = None

        for images, _ in self.trainloader:
            images = images.to(device)  # [m, channels, height, width]
            # Make patches of the images using unfold to get 3x3 patches for each pixel
            patches = F.unfold(images, kernel_size=3, padding=1) # [m, channels*9, num_patches]
            # Reshape patches to [num_patches, m, channels*9]
            patches = patches.permute(2, 0, 1) # [num_patches, m, channels*9]
            # Reshape patches to [num_patches*m, channels, 3, 3]
            patches = patches.reshape(patches.shape[0] * patches.shape[1], images.shape[1], 3, 3) # [num_patches*m, channels, 3, 3]
            # Get the norm of patches accross channel dimensions
            pnorms = torch.sum(patches**2, dim=(1,2,3)) # [num_patches*m]
            # Get the center pixel of each patch
            pcenters = patches[:, :, 1, 1] # [num_patches*m, channels]
            # Compute differences between input and the center pixel of patches
            diffs = (x[:, None, :, :, :] - mu * pcenters[None, :, :, None, None]) # [batch, num_patches*m, channels, height, width]

            # Make circular padding of images to handle the equivariance
            x_padded = F.pad(x, (1, 1, 1, 1), mode='circular') # [batch, channels, height+2, width+2]
            # Make patches of x using unfold to get 3x3 patches for each pixel
            x_patches = F.unfold(x_padded, kernel_size=3, padding=0) # [batch, channels*9, num_patches]
            # Get the norm of patches accross channel dimensions
            x_norms = torch.sum(x_patches, dim=1) ** 2 # [batch, num_patches]
            # Reshape x_norms to [batch, height, width]
            x_norms = x_norms.view(batch, images.shape[2], images.shape[3]) # [batch, height, width]

            # Circular convolution
            pad_input = F.pad(x, (patches.size(3) // 2, patches.size(3) // 2, patches.size(2) // 2, patches.size(2) // 2), mode='circular')
            conv = F.conv2d(pad_input, patches, padding=0)

            # Compute weights using the local patches and the convolution results
            w = - (x_norms[:, None, :, :] - 2 * mu * conv + (mu ** 2) * pnorms[None, :, None, None]) / (2 * sigma ** 2) # [batch, num_patches*m, height, width]

            if subtraction is None:
                subtraction = torch.amax(w, dim=(0, 1), keepdim=False) # scalar
            else:
                new_subtraction = torch.amax(w, dim=(0, 1), keepdim=False) # scalar
                delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
                sum_weights /= torch.exp(delta_subtraction-subtraction)
                sum_weighted_diffs /= torch.exp(delta_subtraction-subtraction)
                subtraction = delta_subtraction
            
            w = torch.exp(w - subtraction) # [batch, num_patches*m, height, width]

            # accumulate weighted differences and weights for each input sample
            sum_weighted_diffs += torch.sum(w[:, :, None, :, :] * diffs, dim=1) # [batch, channels, height, width]
            sum_weights += torch.sum(w, dim=1) # [batch, height, width]

        score = -sum_weighted_diffs / (sigma**2 * sum_weights[:, None, :, :]) # [batch, channels, height, width]

        return score

    def sample(self, x, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        x = x.unsqueeze(0).to(device)  # Add batch dimension

        for t in tqdm(reversed(range(1, self.timesteps)), total=self.timesteps-1, desc="Sampling"):
            score = self.forward(x, t, device=device)

            mu_t, sigma_t = self.mu[t].to(device), self.sigma[t].to(device)
            mu_prev, sigma_prev = self.mu[max(t - 1, 0)].to(device), self.sigma[max(t - 1, 0)].to(device)

            x_0_pred = (x + sigma_t**2 * score) / mu_t
            x = mu_prev * x_0_pred + sigma_prev * (x - mu_t * x_0_pred) / sigma_t

        return x.squeeze(0)  # Remove batch dimension