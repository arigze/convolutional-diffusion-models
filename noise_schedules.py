import torch

def linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    # Returns list of betas
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_noise_schedule(timesteps, s=0.008):
    # Returns list of betas
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exponential_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    # Returns list ofbetas
    return torch.exp(torch.linspace(beta_start, beta_end, timesteps))