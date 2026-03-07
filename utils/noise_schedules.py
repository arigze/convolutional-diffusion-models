import torch
import numpy as np
import math

def exponential_schedule(t):
    if isinstance(t, torch.Tensor):
        return 1 - torch.exp(-2*t)
    return 1 - np.exp(-2*t)

def linear_noise_schedule(t):
    return 0.01 + 0.97*t

def cosine_noise_schedule(t, mode="legacy"):
    if mode == "legacy":
        return 1 - torch.cos((t) / 1.008 * math.pi / 2) ** 2
    return 1 - torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2