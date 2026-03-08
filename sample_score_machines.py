import torch
import matplotlib.pyplot as plt
import random

from data import get_dataset, get_testset
from score_machines import IdealScoreMachine, LocalScoreMachine, EquivariantLocalScoreMachine
from utils.noise_schedules_tensors import linear_noise_schedule, cosine_noise_schedule, exponential_noise_schedule

dataset_name = 'mnist'
dataset = get_dataset(dataset_name, (28, 28))
timesteps = 20
batch_size = 256

seeds = [4786, 1234, 5678, 91011, 121314, 151617, 181920, 212223, 242526, 272829]

for seed in seeds:
    torch.manual_seed(seed)

    timesteps = 100
    model = IdealScoreMachine(
        noise_schedule=cosine_noise_schedule,
        dataset=dataset,
        batch_size=batch_size,
        timesteps=timesteps
    )

    # Start from pure noise of shape [1, 28, 28] (channels, height, width)
    x_noisy = torch.randn(1, 28, 28)

    # Denoise
    with torch.no_grad():
        x_denoised = model.sample(x_noisy)

    torch.save(x_denoised, f"results/tensors/{dataset_name}_IS_{seed}.pt")

    # Visualize
    plt.imshow(x_denoised.squeeze().cpu(), cmap='gray')
    plt.axis('off')
    plt.savefig(f"results/images/{dataset_name}_IS_{seed}.png")