import torch
import matplotlib.pyplot as plt
import random

from data import get_dataset, get_testset
from score_machines import IdealScoreMachine, LocalScoreMachine, EquivariantLocalScoreMachine
from noise_schedules import linear_noise_schedule, cosine_noise_schedule, exponential_noise_schedule

torch.manual_seed(4786)

dataset = get_dataset('mnist', (28, 28))
testset = get_testset('mnist', (28, 28))

timesteps = 100
model = EquivariantLocalScoreMachine(
    noise_schedule=linear_noise_schedule,
    dataset=dataset,
    batch_size=256,
    timesteps=timesteps
)

# Start from pure noise of shape [1, 28, 28] (channels, height, width)
x_noisy = torch.randn(1, 28, 28)

# Denoise
with torch.no_grad():
    x_denoised = model.sample(x_noisy)

# Visualize
plt.imshow(x_denoised.squeeze().cpu(), cmap='gray')
plt.show()

