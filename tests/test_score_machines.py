import torch

from score_machines import IdealScoreMachine, LocalScoreMachine

# Test the IdealScoreMachine implementation
# Test parameters
noise_schedule = lambda t: torch.linspace(0.01, 0.1, t)  # Simple linear noise schedule
dataset = torch.randn(100, 3, 32, 32)  # Dummy dataset of 100 images
batch_size = 10
image_size = (3, 32, 32)
timesteps = 5

# Initialize the IdealScoreMachine
score_machine = IdealScoreMachine(noise_schedule, dataset, batch_size, timesteps)

# Test forward pass
x = torch.randn(batch_size, *image_size)  # Random input batch
t = 2  # Time step to test
score = score_machine.forward(x, t)
print("Score shape:", score.shape) # Should be [batch_size, channels, height, width]

# Test sampling
x = torch.randn(*image_size)  # Random initial noise
sampled_x = score_machine.sample(x)
print("Sampled x shape:", sampled_x.shape) # Should be [channels, height, width]


# Test the LocalScoreMachine implementation
# Test parameters
noise_schedule = lambda t: torch.linspace(0.01, 0.1, t)  # Simple linear noise schedule
dataset = torch.randn(100, 3, 32, 32)  # Dummy dataset of 100 images
batch_size = 10
image_size = (3, 32, 32)
timesteps = 5

# Initialize the LocalScoreMachine
local_score_machine = LocalScoreMachine(noise_schedule, dataset, batch_size, timesteps)

# Test forward pass
x = torch.randn(batch_size, *image_size)  # Random input batch
t = 2  # Time step to test
score = local_score_machine.forward(x, t)
print("Local Score shape:", score.shape) # Should be [batch_size, channels, height, width]

# Test sampling
x = torch.randn(*image_size)  # Random initial noise
sampled_x = local_score_machine.sample(x)
print("Local Sampled x shape:", sampled_x.shape) # Should be [channels, height, width]