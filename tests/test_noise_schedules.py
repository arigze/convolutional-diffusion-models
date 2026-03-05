import torch

"""
Tests for noise schedules to ensure they produce the correct number of betas and that values are in expected ranges.
"""

from noise_schedules import linear_noise_schedule, cosine_noise_schedule, exponential_noise_schedule

# Test linear noise schedule
betas = linear_noise_schedule(10)
print(betas.shape) # Should be [10]
print(betas) # Should be linearly spaced between 0.0001 and 0.02

# Test cosine noise schedule
betas = cosine_noise_schedule(10)
print(betas.shape) # Should be [10]
print(betas) # Should be values between 0 and 0.999 

# Test exponential noise schedule
betas = exponential_noise_schedule(10)
print(betas.shape) # Should be [10]
print(betas) # Should be exponentially spaced between 0.0001 and 0.02