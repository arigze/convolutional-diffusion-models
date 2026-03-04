import torch

from models import UNet, ResNet

# Test UNet with 28x28 input
model = UNet()
x = torch.randn(1, 1, 28, 28)
print(model(x).shape) # Should be [1, 1, 28, 28]

# Test UNet with 32x32 input
model = UNet()
x = torch.randn(1, 1, 32, 32)
print(model(x).shape) # Should be [1, 1, 32, 32]

# Test ResNet with 28x28 input
model = ResNet()
x = torch.randn(1, 1, 28, 28)
print(model(x).shape) # Should be [1, 1, 28, 28]

# Test ResNet with 32x32 input
model = ResNet()
x = torch.randn(1, 1, 32, 32)
print(model(x).shape) # Should be [1, 1, 32, 32]