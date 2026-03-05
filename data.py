import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(name, image_size):
    """
    Returns the train set of a Pytorch dataset.
    name : 'mnist', 'cifar10', 'fashion-mnist'
    image_size : tuple (H, W)
    """
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif name == 'fashion-mnist':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
    return dataset