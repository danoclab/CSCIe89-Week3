"""Load Fashion-MNIST and build train/validation/test DataLoaders.

- Images are converted to float32 tensors with pixel values scaled to [0, 1].
- The original 60,000-image training set is split into 55,000 training and
  5,000 validation images; the 10,000 test images are kept separate.
- A random seed of 42 makes the split and the training shuffle reproducible.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 32
DATA_DIR = "./data"
TRAIN_SIZE = 55_000
VAL_SIZE = 5_000

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ToTensor converts a PIL image (uint8, 0-255) into a float32 tensor of shape
# (1, 28, 28) with pixel values scaled to [0, 1].
transform = transforms.ToTensor()


def get_datasets(data_dir: str = DATA_DIR):
    """Download Fashion-MNIST and return (train, val, test) datasets."""
    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_dataset, val_dataset = random_split(
        full_train,
        [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size: int = BATCH_SIZE, data_dir: str = DATA_DIR):
    """Return (train_loader, val_loader, test_loader); only training is shuffled."""
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    set_seed(SEED)
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Training images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images:       {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {tuple(images.shape)}, dtype: {images.dtype}")
    print(f"Batch labels shape: {tuple(labels.shape)}, dtype: {labels.dtype}")
    print(f"Pixel value range: [{images.min().item():.1f}, {images.max().item():.1f}]")
