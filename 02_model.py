"""Multilayer perceptron for Fashion-MNIST classification.

Architecture:
    Flatten (1 x 28 x 28 -> 784)
    Linear(784 -> 300) + ReLU
    Linear(300 -> 100) + ReLU
    Linear(100 -> 10)

The output layer returns raw logits (no softmax), which is what
nn.CrossEntropyLoss expects during training.
"""

import importlib

import torch
from torch import nn

INPUT_SIZE = 28 * 28
HIDDEN1_SIZE = 300
HIDDEN2_SIZE = 100
NUM_CLASSES = 10


def get_device() -> torch.device:
    """Use the GPU if one is available, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FashionMNISTClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden1_size: int = HIDDEN1_SIZE,
        hidden2_size: int = HIDDEN2_SIZE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.output = nn.Linear(hidden2_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)                # (batch, 1, 28, 28) -> (batch, 784)
        x = self.relu(self.hidden1(x))     # (batch, 300)
        x = self.relu(self.hidden2(x))     # (batch, 100)
        return self.output(x)              # (batch, 10) logits


def build_model(device: torch.device | None = None) -> FashionMNISTClassifier:
    """Create the model and move it to the given device (GPU/CPU)."""
    device = device or get_device()
    return FashionMNISTClassifier().to(device)


if __name__ == "__main__":
    # Module names starting with a digit can't be imported with a normal
    # `import` statement, so load 01_dataset.py through importlib.
    dataset = importlib.import_module("01_dataset")
    dataset.set_seed(dataset.SEED)

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(device)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Sanity check: run one batch from the training DataLoader through the model.
    train_loader, _, _ = dataset.get_dataloaders()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(images)
    print(f"Input batch shape:  {tuple(images.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
