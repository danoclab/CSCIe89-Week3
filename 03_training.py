"""Train the Fashion-MNIST MLP classifier.

- Loss: CrossEntropyLoss
- Optimizer: SGD with a learning rate of 0.1
- Epochs: 5
- After every epoch, records training loss/accuracy and validation loss/accuracy.
"""

import importlib
import json

import torch
from torch import nn

# Module names starting with a digit can't be imported with a normal
# `import` statement, so load the earlier scripts through importlib.
dataset = importlib.import_module("01_dataset")
model_module = importlib.import_module("02_model")

LEARNING_RATE = 0.1
NUM_EPOCHS = 5
MODEL_PATH = "fashion_mnist_mlp.pt"
HISTORY_PATH = "training_history.json"


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch; return (average loss, accuracy) over the epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    """Evaluate without updating weights; return (average loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs):
    """Train the model and return the per-epoch history of metrics."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )

    return history


if __name__ == "__main__":
    dataset.set_seed(dataset.SEED)

    device = model_module.get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = dataset.get_dataloaders()
    model = model_module.build_model(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = train(
        model, train_loader, val_loader, loss_fn, optimizer, device, NUM_EPOCHS
    )

    # Save the trained weights and the metrics for later scripts (plots, testing).
    torch.save(model.state_dict(), MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved model weights to {MODEL_PATH} and history to {HISTORY_PATH}")
