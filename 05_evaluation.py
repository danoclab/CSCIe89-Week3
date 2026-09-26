"""Evaluate the trained Fashion-MNIST classifier and make predictions.

1. Loads the trained weights saved by 03_training.py and switches the model
   to evaluation mode with gradient calculations disabled.
2. Predicts the clothing category of three validation images, converting the
   output logits into probabilities with Softmax and showing the top 4 classes.
3. Compares the predicted labels with the actual labels.
4. Calculates the overall classification accuracy on the test dataset.

Run 03_training.py first to generate the trained weights.
"""

import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

# Module names starting with a digit can't be imported with a normal
# `import` statement, so load the earlier scripts through importlib.
dataset = importlib.import_module("01_dataset")
model_module = importlib.import_module("02_model")
training = importlib.import_module("03_training")

NUM_SAMPLES = 3
TOP_K = 4


def load_trained_model(device, path: str = training.MODEL_PATH):
    """Build the model, load the trained weights and switch to evaluation mode."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run 03_training.py first to train the model."
        )
    model = model_module.build_model(device)
    # map_location lets weights trained on a GPU load on a CPU-only machine.
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()  # evaluation mode (matters for layers like dropout/batch norm)
    return model


def predict_samples(model, val_dataset, device, num_samples=NUM_SAMPLES, top_k=TOP_K):
    """Predict a few validation images and print the top-k classes for each."""
    images = torch.stack([val_dataset[i][0] for i in range(num_samples)])
    labels = torch.tensor([val_dataset[i][1] for i in range(num_samples)])

    with torch.no_grad():
        logits = model(images.to(device))
        probabilities = F.softmax(logits, dim=1)

    # Move results back to the CPU before converting them to Python values.
    probabilities = probabilities.cpu()
    top_probs, top_classes = probabilities.topk(top_k, dim=1)
    predicted = top_classes[:, 0]

    for i in range(num_samples):
        actual = dataset.CLASS_NAMES[labels[i]]
        pred = dataset.CLASS_NAMES[predicted[i]]
        result = "CORRECT" if predicted[i] == labels[i] else "WRONG"

        print(f"\nValidation image {i + 1}")
        print(f"  Actual label:    {actual}")
        print(f"  Predicted label: {pred}  ({result})")
        print(f"  Top {top_k} most probable categories:")
        for rank, (prob, cls) in enumerate(zip(top_probs[i], top_classes[i]), 1):
            print(f"    {rank}. {dataset.CLASS_NAMES[cls]:<12} {prob.item():7.2%}")

    num_correct = (predicted == labels).sum().item()
    print(f"\nSample predictions correct: {num_correct}/{num_samples}")
    return predicted, labels


def compute_accuracy(model, loader, device):
    """Return the fraction of images in the loader that are classified correctly."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    dataset.set_seed(dataset.SEED)

    device = model_module.get_device()
    print(f"Using device: {device}")

    model = load_trained_model(device)
    _, val_dataset, _ = dataset.get_datasets()
    _, _, test_loader = dataset.get_dataloaders()

    print(f"\n=== Predictions for {NUM_SAMPLES} validation images ===")
    predict_samples(model, val_dataset, device)

    test_accuracy = compute_accuracy(model, test_loader, device)
    print("\n=== Test set evaluation ===")
    print(f"Test accuracy: {test_accuracy:.2%} "
          f"on {len(test_loader.dataset):,} test images")
