"""Plot training and validation accuracy per epoch.

Reads the metrics saved by 03_training.py (training_history.json) and draws
training accuracy and validation accuracy on the same line graph.
Run 03_training.py first to generate the history file.
"""

import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

training = importlib.import_module("03_training")

PLOT_PATH = "accuracy_plot.png"

TRAIN_COLOR = "#2a78d6"   # blue
VAL_COLOR = "#eb6834"     # orange
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"


def load_history(path: str = training.HISTORY_PATH) -> dict:
    """Load the per-epoch metrics recorded during training."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run 03_training.py first to train the model."
        )
    with open(path) as f:
        return json.load(f)


def plot_accuracy(history: dict, save_path: str = PLOT_PATH):
    """Draw training vs. validation accuracy per epoch and save it as a PNG."""
    train_acc = [acc * 100 for acc in history["train_acc"]]
    val_acc = [acc * 100 for acc in history["val_acc"]]
    epochs = range(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Different markers as well as colors, so the lines can be told apart
    # without relying on color alone.
    ax.plot(epochs, train_acc, color=TRAIN_COLOR, linewidth=2,
            marker="o", markersize=7, label="Training accuracy")
    ax.plot(epochs, val_acc, color=VAL_COLOR, linewidth=2,
            marker="s", markersize=7, label="Validation accuracy")

    # Label the final value of each line.
    for values, color in ((train_acc, TRAIN_COLOR), (val_acc, VAL_COLOR)):
        other = val_acc if values is train_acc else train_acc
        offset = 8 if values[-1] >= other[-1] else -14
        ax.annotate(f"{values[-1]:.1f}%", xy=(epochs[-1], values[-1]),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=9, color=TEXT_COLOR)

    ax.set_title("Fashion-MNIST MLP: Training vs. Validation Accuracy",
                 fontsize=13, pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(0.7, len(train_acc) + 0.3)

    # Keep the grid and axes light so the data lines stand out.
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved accuracy plot to {save_path}")
    return fig


if __name__ == "__main__":
    history = load_history()
    plot_accuracy(history)
    plt.show()
