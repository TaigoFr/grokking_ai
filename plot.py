from pathlib import Path
from typing import Protocol, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from analysis import clock_coordinates, dominant_embedding_frequencies


class TrainingMetrics(Protocol):
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    relative_weight_change_per_epoch: float


def save_training_plot(history: Sequence[TrainingMetrics], path: Path, prime: int) -> None:
    epochs = [metrics.epoch for metrics in history]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(epochs, [metrics.train_accuracy for metrics in history], label="Train")
    axes[0].plot(epochs, [metrics.test_accuracy for metrics in history], label="Test")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(f"Modular addition, modulus {prime}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [metrics.train_loss for metrics in history], label="Train")
    axes[1].plot(epochs, [metrics.test_loss for metrics in history], label="Test")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    movement = [metrics for metrics in history if metrics.epoch > 0]
    axes[2].plot(
        [metrics.epoch for metrics in movement],
        [metrics.relative_weight_change_per_epoch for metrics in movement],
        color="tab:green",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Relative weight change per epoch")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_clock_plot(token_embedding: Tensor, prime: int, path: Path) -> None:
    frequencies = dominant_embedding_frequencies(token_embedding, prime)
    figure, axes = plt.subplots(1, len(frequencies), figsize=(4 * len(frequencies), 4.2), layout="constrained")
    if len(frequencies) == 1:
        axes = [axes]
    numbers = torch.arange(prime)
    for axis, (frequency, magnitude) in zip(axes, frequencies):
        horizontal, vertical = clock_coordinates(token_embedding, prime, frequency)
        points = axis.scatter(horizontal, vertical, c=numbers, cmap="twilight", s=18)  # Color by residue.
        for number in (0, 1, 2):
            axis.annotate(
                str(number),
                (float(horizontal[number]), float(vertical[number])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        axis.set_aspect("equal")
        axis.set_title(f"frequency {frequency}, magnitude {magnitude:.2f}")
        axis.set_xlabel("cos component")
        axis.set_ylabel("sin component")
        axis.grid(True, alpha=0.3)
    figure.colorbar(points, ax=axes, fraction=0.046, pad=0.04, label="number")
    figure.suptitle(f"Fourier clocks in the number embeddings, modulus {prime}")
    figure.savefig(path, dpi=150)
    plt.close(figure)
