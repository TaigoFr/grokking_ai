import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from analysis import flatten_parameters, relative_weight_change
from model import ModularAdditionTransformer


@dataclass(frozen=True)
class Config:
    prime: int = 113
    train_fraction: float = 0.3
    epochs: int = 40_000
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    model_dimension: int = 128
    head_count: int = 4
    head_dimension: int = 32
    mlp_dimension: int = 512
    model_seed: int = 999
    data_seed: int = 598
    log_every: int = 100


@dataclass(frozen=True)
class Metrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    relative_weight_change_per_epoch: float


def run_training(
    model: ModularAdditionTransformer,
    optimizer: torch.optim.Optimizer,
    train_tokens: Tensor,
    train_labels: Tensor,
    test_tokens: Tensor,
    test_labels: Tensor,
    config: Config,
    metrics_path: Path,
) -> tuple[list[Metrics], float]:
    history: list[Metrics] = []
    previous_parameters = flatten_parameters(model)
    previous_epoch = 0
    started = time.perf_counter()
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(Metrics.__dataclass_fields__))
        writer.writeheader()
        initial = measure(model, 0, train_tokens, train_labels, test_tokens, test_labels, 0.0)
        history.append(initial)
        writer.writerow(asdict(initial))
        for epoch in range(1, config.epochs + 1):
            train_step(model, optimizer, train_tokens, train_labels)
            if epoch % config.log_every != 0 and epoch != config.epochs:
                continue
            current_parameters = flatten_parameters(model)
            # Average the checkpoint displacement over the 100 elapsed optimizer steps.
            movement = relative_weight_change(current_parameters, previous_parameters, epoch - previous_epoch)
            previous_parameters = current_parameters
            previous_epoch = epoch
            metrics = measure(
                model,
                epoch,
                train_tokens,
                train_labels,
                test_tokens,
                test_labels,
                movement,
            )
            history.append(metrics)
            writer.writerow(asdict(metrics))
            metrics_file.flush()
            elapsed = time.perf_counter() - started
            print(
                f"epoch {epoch:5d}  train {metrics.train_accuracy:.3f}  "
                f"test {metrics.test_accuracy:.3f}  movement {movement:.3e}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )
    elapsed_seconds = time.perf_counter() - started
    return history, elapsed_seconds


def train_step(
    model: ModularAdditionTransformer,
    optimizer: torch.optim.Optimizer,
    tokens: Tensor,
    labels: Tensor,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    cross_entropy(model(tokens), labels).backward()  # One full-batch step is one epoch.
    optimizer.step()


@torch.no_grad()
def measure(
    model: ModularAdditionTransformer,
    epoch: int,
    train_tokens: Tensor,
    train_labels: Tensor,
    test_tokens: Tensor,
    test_labels: Tensor,
    movement: float,
) -> Metrics:
    model.eval()
    train_logits = model(train_tokens)
    test_logits = model(test_tokens)
    return Metrics(
        epoch=epoch,
        train_loss=float(cross_entropy(train_logits, train_labels)),
        train_accuracy=accuracy(train_logits, train_labels),
        test_loss=float(cross_entropy(test_logits, test_labels)),
        test_accuracy=accuracy(test_logits, test_labels),
        relative_weight_change_per_epoch=movement,
    )


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)  # Match the reference setup.
    return -log_probabilities.gather(1, labels[:, None]).mean()


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())
