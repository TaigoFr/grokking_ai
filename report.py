import platform
from dataclasses import asdict
from pathlib import Path

import torch

from analysis import dominant_embedding_frequencies
from model import ModularAdditionTransformer
from train import Config, Metrics


def build_summary(
    config: Config,
    history: list[Metrics],
    device: torch.device,
    elapsed_seconds: float,
    parameter_count: int,
    train_pairs: int,
    test_pairs: int,
    cpu_threads: int,
    model: ModularAdditionTransformer,
) -> dict:
    frequencies = dominant_embedding_frequencies(model.token_embedding.weight.detach(), config.prime)
    return {
        "device": str(device),
        "hardware": hardware_name(device),
        "torch_version": torch.__version__,
        "cpu_threads": cpu_threads,
        "elapsed_seconds": elapsed_seconds,
        "epochs": config.epochs,
        "parameter_count": parameter_count,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "memorization_epoch": first_accuracy_epoch(history, "train_accuracy", 0.99),
        "grokking_epoch": first_accuracy_epoch(history, "test_accuracy", 0.99),
        "final_train_accuracy": history[-1].train_accuracy,
        "final_test_accuracy": history[-1].test_accuracy,
        "final_train_loss": history[-1].train_loss,
        "final_test_loss": history[-1].test_loss,
        "dead_period_mean_relative_weight_change_per_epoch": dead_period_mean(history),
        "dominant_embedding_frequencies": [
            {"frequency": frequency, "mean_magnitude": magnitude}
            for frequency, magnitude in frequencies
        ],
        "config": asdict(config),
    }


def first_accuracy_epoch(history: list[Metrics], field: str, threshold: float) -> int | None:
    for metrics in history:
        if getattr(metrics, field) >= threshold:
            return metrics.epoch
    return None


def dead_period_mean(history: list[Metrics]) -> float | None:
    values = [
        metrics.relative_weight_change_per_epoch
        for metrics in history
        if metrics.train_accuracy >= 0.99 and metrics.test_accuracy < 0.1
    ]
    return sum(values) / len(values) if values else None


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def format_stats(summary: dict) -> str:
    config = summary["config"]
    frequencies = ", ".join(
        str(item["frequency"]) for item in summary["dominant_embedding_frequencies"]
    )
    dead_period_mean = summary["dead_period_mean_relative_weight_change_per_epoch"]
    dead_period_mean_text = f"{dead_period_mean:.3e}" if dead_period_mean is not None else "not available"
    return (
        "Modular addition grokking run\n"
        "==============================\n\n"
        f"Device: {summary['device']}\n"
        f"Hardware: {summary['hardware']}\n"
        f"PyTorch: {summary['torch_version']}\n"
        f"CPU threads: {summary['cpu_threads']}\n"
        f"Epochs: {summary['epochs']:,}\n"
        f"Wall time: {summary['elapsed_seconds']:.1f} seconds\n"
        f"Parameters: {summary['parameter_count']:,}\n"
        f"Training pairs: {summary['train_pairs']:,}\n"
        f"Test pairs: {summary['test_pairs']:,}\n"
        f"Train accuracy reached 99%: epoch {summary['memorization_epoch']}\n"
        f"Test accuracy reached 99%: epoch {summary['grokking_epoch']}\n"
        f"Final train accuracy: {100 * summary['final_train_accuracy']:.2f}%\n"
        f"Final test accuracy: {100 * summary['final_test_accuracy']:.2f}%\n"
        f"Final train loss: {summary['final_train_loss']:.3e}\n"
        f"Final test loss: {summary['final_test_loss']:.3e}\n"
        f"Mean dead-period weight change: {dead_period_mean_text}\n"
        f"Dominant embedding frequencies: {frequencies}\n\n"
        "Training configuration\n"
        "----------------------\n"
        f"Prime: {config['prime']}\n"
        f"Train fraction: {config['train_fraction']}\n"
        f"Learning rate: {config['learning_rate']}\n"
        f"Weight decay: {config['weight_decay']}\n"
        f"Adam betas: ({config['beta1']}, {config['beta2']})\n"
        f"Model dimension: {config['model_dimension']}\n"
        f"Attention heads: {config['head_count']}\n"
        f"Head dimension: {config['head_dimension']}\n"
        f"MLP dimension: {config['mlp_dimension']}\n"
        f"Model seed: {config['model_seed']}\n"
        f"Data seed: {config['data_seed']}\n"
        f"Log every: {config['log_every']} epochs\n"
    )
