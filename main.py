import argparse
import json
import os
from pathlib import Path

import torch

from data import make_modular_addition_split
from model import ModularAdditionTransformer, initialize_parameters
from plot import save_clock_plot, save_training_plot
from report import build_summary, format_stats
from train import Config, run_training


def main() -> None:
    args = parse_args()
    config = Config(epochs=args.epochs)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    
    split = make_modular_addition_split(config.prime, config.train_fraction, config.data_seed)
    train_tokens = split.train_tokens.to(device)
    train_labels = split.train_labels.to(device)
    test_tokens = split.test_tokens.to(device)
    test_labels = split.test_labels.to(device)
    
    model = ModularAdditionTransformer(
        vocabulary_size=split.vocabulary_size,
        output_vocabulary_size=config.prime,
        model_dimension=config.model_dimension,
        head_count=config.head_count,
        head_dimension=config.head_dimension,
        mlp_dimension=config.mlp_dimension,
    ).to(device)
    initialize_parameters(model, config.model_seed)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    
    history, elapsed_seconds = run_training(
        model,
        optimizer,
        train_tokens,
        train_labels,
        test_tokens,
        test_labels,
        config,
        output_dir / "metrics.csv",
    )

    save_training_plot(history, output_dir / "training.png", config.prime)
    save_clock_plot(model.token_embedding.weight.detach(), config.prime, output_dir / "clock.png")
    summary = build_summary(
        config,
        history,
        device,
        elapsed_seconds,
        sum(parameter.numel() for parameter in model.parameters()),
        train_labels.shape[0],
        test_labels.shape[0],
        args.cpu_threads,
        model,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "stats.txt").write_text(format_stats(summary))
    print(json.dumps(summary, indent=2), flush=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce modular-addition grokking.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=40_000)
    parser.add_argument("--cpu-threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
