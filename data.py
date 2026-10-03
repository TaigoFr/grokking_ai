from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModularAdditionSplit:
    train_tokens: Tensor
    train_labels: Tensor
    test_tokens: Tensor
    test_labels: Tensor
    prime: int

    @property
    def vocabulary_size(self) -> int:
        return self.prime + 1


def make_modular_addition_split(
    prime: int = 113,
    train_fraction: float = 0.3,
    seed: int = 598,
) -> ModularAdditionSplit:
    values = torch.arange(prime)
    pairs = torch.cartesian_prod(values, values)
    equals = torch.full((pairs.shape[0], 1), prime, dtype=torch.long)  # "=" uses token ID p.
    tokens = torch.cat((pairs, equals), dim=1)
    labels = torch.remainder(pairs[:, 0] + pairs[:, 1], prime)  # (a + b) mod p
    permutation = torch.randperm(pairs.shape[0], generator=torch.Generator().manual_seed(seed))
    train_count = int(train_fraction * pairs.shape[0])
    train_indices = permutation[:train_count]
    test_indices = permutation[train_count:]
    return ModularAdditionSplit(
        train_tokens=tokens[train_indices],
        train_labels=labels[train_indices],
        test_tokens=tokens[test_indices],
        test_labels=labels[test_indices],
        prime=prime,
    )
