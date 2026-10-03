import math

import torch
from torch import Tensor, nn


def flatten_parameters(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1).float().cpu() for parameter in model.parameters()])


# used for monitoring the progress of the training
def relative_weight_change(current: Tensor, previous: Tensor, epoch_gap: int) -> float:
    if epoch_gap <= 0:
        return 0.0
    # ||theta_now - theta_previous|| / (||theta_previous|| * elapsed epochs)
    displacement = torch.linalg.vector_norm(current - previous)
    previous_norm = torch.linalg.vector_norm(previous).clamp_min(1e-12)
    return float(displacement / previous_norm / epoch_gap)


# used for visualizing the embedding of the numbers
def dominant_embedding_frequencies(
    token_embedding: Tensor,
    prime: int,
    count: int = 3,
) -> list[tuple[int, float]]:
    number_embeddings = token_embedding[:prime]
    centered = number_embeddings - number_embeddings.mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=0).abs().mean(dim=1)[1:]  # Drop constant frequency 0.
    values, indices = spectrum.topk(min(count, spectrum.shape[0]))
    return [(int(index) + 1, float(value)) for value, index in zip(values, indices)]


# used for visualizing the embedding of the numbers
def clock_coordinates(token_embedding: Tensor, prime: int, frequency: int) -> tuple[Tensor, Tensor]:
    embeddings = token_embedding[:prime].detach().float().cpu()
    numbers = torch.arange(prime, dtype=torch.float32)
    angles = 2 * math.pi * frequency * numbers / prime
    cosine_direction = embeddings.T @ torch.cos(angles)
    sine_direction = embeddings.T @ torch.sin(angles)
    cosine_direction = cosine_direction / cosine_direction.norm().clamp_min(1e-12)
    sine_direction = sine_direction - cosine_direction * torch.dot(sine_direction, cosine_direction)
    sine_direction = sine_direction / sine_direction.norm().clamp_min(1e-12)
    return embeddings @ cosine_direction, embeddings @ sine_direction
