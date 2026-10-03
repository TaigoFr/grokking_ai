import math

import torch
from torch import Tensor, nn


CONTEXT_LENGTH = 3  # Every input is [a, b, =].


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dimension: int, head_count: int, head_dimension: int) -> None:
        super().__init__()
        self.head_count = head_count
        self.head_dimension = head_dimension
        self.query = nn.Linear(model_dimension, head_count * head_dimension)
        self.key = nn.Linear(model_dimension, head_count * head_dimension)
        self.value = nn.Linear(model_dimension, head_count * head_dimension)
        self.output_projection = nn.Linear(head_count * head_dimension, model_dimension)
        mask = torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, residual: Tensor) -> Tensor:
        batch_size, context_length, _ = residual.shape
        query = self._split_heads(self.query(residual), batch_size, context_length)
        key = self._split_heads(self.key(residual), batch_size, context_length)
        value = self._split_heads(self.value(residual), batch_size, context_length)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(self.head_dimension)
        scores = scores.masked_fill(self.causal_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bhqk,bkhd->bqhd", attention, value)
        mixed = mixed.reshape(batch_size, context_length, self.head_count * self.head_dimension)
        return self.output_projection(mixed)

    def _split_heads(self, projected: Tensor, batch_size: int, context_length: int) -> Tensor:
        return projected.view(batch_size, context_length, self.head_count, self.head_dimension)


class FeedForward(nn.Module):
    def __init__(self, model_dimension: int, mlp_dimension: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(model_dimension, mlp_dimension)
        self.output_projection = nn.Linear(mlp_dimension, model_dimension)

    def forward(self, residual: Tensor) -> Tensor:
        return self.output_projection(torch.relu(self.input_projection(residual)))


class TransformerBlock(nn.Module):
    def __init__(self, model_dimension: int, head_count: int, head_dimension: int, mlp_dimension: int) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(model_dimension, head_count, head_dimension)
        self.feed_forward = FeedForward(model_dimension, mlp_dimension)

    def forward(self, residual: Tensor) -> Tensor:
        residual = residual + self.attention(residual)
        return residual + self.feed_forward(residual)


class ModularAdditionTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        output_vocabulary_size: int,
        model_dimension: int = 128,
        head_count: int = 4,
        head_dimension: int = 32,
        mlp_dimension: int = 512,
    ) -> None:
        super().__init__()
        if head_count * head_dimension != model_dimension:
            raise ValueError("head_count * head_dimension must equal model_dimension")
        self.token_embedding = nn.Embedding(vocabulary_size, model_dimension)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, model_dimension)
        self.block = TransformerBlock(model_dimension, head_count, head_dimension, mlp_dimension)
        self.unembedding = nn.Linear(model_dimension, output_vocabulary_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        residual = self.token_embedding(tokens) + self.position_embedding(positions)
        residual = self.block(residual)
        return self.unembedding(residual)[:, -1, :]  # Predict the answer at the "=" position.


def initialize_parameters(model: ModularAdditionTransformer, seed: int) -> None:
    torch.manual_seed(seed)
    model_dimension = model.token_embedding.weight.shape[1]
    # Reference gain 0.8; dividing by sqrt(width) keeps summed activation variance stable.
    standard_deviation = 0.8 / math.sqrt(model_dimension)
    # Attention and MLP each add a residual branch, so 1/sqrt(2) balances their combined variance.
    residual_standard_deviation = standard_deviation / math.sqrt(2)
    _normal(model.token_embedding.weight, standard_deviation)
    _normal(model.position_embedding.weight, standard_deviation)
    _normal(model.unembedding.weight, standard_deviation)
    attention = model.block.attention
    for projection in (attention.query, attention.key, attention.value):
        _normal(projection.weight, standard_deviation)
        nn.init.zeros_(projection.bias)
    _normal(attention.output_projection.weight, residual_standard_deviation)
    nn.init.zeros_(attention.output_projection.bias)
    feed_forward = model.block.feed_forward
    _normal(feed_forward.input_projection.weight, standard_deviation)
    nn.init.zeros_(feed_forward.input_projection.bias)
    _normal(feed_forward.output_projection.weight, residual_standard_deviation)
    nn.init.zeros_(feed_forward.output_projection.bias)


def _normal(weight: Tensor, standard_deviation: float) -> None:
    nn.init.normal_(weight, mean=0.0, std=standard_deviation)
