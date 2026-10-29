"""A shared item encoder and history encoder; no learned user identifiers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HistoryRecommender(nn.Module):
    def __init__(self, genre_features: torch.Tensor, dimension: int = 32):
        super().__init__()
        count, genres = genre_features.shape
        self.register_buffer("genres", torch.cat([torch.zeros(1, genres), genre_features]))
        self.item_embedding = nn.Embedding(count + 1, dimension, padding_idx=0)
        self.genre_projection = nn.Linear(genres, dimension, bias=False)
        self.item_tower = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh())
        self.user_tower = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh())

    def encode_items(self, indices: torch.Tensor) -> torch.Tensor:
        vectors = self.item_embedding(indices) + self.genre_projection(self.genres[indices])
        return F.normalize(self.item_tower(vectors), dim=-1)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        mask = (history != 0).unsqueeze(-1)
        vectors = self.encode_items(history)
        pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.user_tower(pooled), dim=-1)

    def forward(self, history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        users = self.encode_history(history)
        items = self.encode_items(candidates)
        return (users.unsqueeze(1) * items).sum(dim=-1) / 0.1
