"""Fast baseline: character embeddings -> masked mean -> two-layer projection."""

import torch
from torch import nn
import torch.nn.functional as F

from config import EmbeddingConfig


class MeanEncoder(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_id)
        self.projection = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.embedding_dim),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(ids)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return F.normalize(self.projection(pooled), dim=-1)
