"""Tiny EmbeddingGemma: bidirectional encoder, mean pool, 4x linear neck."""

import math

from torch import nn
import torch.nn.functional as F

from block import Block
from layers import RMSNorm


class TinyEmbeddingGemma(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList(Block(cfg, local=index < cfg.num_layers - 1) for index in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        # Official recipe uses two linear layers with identity activation.
        self.expand = nn.Linear(cfg.hidden_size, cfg.projection_size, bias=False)
        self.compress = nn.Linear(cfg.projection_size, cfg.embedding_dim, bias=False)

    def forward(self, ids, mask, output_dim=None):
        x = self.embedding(ids) * math.sqrt(self.cfg.hidden_size)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x) * mask.unsqueeze(-1)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        vectors = self.compress(self.expand(pooled))
        dimension = output_dim or self.cfg.embedding_dim
        if dimension not in self.cfg.matryoshka_dims:
            raise ValueError(f"output_dim must be one of {self.cfg.matryoshka_dims}")
        return F.normalize(vectors[:, :dimension], dim=-1)
