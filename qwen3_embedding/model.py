"""Tiny Qwen3-Embedding: causal decoder, instruction token, EOS pooling, MRL."""

import torch
from torch import nn
import torch.nn.functional as F

from block import Block
from rms_norm import RMSNorm


class TinyQwen3Embedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def forward(self, ids, mask, output_dim=None):
        x = self.embedding(ids) * mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        last = mask.long().sum(-1) - 1
        pooled = x[torch.arange(x.size(0), device=x.device), last]
        vectors = self.projection(pooled)
        dimension = output_dim or self.cfg.embedding_dim
        if dimension not in self.cfg.matryoshka_dims:
            raise ValueError(f"output_dim must be one of {self.cfg.matryoshka_dims}")
        return F.normalize(vectors[:, :dimension], dim=-1)
