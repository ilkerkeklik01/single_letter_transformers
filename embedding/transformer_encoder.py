"""Tiny modern bidirectional Transformer producing one vector per name."""

import torch
from torch import nn
import torch.nn.functional as F

from block import EncoderBlock
from config import EmbeddingConfig
from rms_norm import RMSNorm


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(ids) * mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x) * mask.unsqueeze(-1)


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = TransformerBackbone(cfg)
        self.pool_score = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def token_features(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(ids, mask)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.token_features(ids, mask)
        scores = self.pool_score(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        return F.normalize(self.projection(pooled), dim=-1)
