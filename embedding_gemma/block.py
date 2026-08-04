"""Gemma sandwich-norm encoder block."""

from torch import nn

from attention import Attention
from layers import GeGLU, RMSNorm


class Block(nn.Module):
    def __init__(self, cfg, local):
        super().__init__()
        self.pre_attention = RMSNorm(cfg.hidden_size, cfg.eps)
        self.attention = Attention(cfg, local)
        self.post_attention = RMSNorm(cfg.hidden_size, cfg.eps)
        self.pre_mlp = RMSNorm(cfg.hidden_size, cfg.eps)
        self.mlp = GeGLU(cfg)
        self.post_mlp = RMSNorm(cfg.hidden_size, cfg.eps)

    def forward(self, x, mask):
        x = x + self.post_attention(self.attention(self.pre_attention(x), mask))
        return x + self.post_mlp(self.mlp(self.pre_mlp(x))) * mask.unsqueeze(-1)
