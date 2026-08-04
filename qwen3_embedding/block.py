from torch import nn

from attention import Attention
from mlp import SwiGLU
from rms_norm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, mask):
        x = x + self.attention(self.attention_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x)) * mask.unsqueeze(-1)
