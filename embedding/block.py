"""Pre-norm encoder block: bidirectional attention followed by SwiGLU."""

from torch import nn

from attention import BidirectionalAttention
from config import EmbeddingConfig
from mlp import SwiGLU
from rms_norm import RMSNorm


class EncoderBlock(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = BidirectionalAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, mask):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x)) * mask.unsqueeze(-1)
        return x
