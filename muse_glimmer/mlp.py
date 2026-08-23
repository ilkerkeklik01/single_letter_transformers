"""SwiGLU feed-forward network - the one part of Muse Glimmer with no twist.

Same recipe as Qwen3 and Gemma: a "gate" projection passed through SiLU,
multiplied elementwise with an "up" projection, then projected back down.
All three linears are bias-free.
"""

from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
