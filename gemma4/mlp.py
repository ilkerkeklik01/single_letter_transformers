"""GeGLU feed-forward network - Gemma's gated MLP.

Identical in shape to Qwen's SwiGLU, but the gate goes through **GELU** (the
tanh approximation) instead of SiLU. Hence Ge-GLU rather than Swi-GLU.

The one twist is `use_double_wide_mlp`: on the layers that share their
keys/values with an earlier layer, the MLP is built **twice as wide**. That is a
deliberate trade - those layers saved a lot of parameters by not carrying their
own k/v projections, so Gemma 4 spends some of the savings back on a fatter
feed-forward. `config.uses_double_wide_mlp(layer_idx)` decides.
"""

from torch import nn
import torch.nn.functional as F

from config import ModelConfig


def gelu_tanh(x):
    """GELU with the tanh approximation ("gelu_pytorch_tanh")."""
    return F.gelu(x, approximate="tanh")


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        width = cfg.intermediate_size * (2 if cfg.uses_double_wide_mlp(layer_idx) else 1)
        self.gate_proj = nn.Linear(cfg.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))
