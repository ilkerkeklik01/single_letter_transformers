"""SwiGLU feed-forward - unchanged from Qwen3, and used two ways.

In the dense 27B it is the whole feed-forward. In the sparse 2.4T it appears
twice over: once as each expert's body (packed into stacked tensors, see
`moe.py`), and once as the **shared expert** that every token passes through
regardless of routing.
"""

from torch import nn
import torch.nn.functional as F

from config import TextConfig


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig, intermediate_size: int = None):
        super().__init__()
        width = intermediate_size if intermediate_size is not None else cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
