"""Full softmax attention - the layer that appears once every four.

Three linear-attention layers give cheap, unbounded-context mixing; this layer
gives the exact, content-addressed lookup that a fixed-size memory cannot. The
architecture alternates them 3:1.

Qwen3.8 specifics:

  * **Output gating, folded into q_proj.** `q_proj` is built *twice* as wide as
    the queries need - `num_heads * head_dim * 2`. The first half is the query;
    the second half becomes `sigmoid(gate)` and multiplies the attention output
    before `o_proj`. So the gate is not a separate parameter you can spot in the
    checkpoint; it is hiding inside a double-width query projection. (This is
    what the released config calls `attn_output_gate = True`.)
  * **QK-Norm** on the per-head query and key vectors, as in Qwen3.
  * **Partial RoPE** - only the first 25% of each head rotates (see rotary.py).
  * **GQA** - 4 key/value heads for 24 (27B) or 64 (2.4T) query heads.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import TextConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Expand KV heads so each query head has a matching KV head (GQA)."""
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_repeat = cfg.num_heads // cfg.num_kv_heads
        self.scaling = cfg.head_dim**-0.5

        # Note the * 2: queries and their output gate share one projection.
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden]. mask: boolean [T, T], True = may attend."""
        B, T, _ = x.shape

        # Split the double-width projection into the query and its gate.
        query, gate = torch.chunk(
            self.q_proj(x).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)

        q = self.q_norm(query.view(B, T, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = repeat_kv(k, self.n_repeat)
        v = repeat_kv(v, self.n_repeat)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)

        out = out.transpose(1, 2).reshape(B, T, -1)
        out = out * torch.sigmoid(gate)          # the gate hidden in q_proj
        return self.o_proj(out)
