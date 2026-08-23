"""Causal self-attention with the four Muse Glimmer twists:

  1. QK-Norm, weight-less + a scale factor
       q and k both go through the SAME weight-less RMSNorm (there is nothing
       to learn, so one module serves both), and q is then multiplied by the
       constant `qk_scale_factor` (3.87 in the released model) ON TOP of the
       usual 1/sqrt(head_dim) attention scaling.

  2. Sliding vs full/NoPE layers
       Three of every four layers use a sliding-window causal mask (each token
       sees only the last `sliding_window` tokens) and RoPE. The fourth layer
       sees everything - but gets NO positional embedding at all (NoPE).

  3. Output gating
       Before the output projection, the merged head outputs are multiplied by
       `sigmoid(gate_proj(x))` - a learned, per-feature gate computed from the
       SAME input the attention read. Heads that aren't useful for this token
       can be turned down to zero.

  4. Decoupled head width
       num_heads * head_dim does not have to equal hidden_size (in the 30B it
       is 4096 vs 6656). q/k/v project hidden -> heads, o projects back.

Plus GQA, as everywhere else: fewer key/value heads than query heads.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Expand KV heads so each query head has a matching KV head (GQA).

    x: [B, num_kv_heads, T, head_dim] -> [B, num_kv_heads * n_repeat, T, head_dim]
    """
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_repeat = cfg.num_heads // cfg.num_kv_heads
        self.qk_scale_factor = cfg.qk_scale_factor

        # This layer's place in the [sliding, sliding, sliding, full] schedule.
        self.layer_type = cfg.layer_types[layer_idx]           # which mask to use
        self.use_rope = cfg.layer_rope_theta[layer_idx] != 0   # full layers are NoPE

        # Projections (no bias). Note gate_proj: hidden -> num_heads * head_dim,
        # same width as the merged attention output it will gate.
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

        # One weight-less QK-norm shared by q and k (it has no parameters).
        self.qk_norm = RMSNorm(eps=cfg.rms_norm_eps, with_scale=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden]. mask: boolean [T, T], True = may attend."""
        B, T, _ = x.shape

        # Project, then split into heads: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm first (q also picks up the extra scale factor), THEN rotate -
        # unless this is a NoPE layer, which skips rotation entirely.
        q = self.qk_norm(q) * self.qk_scale_factor
        k = self.qk_norm(k)
        if self.use_rope:
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        # GQA: grow KV heads to match the number of query heads.
        k = repeat_kv(k, self.n_repeat)
        v = repeat_kv(v, self.n_repeat)

        # The mask (causal or sliding-causal) is built once in model.py and
        # handed in. scaling defaults to 1/sqrt(head_dim), as in the reference.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Merge heads, gate, project out.
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out)
