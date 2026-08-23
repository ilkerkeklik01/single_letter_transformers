"""Vision self-attention: 2D rotary, packed segments, no mask in sight.

Differences from the text attention worth noticing:

  * 2D RoPE. A patch has a (row, col), not a single position. Frequencies are
    computed for HALF the head_dim (`spatial_dim = head_dim // 2`) and the
    angle vector is interleaved `[col, row, col, row]` - each spatial axis
    rotates its own quarter-pairs of the head. The rotation runs in float32.
    (The +1 offset on the ids is applied by the caller, as in the reference.)

  * No masks, no causality. An image patch may look at every other patch in
    its segment - a window tile for window layers, a whole frame for full
    layers. The packed sequence is simply SPLIT at the segment boundaries and
    attention runs per segment.

  * Bias everywhere, no GQA, no QK-norm, no gate - the tower is a plain ViT
    at heart; all the exotic machinery lives on the text side.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import VisionConfig
from rotary import _rotate_half


def precompute_cos_sin_2d(head_dim: int, rows: torch.Tensor, cols: torch.Tensor,
                          theta: float):
    """Return cos, sin of shape [seq, head_dim] for per-patch (row, col) ids."""
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2).float() / spatial_dim))
    freq_r = torch.outer(rows.float(), inv_freq)         # [seq, head_dim/4]
    freq_c = torch.outer(cols.float(), inv_freq)
    angles = torch.cat([freq_c, freq_r, freq_c, freq_r], dim=-1)   # [seq, head_dim]
    return angles.cos(), angles.sin()


def apply_rotary_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape [seq, n_heads, head_dim] in float32, cast back."""
    dtype = x.dtype
    x, cos, sin = x.float(), cos[:, None, :].float(), sin[:, None, :].float()
    return (x * cos + _rotate_half(x) * sin).to(dtype)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim

        # All four projections carry biases (checkpoint names: attn.q_proj,
        # attn.k_proj, attn.v_proj, attn.proj).
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                boundaries: torch.Tensor) -> torch.Tensor:
        """x: [seq, hidden] packed patches. boundaries: cumulative segment ends."""
        seq = x.shape[0]
        q = self.q_proj(x).view(seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(seq, self.num_heads, self.head_dim)

        q = apply_rotary_2d(q, cos, sin)
        k = apply_rotary_2d(k, cos, sin)

        # Attend within each segment independently (bidirectional).
        out = torch.empty_like(q)
        for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
            seg = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1)[None],   # [1, heads, len, head_dim]
                k[start:end].transpose(0, 1)[None],
                v[start:end].transpose(0, 1)[None],
            )
            out[start:end] = seg[0].transpose(0, 1)

        return self.proj(out.reshape(seq, -1))
