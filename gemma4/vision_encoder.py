"""The vision encoder: attention with 2-D rotary positions, and the tower.

**Multidimensional RoPE.** A patch has an (x, y), not a single index, so each
head is split into `ndim = 2` equal halves and each half is rotated by its own
coordinate - the first half by x, the second by y. The two halves are rotated
independently and concatenated back, so one head carries both axes.

Everything else mirrors the text attention: q, k and v are all RMS-normalized
(with `v_norm` weight-less), and `scale = 1.0` because the q/k norms carry the
temperature. What is gone is causality - a patch may see every other patch of
its image - and GQA, since the tower uses as many kv heads as query heads.

The encoder layer is the plain Gemma sandwich: norm before and after each
sub-layer, residual around the pair. No per-layer embeddings, no experts.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import VisionConfig
from rms_norm import RMSNorm
from rotary import _rotate_half
from clipped_linear import ClippableLinear
from mlp import gelu_tanh

NDIM = 2   # patches are positioned in two dimensions


def precompute_cos_sin_2d(head_dim: int, position_ids: torch.Tensor, theta: float):
    """Return cos, sin of shape [B, P, head_dim] for per-patch (x, y) ids.

    Each spatial axis gets its OWN full frequency range computed over
    `head_dim // 2` channels - not one global range split between the axes.
    """
    spatial_dim = head_dim // NDIM
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2,
                                             device=position_ids.device).float() / spatial_dim))
    parts_cos, parts_sin = [], []
    for axis in range(NDIM):
        angles = position_ids[:, :, axis].float()[..., None] * inv_freq   # [B, P, spatial/2]
        angles = torch.cat([angles, angles], dim=-1)                      # [B, P, spatial]
        parts_cos.append(angles.cos())
        parts_sin.append(angles.sin())
    return torch.cat(parts_cos, dim=-1), torch.cat(parts_sin, dim=-1)


def apply_rotary_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x [B, P, heads, head_dim]: first half by x, second half by y."""
    width = 2 * (x.shape[-1] // (2 * NDIM))
    sizes = [width] * NDIM
    out = []
    for x_part, cos_part, sin_part in zip(torch.split(x, sizes, dim=-1),
                                          torch.split(cos, sizes, dim=-1),
                                          torch.split(sin, sizes, dim=-1)):
        cos_part = cos_part.unsqueeze(2)   # broadcast over heads
        sin_part = sin_part.unsqueeze(2)
        out.append(x_part * cos_part + _rotate_half(x_part) * sin_part)
    return torch.cat(out, dim=-1)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_repeat = cfg.num_heads // cfg.num_kv_heads

        clip = cfg.use_clipped_linears
        self.q_proj = ClippableLinear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, clip)
        self.k_proj = ClippableLinear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, clip)
        self.v_proj = ClippableLinear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, clip)
        self.o_proj = ClippableLinear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, clip)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, with_scale=False)

    def forward(self, x, cos, sin, mask):
        B, P, _ = x.shape
        shape = (B, P, -1, self.head_dim)

        q = apply_rotary_2d(self.q_norm(self.q_proj(x).view(shape)), cos, sin).transpose(1, 2)
        k = apply_rotary_2d(self.k_norm(self.k_proj(x).view(shape)), cos, sin).transpose(1, 2)
        v = self.v_norm(self.v_proj(x).view(shape)).transpose(1, 2)

        if self.n_repeat > 1:
            k = k.repeat_interleave(self.n_repeat, dim=1)
            v = v.repeat_interleave(self.n_repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = out.transpose(1, 2).reshape(B, P, self.num_heads * self.head_dim)
        return self.o_proj(out)


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        clip = cfg.use_clipped_linears
        self.gate_proj = ClippableLinear(cfg.hidden_size, cfg.intermediate_size, clip)
        self.up_proj = ClippableLinear(cfg.hidden_size, cfg.intermediate_size, clip)
        self.down_proj = ClippableLinear(cfg.intermediate_size, cfg.hidden_size, clip)

    def forward(self, x):
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


class VisionEncoderLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, mask):
        residual = x
        x = self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cos, sin, mask))
        x = residual + x

        residual = x
        x = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return residual + x


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([VisionEncoderLayer(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x, position_ids, valid):
        """valid: [B, P] bool, True where the patch is real (not padding)."""
        cos, sin = precompute_cos_sin_2d(self.cfg.head_dim, position_ids, self.cfg.rope_theta)
        # Bidirectional: every real patch may attend to every real patch.
        mask = valid[:, None, None, :].expand(-1, 1, valid.shape[1], -1)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x
