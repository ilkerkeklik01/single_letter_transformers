"""Rotary Position Embeddings, in two flavors - one per attention type.

Sliding layers use ordinary RoPE with a small base (10k): every feature pair
rotates. Full-attention layers use **proportional RoPE** with a huge base (1M)
and `partial_rotary_factor = 0.25`.

"Proportional" is a neat trick worth understanding, because it is NOT
implemented by slicing the head in two. The frequency table is built at full
head width, but only the first quarter of the angle pairs get a real frequency;
the rest are set to **exactly zero**. A zero frequency means `cos = 1, sin = 0`,
so those features come out of the rotation untouched. The result is that 25% of
each head carries position and 75% is position-free (NoPE) - implemented as one
uniform multiply, with no reshaping and no branch.

  * precompute_cos_sin -> build cos/sin tables for one layer type
  * apply_rotary       -> rotate a query or key tensor with those tables
"""

import torch


def precompute_cos_sin(head_dim: int, max_seq_len: int, theta: float,
                       partial_rotary_factor: float = 1.0):
    """Return cos, sin tables of shape [max_seq_len, head_dim].

    With partial_rotary_factor < 1, the trailing angle pairs get frequency 0,
    which leaves those features unrotated.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2).float() / head_dim))

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles)])

    positions = torch.arange(max_seq_len).float()        # [max_seq_len]
    angles = torch.outer(positions, inv_freq)            # [max_seq_len, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)         # [max_seq_len, head_dim]
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Turn [x1, x2] (split in half) into [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, T, n_heads, head_dim].

    Note the layout: Gemma 4 rotates BEFORE transposing heads forward, so the
    tables broadcast over the head axis (dim 2), not the sequence axis.
    """
    cos = cos[None, :, None, :]   # [1, T, 1, head_dim]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin
