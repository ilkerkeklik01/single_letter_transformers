"""Rotary positions in Qwen3.8: partial, and interleaved across three axes.

Two things are unusual.

**Partial rotary.** Only the first `partial_rotary_factor` of each head rotates
- 25% in both releases, so 64 of every 256 channels. Unlike Gemma 4 (which zeroes
the unused frequencies and multiplies uniformly), Qwen splits the head, rotates
the front slice, and concatenates the untouched tail back on. The cos/sin tables
are therefore `rotary_dim` wide, not `head_dim` wide.

**Interleaved MRoPE.** Positions are 3-D: time, height and width. A text token
has the same index on all three axes, so for pure text this collapses to
ordinary RoPE - but for an image each patch carries a distinct (t, h, w).

The interleaving is the interesting part. A naive multi-axis RoPE gives each
axis a contiguous block of frequencies (`TTT...HHH...WWW`), which hands the time
axis all the high frequencies and the width axis all the low ones. Qwen instead
deals them out round-robin (`THWTHWTHW...`) so every axis gets the full spectrum.
`mrope_section` says how many triples each axis takes; the leftovers at the end
stay with time.
"""

import torch


def precompute_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    """One inverse frequency per rotating feature pair."""
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section) -> torch.Tensor:
    """[3, B, T, rotary_dim/2] -> [B, T, rotary_dim/2], axes dealt round-robin.

    Start from the time axis, then overwrite every 3rd slot with height and
    every 3rd slot with width, offset by one and two.
    """
    out = freqs[0].clone()                       # start with T everywhere
    for axis, offset in enumerate((1, 2), start=1):   # then H, then W
        length = mrope_section[axis] * 3
        out[..., offset:length:3] = freqs[axis, ..., offset:length:3]
    return out


def compute_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor,
                    mrope_section) -> tuple:
    """position_ids [3, B, T] (time/height/width) -> cos, sin of [B, T, rotary_dim]."""
    inv_freq = inv_freq.to(position_ids.device)
    # [3, B, T, 1] x [rotary_dim/2] -> [3, B, T, rotary_dim/2]
    freqs = position_ids.float()[..., None] * inv_freq
    freqs = apply_interleaved_mrope(freqs, mrope_section)
    emb = torch.cat([freqs, freqs], dim=-1)      # [B, T, rotary_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Turn [x1, x2] (split in half) into [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading `rotary_dim` channels of x [B, heads, T, head_dim].

    The rest of each head is passed through untouched and concatenated back.
    """
    cos = cos.unsqueeze(1)                       # broadcast over heads
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)
