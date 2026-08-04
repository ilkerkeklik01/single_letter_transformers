"""Rotary position embeddings for a short bidirectional sequence."""

import torch


def rotary_cache(length: int, dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = torch.outer(torch.arange(length, device=device).float(), frequencies)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
