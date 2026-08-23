"""Patch embedding + interpolated learned position embeddings.

Muse Glimmer's tower does NOT use a Conv2d patchifier: the image processor
already flattens each patch (frames * RGB * patch * patch values) and the
"patchifier" is a single bias-free Linear over that vector.

Positions are a learned `side x side` table (32x32 in the 30B). Real images
rarely arrive as exactly side x side patches, so the table is BILINEARLY
INTERPOLATED to the actual h x w grid: each target patch reads the 4 table
entries around its (fractional) position and blends them. The sampling
matches `F.grid_sample(align_corners=False, padding_mode="zeros")` - half-pixel
centers, and corners that fall outside the table contribute zero (that is what
the validity masks do).
"""

import torch
from torch import nn

from config import VisionConfig


class PatchEmbedder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.side = cfg.pos_emb_side
        self.patch_embedding = nn.Linear(cfg.patch_dim, cfg.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(cfg.pos_emb_side**2, cfg.hidden_size)

    def _axis_taps(self, n: int, device):
        """For n target positions on one axis: the two source rows they sit
        between (floor/ceil, clamped into the table) and the blend weights,
        with out-of-table taps marked invalid (their weight becomes 0)."""
        side = self.side
        grid = (torch.arange(n, device=device).float() + 0.5) * (side / n) - 0.5
        floor = torch.floor(grid).long()
        frac = grid - floor.float()
        taps, weights, valid = [], [], []
        for tap, weight in ((floor, 1 - frac), (floor + 1, frac)):
            valid.append((tap >= 0) & (tap <= side - 1))
            taps.append(tap.clamp(0, side - 1))
            weights.append(weight)
        return taps, weights, valid

    def _interpolated_positions(self, t: int, h: int, w: int, device) -> torch.Tensor:
        """[t*h*w, hidden] of blended position embeddings for an h x w grid."""
        h_taps, h_weights, h_valids = self._axis_taps(h, device)
        w_taps, w_weights, w_valids = self._axis_taps(w, device)

        pos = torch.zeros(h * w, self.position_embedding_table.embedding_dim, device=device)
        for h_tap, h_weight, h_valid in zip(h_taps, h_weights, h_valids):
            for w_tap, w_weight, w_valid in zip(w_taps, w_weights, w_valids):
                idx = (h_tap[:, None] * self.side + w_tap[None, :]).flatten()
                weight = (h_weight[:, None] * w_weight[None, :]
                          * (h_valid[:, None] & w_valid[None, :])).flatten()
                pos = pos + self.position_embedding_table(idx) * weight[:, None]
        return pos.repeat(t, 1)   # every frame of a video shares the same grid

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values: [total_patches, patch_dim] packed across all images."""
        embeddings = self.patch_embedding(pixel_values)
        pos = torch.cat(
            [self._interpolated_positions(t, h, w, pixel_values.device)
             for t, h, w in grid_thw.tolist()]
        )
        return embeddings + pos.to(embeddings.dtype)
