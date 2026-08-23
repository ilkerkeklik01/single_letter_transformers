"""The full vision tower: packed patches in, merged vision tokens out.

Flow:  flat patches -> linear patch embed + interpolated positions -> ln_pre
       -> reorder into window tiles -> N encoder layers (window tiles or whole
       frames, per the schedule) -> undo the reorder -> ln_post
       -> pixel shuffle (2x2 patches fold into channels)

The pixel shuffle at the end is what connects vision to language: every
`merge_size x merge_size` block of neighboring patches is CONCATENATED along
the channel dimension into one vector of size `hidden * merge^2` (6144 in the
30B) - so a 32x32-patch image becomes 16x16 language-model tokens, each fat
enough to carry its whole block. The channel layout is dim-major (all 4
positions of channel 0, then channel 1, ...), matching the released adapter
weights.
"""

import torch
from torch import nn

from config import VisionConfig
from vision_embed import PatchEmbedder
from vision_block import VisionEncoderLayer
from vision_attention import precompute_cos_sin_2d
from vision_windows import frame_boundaries, position_ids, window_reorder


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedder = PatchEmbedder(cfg)
        self.ln_pre = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList([VisionEncoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.ln_post = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def pixel_shuffle(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """[total_patches, dim] -> [total_patches / merge^2, dim * merge^2]."""
        factor = self.cfg.merge_size
        dim = x.shape[-1]
        out = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            frame = x[offset:offset + t * h * w]
            # Gather each merge block's patches next to each other...
            perm = torch.arange(h * w, device=x.device)
            perm = perm.view(h // factor, factor, w // factor, factor).permute(0, 2, 1, 3).reshape(-1)
            if t > 1:   # same permutation for every frame, shifted per frame
                perm = (perm[None] + torch.arange(t, device=x.device)[:, None] * h * w).reshape(-1)
            # ...then fold the block into the channel dim (dim-major layout).
            merged = frame[perm].view(-1, factor * factor, dim)
            merged = merged.permute(0, 2, 1).reshape(-1, dim * factor * factor)
            out.append(merged)
            offset += t * h * w
        return torch.cat(out)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values: [total_patches, patch_dim], grid_thw: [n_images, 3]."""
        frames = frame_boundaries(grid_thw)
        windows_index, windows = window_reorder(grid_thw, self.cfg.pos_emb_side)

        x = self.ln_pre(self.patch_embedder(pixel_values, grid_thw))
        x = x[windows_index]   # window tiles are now contiguous

        # 2D rotary tables, in window order, with the reference's +1 offset.
        pos = position_ids(grid_thw)[windows_index] + 1
        cos, sin = precompute_cos_sin_2d(self.cfg.head_dim, pos[:, 0], pos[:, 1],
                                         self.cfg.rope_theta)

        boundaries = {"full_attention": frames, "window_attention": windows}
        for layer_type, layer in zip(self.cfg.layer_types, self.layers):
            x = layer(x, cos, sin, boundaries[layer_type])

        x = x[torch.argsort(windows_index)]   # back to row-major order
        x = self.ln_post(x)
        return self.pixel_shuffle(x, grid_thw)
