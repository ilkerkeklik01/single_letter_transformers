"""The Qwen3.8 vision tower (27B only) - a packed, merge-ordered ViT.

Four ideas worth reading slowly.

**Patches are 3-D.** One `Conv3d` of shape
`(temporal_patch_size, patch_size, patch_size)` embeds a *clip*, not a frame, so
images and video take exactly the same path - a still image is a clip whose
`temporal_patch_size` frames happen to be identical. That is why a "patch" is
`3 * 2 * 16 * 16 = 1536` numbers rather than `3 * 16 * 16`.

**Positions are learned, then resized.** `pos_embed` is a square `48 x 48` table
(2304 entries). Real images are not 48x48 patches, so the table is **bilinearly
resampled** to whatever grid each image has: every patch gathers the 4 table
entries around its fractional position and blends them. `align_corners=True`
here, so the grid's corners map exactly onto the table's corners.

**Patches arrive in merge-block order.** Not raster order. Because the tower ends
by folding each `2 x 2` block of patches into one token, the patches are
*already* laid out block by block, so the merge is a plain reshape. This is why
`interpolation_indices` has to undo `(block_row, block_col, in_row, in_col)` to
recover each patch's true (row, col).

**Attention is packed and per-image.** All images in a batch live in one long
sequence; `cu_seqlens` marks the boundaries, and attention runs separately inside
each segment so patches never see another image. It is bidirectional, with 2-D
rotary positions and no GQA.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import VisionConfig
from rotary import _rotate_half


def interpolation_indices(grid_thw: torch.Tensor, side: int, merge: int):
    """Per-patch (indices, weights) that resample a `side x side` table to each grid.

    Reproduces `F.interpolate(mode="bilinear", align_corners=True)` as a gather
    plus a weighted sum, and accounts for patches arriving in merge-block order.
    """
    device = grid_thw.device
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(int(counts.sum()), device=device) - starts) % (heights * widths)

    # Undo the merge-block ordering to recover each patch's (row, col).
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row, col = block_row * merge + in_row, block_col * merge + in_col

    def axis_taps(index, size):
        # align_corners=True: endpoints map exactly onto 0 and side-1.
        src = index.float() * (side - 1) / torch.clamp(size - 1, min=1)
        floor = torch.floor(src)
        taps = (floor.long()[:, None] + torch.arange(0, 2, device=device)).clamp(0, side - 1)
        distance = (src[:, None] - floor[:, None] - torch.arange(0, 2, device=device)).abs()
        return taps, (1 - distance).clamp(min=0)

    h_taps, h_weights = axis_taps(row, heights)
    w_taps, w_weights = axis_taps(col, widths)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def vision_position_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """[total_patches, 2] of (row, col), in merge-block order."""
    out = []
    for t, h, w in grid_thw.tolist():
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        shape = (h // merge, merge, w // merge, merge)
        rows = rows.reshape(shape).transpose(1, 2).flatten()
        cols = cols.reshape(shape).transpose(1, 2).flatten()
        out.append(torch.stack([rows, cols], dim=-1).repeat(t, 1))
    return torch.cat(out)


def frame_boundaries(grid_thw: torch.Tensor) -> torch.Tensor:
    """Cumulative segment ends - one attention segment per frame."""
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(0), (1, 0))


class PatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        kernel = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size,
                              kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [total_patches, patch_dim] -> [total_patches, hidden]."""
        cfg = self.cfg
        x = x.view(-1, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, cfg.hidden_size)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x, cos, sin, boundaries):
        seq = x.shape[0]
        q, k, v = self.qkv(x).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        # Rotate in float32, as the reference does.
        dtype = q.dtype
        qf, kf = q.float(), k.float()
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = (qf * c + _rotate_half(qf) * s).to(dtype)
        k = (kf * c + _rotate_half(kf) * s).to(dtype)

        q, k, v = [t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)]

        # Attend within each frame independently (bidirectional).
        lengths = (boundaries[1:] - boundaries[:-1]).tolist()
        parts = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        out = torch.cat([
            F.scaled_dot_product_attention(qi, ki, vi, scale=self.scaling)
            for qi, ki, vi in zip(*parts)], dim=2)

        return self.proj(out.transpose(1, 2).reshape(seq, -1).contiguous())


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, cos, sin, boundaries):
        x = x + self.attn(self.norm1(x), cos, sin, boundaries)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """Fold each merge x merge block of patches into one language-model token."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size * cfg.spatial_merge_size**2
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def forward(self, x):
        # Patches already arrive block by block, so the merge is just a view.
        x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = PatchMerger(cfg)

        head_dim = cfg.hidden_size // cfg.num_heads
        self.register_buffer(
            "inv_freq",
            1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))),
            persistent=False)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values [total_patches, patch_dim] -> [num_merged_tokens, out_hidden]."""
        merge = self.cfg.spatial_merge_size

        indices, weights = interpolation_indices(grid_thw, self.num_grid_per_side, merge)
        x = self.patch_embed(pixel_values)
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        x = x + pos.to(x.dtype)

        # 2-D rotary: (row, col) each get half the head, concatenated then doubled.
        position_ids = vision_position_ids(grid_thw, merge).to(x.device)
        freqs = (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()

        boundaries = frame_boundaries(grid_thw)
        for block in self.blocks:
            x = block(x, cos, sin, boundaries)

        return self.merger(x)
