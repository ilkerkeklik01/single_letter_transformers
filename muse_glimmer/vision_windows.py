"""Token bookkeeping for the vision tower: segments, positions, windows.

The tower never sees a [batch, seq, ...] tensor. All patches from all images
(and video frames) are PACKED into one long sequence, and attention is told
where each image starts and ends via boundary lists ("cu_seqlens" - cumulative
sequence lengths, the same convention flash-attention uses):

  * frame_boundaries  : one attention segment per FRAME (t segments for a
                        t-frame video, one for a still image). Full-attention
                        layers use these - every patch of a frame sees every
                        other patch of that frame, never a different image.

  * window_reorder    : window-attention layers see even less - only a
                        `window_side` x `window_side` tile of patches. Instead
                        of masking, the tokens are physically REORDERED so each
                        window is contiguous, attention runs per window
                        (window_boundaries), and the order is undone afterward
                        (torch.argsort of the same index). Partial windows at
                        the right/bottom edge simply become smaller segments.

  * position_ids      : the (row, col) of every patch, for the 2D rotary
                        embedding. Frames of a video share the same grid.
"""

import torch


def frame_boundaries(grid_thw: torch.Tensor) -> torch.Tensor:
    """[0, n1, n1+n2, ...] where each n is h*w patches of one frame."""
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return torch.nn.functional.pad(seqlens.cumsum(0), (1, 0))


def position_ids(grid_thw: torch.Tensor) -> torch.Tensor:
    """[total_patches, 2] of (row, col), row-major per frame."""
    out = []
    for t, h, w in grid_thw.tolist():
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        out.append(torch.stack([rows.flatten(), cols.flatten()], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def window_reorder(grid_thw: torch.Tensor, window_side: int):
    """Group each frame's patches into window tiles.

    Returns (index, boundaries): `index` is a permutation of all patch
    positions such that every window is contiguous, and `boundaries` are the
    cumulative window lengths for attention. Undo the permutation with
    `torch.argsort(index)`.
    """
    index_parts = []
    boundaries = [0]
    offset = 0
    for t, h, w in grid_thw.tolist():
        pad_h, pad_w = (-h) % window_side, (-w) % window_side
        # Lay the frame's patch indices on the grid, pad the ragged edges with
        # -1, cut into window tiles, then drop the padding again.
        idx = torch.arange(t * h * w).reshape(t, h, w)
        idx = torch.nn.functional.pad(idx, (0, pad_w, 0, pad_h), value=-1)
        idx = idx.reshape(t, (h + pad_h) // window_side, window_side,
                          (w + pad_w) // window_side, window_side)
        idx = idx.permute(0, 1, 3, 2, 4).reshape(t, -1, window_side, window_side)

        for window in idx.reshape(-1, window_side * window_side):
            window = window[window != -1]
            index_parts.append(window + offset)
            boundaries.append(boundaries[-1] + len(window))
        offset += t * h * w

    return torch.cat(index_parts), torch.tensor(boundaries)
