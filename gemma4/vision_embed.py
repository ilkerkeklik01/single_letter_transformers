"""Getting pixels into - and soft tokens out of - the vision tower.

Two modules that bracket the encoder.

`PatchEmbedder` turns already-flattened patches into hidden vectors. Two details
are Gemma-specific:

  * pixels are rescaled in MODEL code, not by the processor: `2 * (x - 0.5)`
    maps [0, 1] to [-1, 1]. The processor deliberately does no normalization.
  * positions come from **two separate 1-D tables**, one for the x coordinate
    and one for the y, stored together as a single `(2, size, hidden)`
    parameter. A patch's position embedding is the SUM of its x row and its y
    row. That is much cheaper than a full 2-D table and lets the tower handle
    any aspect ratio up to `position_embedding_size` patches on a side.

Padding patches are marked `(-1, -1)`. Their indices are clamped to 0 so the
lookup stays in range, then their embeddings are zeroed outright.

`Pooler` is what makes vision affordable for the language model: it averages
each `k x k` block of neighboring patches into one soft token, so a grid of
patches becomes `num_patches / k^2` tokens. The averaging is written as a
one-hot matmul rather than a reshape, because patches arrive in a padded,
ragged batch and their (x, y) coordinates - not their order - decide which
block they belong to. Finally it scales by `sqrt(hidden_size)` in float32,
because that multiply can overflow float16.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import VisionConfig


class PatchEmbedder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.patch_dim, cfg.hidden_size, bias=False)
        # Index 0 is the x table, index 1 the y table.
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, cfg.position_embedding_size, cfg.hidden_size))

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor,
                padding: torch.Tensor) -> torch.Tensor:
        """pixel_values [B, P, patch_dim], position_ids [B, P, 2], padding [B, P]."""
        pixel_values = 2 * (pixel_values - 0.5)          # [0,1] -> [-1,1], in model code
        hidden = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))

        clamped = position_ids.clamp(min=0)              # -1 marks padding
        pos = (F.embedding(clamped[..., 0], self.position_embedding_table[0])
               + F.embedding(clamped[..., 1], self.position_embedding_table[1]))
        pos = torch.where(padding.unsqueeze(-1), 0.0, pos)
        return hidden + pos


class Pooler(nn.Module):
    """Average k x k blocks of patches into soft tokens, then scale."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.root_hidden_size = cfg.hidden_size**0.5

    def _avg_pool(self, hidden: torch.Tensor, position_ids: torch.Tensor, length: int):
        seq_len = hidden.shape[1]
        k = int((seq_len // length) ** 0.5)
        if k * k * length != seq_len:
            raise ValueError(
                f"Cannot pool {tuple(hidden.shape)} to {length}: k={k}^2 times "
                f"length={length} must equal {seq_len}.")

        clamped = position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        block = torch.div(clamped, k, rounding_mode="floor")
        # Flatten the 2-D block coordinate into one index per output token.
        block = block[..., 0] + (max_x // k) * block[..., 1]

        weights = F.one_hot(block.long(), length).float() / (k * k)
        pooled = weights.transpose(1, 2) @ hidden.float()
        # An output slot no patch mapped into is padding.
        mask = torch.logical_not((weights == 0).all(dim=1))
        return pooled.to(hidden.dtype), mask

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                padding: torch.Tensor, output_length: int):
        if output_length > hidden.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens ({output_length}) than there are "
                f"patches ({hidden.shape[1]}).")

        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        if hidden.shape[1] != output_length:
            hidden, padding = self._avg_pool(hidden, position_ids, output_length)

        # sqrt(hidden) can push past the float16 max, so stay in float32 and let
        # the caller standardize and cast back.
        return hidden.float() * self.root_hidden_size, padding
