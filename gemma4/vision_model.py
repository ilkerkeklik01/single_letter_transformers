"""The complete vision tower: pixels in, soft tokens out.

Flow:  flat patches -> patch embed + 2-D positions -> bidirectional encoder
       -> average-pool k x k blocks -> drop padding -> (optional standardize)

The number of soft tokens is not a config constant: it is
`num_patches // pooling_kernel_size**2`, so a bigger image simply produces more
tokens. The released towers pool 3x3, and the e2b processor feeds 2520 patches
to get the 280 soft tokens per image the text side expects.

The last step is worth noting: `hidden[mask]` FLATTENS the batch. Different
images in a batch keep different numbers of real patches, so the tower returns
one long `[total_valid_tokens, hidden]` run rather than a padded rectangle -
exactly the shape the multimodal embedder wants for scattering into the token
stream.
"""

import torch
from torch import nn

from config import VisionConfig
from vision_embed import PatchEmbedder, Pooler
from vision_encoder import VisionEncoder


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedder = PatchEmbedder(cfg)
        self.encoder = VisionEncoder(cfg)
        self.pooler = Pooler(cfg)
        if cfg.standardize:
            # Calibration constants, saved but not trained (like the clip bounds).
            self.register_buffer("std_bias", torch.zeros(cfg.hidden_size))
            self.register_buffer("std_scale", torch.ones(cfg.hidden_size))

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """pixel_values [B, P, patch_dim], position_ids [B, P, 2] (-1 = padding).

        Returns [total_valid_soft_tokens, hidden] - batch-flattened.
        """
        k = self.cfg.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (k * k)

        padding = (position_ids == -1).all(dim=-1)
        x = self.patch_embedder(pixel_values, position_ids, padding)
        x = self.encoder(x, position_ids, valid=~padding)

        x, pooled_padding = self.pooler(x, position_ids, padding, output_length)
        x = x[pooled_padding]                     # drop padding, flattening the batch

        if self.cfg.standardize:
            x = (x - self.std_bias.float()) * self.std_scale.float()
        return x.to(self.patch_embedder.input_proj.weight.dtype)
