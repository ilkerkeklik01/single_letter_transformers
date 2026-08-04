"""Vision patch encoder with a Qwen-VL-style 2x2 spatial merger."""

import torch
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(1, cfg.vision_dim, cfg.patch_size, stride=cfg.patch_size, bias=False)
        grid_h, grid_w = cfg.image_height // cfg.patch_size, cfg.image_width // cfg.patch_size
        self.grid_h, self.grid_w = grid_h, grid_w
        self.position = nn.Parameter(torch.randn(1, grid_h * grid_w, cfg.vision_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.vision_dim, cfg.vision_heads, 2 * cfg.vision_dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.merger = nn.Sequential(
            nn.Linear(4 * cfg.vision_dim, cfg.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
        )

    def forward(self, images):
        patches = self.patch(images).flatten(2).transpose(1, 2) + self.position
        patches = self.encoder(patches)
        batch, _, dim = patches.shape
        grid = patches.view(batch, self.grid_h, self.grid_w, dim)
        # Every adjacent 2x2 patch group becomes one language-backbone token.
        grid = grid.view(batch, self.grid_h // 2, 2, self.grid_w // 2, 2, dim)
        merged = grid.permute(0, 1, 3, 2, 4, 5).reshape(batch, -1, 4 * dim)
        return self.merger(merged)
