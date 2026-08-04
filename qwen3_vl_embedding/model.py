"""Tiny multimodal dual tower with shared EOS pooling and MRL projection."""

import torch
from torch import nn
import torch.nn.functional as F

from backbone import QwenBackbone
from vision import VisionEncoder


class TinyQwen3VLEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.text_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_id)
        self.vision = VisionEncoder(cfg)
        self.text_modality = nn.Parameter(torch.randn(cfg.hidden_size) * .02)
        self.image_modality = nn.Parameter(torch.randn(cfg.hidden_size) * .02)
        self.eos = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * .02)
        self.backbone = QwenBackbone(cfg)
        self.projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def _pool(self, tokens, mask, output_dim):
        eos = self.eos.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((tokens, eos), dim=1)
        mask = torch.cat((mask, torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)), dim=1)
        hidden = self.backbone(tokens, mask)[:, -1]
        dimension = output_dim or self.cfg.embedding_dim
        if dimension not in self.cfg.matryoshka_dims:
            raise ValueError(f"output_dim must be one of {self.cfg.matryoshka_dims}")
        return F.normalize(self.projection(hidden)[:, :dimension], dim=-1)

    def encode_text(self, ids, mask, output_dim=None):
        tokens = self.text_embedding(ids) + self.text_modality
        return self._pool(tokens, mask, output_dim)

    def encode_image(self, images, output_dim=None):
        tokens = self.vision(images) + self.image_modality
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return self._pool(tokens, mask, output_dim)

    def encode_fused(self, ids, mask, images, output_dim=None):
        image_tokens = self.vision(images) + self.image_modality
        text_tokens = self.text_embedding(ids) + self.text_modality
        tokens = torch.cat((image_tokens, text_tokens), dim=1)
        image_mask = torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return self._pool(tokens, torch.cat((image_mask, mask), dim=1), output_dim)
