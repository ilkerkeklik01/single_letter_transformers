"""The complete Muse Glimmer: vision tower + bridge + language model.

How an image reaches the text stack:

  1. The tower turns packed patches into merged tokens of size hidden*merge^2.
  2. The ADAPTER (two bias-free linears, GELU after EACH - note the outer
     activation) reshapes that into projector space.
  3. The PROJECTION maps projector space into the text model's hidden size.
  4. A weight-less RMSNorm (`perception_emb_norm`) puts image tokens at unit
     RMS - the exact mirror of what `embed_norm` does to text tokens, so both
     modalities enter the decoder at the same scale.
  5. The features are written into the token sequence wherever the
     placeholder `image_token_id` (or `video_token_id`) appears - one
     placeholder per merged vision token, counts must match exactly.

Submodule names mirror the released checkpoint (language_model, vision_tower,
vision_adapter, vision_projection under a stripped "model." prefix), so
hf_weights.py can load the full 30B into this class the same way it loads the
text-only stack - see VLConfig.muse_glimmer_30b().
"""

import torch
from torch import nn

from config import VLConfig
from model import MuseGlimmer
from rms_norm import RMSNorm
from vision_model import VisionTower


class VisionAdapter(nn.Module):
    def __init__(self, cfg: VLConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.out_hidden_size, cfg.projector_hidden_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.projector_hidden_size, cfg.projector_hidden_size, bias=False)

    def forward(self, x):
        return self.act(self.fc2(self.act(self.fc1(x))))


class MuseGlimmerVL(nn.Module):
    def __init__(self, cfg: VLConfig):
        super().__init__()
        self.cfg = cfg
        self.language_model = MuseGlimmer(cfg.text)
        self.vision_tower = VisionTower(cfg.vision)
        self.vision_adapter = VisionAdapter(cfg)
        self.vision_projection = nn.Linear(cfg.projector_hidden_size,
                                           cfg.text.hidden_size, bias=False)
        # Weight-less, like embed_norm: no parameters, just unit RMS.
        self.perception_emb_norm = RMSNorm(eps=cfg.text.rms_norm_eps, with_scale=False)

    def get_image_features(self, pixel_values: torch.Tensor,
                           grid_thw: torch.Tensor) -> torch.Tensor:
        """Packed patches -> [n_merged_tokens, text_hidden], ready to scatter.

        Videos take the identical path - a video is just a grid with t > 1.
        """
        features = self.vision_tower(pixel_values, grid_thw)
        features = self.vision_adapter(features)
        features = self.vision_projection(features)
        return self.perception_emb_norm(features)

    def _scatter(self, x: torch.Tensor, idx: torch.Tensor, token_id: int,
                 pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Write vision features into x wherever idx == token_id (row-major)."""
        features = self.get_image_features(pixel_values, grid_thw).to(x.dtype)
        mask = idx == token_id
        assert int(mask.sum()) == features.shape[0], (
            f"{int(mask.sum())} placeholder tokens but {features.shape[0]} vision tokens"
        )
        x[mask] = features
        return x

    def embed(self, idx: torch.Tensor,
              pixel_values: torch.Tensor = None, image_grid_thw: torch.Tensor = None,
              pixel_values_videos: torch.Tensor = None, video_grid_thw: torch.Tensor = None,
              ) -> torch.Tensor:
        """Token ids (+ optional images/videos) -> [B, T, hidden] decoder input."""
        # Placeholder ids live outside the vocabulary; zero them for the
        # lookup - their embeddings are overwritten by vision features anyway.
        ids = idx.clone()
        multimodal = (ids == self.cfg.image_token_id) | (ids == self.cfg.video_token_id)
        ids[multimodal] = 0
        x = self.language_model.embed_norm(self.language_model.embed_tokens(ids))

        if pixel_values is not None:
            x = self._scatter(x, idx, self.cfg.image_token_id, pixel_values, image_grid_thw)
        if pixel_values_videos is not None:
            x = self._scatter(x, idx, self.cfg.video_token_id, pixel_values_videos, video_grid_thw)
        return x

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, **vision_inputs):
        """idx: [B, T] of token ids (with placeholders where images go).

        vision_inputs: pixel_values + image_grid_thw and/or
        pixel_values_videos + video_grid_thw, as in `embed`.
        Returns (logits, loss) exactly like the text-only model.
        """
        x = self.embed(idx, **vision_inputs)
        return self.language_model(inputs_embeds=x, targets=targets)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None, eos_id: int = None,
                 **vision_inputs) -> torch.Tensor:
        """Autoregressively extend idx [B, T], images embedded once up front.

        New tokens are embedded and appended to the prompt's embedding
        sequence, so the vision tower runs exactly once. The prompt plus all
        new tokens must fit in cfg.text.max_seq_len.
        """
        import torch.nn.functional as F

        assert idx.shape[1] + max_new_tokens <= self.cfg.text.max_seq_len, \
            "prompt + max_new_tokens must fit max_seq_len (no truncation with images)"
        x = self.embed(idx, **vision_inputs)
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            logits, _ = self.language_model(inputs_embeds=x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                next_id[finished] = eos_id
                finished = finished | (next_id.squeeze(1) == eos_id)

            idx = torch.cat([idx, next_id], dim=1)
            new_embed = self.language_model.embed_norm(self.language_model.embed_tokens(next_id))
            x = torch.cat([x, new_embed], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return idx
