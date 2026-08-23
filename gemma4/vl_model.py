"""The complete Gemma 4: text stack + vision tower + audio tower.

Both encoders end the same way, through a `MultimodalEmbedder`: a weight-less
RMSNorm followed by one bias-free Linear into the text hidden size. That is the
entire bridge - two lines, and the only trainable part is one matrix per
modality. Vision enters at its own hidden width (768); audio enters at
`output_proj_dims` (1536), which is why the audio tower ends with a projection
of its own.

Features are then scattered into the token sequence wherever the matching
placeholder id appears - `image_token_id` for image (and `video_token_id` for
video, which takes the identical path) and `audio_token_id` for audio. One
placeholder per soft token; the counts must line up exactly.

Submodule names mirror the released checkpoint (`language_model`,
`vision_tower`, `audio_tower`, `embed_vision`, `embed_audio` under a stripped
"model." prefix), so `hf_weights.py` loads the full thing.
"""

import torch
from torch import nn

from config import VLConfig
from model import TinyGemma
from rms_norm import RMSNorm
from vision_model import VisionTower
from audio_model import AudioTower


class MultimodalEmbedder(nn.Module):
    """Normalize encoder output and project it into the language model's space."""

    def __init__(self, multimodal_hidden_size: int, text_hidden_size: int, eps: float):
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(multimodal_hidden_size, eps,
                                                     with_scale=False)
        self.embedding_projection = nn.Linear(multimodal_hidden_size, text_hidden_size,
                                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


class Gemma4(nn.Module):
    def __init__(self, cfg: VLConfig):
        super().__init__()
        self.cfg = cfg
        self.language_model = TinyGemma(cfg.text)
        self.vision_tower = VisionTower(cfg.vision)
        self.embed_vision = MultimodalEmbedder(
            cfg.vision.hidden_size, cfg.text.hidden_size, cfg.vision.rms_norm_eps)
        # The audio tower is optional: 26B-A4B ships without one, and the
        # reference likewise builds nothing when `audio_config` is null.
        if cfg.audio is not None:
            self.audio_tower = AudioTower(cfg.audio)
            self.embed_audio = MultimodalEmbedder(
                cfg.audio.output_proj_dims, cfg.text.hidden_size, cfg.audio.rms_norm_eps)

    def get_image_features(self, pixel_values: torch.Tensor,
                           pixel_position_ids: torch.Tensor) -> torch.Tensor:
        """Patches -> [num_soft_tokens, text_hidden], ready to scatter."""
        return self.embed_vision(self.vision_tower(pixel_values, pixel_position_ids))

    def get_audio_features(self, input_features: torch.Tensor,
                           input_features_mask: torch.Tensor = None) -> torch.Tensor:
        """Spectrogram -> [num_soft_tokens, text_hidden], padding already dropped."""
        if self.cfg.audio is None:
            raise RuntimeError("this model was built without an audio tower")
        hidden, valid = self.audio_tower(input_features, input_features_mask)
        hidden = self.embed_audio(hidden)
        return hidden[valid.bool()] if valid is not None else hidden.reshape(-1, hidden.shape[-1])

    def _scatter(self, x, idx, token_id, features):
        mask = idx == token_id
        assert int(mask.sum()) == features.shape[0], (
            f"{int(mask.sum())} placeholders for token {token_id} but "
            f"{features.shape[0]} encoder tokens")
        x[mask] = features.to(x.dtype)
        return x

    def embed(self, idx: torch.Tensor,
              pixel_values: torch.Tensor = None, pixel_position_ids: torch.Tensor = None,
              input_features: torch.Tensor = None, input_features_mask: torch.Tensor = None,
              ) -> torch.Tensor:
        """Token ids (+ optional image/audio) -> [B, T, hidden] decoder input."""
        cfg = self.cfg
        lm = self.language_model
        # Placeholder ids live outside the vocabulary, so they are replaced with
        # the PAD id before the lookup; encoder features overwrite those
        # positions a few lines later anyway.
        pad_id = cfg.text.pad_token_id
        ids = idx.clone()
        placeholders = ((ids == cfg.image_token_id) | (ids == cfg.audio_token_id)
                        | (ids == cfg.video_token_id))
        ids[placeholders] = pad_id

        x = lm.embed_tokens(ids) * lm.embed_scale

        # Only the IDENTITY half of PLE is built here, from the pad-substituted
        # ids - so at a media position it is the pad token's per-layer vector.
        # The context half is projected inside the text model, from the
        # embeddings after the scatter below, and therefore does see the media.
        identity = (lm.per_layer_identity(ids)
                    if cfg.text.hidden_size_per_layer_input else None)

        if pixel_values is not None:
            x = self._scatter(x, idx, cfg.image_token_id,
                              self.get_image_features(pixel_values, pixel_position_ids))
        if input_features is not None:
            x = self._scatter(x, idx, cfg.audio_token_id,
                              self.get_audio_features(input_features, input_features_mask))
        return x, identity

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, **modality_inputs):
        """idx: [B, T] token ids, with placeholders where encoder output goes.

        modality_inputs: pixel_values + pixel_position_ids, and/or
        input_features + input_features_mask. Returns (logits, loss).
        """
        x, identity = self.embed(idx, **modality_inputs)
        return self.language_model(inputs_embeds=x, per_layer_inputs=identity, targets=targets)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None, eos_id: int = None, **modality_inputs) -> torch.Tensor:
        """Autoregressively extend idx, running the encoders once up front."""
        import torch.nn.functional as F

        assert idx.shape[1] + max_new_tokens <= self.cfg.text.max_seq_len, \
            "prompt + max_new_tokens must fit max_seq_len (no truncation with media)"
        x, identity = self.embed(idx, **modality_inputs)
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            logits, _ = self.language_model(inputs_embeds=x, per_layer_inputs=identity)
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

            new_x = self.language_model.embed_tokens(next_id) * self.language_model.embed_scale
            x = torch.cat([x, new_x], dim=1)
            if identity is not None:
                identity = torch.cat(
                    [identity, self.language_model.per_layer_identity(next_id)], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return idx
