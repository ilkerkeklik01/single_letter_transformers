"""The complete Qwen3.8: text stack (+ MTP head) and, on the 27B, a vision tower.

The bridge here is thinner than in the other folders: there is no adapter and no
projection module at all. The vision tower's own `merger` already outputs
`out_hidden_size`, which the config requires to equal the text hidden size, so
merged patches are written straight into the token sequence. The whole
"multimodal" wiring is one masked scatter.

What *is* more involved than elsewhere is positions. The text model wants a
`[3, B, T]` tensor of (time, height, width) indices. Text tokens get the same
index on all three axes, but an image's patches get real 2-D coordinates, and
everything after an image is pushed forward by the image's extent rather than by
its token count. `image_position_ids` below does that bookkeeping; it is the
reason the rotary code had to be 3-D all along.
"""

import torch
from torch import nn

from config import VLConfig
from model import Qwen38Text
from vision import VisionTower


class Qwen38(nn.Module):
    def __init__(self, cfg: VLConfig):
        super().__init__()
        self.cfg = cfg
        self.language_model = Qwen38Text(cfg.text)
        # 2.4T-A95B ships no vision tower, and neither do we in that case.
        if cfg.vision is not None:
            if cfg.vision.out_hidden_size != cfg.text.hidden_size:
                raise ValueError(
                    f"vision out_hidden_size ({cfg.vision.out_hidden_size}) must equal "
                    f"text hidden_size ({cfg.text.hidden_size})")
            self.visual = VisionTower(cfg.vision)

    def get_image_features(self, pixel_values: torch.Tensor,
                           grid_thw: torch.Tensor) -> torch.Tensor:
        """Patches -> [num_merged_tokens, text_hidden], ready to scatter."""
        if self.cfg.vision is None:
            raise RuntimeError("this model was built without a vision tower")
        return self.visual(pixel_values, grid_thw)

    def image_position_ids(self, idx: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build [3, B, T] (time, height, width) position ids for a batch.

        Text tokens advance all three axes together. An image's tokens take their
        real (t, h, w) grid coordinates, and the text after it resumes from the
        largest coordinate the image reached - so a wide image costs more
        positional room than a narrow one with the same token count.
        """
        cfg = self.cfg
        merge = cfg.vision.spatial_merge_size
        B, T = idx.shape
        out = torch.zeros(3, B, T, dtype=torch.long, device=idx.device)

        for b in range(B):
            row = idx[b]
            pos = 0            # next free position on every axis
            i = 0
            image = 0
            while i < T:
                if row[i] == cfg.image_token_id or row[i] == cfg.video_token_id:
                    t, h, w = (int(v) for v in grid_thw[image])
                    gh, gw = h // merge, w // merge
                    n = t * gh * gw
                    ts = torch.arange(t, device=idx.device).repeat_interleave(gh * gw)
                    hs = torch.arange(gh, device=idx.device).repeat_interleave(gw).repeat(t)
                    ws = torch.arange(gw, device=idx.device).repeat(t * gh)
                    out[0, b, i:i + n] = ts + pos
                    out[1, b, i:i + n] = hs + pos
                    out[2, b, i:i + n] = ws + pos
                    pos += max(t, gh, gw)     # the image's largest extent
                    i += n
                    image += 1
                else:
                    out[:, b, i] = pos
                    pos += 1
                    i += 1
        return out

    def embed(self, idx: torch.Tensor, pixel_values: torch.Tensor = None,
              grid_thw: torch.Tensor = None):
        """Token ids (+ optional image) -> (embeddings, 3-D position ids)."""
        cfg = self.cfg
        mask = (idx == cfg.image_token_id) | (idx == cfg.video_token_id)

        # In the released models the placeholder ids sit inside the vocabulary,
        # so the lookup would succeed; zeroing them anyway costs nothing (those
        # rows are overwritten below) and keeps small vocabularies working.
        ids = torch.where(mask, torch.zeros_like(idx), idx)
        x = self.language_model.embed_tokens(ids)

        if pixel_values is None:
            B, T = idx.shape
            return x, self.language_model.default_position_ids(B, T, idx.device)

        features = self.get_image_features(pixel_values, grid_thw).to(x.dtype)
        assert int(mask.sum()) == features.shape[0], (
            f"{int(mask.sum())} image placeholders but {features.shape[0]} vision tokens")
        x[mask] = features
        return x, self.image_position_ids(idx, grid_thw)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                pixel_values: torch.Tensor = None, grid_thw: torch.Tensor = None):
        """idx: [B, T] token ids, with placeholders where image patches go."""
        x, position_ids = self.embed(idx, pixel_values, grid_thw)
        return self.language_model(inputs_embeds=x, position_ids=position_ids,
                                   targets=targets)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None, eos_id: int = None,
                 pixel_values: torch.Tensor = None, grid_thw: torch.Tensor = None):
        """Autoregressively extend idx, running the vision tower once up front."""
        import torch.nn.functional as F

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx, pixel_values=pixel_values, grid_thw=grid_thw)
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
            if eos_id is not None and bool(finished.all()):
                break
        return idx
