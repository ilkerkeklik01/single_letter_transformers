"""The Qwen3.8 text model - tiny, 27B or 2.4T, decided only by the config.

Flow:  token ids -> embeddings -> N decoder layers (3 linear : 1 full attention)
       -> final RMSNorm -> untied lm_head -> logits

Two details that only exist because this model was designed for images too:

**Position ids are 3-D.** The rotary tables are built from a `[3, B, T]` tensor
of (time, height, width) indices. For pure text all three rows are identical, so
the interleaved MRoPE collapses to ordinary RoPE and costs nothing - but the
plumbing has to be there, because an image's patches carry genuinely different
indices on each axis.

**Only the full-attention layers get a mask.** The linear layers have no
attention matrix to mask: causality is structural, since the recurrence only
ever walks forward. They take a padding mask instead, which zeroes padded
positions before they can be written into the memory.

Parameter names match the released checkpoints' language-model keys one-for-one,
which is what lets `hf_weights.py` load the real thing.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import TextConfig
from rms_norm import RMSNorm
from block import DecoderLayer
from rotary import precompute_inv_freq, compute_cos_sin


class Qwen38Text(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.register_buffer("inv_freq",
                             precompute_inv_freq(cfg.rotary_dim, cfg.rope_theta),
                             persistent=False)

        # The multi-token-prediction head, if this checkpoint has one.
        if cfg.mtp_num_hidden_layers:
            from mtp import MTPHead     # imported here to keep the import graph acyclic
            self.mtp = MTPHead(cfg)

    def rotary_tables(self, position_ids: torch.Tensor):
        """position_ids [3, B, T] -> (cos, sin), each [B, T, rotary_dim]."""
        return compute_cos_sin(position_ids, self.inv_freq, self.cfg.mrope_section)

    def default_position_ids(self, B: int, T: int, device) -> torch.Tensor:
        """Plain text positions: the same index on all three axes."""
        pos = torch.arange(T, device=device)
        return pos.view(1, 1, -1).expand(3, B, -1)

    def forward(self, idx: torch.Tensor = None, targets: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None, position_ids: torch.Tensor = None,
                valid: torch.Tensor = None):
        """idx: [B, T] token ids. Returns (logits, loss).

        The multimodal wrapper passes `inputs_embeds` (with image features already
        scattered in) and the 3-D `position_ids` it computed for them.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(idx)
        x = inputs_embeds
        B, T, _ = x.shape

        if position_ids is None:
            position_ids = self.default_position_ids(B, T, x.device)
        cos, sin = self.rotary_tables(position_ids)

        # Only softmax attention needs a mask; the recurrence is causal by shape.
        pos = torch.arange(T, device=x.device)
        causal = pos[None, :] <= pos[:, None]

        for layer in self.layers:
            x = layer(x, cos, sin, causal, valid=valid)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def predict_next_next(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the MTP head: logits for the token TWO ahead of each position.

        The main stack predicts token i+1 at position i; this predicts i+2, by
        fusing the main model's hidden state at i with the embedding of the token
        that actually sits at i+1. The last position has no i+1 to look at, so it
        is fed a zero embedding and its output should be ignored.
        """
        if not self.cfg.mtp_num_hidden_layers:
            raise RuntimeError("this model was built without an MTP head")
        B, T = idx.shape

        x = self.embed_tokens(idx)
        cos, sin = self.rotary_tables(self.default_position_ids(B, T, idx.device))
        pos = torch.arange(T, device=idx.device)
        causal = pos[None, :] <= pos[:, None]

        for layer in self.layers:
            x = layer(x, cos, sin, causal)
        hidden = self.norm(x)

        # Embeddings shifted one left: position i sees the token at i+1.
        next_embeds = torch.cat(
            [self.embed_tokens(idx[:, 1:]), torch.zeros_like(x[:, :1])], dim=1)

        return self.lm_head(self.mtp(hidden, next_embeds, cos, sin, causal))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None,
                 eos_id: int = None) -> torch.Tensor:
        """Autoregressively extend idx [B, T].

        If eos_id is given, each row stops once it emits that token (finished
        rows keep getting padded with eos), and we exit early once all rows stop.
        """
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                next_id[finished] = eos_id                       # keep finished rows on eos
                finished = finished | (next_id.squeeze(1) == eos_id)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break                                            # everyone hit eos
        return idx
