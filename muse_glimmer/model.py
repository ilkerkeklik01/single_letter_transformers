"""The full Muse Glimmer text model - tiny or 30B, decided only by the config.

Flow:  token ids -> embeddings -> WEIGHT-LESS RMSNORM (!) -> N decoder blocks
       (each sliding+RoPE or full+NoPE) -> final RMSNorm -> UNTIED lm_head
       -> logits * output_multiplier -> tanh softcap -> logits

Three head-end details that set Muse Glimmer apart:

  * Normed embeddings: the embedding lookup is followed by a weight-less
    RMSNorm, so every token enters the stack at unit RMS regardless of how the
    embedding row grew during training. (No parameters - it never shows up in
    the state_dict.)

  * Untied head: `lm_head` is its own matrix, NOT shared with `embed_tokens`.
    In the 30B those are 2.5 GB each.

  * Softcapped logits: logits are scaled by `output_multiplier`
    (= 1/sqrt(hidden/256)) and squashed with a Gemma-style tanh cap,
    `cap * tanh(logits / cap)`, so nothing ever leaves [-20, 20].

Parameter names here (embed_tokens, layers.N.self_attn.q_proj, ..., norm,
lm_head) match the released checkpoint's language-model keys one-for-one -
the checkpoint just prefixes them with "model.language_model." (see
hf_weights.py). That is what makes "load the 30B by only changing the config"
literally true.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from block import TransformerBlock
from rotary import precompute_cos_sin


class MuseGlimmer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # Weight-less norm on top of the embeddings (adds no parameters).
        self.embed_norm = RMSNorm(eps=cfg.rms_norm_eps, with_scale=False)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # No weight tying - Muse Glimmer's input and output embeddings are separate.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE tables are fixed, so precompute them once and store as buffers.
        # One global table serves every RoPE layer; NoPE layers just skip it.
        cos, sin = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build_masks(self, T: int, device) -> dict:
        """Boolean [T, T] masks (True = may attend), one per layer type.

        full    : plain causal - see everything up to and including yourself.
        sliding : causal AND within the window - see only the last
                  `sliding_window` tokens (yourself included).
        """
        pos = torch.arange(T, device=device)
        causal = pos[None, :] <= pos[:, None]
        sliding = causal & (pos[None, :] > pos[:, None] - self.cfg.sliding_window)
        return {"full_attention": causal, "sliding_attention": sliding}

    def forward(self, idx: torch.Tensor = None, targets: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None):
        """idx: [B, T] of token ids. Returns (logits, loss).

        loss is None unless targets ([B, T]) are provided. The multimodal
        wrapper (vl_model.py) passes `inputs_embeds` [B, T, hidden] instead of
        ids - already normed, with image features scattered in - and the
        embedding step is skipped.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_norm(self.embed_tokens(idx))  # [B, T, hidden], unit RMS
        x = inputs_embeds
        T = x.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]   # slice RoPE tables to current length
        masks = self._build_masks(T, x.device)

        for layer in self.layers:
            x = layer(x, cos, sin, masks[layer.self_attn.layer_type])
        x = self.norm(x)

        logits = self.lm_head(x)                          # [B, T, vocab]
        # Scale, then softcap: cap * tanh(logits * mult / cap) keeps every
        # logit inside [-cap, cap] while staying smooth (and differentiable).
        logits = logits * self.cfg.output_multiplier
        cap = self.cfg.final_logit_softcapping
        logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [B*T, vocab]
                targets.reshape(-1),                  # [B*T]
            )
        return logits, loss

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
            # Never feed more than max_seq_len tokens of context.
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # logits for the next token

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample one token

            if eos_id is not None:
                next_id[finished] = eos_id                       # keep finished rows on eos
                finished = finished | (next_id.squeeze(1) == eos_id)

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break                                            # everyone hit eos
        return idx
