"""The full Gemma 4 text model - tiny or 26B, decided only by the config.

Flow:  token ids -> embeddings * sqrt(hidden) -> N decoder blocks (sliding or
       full, each optionally fed a per-layer embedding) -> final RMSNorm
       -> TIED lm_head -> tanh softcap -> logits

Two things happen up here that are worth reading slowly.

**Per-Layer Embeddings (PLE).** Every layer gets its own extra vector per
token, and that vector is built from two independent halves that are averaged:

  1. *Token identity* - a second embedding table, `embed_tokens_per_layer`,
     looked up with the same ids. It is packed: one row holds all layers'
     vectors end to end (`num_layers * ple_dim` wide), reshaped afterward. It
     is scaled by `sqrt(ple_dim)` on the way out, mirroring the main embedding.
  2. *Context* - a linear projection of the ALREADY-EMBEDDED input, scaled by
     `1/sqrt(hidden)` and normalized.

They are added and multiplied by `1/sqrt(2)` - an average that keeps the
variance of the sum in check. So each layer's gate is steered partly by which
token this is, and partly by what the embedding layer made of it.

The e2b release spends a lot of its parameters here: `embed_tokens_per_layer`
is 262144 x 8960, which is bigger than the main embedding table.

**Two RoPE tables, and two masks.** Sliding layers use base-10k full rotation
and a windowed mask; full layers use base-1M proportional rotation (only 25% of
each head rotates) and a plain causal mask. Both are precomputed once here and
handed to whichever layer needs them.

Parameter names match the released checkpoint's language-model keys one-for-one
(the checkpoint prefixes them with "model.language_model."), which is what lets
`hf_weights.py` load the real thing - see `ModelConfig.gemma4_e2b()`.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from block import TransformerBlock
from rotary import precompute_cos_sin


class TinyGemma(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_scale = math.sqrt(cfg.hidden_size)   # Gemma scales embeddings up

        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight    # weight tying

        # --- per-layer embeddings (skipped entirely when the width is 0) ---
        if cfg.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                cfg.vocab_size_per_layer_input,
                cfg.num_layers * cfg.hidden_size_per_layer_input)
            self.per_layer_embed_scale = math.sqrt(cfg.hidden_size_per_layer_input)
            self.per_layer_model_projection = nn.Linear(
                cfg.hidden_size, cfg.num_layers * cfg.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection_norm = RMSNorm(
                cfg.hidden_size_per_layer_input, cfg.rms_norm_eps)

        # --- one RoPE table per attention type ---
        for layer_type in ("sliding_attention", "full_attention"):
            # Full layers are wider per head AND only partially rotated.
            is_full = layer_type == "full_attention"
            cos, sin = precompute_cos_sin(
                cfg.global_head_dim if is_full else cfg.head_dim,
                cfg.max_seq_len,
                cfg.rope_theta_for(layer_type),
                cfg.partial_rotary_factor if is_full else 1.0,
            )
            self.register_buffer(f"{layer_type}_cos", cos, persistent=False)
            self.register_buffer(f"{layer_type}_sin", sin, persistent=False)

    def _build_masks(self, T: int, device) -> dict:
        """Boolean [T, T] masks (True = may attend), one per attention type."""
        pos = torch.arange(T, device=device)
        causal = pos[None, :] <= pos[:, None]
        sliding = causal & (pos[None, :] > pos[:, None] - self.cfg.sliding_window)
        return {"full_attention": causal, "sliding_attention": sliding}

    def per_layer_identity(self, idx: torch.Tensor) -> torch.Tensor:
        """Half one of PLE: token identity, from the packed second table.

        [B, T] ids -> [B, T, num_layers, ple_dim].
        """
        cfg = self.cfg
        out = self.embed_tokens_per_layer(idx) * self.per_layer_embed_scale
        return out.view(*idx.shape, cfg.num_layers, cfg.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, embeds: torch.Tensor,
                                 identity: torch.Tensor = None) -> torch.Tensor:
        """Half two of PLE: context, projected from the embeddings - then averaged
        with the identity half.

        Kept separate from `per_layer_identity` for a reason that only shows up
        in the multimodal model: identity is computed from the token ids BEFORE
        image and audio features are scattered in, but context is projected from
        the embeddings AFTER, so the per-layer signal does see the media.
        """
        cfg = self.cfg
        context = self.per_layer_model_projection(embeds) * (cfg.hidden_size**-0.5)
        context = context.view(*embeds.shape[:-1], cfg.num_layers,
                               cfg.hidden_size_per_layer_input)
        context = self.per_layer_projection_norm(context)
        if identity is None:
            return context
        return (context + identity) * (2.0**-0.5)   # average the two halves

    def forward(self, idx: torch.Tensor = None, targets: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None, per_layer_inputs: torch.Tensor = None):
        """idx: [B, T] of token ids. Returns (logits, loss).

        loss is None unless targets ([B, T]) are provided. The multimodal
        wrapper (vl_model.py) passes `inputs_embeds` [B, T, hidden] - already
        scaled, with encoder features scattered in - plus `per_layer_inputs`
        holding the identity half of PLE, and the embedding step is skipped.
        """
        cfg = self.cfg

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(idx) * self.embed_scale
            if cfg.hidden_size_per_layer_input and per_layer_inputs is None:
                per_layer_inputs = self.per_layer_identity(idx)
        x = inputs_embeds
        # The context half is always projected here, from whatever embeddings
        # actually reached the decoder.
        ple = (self.project_per_layer_inputs(x, per_layer_inputs)
               if cfg.hidden_size_per_layer_input else None)
        T = x.shape[1]

        masks = self._build_masks(T, x.device)
        # Layers that share KV read from here; the storing layers fill it in.
        shared_kv = {}

        for i, layer in enumerate(self.layers):
            layer_type = cfg.layer_types[i]
            x = layer(
                x,
                getattr(self, f"{layer_type}_cos")[:T],
                getattr(self, f"{layer_type}_sin")[:T],
                masks[layer_type],
                shared_kv,
                per_layer_input=ple[:, :, i, :] if ple is not None else None,
            )

        x = self.norm(x)
        logits = self.lm_head(x)

        # Gemma's tanh softcap: squash logits into [-cap, cap], smoothly.
        if cfg.final_logit_softcapping is not None:
            cap = cfg.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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
