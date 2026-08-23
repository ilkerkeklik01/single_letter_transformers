"""A Gemma 4 decoder layer: sandwich norms, an optional MoE arm, an optional
per-layer-embedding gate, and a scalar on the way out.

The spine is Gemma's familiar sandwich normalization - normalize before AND
after each sub-layer, with the residual wrapping the pair:

    x = x + post_attn_norm( attn( input_norm(x) ) )
    x = x + post_ffn_norm(  ffn(  pre_ffn_norm(x) ) )

Then two optional stages that make Gemma 4 what it is:

**The MoE arm.** When enabled, `ffn` above is not just the dense MLP - the
sparse experts run too, and the two are SUMMED before the shared post-norm.
Note where the router looks: at the raw residual (the layer's input), not at
the normalized copy the dense MLP consumed. The experts get their own separate
normalization of that same residual. So the two arms genuinely see different
inputs.

**The PLE gate.** When per-layer embeddings are on, the layer ends with an
extra residual block whose job is to let a small per-(token, layer) vector
modulate the stream: project the hidden state down to the narrow PLE width,
GELU it, MULTIPLY by this layer's slice of the per-layer embedding, project
back up, normalize, add. The multiply is the point - the per-layer embedding
acts as a gate on a learned function of the hidden state, not as another
additive bias.

**`layer_scalar`.** Finally the output is multiplied by a one-element buffer.
It ships as ones in the released checkpoints (so it is a no-op there), but it
is a real, saved tensor, so it is part of the state dict.
"""

import torch
from torch import nn

from config import ModelConfig
from rms_norm import RMSNorm
from attention import Attention
from mlp import MLP, gelu_tanh
from moe import Router, Experts


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg, layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Ships as ones; still a saved tensor, so it lives in the state dict.
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

        self.hidden_size_per_layer_input = cfg.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                cfg.hidden_size, cfg.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(
                cfg.hidden_size_per_layer_input, cfg.hidden_size, bias=False)
            self.post_per_layer_input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.enable_moe_block = cfg.enable_moe_block
        if self.enable_moe_block:
            self.router = Router(cfg)
            self.experts = Experts(cfg)
            self.post_feedforward_layernorm_1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, mask, shared_kv, per_layer_input=None):
        # --- attention, sandwiched ---
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask, shared_kv)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # --- feed-forward (dense, plus experts when enabled), sandwiched ---
        residual = x
        h = self.mlp(self.pre_feedforward_layernorm(x))

        if self.enable_moe_block:
            dense = self.post_feedforward_layernorm_1(h)
            # The sparse arm reads the layer input itself, not the normed copy.
            flat = residual.reshape(-1, residual.shape[-1])
            top_k_weights, top_k_index = self.router(flat)
            sparse = self.experts(self.pre_feedforward_layernorm_2(flat), top_k_index, top_k_weights)
            sparse = self.post_feedforward_layernorm_2(sparse.reshape(residual.shape))
            h = dense + sparse

        x = residual + self.post_feedforward_layernorm(h)

        # --- per-layer embedding gate ---
        if self.hidden_size_per_layer_input:
            residual = x
            h = gelu_tanh(self.per_layer_input_gate(x))
            h = h * per_layer_input                 # the PLE vector gates it
            h = self.per_layer_projection(h)
            x = residual + self.post_per_layer_input_norm(h)

        return x * self.layer_scalar
