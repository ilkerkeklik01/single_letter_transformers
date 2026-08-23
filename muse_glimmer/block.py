"""A Muse Glimmer decoder layer: Gemma-style "sandwich" normalization,
with centered norms and two different epsilons.

    x = x + post_attn_norm( attn( input_norm(x) ) )
    x = x + post_ffn_norm(  mlp(  pre_ffn_norm(x) ) )

All four norms are CenteredRMSNorm (`x_hat * (1 + w)`, weight starts at zero).
The two PRE-norms use `rms_norm_eps` (1e-5); the two POST-norms - the ones
standing between a sub-layer's output and the residual stream - use the much
tighter `post_norm_eps` (1e-8), so they stay honest even when a gated
attention output comes through very small.

The attention module is named `self_attn` and the norms carry their full
checkpoint names, so this block's state_dict matches the released weights
key for key.
"""

import torch
from torch import nn

from config import ModelConfig
from rms_norm import CenteredRMSNorm
from attention import Attention
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = CenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(cfg.hidden_size, cfg.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = CenteredRMSNorm(cfg.hidden_size, cfg.post_norm_eps)

    def forward(self, x, cos, sin, mask):
        h = self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.post_attention_layernorm(h)

        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(h)
        return x
