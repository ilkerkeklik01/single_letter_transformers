"""A Qwen3.8 decoder layer - plain pre-norm, with a swappable token mixer.

The skeleton is the classic Qwen3 shape, unchanged since Qwen3 dense:

    x = x + mixer(norm(x))
    x = x + mlp(norm(x))

What changes per layer is the **mixer**: three layers out of four use
`linear_attn` (Gated DeltaNet), the fourth uses `self_attn` (softmax attention).
The attribute name differs too, not just the class - that is how the checkpoint
distinguishes them, and why `layer_types` has to be known before any weight is
loaded.

The feed-forward is either the dense SwiGLU or the sparse MoE block, decided once
by the config for the whole model.

Note what is NOT here compared with the other folders: no sandwich norms, no
per-layer embeddings, no layer scalar. Qwen keeps the block itself boring and
spends its ideas inside the mixer.
"""

from torch import nn

from config import TextConfig
from rms_norm import RMSNorm
from attention import Attention
from gated_deltanet import GatedDeltaNet
from mlp import MLP
from moe import SparseMoeBlock


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int):
        super().__init__()
        self.block_type = cfg.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = Attention(cfg)

        self.mlp = SparseMoeBlock(cfg) if cfg.enable_moe else MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, mask, valid=None):
        residual = x
        h = self.input_layernorm(x)
        if self.block_type == "linear_attention":
            h = self.linear_attn(h, valid=valid)
        else:
            h = self.self_attn(h, cos, sin, mask)
        x = residual + h

        return x + self.mlp(self.post_attention_layernorm(x))
