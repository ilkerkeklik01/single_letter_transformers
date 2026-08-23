"""A vision encoder layer: classic pre-LN ViT, nothing sandwiched.

    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

Both norms are plain LayerNorm (with mean subtraction and bias - the only
place in the whole model that uses it), and the MLP is a plain two-layer
GELU net with biases: no gating, no SwiGLU. The contrast with the text
block is the lesson: all of Muse Glimmer's novelty budget went to the
language side.
"""

from torch import nn

from config import VisionConfig
from vision_attention import VisionAttention


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, cos, sin, boundaries):
        x = x + self.attn(self.norm1(x), cos, sin, boundaries)
        x = x + self.mlp(self.norm2(x))
        return x
