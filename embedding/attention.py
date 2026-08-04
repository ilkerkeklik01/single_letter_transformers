"""Non-causal self-attention with QK-Norm and RoPE."""

import torch
from torch import nn
import torch.nn.functional as F

from config import EmbeddingConfig
from rms_norm import RMSNorm
from rotary import apply_rotary, rotary_cache


class BidirectionalAttention(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.output = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        cos, sin = rotary_cache(length, self.head_dim, self.rope_theta, x.device)
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)

        # True means "allowed" for scaled_dot_product_attention's boolean mask.
        attention_mask = mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        return self.output(attended) * mask.unsqueeze(-1)
