"""Qwen-style causal GQA with QK-Norm and RoPE."""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from rotary import apply, cache


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.theta = cfg.rope_theta
        self.q = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x, mask):
        batch, length, hidden = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cache(length, self.head_dim, self.theta, x.device)
        q, k = apply(self.q_norm(q), cos, sin), apply(self.k_norm(k), cos, sin)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.o(out.transpose(1, 2).reshape(batch, length, hidden)) * mask.unsqueeze(-1)
