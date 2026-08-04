"""Bidirectional GQA; early layers use a local window, the last is global."""

import torch
from torch import nn
import torch.nn.functional as F

from layers import RMSNorm, apply_rope


class Attention(nn.Module):
    def __init__(self, cfg, local):
        super().__init__()
        self.heads, self.kv_heads, self.dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.theta, self.window = cfg.rope_theta, cfg.local_window if local else None
        self.q = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(cfg.head_dim, cfg.eps), RMSNorm(cfg.head_dim, cfg.eps)

    def forward(self, x, mask):
        batch, length, hidden = x.shape
        q = self.q(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        q, k = apply_rope(self.q_norm(q), self.theta), apply_rope(self.k_norm(k), self.theta)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        allowed = mask[:, None, None, :].expand(batch, 1, length, length)
        if self.window is not None:
            position = torch.arange(length, device=x.device)
            allowed = allowed & ((position[:, None] - position[None, :]).abs() <= self.window)[None, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.o(out.transpose(1, 2).reshape(batch, length, hidden)) * mask.unsqueeze(-1)
