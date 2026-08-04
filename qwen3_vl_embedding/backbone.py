"""Small causal Qwen backbone shared by text, image and fused token streams."""

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).to(x.dtype) * self.weight


def rope(x, theta):
    length, dim = x.size(-2), x.size(-1)
    frequency = 1.0 / theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
    angle = torch.outer(torch.arange(length, device=x.device).float(), frequency)
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(cfg.head_dim, cfg.eps), RMSNorm(cfg.head_dim, cfg.eps)

    def forward(self, x, mask):
        cfg = self.cfg
        batch, length, hidden = x.shape
        q = self.q(x).view(batch, length, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = rope(self.q_norm(q), cfg.rope_theta), rope(self.k_norm(k), cfg.rope_theta)
        repeat = cfg.num_heads // cfg.num_kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.o(out.transpose(1, 2).reshape(batch, length, hidden)) * mask.unsqueeze(-1)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.hidden_size, cfg.eps)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.hidden_size, cfg.eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.mlp(self.n2(x)) * mask.unsqueeze(-1)


class QwenBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)

    def forward(self, tokens, mask):
        for block in self.blocks: tokens = block(tokens, mask)
        return self.norm(tokens)
