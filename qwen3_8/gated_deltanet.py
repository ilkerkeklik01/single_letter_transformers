"""Gated DeltaNet - the linear-attention layer that fills 3 of every 4 layers.

Softmax attention keeps every past token and compares against all of them, which
costs O(T^2). This keeps **one fixed-size memory matrix** `S` (shape
`[key_dim, value_dim]` per head) and updates it token by token, so cost is linear
in sequence length and nothing grows with context. The update is the *gated delta
rule*:

    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1}^T k_t) k_t^T

  * `alpha_t` (written `g` below, in log space) is a **decay gate** - how much of
    the old memory to keep. It comes from `A_log` and `dt_bias` through a softplus,
    the same discretization Mamba uses.
  * `beta_t` is a **write gate** - how strongly to write this token in.
  * the `(v_t - prediction)` term is the **delta**: only store what the memory got
    wrong. That is what stops a fixed-size memory from saturating.

Reading back is `q_t^T S_t`.

Two implementations live here, and they compute the same thing:

  * `recurrent_delta_rule` is the literal loop above - one step per token. Slow,
    but it is the definition, and it is what runs when generating one token at a
    time.
  * `chunked_delta_rule` is what actually runs during training. It splits the
    sequence into chunks of 64 and does the within-chunk part as one matmul,
    carrying the memory across chunk boundaries. The `attn` inversion loop builds
    the strictly-lower-triangular operator that "undoes" the sequential
    dependency inside a chunk - the trick that makes the delta rule parallel.

Around the recurrence sit three things worth noticing: a **short depthwise causal
conv** over q/k/v before the recurrence (a cheap local mixer, kernel 4), **L2
normalization** of q and k rather than a softmax, and a final `RMSNormGated`
whose SiLU gate comes from its own projection `in_proj_z`.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import TextConfig
from rms_norm import RMSNormGated


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Scale each vector to unit length (the FLA library's exact form)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_delta_rule(query, key, value, g, beta, initial_state=None):
    """The delta rule written literally: one step per token.

    All inputs are [B, T, heads, dim]; g and beta are [B, T, heads].
    Returns [B, T, heads, value_dim].
    """
    dtype = query.dtype
    query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    B, heads, T, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (k_dim**-0.5)

    out = torch.zeros(B, heads, T, v_dim, dtype=value.dtype, device=value.device)
    state = (torch.zeros(B, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
             if initial_state is None else initial_state.to(value))

    for t in range(T):
        k_t, v_t = key[:, :, t], value[:, :, t]
        state = state * g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)   # decay
        predicted = (state * k_t.unsqueeze(-1)).sum(dim=-2)            # what memory says
        delta = (v_t - predicted) * beta[:, :, t].unsqueeze(-1)        # what it got wrong
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)        # write the fix
        out[:, :, t] = (state * query[:, :, t].unsqueeze(-1)).sum(dim=-2)   # read back

    return out.transpose(1, 2).contiguous().to(dtype)


def chunked_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None):
    """The same rule, computed a chunk at a time so it parallelizes.

    Within a chunk the sequential dependency is inverted algebraically (the
    `attn` loop below); across chunks the memory is carried forward as usual.
    """
    dtype = query.dtype
    query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    B, heads, T, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    padded_T = T + pad
    query = query * (k_dim**-0.5)

    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Cumulative decay inside each chunk, and the pairwise decay between positions.
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # Invert the within-chunk sequential dependency into one triangular operator.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=0)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = (torch.zeros(B, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
             if initial_state is None else initial_state.to(value))
    out = torch.zeros_like(value)

    for i in range(padded_T // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # decay_mask is already lower-triangular, so this is causal on its own.
        within = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ state              # what memory already predicts
        v_new = v_i - v_prime                              # the chunk's delta
        across = (q_i * g[:, :, i, :, None].exp()) @ state  # contribution of old memory
        out[:, :, i] = across + within @ v_new
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)

    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :T]
    return out.transpose(1, 2).contiguous().to(dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = cfg.linear_key_dim
        self.value_dim = cfg.linear_value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim

        # One projection produces query, key and value side by side...
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, cfg.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)   # output gate
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)  # write gate
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)  # decay gate

        # ...and one depthwise conv mixes each channel over the last few tokens.
        self.conv1d = nn.Conv1d(cfg.conv_dim, cfg.conv_dim, bias=False,
                                kernel_size=self.conv_kernel_size, groups=cfg.conv_dim,
                                padding=self.conv_kernel_size - 1)

        # Mamba-style discretization parameters, one per value head.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

        self.norm = RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor = None) -> torch.Tensor:
        """x: [B, T, hidden]. valid: [B, T] bool padding mask, or None."""
        if valid is not None:
            x = x * valid[:, :, None].to(x.dtype)   # padded rows must not enter the memory
        B, T, _ = x.shape

        qkv = self.in_proj_qkv(x).transpose(1, 2)          # [B, conv_dim, T]
        # Causal depthwise conv: pad on the left, then trim the overhang.
        qkv = F.silu(self.conv1d(qkv)[:, :, :T]).transpose(1, 2)

        query, key, value = torch.split(
            qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, T, -1, self.head_k_dim)
        key = key.reshape(B, T, -1, self.head_k_dim)
        value = value.reshape(B, T, -1, self.head_v_dim)

        beta = self.in_proj_b(x).sigmoid()
        # .float() first: in fp16 the exponential can otherwise underflow to -inf.
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)

        # Value heads can outnumber key heads; share each key head across its group.
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        out = chunked_delta_rule(query, key, value, g=g, beta=beta,
                                 chunk_size=self.cfg.chunk_size)

        z = self.in_proj_z(x).reshape(B, T, -1, self.head_v_dim)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(B, T, -1))
