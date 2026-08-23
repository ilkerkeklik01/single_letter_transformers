"""Causal self-attention with Gemma 4's four twists.

  1. No 1/sqrt(head_dim) scaling.
     `scaling = 1.0`. Logits are the raw q·k dot product. This is safe only
     because q and k are both RMS-normalized per head first, and the q_norm /
     k_norm weights are free to learn whatever scale the layer wants. The
     temperature became a parameter instead of a constant.

  2. Q, K and V are all normalized - but not identically.
     `q_norm` and `k_norm` are ordinary RMSNorms with weights. `v_norm` is
     WEIGHT-LESS: values get put on a common scale, with nothing learnable.

  3. Layers can share keys and values with an earlier layer.
     The last `num_kv_shared_layers` layers build **no k/v projections and no
     k/v norms at all**. They reuse the keys/values computed by the last
     non-shared layer of their own attention type. Two layer types means two
     stashes, which is why `shared_kv` is a dict keyed by layer type. On the
     e2b release this is 20 of 35 layers - a large parameter saving, partly
     spent again on the double-wide MLPs those same layers get.

  4. Optionally, K and V are literally the same projection (`attention_k_eq_v`).
     On full-attention layers of the 26B model there is no `v_proj`: the raw
     `k_proj` output is used as the value. Note the ordering - the value is the
     PRE-norm k_proj output; keys then get `k_norm` and RoPE on top, while
     values get `v_norm` and nothing else.

Also here, quietly: full-attention layers are a different shape from sliding
ones - wider heads (512 vs 256 in both releases) and sometimes fewer KV heads.
`cfg.head_dim_for(i)` / `cfg.num_kv_heads_for(i)` decide.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Expand KV heads so each query head has a matching KV head (GQA).

    x: [B, num_kv_heads, T, head_dim] -> [B, num_kv_heads * n_repeat, T, head_dim]
    """
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim_for(layer_idx)
        self.num_kv_heads = cfg.num_kv_heads_for(layer_idx)
        self.n_repeat = cfg.num_heads // self.num_kv_heads

        self.is_kv_shared = cfg.is_kv_shared(layer_idx)
        self.stores_shared_kv = cfg.stores_shared_kv(layer_idx)
        # k=v applies to full-attention layers only.
        self.k_eq_v = cfg.attention_k_eq_v and self.layer_type == "full_attention"

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

        # Layers that share KV states build no k/v machinery whatsoever.
        if not self.is_kv_shared:
            self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.v_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False)
            if not self.k_eq_v:
                self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor, shared_kv: dict) -> torch.Tensor:
        """x: [B, T, hidden]. mask: boolean [T, T], True = may attend.

        `shared_kv` maps layer type -> (key, value) for the KV-sharing layers to
        read; the storing layer writes into it as a side effect.
        """
        B, T, _ = x.shape
        shape = (B, T, -1, self.head_dim)

        # RoPE is applied in [B, T, heads, dim] layout, before moving heads forward.
        q = self.q_proj(x).view(shape)
        q = apply_rotary(self.q_norm(q), cos, sin).transpose(1, 2)

        if self.is_kv_shared:
            k, v = shared_kv[self.layer_type]
        else:
            k = self.k_proj(x).view(shape)
            # k=v layers reuse the RAW projection as the value, before k_norm.
            v = k if self.k_eq_v else self.v_proj(x).view(shape)
            k = apply_rotary(self.k_norm(k), cos, sin).transpose(1, 2)
            v = self.v_norm(v).transpose(1, 2)
            if self.stores_shared_kv:
                shared_kv[self.layer_type] = (k, v)

        # GQA: grow KV heads to match the query heads.
        k = repeat_kv(k, self.n_repeat)
        v = repeat_kv(v, self.n_repeat)

        # scale=1.0: the q/k norms carry the temperature (see the note above).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)
