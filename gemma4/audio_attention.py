"""Chunked local attention with relative positions - the audio tower's core.

Nothing here looks like the text attention. Three ideas stack up.

**Chunking.** The sequence is cut into blocks of `chunk_size`. Queries are
grouped by block; keys and values are gathered as a sliding CONTEXT window
around each block - `attention_context_left - 1` frames before it and
`attention_context_right` after. So instead of one `[T, T]` attention matrix
there are `T/chunk` small `[chunk, context]` ones, which is linear in sequence
length. `_extract_block_context` does the gathering with `unfold`, which builds
overlapping views without copying the whole sequence per block.

**Relative positions, Transformer-XL style.** Position is not rotary and not
additive-on-input. The logits are the sum of two terms:

    matrix_ac = query . key           (content vs content)
    matrix_bd = query . relative_key  (content vs relative offset)

where `relative_key` is a learned projection of a fixed sinusoidal table
indexed by *distance*, not absolute position. `_rel_shift` is the classic trick
that turns the `[chunk, num_distances]` result into the `[chunk, context]`
alignment the first term uses, by padding one extra column and re-viewing the
tensor so every row slides one step - no gather, no index arithmetic.

**Softcapped logits and a learned per-dimension scale.** Logits pass through
`50 * tanh(x / 50)` before the softmax, and queries are scaled by
`softplus(per_dim_scale)` - a learned width for every feature of the head - on
top of the usual constant. The two odd-looking constants (`/ log 2` on the query
scale, `log(1 + e) / log 2` on the key scale) come from the reference and make
`per_dim_scale = 0` the neutral starting point.

The whole computation runs in float32.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from config import AudioConfig
from clipped_linear import ClippableLinear


class RelPositionalEncoding(nn.Module):
    """A fixed sinusoidal table indexed by DISTANCE, from context_size//2 down to 0."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.context_size = cfg.context_size
        num_timescales = cfg.hidden_size // 2
        log_increment = math.log(10000.0) / max(num_timescales - 1, 1)
        inv_timescales = torch.exp(torch.arange(num_timescales) * -log_increment)
        self.register_buffer("inv_timescales", inv_timescales[None, None, :], persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distances = torch.arange(self.context_size // 2, -1, -1, device=x.device)[..., None]
        scaled = distances * self.inv_timescales.to(x.device)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).to(x.dtype)


class AudioAttention(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.chunk_size = cfg.attention_chunk_size
        self.max_past = cfg.attention_context_left - 1
        self.max_future = cfg.attention_context_right
        self.context_size = self.chunk_size + self.max_past + self.max_future

        # Chosen so that per_dim_scale = 0 is the neutral setting.
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.softcap = cfg.attention_logit_cap

        clip = cfg.use_clipped_linears
        hidden = cfg.hidden_size
        self.q_proj = ClippableLinear(hidden, self.num_heads * self.head_dim, clip)
        self.k_proj = ClippableLinear(hidden, self.num_heads * self.head_dim, clip)
        self.v_proj = ClippableLinear(hidden, self.num_heads * self.head_dim, clip)
        self.post = ClippableLinear(hidden, hidden, clip)

        self.relative_k_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def _to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, H, D] -> [B, num_blocks, chunk, H, D] (right-padded)."""
        B, T, H, D = x.shape
        num_blocks = (T + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - T
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x.reshape(B, num_blocks, self.chunk_size, H, D).contiguous()

    def _block_context(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, H, D] -> [B, num_blocks, context, H, D], windows overlapping."""
        x = F.pad(x, (0, 0, 0, 0, self.max_past, self.max_future + self.chunk_size - 1))
        x = x.unfold(1, self.context_size, self.chunk_size)   # [B, nb, H, D, context]
        return torch.movedim(x, -1, 2).contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, nb, chunk, distances] -> [B, H, nb, chunk, context].

        Pad one extra column, then re-view: each row ends up offset by one
        relative to the last, which is exactly the query-key alignment.
        """
        B, H, nb, block, positions = x.shape
        x = F.pad(x, (0, self.context_size + 1 - positions))
        x = x.view(B, H, nb, block * (self.context_size + 1))
        x = x[..., : block * self.context_size]
        return x.view(B, H, nb, block, self.context_size)

    def forward(self, x: torch.Tensor, position_embeddings: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """x [B, T, hidden]; mask [B, 1, num_blocks, chunk, context] bool or None."""
        B, T, _ = x.shape
        shape = (B, T, self.num_heads, self.head_dim)

        q = self.q_proj(x).float().view(shape)
        k = self.k_proj(x).float().view(shape)
        v = self.v_proj(x).float().view(shape)

        q = q * self.q_scale * F.softplus(self.per_dim_scale)
        k = k * self.k_scale

        q = self._to_blocks(q)          # [B, nb, chunk, H, D]
        k = self._block_context(k)      # [B, nb, context, H, D]
        v = self._block_context(v)
        num_blocks = q.shape[1]

        rel_k = self.relative_k_proj(position_embeddings)
        rel_k = rel_k.view(-1, self.num_heads, self.head_dim).to(q.dtype)

        queries = q.permute(0, 3, 1, 2, 4)                       # [B, H, nb, chunk, D]
        matrix_ac = queries @ k.permute(0, 3, 1, 4, 2)           # [B, H, nb, chunk, context]

        flat = queries.reshape(B, self.num_heads, -1, self.head_dim)
        matrix_bd = flat @ rel_k.permute(1, 2, 0)                # [B, H, nb*chunk, distances]
        matrix_bd = matrix_bd.reshape(B, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        logits = matrix_ac + matrix_bd
        logits = self.softcap * torch.tanh(logits / self.softcap)
        if mask is not None:
            logits = logits.masked_fill(mask.logical_not(),
                                        self.cfg.attention_invalid_logits_value)

        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(v.dtype)
        out = weights @ v.permute(0, 3, 1, 2, 4)                 # [B, H, nb, chunk, D]
        out = out.permute(0, 2, 3, 1, 4).reshape(B, num_blocks * self.chunk_size, -1)
        return self.post(out[:, :T].contiguous().to(x.dtype))
