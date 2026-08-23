"""Mixture of Experts - the feed-forward of the 2.4T-A95B release.

512 experts, 10 active per token, plus a **shared expert** every token always
uses. That shared path is the piece worth understanding: routed experts can
specialize freely only if something else is guaranteed to handle whatever is
common to all tokens, and the shared expert is that something. Its contribution
is itself gated - `sigmoid(shared_expert_gate(x))` - so each token decides how
much of the generic path it wants.

The name "A95B" is this design's whole point: 2.4T total parameters, but only
about 95B active for any one token.

`Experts` stores all experts as stacked 3-D tensors rather than a ModuleList,
with each expert's gate and up projections packed into a single `gate_up_proj`
that is split with `.chunk(2)` at use time. The loop walks only the experts some
token actually selected - readable rather than fast.

Note the router has no bias and no normalization layer: it is one bare
`[num_experts, hidden]` weight, softmaxed in float32, top-k, then renormalized so
the kept weights sum to 1.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import TextConfig
from mlp import MLP


class TopKRouter(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.hidden_dim = cfg.hidden_size
        self.weight = nn.Parameter(torch.zeros(cfg.num_experts, cfg.hidden_size))

    def forward(self, x: torch.Tensor):
        """x: [tokens, hidden] -> (weights, indices), both [tokens, top_k]."""
        logits = F.linear(x.reshape(-1, self.hidden_dim), self.weight)
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
        top_value, top_index = torch.topk(probs, self.top_k, dim=-1)
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        return top_value.to(logits.dtype), top_index


class Experts(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        # All experts in two stacked tensors; gate and up are packed together.
        self.gate_up_proj = nn.Parameter(
            torch.empty(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size))
        self.down_proj = nn.Parameter(
            torch.empty(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size))
        # Raw Parameters, so nothing initializes them for us.
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """x: [tokens, hidden]; top_k_*: [tokens, k]. Returns [tokens, hidden]."""
        out = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            chosen = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in chosen:
            expert_idx = expert_idx[0]
            slot, token_idx = torch.where(expert_mask[expert_idx])
            tokens = x[token_idx]
            gate, up = F.linear(tokens, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            hidden = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            hidden = hidden * top_k_weights[token_idx, slot, None]
            out.index_add_(0, token_idx, hidden.to(out.dtype))
        return out


class SparseMoeBlock(nn.Module):
    """Routed experts plus a gated shared expert, summed."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.gate = TopKRouter(cfg)
        self.experts = Experts(cfg)
        self.shared_expert = MLP(cfg, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, hidden = x.shape
        flat = x.view(-1, hidden)

        shared = self.shared_expert(flat)
        weights, index = self.gate(flat)
        routed = self.experts(flat, index, weights)

        shared = F.sigmoid(self.shared_expert_gate(flat)) * shared
        return (routed + shared).reshape(B, T, hidden)
