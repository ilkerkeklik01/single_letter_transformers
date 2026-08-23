"""Mixture of Experts - the sparse half of a Gemma 4 MoE layer.

The surprise here, if you have seen MoE elsewhere: in Gemma 4 the experts do
**not replace** the dense MLP. Both run on every token and their outputs are
added together (see `block.py`). So an MoE layer has a dense feed-forward path
that every token always takes, plus a sparse path where each token picks its
`top_k` experts out of `num_experts`.

Two pieces:

  * `Router` decides which experts each token uses. It normalizes the input
    (weight-less RMSNorm), rescales it by a learned per-feature `scale` times
    `1/sqrt(hidden)`, projects to one score per expert, softmaxes in float32,
    takes the top-k, renormalizes those k weights to sum to 1, and finally
    multiplies by a learned `per_expert_scale`. That last multiply is why the
    weights need not sum to 1 by the time the experts see them.

  * `Experts` holds all experts as single stacked 3D tensors rather than a list
    of Linear modules - `gate_up_proj` is [E, 2*I, H] and `down_proj` is
    [E, H, I]. Each expert is a GeGLU with its gate and up halves packed into
    one matrix, split with `.chunk(2)` at use time.

The loop below walks only the experts that at least one token actually chose,
gathering that expert's tokens, running them, and scattering the weighted
results back. It is the readable form, not the fast one.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from mlp import gelu_tanh


class Router(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.scalar_root_size = cfg.hidden_size**-0.5
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(cfg.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(cfg.num_experts))

    def forward(self, x: torch.Tensor):
        """x: [tokens, hidden] -> (top_k_weights, top_k_index), both [tokens, k]."""
        x = self.norm(x)
        x = x * self.scale * self.scalar_root_size

        scores = self.proj(x)                                     # [tokens, experts]
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)

        top_k_weights, top_k_index = torch.topk(probabilities, k=self.cfg.top_k_experts, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


class Experts(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        # All experts stacked: one big tensor each, not a ModuleList. These are
        # raw Parameters rather than Linear layers, so nothing initializes them
        # for us - fill them here, the way the reference's _init_weights does.
        self.gate_up_proj = nn.Parameter(
            torch.empty(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size))
        self.down_proj = nn.Parameter(
            torch.empty(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size))
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=cfg.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=cfg.initializer_range)

    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """x: [tokens, hidden]; top_k_*: [tokens, k]. Returns [tokens, hidden]."""
        out = torch.zeros_like(x)

        with torch.no_grad():
            # [experts, k, tokens] - who picked whom, and in which top-k slot.
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            chosen = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in chosen:
            expert_idx = expert_idx[0]
            slot, token_idx = torch.where(expert_mask[expert_idx])
            tokens = x[token_idx]
            gate, up = F.linear(tokens, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            hidden = gelu_tanh(gate) * up
            hidden = F.linear(hidden, self.down_proj[expert_idx])
            hidden = hidden * top_k_weights[token_idx, slot, None]
            out.index_add_(0, token_idx, hidden.to(out.dtype))

        return out
