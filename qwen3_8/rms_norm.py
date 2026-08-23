"""The two RMSNorms Qwen3.8 uses.

`RMSNorm` keeps the Qwen/Gemma quirk that Gemma 4 dropped: the weight is
**zero-initialized** and applied as `x_hat * (1 + w)`, so an untrained norm is
exactly the identity. Note also the order - the multiply happens in float32 and
the cast comes last (`(x_hat * w).to(dtype)`, not `x_hat.to(dtype) * w`), which
is the thing the reference's comment about PR #29402 is pointing at.

`RMSNormGated` is used at the end of every Gated DeltaNet layer, and it is not
just a norm with a gate bolted on - the ORDER matters and is easy to get wrong:

    normalize -> scale by weight -> multiply by silu(gate)

The gate is applied *after* the learned scale, not before, and the gate goes
through SiLU rather than sigmoid. Its weight is ones-initialized, unlike the
plain RMSNorm above.
"""

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))   # identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class RMSNormGated(nn.Module):
    """RMSNorm whose output is gated by a SiLU'd side input."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))    # ones here, not zeros
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        x = self.weight * x.to(dtype)                  # scale first...
        x = x * F.silu(gate.to(torch.float32))         # ...then gate
        return x.to(dtype)
