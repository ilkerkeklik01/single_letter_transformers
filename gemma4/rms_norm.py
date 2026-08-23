"""RMSNorm, Gemma 4 flavor.

Worth flagging if you know earlier Gemmas: **the `(1 + weight)` quirk is gone.**
Gemma 1-3 stored a zero-initialized weight and applied `x_hat * (1 + w)`;
Gemma 4 stores a ones-initialized weight and applies `x_hat * w`, like everyone
else. If you port weights between the families by hand, that difference will
bite you.

`with_scale=False` drops the parameter entirely. Gemma 4 uses that weight-less
form in two places: the value norm inside attention, and the router's input
norm.

The `torch.pow(mean_squared, -0.5)` is copied verbatim from the reference (with
its comment) rather than written as `rsqrt`, so the arithmetic matches
bit-for-bit.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int = None, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler
        # differences between Torch and JAX
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float())          # compute in float32
        if self.with_scale:
            out = out * self.weight.float()
        return out.type_as(x)                # cast back
