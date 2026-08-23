"""The two RMSNorm flavors Muse Glimmer uses.

Both rescale a vector by its root-mean-square (no mean subtraction, no bias)
and both do the math in float32 before casting back. They differ only in what
the learnable scale looks like:

  * RMSNorm         : classic `x_hat * w`, weight initialized to ONES.
                      With `with_scale=False` there is no weight at all - Muse
                      Glimmer uses that weight-less form on the embeddings, on
                      per-head queries/keys (QK-norm), and on projected image
                      features. Only the final pre-head norm keeps its weight.

  * CenteredRMSNorm : Gemma-style `x_hat * (1 + w)`, weight initialized to
                      ZEROS (so it starts as a plain normalize). All four norms
                      inside a decoder block are this flavor.

A tiny but deliberate detail copied from the reference: the epsilon goes INSIDE
the square root - `x / sqrt(mean(x^2) + eps)` - and CenteredRMSNorm multiplies
by the weight while still in float32, then casts (`(x_hat * w).to(dtype)`,
not `x_hat.to(dtype) * w`).
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(dtype)


class CenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # starts as identity scale (1 + 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)
