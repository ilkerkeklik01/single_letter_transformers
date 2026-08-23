"""A Linear that can clamp its input and its output to a learned range.

Both Gemma 4 towers (vision and audio) use this instead of a bare `nn.Linear`.
The four bounds are **buffers, not parameters** - they are not trained by
gradient descent, they are calibration numbers measured after training and
shipped with the checkpoint. They default to +/- infinity, which makes the
clamps no-ops, so a freshly built model behaves exactly like a plain Linear
until real bounds are loaded.

The point is numerical safety in low precision: the released towers run in
bfloat16, and clamping keeps activations inside a range the calibration
actually saw. Note the shape of the wrapper - the real weight lives at
`self.linear.weight`, so checkpoint keys look like `q_proj.linear.weight`, one
level deeper than you might expect.
"""

import torch
from torch import nn


class ClippableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_clipped_linears: bool):
        super().__init__()
        self.use_clipped_linears = use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if use_clipped_linears:
            # Saved with the checkpoint; infinite (i.e. inactive) by default.
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x
