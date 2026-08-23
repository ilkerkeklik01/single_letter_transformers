"""The convolutional parts of the audio Conformer.

Two very different jobs.

`SubSampleConvProjection` is the front door: a mel spectrogram
`[batch, frames, mel_bins]` goes in, and a much shorter sequence of hidden
vectors comes out. Two stride-2 2-D convolutions quarter BOTH axes, so 100
frames of audio become 25 tokens - which is the whole reason a long waveform is
affordable for the language model. Note the norm: `LayerNorm` over the CHANNEL
axis (so the tensor is permuted to put channels last and back), with a weight
but no bias, then ReLU.

That `(channels[0] // 4) * channels[1]` width for the final projection encodes
an assumption: the feature axis started at `channels[0]` mel bins and got
quartered by the two stride-2 layers.

`LightConv1d` is the Conformer's signature "lightweight convolution" module,
which sits between attention and the second feed-forward and gives each layer a
cheap way to mix neighbouring frames:

    norm -> linear to 2x width -> GLU -> depthwise causal conv -> clamp
         -> norm -> activation -> linear back -> add residual

The convolution is **depthwise** (`groups = hidden_size`, one filter per
channel, no cross-channel mixing) and **causal**: `CausalConv1d` pads only on
the left, so a frame never sees the future. That matters because the released
audio tower is meant for streaming.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import AudioConfig
from rms_norm import RMSNorm
from clipped_linear import ClippableLinear


def activation(name: str):
    return {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[name]


class CausalConv1d(nn.Conv1d):
    """Conv1d that pads only on the left, so no output sees a future input."""

    @property
    def left_pad(self) -> int:
        effective_kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class SubSampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_eps: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                              stride=(2, 2), padding=1, bias=False)
        # LayerNorm over channels, with a weight but deliberately no bias.
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            x = x * mask[:, None, :, None]        # zero the padded frames
        x = self.conv(x)
        # Norm wants channels last; the conv wants them second.
        x = self.act(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        if mask is not None:
            mask = mask[:, ::2]                   # the time axis just halved
        return x, mask


class SubSampleConvProjection(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        channels = cfg.subsampling_conv_channels
        self.layer0 = SubSampleConvLayer(1, channels[0], cfg.rms_norm_eps)
        self.layer1 = SubSampleConvLayer(channels[0], channels[1], cfg.rms_norm_eps)
        # The feature axis has been quartered by the two stride-2 layers.
        self.input_proj_linear = nn.Linear((channels[0] // 4) * channels[1],
                                           cfg.hidden_size, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None):
        """features [B, frames, mel_bins] -> ([B, frames/4, hidden], mask)."""
        x = features.unsqueeze(1)                 # add the channel axis
        x, mask = self.layer0(x, mask)
        x, mask = self.layer1(x, mask)

        B, _, time, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, time, -1)
        return self.input_proj_linear(x), mask


class LightConv1d(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        clip = cfg.use_clipped_linears
        self.linear_start = ClippableLinear(cfg.hidden_size, cfg.hidden_size * 2, clip)
        self.linear_end = ClippableLinear(cfg.hidden_size, cfg.hidden_size, clip)
        self.depthwise_conv1d = CausalConv1d(
            cfg.hidden_size, cfg.hidden_size, kernel_size=cfg.conv_kernel_size,
            groups=cfg.hidden_size, bias=False)   # one filter per channel

        self.pre_layer_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.conv_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.act_fn = activation(cfg.hidden_act)
        self.gradient_clipping = cfg.gradient_clipping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_layer_norm(x)
        x = F.glu(self.linear_start(x), dim=-1)   # 2x width, gated back down to 1x
        x = self.depthwise_conv1d(x.transpose(1, 2)).transpose(1, 2)

        clip = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        x = torch.clamp(x, -clip, clip)
        x = self.act_fn(self.conv_norm(x))
        return self.linear_end(x) + residual
