"""The complete audio tower: a Conformer stack over subsampled spectrograms.

Flow:  mel spectrogram -> two stride-2 convs (time and features quartered)
       -> N Conformer layers -> linear projection to `output_proj_dims`

A Conformer layer is `FFN -> attention -> lightweight conv -> FFN`, which is
where the name comes from: convolution for local detail, attention for reach.
Two details set it apart from a transformer layer:

  * **Macaron residuals.** Both feed-forwards add only `residual_weight` (0.5)
    of their output back to the stream, so the two half-steps together are worth
    roughly one full feed-forward. The module is a sandwich - a half FFN at each
    end, attention and convolution in the middle.
  * **Clamp, everywhere.** Activations are clamped to +/- `gradient_clipping`
    before every norm. At the default 1e10 that only bites in low precision,
    but the calls are part of the architecture and are reproduced here.

One quirk faithfully preserved: the feed-forward and layer norms are built
WITHOUT passing `rms_norm_eps`, so they use RMSNorm's own 1e-6 default, while
the convolution module's norms do take the config value. Both are 1e-6 in every
released config, so it never shows - until someone changes the config.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import AudioConfig
from rms_norm import RMSNorm
from clipped_linear import ClippableLinear
from audio_conv import SubSampleConvProjection, LightConv1d, activation
from audio_attention import AudioAttention, RelPositionalEncoding


class AudioFeedForward(nn.Module):
    """A half-step ("macaron") feed-forward: 4x width, and a weighted residual."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        clip = cfg.use_clipped_linears
        self.ffw_layer_1 = ClippableLinear(cfg.hidden_size, cfg.hidden_size * 4, clip)
        self.ffw_layer_2 = ClippableLinear(cfg.hidden_size * 4, cfg.hidden_size, clip)
        # Note: no eps passed, matching the reference - RMSNorm's 1e-6 default.
        self.pre_layer_norm = RMSNorm(cfg.hidden_size)
        self.post_layer_norm = RMSNorm(cfg.hidden_size)
        self.act_fn = activation(cfg.hidden_act)
        self.gradient_clipping = cfg.gradient_clipping
        self.post_layer_scale = cfg.residual_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clip = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        residual = x
        x = self.pre_layer_norm(torch.clamp(x, -clip, clip))
        x = self.ffw_layer_2(self.act_fn(self.ffw_layer_1(x)))
        x = self.post_layer_norm(torch.clamp(x, -clip, clip))
        return x * self.post_layer_scale + residual


class AudioLayer(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.feed_forward1 = AudioFeedForward(cfg)
        self.feed_forward2 = AudioFeedForward(cfg)
        self.self_attn = AudioAttention(cfg)
        self.lconv1d = LightConv1d(cfg)
        self.norm_pre_attn = RMSNorm(cfg.hidden_size)
        self.norm_post_attn = RMSNorm(cfg.hidden_size)
        self.norm_out = RMSNorm(cfg.hidden_size)
        self.gradient_clipping = cfg.gradient_clipping

    def forward(self, x, position_embeddings, mask):
        clip = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)

        x = self.feed_forward1(x)                       # first half-step FFN
        residual = x
        x = self.norm_pre_attn(torch.clamp(x, -clip, clip))
        x = self.self_attn(x, position_embeddings, mask)
        x = self.norm_post_attn(torch.clamp(x, -clip, clip))
        x = x + residual

        x = self.lconv1d(x)                             # local convolution
        x = self.feed_forward2(x)                       # second half-step FFN
        return self.norm_out(torch.clamp(x, -clip, clip))


class AudioTower(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_conv_projection = SubSampleConvProjection(cfg)
        self.rel_pos_enc = RelPositionalEncoding(cfg)
        self.layers = nn.ModuleList([AudioLayer(cfg) for _ in range(cfg.num_layers)])
        self.output_proj = nn.Linear(cfg.hidden_size, cfg.output_proj_dims, bias=True)

    def _blocked_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """Build the [B, 1, num_blocks, chunk, context] attention mask.

        Laid out blockwise, block `b` holds queries `b*chunk + i` and keys
        `b*chunk + j - max_past`. The window itself is written exactly as the
        reference writes it, in terms of `dist = query - key`, and note that
        both bounds are EXCLUSIVE: with `attention_context_left = 13` a query
        sees 12 frames of history, not 13. With `attention_context_right = 0`
        the future half can never fire, so the tower is strictly causal.

        Padding masks KEYS only - a query sitting on a padded frame still
        attends normally, and its output is simply discarded downstream.
        """
        B, T = valid.shape
        chunk, context = self.cfg.attention_chunk_size, self.cfg.context_size
        left_window, right_window = self.cfg.attention_context_left - 1, self.cfg.attention_context_right
        num_blocks = (T + chunk - 1) // chunk
        device = valid.device

        block = torch.arange(num_blocks, device=device)[:, None, None]
        i = torch.arange(chunk, device=device)[None, :, None]
        j = torch.arange(context, device=device)[None, None, :]

        q_pos = block * chunk + i                       # [nb, chunk, 1]
        k_pos = block * chunk + j - left_window

        dist = q_pos - k_pos
        window = (((dist >= 0) & (dist < left_window))
                  | ((dist < 0) & (-dist < right_window)))
        allowed = window & (k_pos >= 0) & (k_pos < T) & (q_pos < T)

        # A key is only usable if that frame is real, not padding.
        key_valid = F.pad(valid, (0, num_blocks * chunk - T), value=False)
        key_valid = key_valid[:, k_pos.clamp(0, T - 1).expand(num_blocks, chunk, context)]
        return (allowed[None] & key_valid).unsqueeze(1)

    def forward(self, features: torch.Tensor, valid: torch.Tensor = None):
        """features [B, frames, mel_bins]; valid [B, frames] bool or None.

        Returns (hidden [B, frames/4, output_proj_dims], valid [B, frames/4]).
        """
        x, valid = self.subsample_conv_projection(features, valid)
        position_embeddings = self.rel_pos_enc(x)

        # The mask is never optional: even with no padding, attention is still
        # restricted to each chunk's context window. Absent a padding mask,
        # every frame counts as real.
        frames_valid = (valid.bool() if valid is not None
                        else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device))
        mask = self._blocked_mask(frames_valid)

        for layer in self.layers:
            x = layer(x, position_embeddings, mask)
        return self.output_proj(x), valid
