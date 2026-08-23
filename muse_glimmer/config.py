"""Model configuration for the Muse Glimmer text transformer.

One dataclass, two ways to fill it:

  * `ModelConfig()`               -> tiny defaults, trains on a CPU in seconds
  * `ModelConfig.muse_glimmer_30b()` -> the exact released 30B text stack

The whole point of this folder is that the MODEL CODE never changes between
those two - only these numbers do. The tiny defaults keep every architectural
ingredient alive (sliding + full/NoPE layers, GQA, gated attention, softcap),
just at name-generator scale.

Two fields are derived rather than typed in, exactly like the reference
`MuseGlimmerTextConfig` in transformers:

  * `layer_types`      : every 4th layer COUNTED BACKWARD FROM THE LAST is
                         "full_attention", the rest are "sliding_attention".
                         For 52 layers that is the [s, s, s, f] repeating
                         pattern; for our 4 tiny layers it is [s, s, s, f] too.
  * `layer_rope_theta` : the full-attention layers get theta = 0, meaning NoPE
                         (no rotary at all); sliding layers get the global theta.

  * `output_multiplier`: defaults to 1/sqrt(hidden_size / 256), which for
                         hidden_size = 6656 is exactly the released checkpoint's
                         0.19611613513818404.
"""

import math
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 30          # 30 Turkish chars (incl. newline)
    hidden_size: int = 32         # model / embedding dimension
    num_layers: int = 4           # 3 sliding + 1 full(NoPE), the smallest full pattern
    num_heads: int = 4            # number of query heads
    num_kv_heads: int = 2         # number of key/value heads (GQA)
    head_dim: int = 8             # dimension per head (decoupled from hidden_size!)
    intermediate_size: int = 64   # SwiGLU hidden dimension
    max_seq_len: int = 24         # longest sequence we ever feed in (names are short)
    rope_theta: float = 10000.0   # RoPE base frequency (sliding layers only)
    sliding_window: int = 8       # sliding layers see the last 8 tokens (incl. self)
    rms_norm_eps: float = 1e-5    # epsilon for the pre-norms / final norm / QK-norm
    post_norm_eps: float = 1e-8   # epsilon for the two POST-norms in each block

    # The Muse Glimmer signature knobs.
    qk_scale_factor: float = 3.87            # extra multiplier on Q after QK-norm
    final_logit_softcapping: float = 20.0    # logits are squashed into [-20, 20]
    output_multiplier: float = None          # None -> 1/sqrt(hidden_size/256)

    # Derived per-layer schedules (override only if you know why).
    layer_types: list = field(default=None)
    layer_rope_theta: list = field(default=None)

    def __post_init__(self):
        if self.output_multiplier is None:
            self.output_multiplier = 1.0 / math.sqrt(self.hidden_size / 256)
        # Full attention every 4th layer counted backward from the last.
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (self.num_layers - 1 - i) % 4 == 0 else "sliding_attention"
                for i in range(self.num_layers)
            ]
        # Those same full-attention layers are NoPE (theta 0 = no rotary).
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if (self.num_layers - 1 - i) % 4 == 0 else self.rope_theta
                for i in range(self.num_layers)
            ]

    @classmethod
    def muse_glimmer_30b(cls) -> "ModelConfig":
        """The released Muse-Glimmer-30B text stack, number for number.

        Same class, same model code - `MuseGlimmer(ModelConfig.muse_glimmer_30b())`
        builds a module whose state_dict keys and shapes match the language-model
        half of the `unsloth/Muse-Glimmer-30B` checkpoint exactly (the checkpoint
        prefixes them with "model.language_model.", see hf_weights.py).
        """
        return cls(
            vocab_size=202_048,
            hidden_size=6656,
            num_layers=52,
            num_heads=32,
            num_kv_heads=2,
            head_dim=128,             # note: 32 * 128 = 4096 != 6656 - that's intentional
            intermediate_size=19_968,
            max_seq_len=131_072,
            rope_theta=500_000.0,
            sliding_window=2048,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            qk_scale_factor=3.87,
            final_logit_softcapping=20.0,
            # output_multiplier derives to 0.19611613513818404 = 1/sqrt(6656/256)
        )


@dataclass
class VisionConfig:
    """The Muse Glimmer vision tower: a windowed ViT with 2D RoPE.

    Its layer schedule runs the OPPOSITE way from the text stack: full attention
    on every 4th layer counted FORWARD ((i+1) % 4 == 0), plus always the last
    layer, window attention everywhere else. Windows are `pos_emb_side` patches
    on a side - exactly the span the position-embedding table was trained on.
    """

    hidden_size: int = 32         # vision embedding dimension
    intermediate_size: int = 64   # plain-GELU MLP hidden dimension (no gating here)
    num_layers: int = 2           # window + full - the smallest mix of both types
    num_heads: int = 2            # head_dim = hidden / heads (16); no GQA in the tower
    patch_size: int = 2           # pixels per patch side
    patch_temporal: int = 1       # frames folded into one patch (2 in the 30B, for video)
    pos_emb_side: int = 4         # the learned position grid is side x side...
    merge_size: int = 2           # ...and 2x2 patches merge into one LLM token at the end
    layer_norm_eps: float = 1e-5  # the tower uses plain LayerNorm, not RMSNorm
    rope_theta: float = 10000.0   # 2D rotary base (independent of the text theta)

    layer_types: list = field(default=None)   # derived

    def __post_init__(self):
        # Full attention every 4th layer counted FORWARD, plus the last layer.
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 or i == self.num_layers - 1
                else "window_attention"
                for i in range(self.num_layers)
            ]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        """Flattened input size of one patch: frames * RGB * patch * patch."""
        return self.patch_temporal * 3 * self.patch_size**2

    @classmethod
    def muse_glimmer_30b(cls) -> "VisionConfig":
        """The released 30B vision tower (patch_dim = 2*3*14*14 = 1176)."""
        return cls(
            hidden_size=1536,
            intermediate_size=8960,
            num_layers=50,
            num_heads=16,
            patch_size=14,
            patch_temporal=2,
            pos_emb_side=32,
            merge_size=2,
            layer_norm_eps=1e-5,
            rope_theta=10000.0,
        )


@dataclass
class VLConfig:
    """The complete vision+text model: a text config, a vision config, and the
    little bridge between them (adapter widths and placeholder token ids).

    The image/video token ids sit deliberately OUTSIDE the text vocabulary:
    both our model and the reference zero those positions before the embedding
    lookup (their embeddings get overwritten by image features anyway), so they
    never index the embedding table - and no real token can ever be mistaken
    for an image slot.
    """

    text: ModelConfig = field(default_factory=ModelConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    projector_hidden_size: int = 16   # adapter width (4096 in the 30B)
    image_token_id: int = 200_092     # placeholder id for one merged image token
    video_token_id: int = 200_091     # same, for video frames

    @property
    def out_hidden_size(self) -> int:
        """Width of one merged vision token: hidden * merge^2 (6144 in the 30B)."""
        return self.vision.hidden_size * self.vision.merge_size**2

    @classmethod
    def muse_glimmer_30b(cls) -> "VLConfig":
        """The complete released model - `MuseGlimmerVL(VLConfig.muse_glimmer_30b())`
        matches ALL 1436 tensors of the `unsloth/Muse-Glimmer-30B` checkpoint."""
        return cls(
            text=ModelConfig.muse_glimmer_30b(),
            vision=VisionConfig.muse_glimmer_30b(),
            projector_hidden_size=4096,
            image_token_id=200_092,
            video_token_id=200_091,
        )
