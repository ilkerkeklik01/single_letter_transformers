"""Model configuration for Qwen3.8.

Qwen3.8 is not a new architecture. Its `config.json` says
`model_type = "qwen3_5"` and `architectures = ["Qwen3_5ForConditionalGeneration"]`
- it is the **Qwen3.5 architecture with new weights**, which is why this folder
reads so much like [`../qwen3_5/`](../qwen3_5/), only exact instead of simplified.

Several ways to fill the config:

  * `TextConfig()`                  -> tiny dense defaults, trains on CPU in seconds
  * `TextConfig.tiny_moe()`         -> tiny, with the sparse recipe switched on
  * `TextConfig.qwen3_8_27b()`      -> the released Qwen/Qwen3.8-27B text stack
  * `TextConfig.qwen3_8_2_4t_a95b()`-> the released Qwen/Qwen3.8-2.4T-A95B stack

The model code never changes between them. The two releases differ mostly in
scale and in one switch:

                        27B (dense)        2.4T-A95B (sparse)
    layers               64                 92
    hidden               5120               8192
    feed-forward         dense, 17408       512 experts, top-10, + shared expert
    linear value heads   48                 128
    vision tower         yes                no
    MTP head             yes                yes

Everything derived here follows the reference `Qwen3_5TextConfig` rules:
`layer_types` is 3 linear-attention layers to every 1 full-attention layer, and
the rotary dimension is only `partial_rotary_factor` of each head.
"""

from dataclasses import dataclass, field


@dataclass
class TextConfig:
    vocab_size: int = 30           # 30 Turkish chars (incl. newline)
    hidden_size: int = 32          # model / embedding dimension
    num_layers: int = 4            # 3 linear-attention + 1 full-attention
    intermediate_size: int = 64    # SwiGLU hidden dimension (dense layers)
    max_seq_len: int = 24          # longest sequence we ever feed in

    # --- full-attention layers ---
    num_heads: int = 4             # query heads
    num_kv_heads: int = 2          # key/value heads (GQA)
    head_dim: int = 32             # decoupled from hidden_size, as in the releases
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25   # only the first 25% of each head rotates
    mrope_section: tuple = (2, 1, 1)      # time / height / width split, for images

    # --- linear-attention (Gated DeltaNet) layers ---
    linear_num_key_heads: int = 2
    linear_num_value_heads: int = 4       # a multiple of the key heads
    linear_key_head_dim: int = 8
    linear_value_head_dim: int = 8
    linear_conv_kernel_dim: int = 4       # short depthwise conv before the recurrence
    chunk_size: int = 64                  # chunk length of the parallel delta rule

    full_attention_interval: int = 4      # every 4th layer is full attention
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    # --- Mixture of Experts (None -> dense feed-forward everywhere) ---
    num_experts: int = None
    num_experts_per_tok: int = None
    moe_intermediate_size: int = None
    shared_expert_intermediate_size: int = None

    # --- Multi-Token Prediction head (0 -> no MTP module) ---
    mtp_num_hidden_layers: int = 1

    layer_types: list = field(default=None)   # derived

    def __post_init__(self):
        # 3 linear : 1 full, exactly as the reference derives it.
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (i + 1) % self.full_attention_interval else "full_attention"
                for i in range(self.num_layers)
            ]

    @property
    def enable_moe(self) -> bool:
        return self.num_experts is not None

    @property
    def rotary_dim(self) -> int:
        """How many channels of each head actually rotate."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        """Width of the depthwise conv: query and key and value, side by side."""
        return self.linear_key_dim * 2 + self.linear_value_dim

    def validate(self) -> None:
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                f"linear_num_value_heads ({self.linear_num_value_heads}) must be a "
                f"multiple of linear_num_key_heads ({self.linear_num_key_heads})")
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"mrope_section {tuple(self.mrope_section)} must sum to rotary_dim // 2 "
                f"= {self.rotary_dim // 2}")

    # -- presets --------------------------------------------------------------

    @classmethod
    def tiny_moe(cls) -> "TextConfig":
        """Tiny, but wired the 2.4T-A95B way: sparse experts plus a shared one."""
        return cls(num_experts=8, num_experts_per_tok=2,
                   moe_intermediate_size=16, shared_expert_intermediate_size=16)

    @classmethod
    def qwen3_8_27b(cls) -> "TextConfig":
        """The released Qwen/Qwen3.8-27B text stack, number for number."""
        return cls(
            vocab_size=248_320,
            hidden_size=5120,
            num_layers=64,
            intermediate_size=17_408,
            max_seq_len=262_144,
            num_heads=24,
            num_kv_heads=4,
            head_dim=256,
            rope_theta=10_000_000.0,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            full_attention_interval=4,
            mtp_num_hidden_layers=1,
        )

    @classmethod
    def qwen3_8_2_4t_a95b(cls) -> "TextConfig":
        """The released Qwen/Qwen3.8-2.4T-A95B text stack, number for number."""
        return cls(
            vocab_size=248_320,
            hidden_size=8192,
            num_layers=92,
            intermediate_size=None,      # every layer is sparse
            max_seq_len=262_144,
            num_heads=64,
            num_kv_heads=4,
            head_dim=256,
            rope_theta=10_000_000.0,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            linear_num_key_heads=16,
            linear_num_value_heads=128,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            full_attention_interval=4,
            num_experts=512,
            num_experts_per_tok=10,
            moe_intermediate_size=2048,
            shared_expert_intermediate_size=2048,
            mtp_num_hidden_layers=1,
        )


@dataclass
class VisionConfig:
    """The Qwen3.8 vision tower - a Qwen3-VL style ViT.

    Patches are 3-D (`temporal_patch_size` frames x `patch_size` x `patch_size`),
    so one Conv3d handles images and video alike; a still image is just a clip
    whose frames are identical. Positions come from a learned table that is
    bilinearly resized to whatever grid the image actually has, and the final
    merger folds `spatial_merge_size**2` neighbouring patches into one token.
    """

    depth: int = 2                    # encoder blocks
    hidden_size: int = 32
    intermediate_size: int = 64
    num_heads: int = 2
    in_channels: int = 3
    patch_size: int = 4
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 32         # must equal the text hidden size
    num_position_embeddings: int = 64  # a (side x side) learned grid, side = sqrt(this)

    @property
    def patch_dim(self) -> int:
        """Flattened size of one 3-D patch."""
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @classmethod
    def qwen3_8_27b(cls) -> "VisionConfig":
        """The released Qwen3.8-27B vision tower."""
        return cls(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=5120,
            num_position_embeddings=2304,
        )


@dataclass
class VLConfig:
    """The complete model: a text stack plus (optionally) a vision tower.

    27B ships with vision; 2.4T-A95B is text-only, so `vision` is None there and
    nothing vision-shaped is built - exactly like the released checkpoints.
    """

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)   # None -> text only
    image_token_id: int = 248_056
    video_token_id: int = 248_057
    vision_start_token_id: int = 248_053
    vision_end_token_id: int = 248_054

    @classmethod
    def qwen3_8_27b(cls) -> "VLConfig":
        """The complete released Qwen/Qwen3.8-27B: text + vision + MTP."""
        return cls(text=TextConfig.qwen3_8_27b(), vision=VisionConfig.qwen3_8_27b())

    @classmethod
    def qwen3_8_2_4t_a95b(cls) -> "VLConfig":
        """The complete released Qwen/Qwen3.8-2.4T-A95B: sparse text + MTP, no vision."""
        return cls(text=TextConfig.qwen3_8_2_4t_a95b(), vision=None)
