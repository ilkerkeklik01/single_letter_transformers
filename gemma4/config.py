"""Model configuration for the Gemma 4 text transformer.

One dataclass, several ways to fill it:

  * `ModelConfig()`            -> tiny dense defaults, trains on a CPU in seconds
  * `ModelConfig.tiny_moe()`   -> tiny, but with the MoE + k=v recipe switched on
  * `ModelConfig.gemma4_e2b()` -> the released google/gemma-4-e2b-it text stack
  * `ModelConfig.gemma4_26b_a4b()` -> the released google/gemma-4-26B-A4B-it text stack

The model code never changes between them - only these numbers do. That matters
more here than in the other folders, because the two released Gemma 4 models use
almost DISJOINT feature sets:

                          e2b (dense)      26B-A4B (sparse)
    per-layer embeddings   yes (256)        no (0)
    double-wide MLP        yes              no
    KV-shared layers       20 of 35         0
    MoE experts            no               128, top-8
    k and v share weights  no               yes (on full-attention layers)

Both are the same architecture; they just turn different switches on. Anything
this file derives (`layer_types`, per-layer `head_dim` / `num_kv_heads`, which
layers share KV, which layers go double-wide) follows the same rules the
reference `Gemma4TextConfig` uses, so it stays right at any depth you pick.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 30           # 30 Turkish chars (incl. newline)
    hidden_size: int = 32          # model / embedding dimension
    num_layers: int = 8            # 6 sliding + 2 full, with 3 KV-shared at the end
    num_heads: int = 4             # number of query heads
    num_kv_heads: int = 2          # key/value heads on SLIDING layers (GQA)
    head_dim: int = 8              # per-head dim on SLIDING layers
    intermediate_size: int = 64    # GeGLU hidden dimension
    max_seq_len: int = 24          # longest sequence we ever feed in
    rms_norm_eps: float = 1e-6     # epsilon inside RMSNorm

    # Local (sliding) vs global (full) attention.
    sliding_window: int = 8        # sliding layers only see the last 8 tokens
    sliding_pattern: int = 5       # every 5th layer is full; the last is always full
    rope_theta_sliding: float = 10_000.0     # RoPE base on sliding layers
    rope_theta_full: float = 1_000_000.0     # RoPE base on full layers
    partial_rotary_factor: float = 0.25      # full layers rotate only the first 25%

    # Full-attention layers are WIDER per head, and may use fewer KV heads.
    global_head_dim: int = 16          # head_dim on full layers (512 in both releases)
    num_global_kv_heads: int = None    # None -> same as num_kv_heads

    # Per-Layer Embeddings (PLE). 0 disables the whole mechanism.
    hidden_size_per_layer_input: int = 8
    vocab_size_per_layer_input: int = None   # None -> same as vocab_size

    # KV sharing: the last `num_kv_shared_layers` layers reuse an earlier layer's
    # keys/values instead of projecting their own.
    num_kv_shared_layers: int = 3
    use_double_wide_mlp: bool = True   # KV-shared layers get a 2x wider MLP
    attention_k_eq_v: bool = False     # full layers reuse k_proj output as v

    # Mixture of Experts (runs ALONGSIDE the dense MLP, not instead of it).
    enable_moe_block: bool = False
    num_experts: int = None
    top_k_experts: int = None
    moe_intermediate_size: int = None

    hidden_activation: str = "gelu_pytorch_tanh"   # GeGLU, not SwiGLU
    final_logit_softcapping: float = 30.0          # None disables the cap
    initializer_range: float = 0.02                # std for the stacked expert weights
    pad_token_id: int = 0                          # substituted at media placeholders

    layer_types: list = field(default=None)        # derived

    def __post_init__(self):
        if self.vocab_size_per_layer_input is None:
            self.vocab_size_per_layer_input = self.vocab_size
        # Every `sliding_pattern`-th layer is full attention; the last layer is
        # always full (the reference forces this and warns if you fight it).
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % self.sliding_pattern else "full_attention"
                for i in range(self.num_layers)
            ]
        self.layer_types[-1] = "full_attention"

    # -- per-layer shape overrides -------------------------------------------
    # Full-attention layers are configured differently from sliding ones. The
    # reference stores this as a `per_layer_config` dict; two small methods are
    # clearer, and give the same answers.

    def head_dim_for(self, layer_idx: int) -> int:
        return (self.global_head_dim if self.layer_types[layer_idx] == "full_attention"
                else self.head_dim)

    def num_kv_heads_for(self, layer_idx: int) -> int:
        # The kv-head override only applies to models that declare `attention_k_eq_v`
        # AND give an explicit global count - exactly the reference's gating.
        if (self.layer_types[layer_idx] == "full_attention"
                and self.attention_k_eq_v and self.num_global_kv_heads is not None):
            return self.num_global_kv_heads
        return self.num_kv_heads

    def rope_theta_for(self, layer_type: str) -> float:
        return (self.rope_theta_full if layer_type == "full_attention"
                else self.rope_theta_sliding)

    # -- KV sharing bookkeeping ----------------------------------------------

    @property
    def first_kv_shared_layer(self) -> int:
        return self.num_layers - self.num_kv_shared_layers

    def is_kv_shared(self, layer_idx: int) -> bool:
        """Does this layer reuse someone else's keys/values (and so build no k/v)?"""
        return layer_idx >= self.first_kv_shared_layer >= 0

    def uses_double_wide_mlp(self, layer_idx: int) -> bool:
        # Note the strict `> 0` (the reference's, deliberately): a model where
        # EVERY layer is kv-shared gets no double-wide MLP at all.
        return self.use_double_wide_mlp and layer_idx >= self.first_kv_shared_layer > 0

    def stores_shared_kv(self, layer_idx: int) -> bool:
        """Is this the layer whose keys/values later layers of its type will reuse?

        It is the LAST non-shared layer of each attention type.
        """
        if self.is_kv_shared(layer_idx):
            return False
        prefix = self.layer_types[:self.first_kv_shared_layer]
        return layer_idx == len(prefix) - 1 - prefix[::-1].index(self.layer_types[layer_idx])

    def validate(self) -> None:
        """Every KV-shared layer must have an earlier layer of its type to copy."""
        prefix = set(self.layer_types[:self.first_kv_shared_layer])
        for i in range(self.first_kv_shared_layer, self.num_layers):
            if self.layer_types[i] not in prefix:
                raise ValueError(
                    f"layer {i} is KV-shared and of type {self.layer_types[i]!r}, but no "
                    f"layer before {self.first_kv_shared_layer} has that type to share from. "
                    "Lower num_kv_shared_layers."
                )

    # -- presets --------------------------------------------------------------

    @classmethod
    def tiny_moe(cls) -> "ModelConfig":
        """Tiny, but wired the 26B-A4B way: MoE experts, k=v, and no PLE."""
        return cls(
            num_layers=6,
            sliding_pattern=6,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
            num_global_kv_heads=1,
            enable_moe_block=True,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=16,
        )

    @classmethod
    def gemma4_e2b(cls) -> "ModelConfig":
        """The released google/gemma-4-e2b-it text stack, number for number."""
        return cls(
            vocab_size=262_144,
            hidden_size=1536,
            num_layers=35,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            intermediate_size=6144,
            max_seq_len=131_072,
            sliding_window=512,
            sliding_pattern=5,
            global_head_dim=512,
            num_global_kv_heads=None,
            hidden_size_per_layer_input=256,
            vocab_size_per_layer_input=262_144,
            num_kv_shared_layers=20,
            use_double_wide_mlp=True,
            attention_k_eq_v=False,
        )

    @classmethod
    def gemma4_26b_a4b(cls) -> "ModelConfig":
        """The released google/gemma-4-26B-A4B-it text stack, number for number."""
        return cls(
            vocab_size=262_144,
            hidden_size=2816,
            num_layers=30,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            intermediate_size=2112,
            max_seq_len=131_072,
            sliding_window=1024,
            sliding_pattern=6,
            global_head_dim=512,
            num_global_kv_heads=2,
            hidden_size_per_layer_input=0,      # this one has no PLE
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=128,
            top_k_experts=8,
            moe_intermediate_size=704,
        )


@dataclass
class VisionConfig:
    """The Gemma 4 vision tower: a bidirectional ViT with 2-D rotary positions.

    It shares the text stack's sandwich norms and its q/k/v-norm attention (also
    with `scale = 1.0`), but drops causality, GQA and every text-only extra.
    Two things are specific to it: positions are looked up from a pair of
    learned 1-D tables (one for x, one for y) and summed, and the encoder output
    is average-pooled over `pooling_kernel_size**2` patch blocks before it ever
    reaches the language model.
    """

    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    num_kv_heads: int = 2          # the tower has no GQA: kv heads == query heads
    head_dim: int = 16
    patch_size: int = 4            # pixels per patch side; input is 3 * patch^2 per patch
    pooling_kernel_size: int = 3   # k x k patches average into one soft token
    position_embedding_size: int = 64   # rows in each of the two position tables
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    use_clipped_linears: bool = True    # both released towers enable this
    standardize: bool = False

    @property
    def patch_dim(self) -> int:
        """Flattened size of one patch: RGB * patch * patch."""
        return 3 * self.patch_size**2

    @classmethod
    def gemma4_e2b(cls) -> "VisionConfig":
        """e2b's vision tower: smaller, with clipped linears and no standardize."""
        return cls(
            hidden_size=768,
            intermediate_size=3072,
            num_layers=16,
            num_heads=12,
            num_kv_heads=12,
            head_dim=64,
            patch_size=16,
            pooling_kernel_size=3,
            position_embedding_size=10240,
            rms_norm_eps=1e-6,
            rope_theta=100.0,
            use_clipped_linears=True,
            standardize=False,
        )

    @classmethod
    def gemma4_26b_a4b(cls) -> "VisionConfig":
        """26B-A4B's vision tower: bigger, and the switches are flipped the
        other way - no clipped linears, but the pooled features ARE standardized."""
        return cls(
            hidden_size=1152,
            intermediate_size=4304,
            num_layers=27,
            num_heads=16,
            num_kv_heads=16,
            head_dim=72,
            patch_size=16,
            pooling_kernel_size=3,
            position_embedding_size=10240,
            rms_norm_eps=1e-6,
            rope_theta=100.0,
            use_clipped_linears=False,
            standardize=True,
        )


@dataclass
class AudioConfig:
    """The Gemma 4 audio tower: a Conformer, not a plain transformer.

    Each layer is `FFN -> attention -> lightweight conv -> FFN`, with half-step
    residuals on the feed-forwards (the `residual_weight = 0.5` "macaron"
    pattern). Attention is CHUNKED and local: the sequence is cut into blocks of
    `attention_chunk_size`, and each block attends over itself plus
    `attention_context_left - 1` frames of history and `attention_context_right`
    of future - so by default it is strictly causal in time.

    Position information is relative and sinusoidal (Transformer-XL style, with
    the `matrix_ac + matrix_bd` decomposition), not rotary.
    """

    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 2
    subsampling_conv_channels: tuple = (8, 4)   # two stride-2 conv layers
    conv_kernel_size: int = 5                   # depthwise causal conv width
    residual_weight: float = 0.5                # macaron half-step residual
    attention_chunk_size: int = 4
    attention_context_left: int = 5             # frames of history (minus 1)
    attention_context_right: int = 0            # frames of future (0 = causal)
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9
    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 48                  # width handed to the language model
    hidden_act: str = "silu"

    @property
    def context_size(self) -> int:
        """Keys visible to one chunk: itself, plus left history and right future."""
        return (self.attention_chunk_size + self.attention_context_left - 1
                + self.attention_context_right)

    @property
    def mel_bins(self) -> int:
        """Input feature width the subsampler expects.

        `input_proj_linear` is built as `(channels[0] // 4) * channels[1]` wide,
        and the two stride-2 convolutions divide the feature axis by 4 - so the
        reference is implicitly assuming the input has `channels[0]` mel bins.
        """
        return self.subsampling_conv_channels[0]

    @classmethod
    def gemma4(cls) -> "AudioConfig":
        """The released Gemma 4 audio tower (shared by e2b and 26B-A4B)."""
        return cls(
            hidden_size=1024,
            num_layers=12,
            num_heads=8,
            subsampling_conv_channels=(128, 32),
            conv_kernel_size=5,
            residual_weight=0.5,
            attention_chunk_size=12,
            attention_context_left=13,
            attention_context_right=0,
            attention_logit_cap=50.0,
            attention_invalid_logits_value=-1.0e9,
            use_clipped_linears=True,
            rms_norm_eps=1e-6,
            gradient_clipping=1e10,
            output_proj_dims=1536,
            hidden_act="silu",
        )


@dataclass
class VLConfig:
    """The complete Gemma 4: a text stack, a vision tower, an audio tower, and
    the two small projections that bring each modality into text space.

    The modality placeholder ids sit outside the toy vocabulary on purpose: both
    our model and the reference zero them before the embedding lookup, because
    those positions get overwritten by encoder features anyway.
    """

    text: ModelConfig = field(default_factory=ModelConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)   # None -> no audio tower
    image_token_id: int = 258_880
    audio_token_id: int = 258_881
    video_token_id: int = 258_884

    @classmethod
    def tiny_26b_style(cls) -> "VLConfig":
        """Tiny, but wired the 26B-A4B way: MoE text, a standardizing vision
        tower with no clipped linears, and no audio tower at all."""
        return cls(
            text=ModelConfig.tiny_moe(),
            vision=VisionConfig(use_clipped_linears=False, standardize=True),
            audio=None,
        )

    @classmethod
    def gemma4_e2b(cls) -> "VLConfig":
        """The complete released google/gemma-4-e2b-it: text + vision + audio."""
        return cls(text=ModelConfig.gemma4_e2b(),
                   vision=VisionConfig.gemma4_e2b(), audio=AudioConfig.gemma4())

    @classmethod
    def gemma4_26b_a4b(cls) -> "VLConfig":
        """The complete released google/gemma-4-26B-A4B-it.

        This one ships **no audio tower at all** - the released config has
        `audio_config: null`, and the reference simply does not build one. The
        towers are per-model too, not a shared component: this vision encoder is
        wider and deeper than e2b's, and flips both of its switches.
        """
        return cls(text=ModelConfig.gemma4_26b_a4b(),
                   vision=VisionConfig.gemma4_26b_a4b(), audio=None)
