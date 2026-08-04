from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int = 0
    hidden_size: int = 64
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    vision_dim: int = 32
    vision_heads: int = 4
    image_height: int = 32
    image_width: int = 96
    patch_size: int = 8
    max_text_length: int = 24
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    temperature: float = 0.07
    matryoshka_dims: tuple[int, ...] = (8, 16, 32)
