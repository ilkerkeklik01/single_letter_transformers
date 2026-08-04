"""Tiny Qwen3-Embedding configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int = 0
    eos_id: int = 2
    hidden_size: int = 64
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    max_seq_len: int = 32
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    temperature: float = 0.07
    matryoshka_dims: tuple[int, ...] = (8, 16, 32)

    def __post_init__(self):
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_heads * head_dim must equal hidden_size")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
