from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int = 0
    hidden_size: int = 64
    projection_size: int = 256
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    max_seq_len: int = 32
    local_window: int = 8
    rope_theta: float = 10_000.0
    eps: float = 1e-6
    temperature: float = 0.07
    gor_weight: float = 0.05
    matryoshka_dims: tuple[int, ...] = (8, 16, 32)
