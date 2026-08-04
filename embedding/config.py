"""Configuration shared by the tiny embedding architectures."""

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    vocab_size: int
    pad_id: int = 0
    max_seq_len: int = 24
    hidden_size: int = 64
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 16
    intermediate_size: int = 128
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    temperature: float = 0.07
    matryoshka_dims: tuple[int, ...] = (8, 16, 32)

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_heads * head_dim must equal hidden_size")
        if not self.matryoshka_dims or self.matryoshka_dims[-1] != self.embedding_dim:
            raise ValueError("matryoshka_dims must end with embedding_dim")
        if tuple(sorted(set(self.matryoshka_dims))) != self.matryoshka_dims:
            raise ValueError("matryoshka_dims must be unique and increasing")
