"""One construction point for scripts and checkpoint loading."""

from config import EmbeddingConfig
from late_interaction import LateInteractionEncoder
from mean_encoder import MeanEncoder
from transformer_encoder import TransformerEncoder


def build_model(architecture: str, cfg: EmbeddingConfig):
    if architecture == "mean":
        return MeanEncoder(cfg)
    if architecture == "transformer":
        return TransformerEncoder(cfg)
    if architecture == "late":
        return LateInteractionEncoder(cfg)
    raise ValueError(f"unknown architecture: {architecture}")
