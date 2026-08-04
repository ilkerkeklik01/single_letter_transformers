"""Small, explicit checkpoint format shared by training and demos."""

from dataclasses import asdict
from pathlib import Path

import torch

from config import EmbeddingConfig
from model_factory import build_model
from tokenizer import CharTokenizer


def save_checkpoint(path, model, architecture: str, cfg: EmbeddingConfig, tokenizer: CharTokenizer) -> None:
    torch.save(
        {
            "architecture": architecture,
            "config": asdict(cfg),
            "chars": tokenizer.tokens[2:],
            "model": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    tokenizer = CharTokenizer(payload["chars"])
    config_values = dict(payload["config"])
    config_values["matryoshka_dims"] = tuple(config_values["matryoshka_dims"])
    cfg = EmbeddingConfig(**config_values)
    model = build_model(payload["architecture"], cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, cfg, payload["architecture"]
