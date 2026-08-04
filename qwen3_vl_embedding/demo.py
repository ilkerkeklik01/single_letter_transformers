"""Retrieve rendered name images with text queries."""

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import TinyQwen3VLEmbedding
from render import render_batch
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "checkpoint", nargs="?", type=Path, default=ROOT / "tiny_qwen3_vl_embedding.pt"
)
args = parser.parse_args()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
values = payload["config"]
values["matryoshka_dims"] = tuple(values["matryoshka_dims"])
cfg = ModelConfig(**values)
tokenizer = CharTokenizer(payload["chars"])
model = TinyQwen3VLEmbedding(cfg)
model.load_state_dict(payload["model"])
model.eval()
queries = ["ayse", "mehmed", "cagla"]
candidates = ["ali", "ayşe", "çağla", "mehmet", "veli"]
ids, mask = tokenizer.batch_encode(queries, "query")
images = render_batch(candidates)
with torch.no_grad():
    scores = model.encode_text(ids, mask) @ model.encode_image(images).T
for query, row in zip(queries, scores):
    best = row.argmax().item()
    print(f"{query:>7s} -> image({candidates[best]}) cosine={row[best]:.3f}")
