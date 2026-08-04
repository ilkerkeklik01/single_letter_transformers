"""Show instruction-aware retrieval and Matryoshka dimensions."""

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import TinyQwen3Embedding
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "checkpoint", nargs="?", type=Path, default=ROOT / "tiny_qwen3_embedding.pt"
)
args = parser.parse_args()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
values = payload["config"]
values["matryoshka_dims"] = tuple(values["matryoshka_dims"])
cfg = ModelConfig(**values)
tokenizer = CharTokenizer(payload["chars"])
model = TinyQwen3Embedding(cfg)
model.load_state_dict(payload["model"])
model.eval()
queries = ["ayse", "mehmed", "cagla"]
documents = ["ayşe", "mehmet", "çağla", "ali", "veli"]

with torch.no_grad():
    q_ids, q_mask = tokenizer.batch_encode(queries, "query")
    d_ids, d_mask = tokenizer.batch_encode(documents, "document")
    for dimension in cfg.matryoshka_dims:
        scores = model(q_ids, q_mask, dimension) @ model(d_ids, d_mask, dimension).T
        print(f"\n{dimension} dimensions")
        for query, row in zip(queries, scores):
            best = row.argmax().item()
            print(f"  {query:>7s} -> {documents[best]:<7s} cosine={row[best]:.3f}")
