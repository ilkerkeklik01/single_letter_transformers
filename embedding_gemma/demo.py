"""Compare retrieval at every supported Matryoshka dimension."""

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import TinyEmbeddingGemma
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "checkpoint", nargs="?", type=Path, default=ROOT / "tiny_embedding_gemma.pt"
)
args = parser.parse_args()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
values = payload["config"]
values["matryoshka_dims"] = tuple(values["matryoshka_dims"])
cfg = ModelConfig(**values)
tokenizer = CharTokenizer(payload["chars"])
model = TinyEmbeddingGemma(cfg)
model.load_state_dict(payload["model"])
model.eval()
queries = ["ayse", "mehmed", "cagla"]
documents = ["ayşe", "mehmet", "çağla", "ali", "veli"]
q_ids, q_mask = tokenizer.batch_encode(queries, "query")
d_ids, d_mask = tokenizer.batch_encode(documents, "document")
with torch.no_grad():
    for dimension in cfg.matryoshka_dims:
        scores = model(q_ids, q_mask, dimension) @ model(d_ids, d_mask, dimension).T
        print(f"\n{dimension} dimensions")
        for query, row in zip(queries, scores):
            best = row.argmax().item()
            print(f"  {query:>7s} -> {documents[best]:<7s} cosine={row[best]:.3f}")
