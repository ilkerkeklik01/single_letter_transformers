"""Embed every clean name and persist an exact cosine-search index."""

import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from vector_store import VectorIndex


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data" / "temiz_isimler.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "name_index.pt")
    args = parser.parse_args()

    model, tokenizer, cfg, architecture = load_checkpoint(args.checkpoint)
    if architecture == "late":
        raise SystemExit("late interaction stores one vector per token; use demo.py for MaxSim search")

    names = [line for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(names), 256):
            ids, mask = tokenizer.batch_encode(names[start : start + 256], cfg.max_seq_len)
            chunks.append(model(ids, mask))
    index = VectorIndex(torch.cat(chunks), names)
    index.save(args.output)
    print(f"indexed={len(names)} dimension={index.vectors.size(1)} saved={args.output}")


if __name__ == "__main__":
    main()
