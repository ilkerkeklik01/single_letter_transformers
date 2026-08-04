"""Train the tiny EmbeddingGemma recipe on noisy/clean names."""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch

from config import ModelConfig
from losses import embedding_gemma_loss
from model import TinyEmbeddingGemma
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "temiz_isimler.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=ROOT / "tiny_embedding_gemma.pt")
    args = parser.parse_args()
    rng = random.Random(1337)
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [x for x in DATA.read_text(encoding="utf-8").splitlines() if x]
    tokenizer = CharTokenizer.from_file(DATA)
    cfg = ModelConfig(tokenizer.vocab_size, tokenizer.pad_id)
    model = TinyEmbeddingGemma(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        clean = rng.sample(names, args.batch_size)
        noisy = [name[:-1] if len(name) > 2 and rng.random() < .5 else name for name in clean]
        q_ids, q_mask = tokenizer.batch_encode(noisy, "query", cfg.max_seq_len, device)
        d_ids, d_mask = tokenizer.batch_encode(clean, "document", cfg.max_seq_len, device)
        loss = embedding_gemma_loss(model(q_ids, q_mask), model(d_ids, d_mask), cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step={step:4d} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "chars": tokenizer.tokens[4:]}, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
