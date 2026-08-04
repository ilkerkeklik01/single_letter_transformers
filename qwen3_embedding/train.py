"""Train TinyQwen3Embedding on clean/noisy Turkish-name pairs."""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch

from config import ModelConfig
from losses import matryoshka_loss
from model import TinyQwen3Embedding
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "temiz_isimler.txt"
LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"


def noisy(name, rng):
    if len(name) > 1 and rng.random() < 0.5:
        index = rng.randrange(len(name))
        return name[:index] + name[index + 1 :]
    index = rng.randrange(len(name))
    return name[:index] + rng.choice(LETTERS) + name[index + 1 :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=ROOT / "tiny_qwen3_embedding.pt")
    args = parser.parse_args()
    rng = random.Random(1337)
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [name for name in DATA.read_text(encoding="utf-8").splitlines() if name]
    tokenizer = CharTokenizer.from_file(DATA)
    cfg = ModelConfig(tokenizer.vocab_size, tokenizer.pad_id, tokenizer.eos_id)
    model = TinyQwen3Embedding(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, args.steps + 1):
        clean = rng.sample(names, args.batch_size)
        queries = [noisy(name, rng) for name in clean]
        q_ids, q_mask = tokenizer.batch_encode(queries, "query", cfg.max_seq_len, device)
        d_ids, d_mask = tokenizer.batch_encode(clean, "document", cfg.max_seq_len, device)
        # Full vectors are returned; the loss re-normalizes every nested prefix.
        q = model(q_ids, q_mask)
        d = model(d_ids, d_mask)
        loss = matryoshka_loss(q, d, cfg.matryoshka_dims, cfg.temperature)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step={step:4d} loss={loss.item():.4f}")

    torch.save({"model": model.state_dict(), "config": asdict(cfg), "chars": tokenizer.tokens[5:]}, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
