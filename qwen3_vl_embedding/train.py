"""Align noisy text queries with rendered clean-name images."""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import TinyQwen3VLEmbedding
from render import render_batch
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "temiz_isimler.txt"


def loss_at_dimensions(model, ids, mask, images, cfg):
    labels = torch.arange(ids.size(0), device=ids.device)
    losses = []
    text_full, image_full = model.encode_text(ids, mask), model.encode_image(images)
    for dimension in cfg.matryoshka_dims:
        text = F.normalize(text_full[:, :dimension], dim=-1)
        image = F.normalize(image_full[:, :dimension], dim=-1)
        logits = text @ image.T / cfg.temperature
        losses.append(.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "tiny_qwen3_vl_embedding.pt")
    args = parser.parse_args()
    rng = random.Random(1337)
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [x for x in DATA.read_text(encoding="utf-8").splitlines() if x]
    tokenizer = CharTokenizer.from_file(DATA)
    cfg = ModelConfig(tokenizer.vocab_size, tokenizer.pad_id)
    model = TinyQwen3VLEmbedding(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        clean = rng.sample(names, args.batch_size)
        noisy = [name.translate(str.maketrans("çğıöşü", "cgiosu")) for name in clean]
        ids, mask = tokenizer.batch_encode(noisy, "query", cfg.max_text_length, device)
        images = render_batch(clean, cfg.image_height, cfg.image_width, device)
        loss = loss_at_dimensions(model, ids, mask, images, cfg)
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
