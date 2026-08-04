"""Train a tiny name embedding model with in-batch contrastive learning.

Examples:
    python train.py mean --steps 400
    python train.py transformer --steps 800
    python train.py late --steps 800
"""

import argparse
from pathlib import Path
import random

import torch

from augment import corrupt_name, make_positive_pairs
from checkpoint import save_checkpoint
from config import EmbeddingConfig
from late_interaction import LateInteractionEncoder
from losses import late_interaction_info_nce, matryoshka_info_nce
from model_factory import build_model
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data" / "temiz_isimler.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture", choices=("mean", "transformer", "late"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def retrieval_at_one(model, architecture, tokenizer, names, rng, device, limit=160):
    """Can a noisy spelling retrieve its clean source among held-out names?"""
    model.eval()
    clean = names[:limit]
    noisy = [corrupt_name(name, rng) for name in clean]
    left_ids, left_mask = tokenizer.batch_encode(noisy, device=device)
    right_ids, right_mask = tokenizer.batch_encode(clean, device=device)

    if architecture == "late":
        left = model(left_ids, left_mask)
        right = model(right_ids, right_mask)
        scores = model.pairwise_scores(left, right, left_mask, right_mask)
    else:
        scores = model(left_ids, left_mask) @ model(right_ids, right_mask).T
    return (scores.argmax(dim=1) == torch.arange(len(clean), device=device)).float().mean().item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    names = [line.strip() for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng.shuffle(names)
    split = round(0.8 * len(names))
    train_names, validation_names = names[:split], names[split:]

    tokenizer = CharTokenizer.from_file(DATA_FILE)
    cfg = EmbeddingConfig(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model = build_model(args.architecture, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} architecture={args.architecture} parameters={parameters:,}")

    for step in range(1, args.steps + 1):
        anchors, positives = make_positive_pairs(train_names, args.batch_size, rng)
        anchor_ids, anchor_mask = tokenizer.batch_encode(anchors, cfg.max_seq_len, device)
        positive_ids, positive_mask = tokenizer.batch_encode(positives, cfg.max_seq_len, device)

        if args.architecture == "late":
            anchor_tokens = model(anchor_ids, anchor_mask)
            positive_tokens = model(positive_ids, positive_mask)
            scores_ab = model.pairwise_scores(anchor_tokens, positive_tokens, anchor_mask, positive_mask)
            scores_ba = model.pairwise_scores(positive_tokens, anchor_tokens, positive_mask, anchor_mask)
            loss = late_interaction_info_nce(scores_ab, scores_ba, cfg.temperature)
        else:
            anchor_vectors = model(anchor_ids, anchor_mask)
            positive_vectors = model(positive_ids, positive_mask)
            loss = matryoshka_info_nce(
                anchor_vectors, positive_vectors, cfg.matryoshka_dims, cfg.temperature
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            score = retrieval_at_one(
                model, args.architecture, tokenizer, validation_names, rng, device
            )
            print(f"step={step:4d} loss={loss.item():.4f} validation_recall@1={score:.3f}")
            model.train()

    output = args.output or ROOT / f"tiny_{args.architecture}_embedding.pt"
    save_checkpoint(output, model, args.architecture, cfg, tokenizer)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
