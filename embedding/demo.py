"""Compare letters/names and retrieve the closest names from the corpus."""

import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data" / "temiz_isimler.txt"
LETTER_PAIRS = (("c", "ç"), ("g", "ğ"), ("i", "ı"), ("o", "ö"), ("a", "z"))
NAME_PAIRS = (("mehmet", "memet"), ("ayşe", "ayşe"), ("elif", "eliff"), ("ali", "veli"))


@torch.no_grad()
def pair_scores(model, architecture, tokenizer, cfg, pairs):
    left_texts = [left for left, _ in pairs]
    right_texts = [right for _, right in pairs]
    left_ids, left_mask = tokenizer.batch_encode(left_texts, cfg.max_seq_len)
    right_ids, right_mask = tokenizer.batch_encode(right_texts, cfg.max_seq_len)
    if architecture == "late":
        left = model(left_ids, left_mask)
        right = model(right_ids, right_mask)
        matrix = model.pairwise_scores(left, right, left_mask, right_mask)
    else:
        matrix = model(left_ids, left_mask) @ model(right_ids, right_mask).T
    return matrix.diag().tolist()


@torch.no_grad()
def nearest_names(model, architecture, tokenizer, cfg, queries, names, k=5):
    query_ids, query_mask = tokenizer.batch_encode(queries, cfg.max_seq_len)
    query_vectors = model(query_ids, query_mask)
    rows = [[] for _ in queries]

    for start in range(0, len(names), 256):
        candidates = names[start : start + 256]
        ids, mask = tokenizer.batch_encode(candidates, cfg.max_seq_len)
        candidate_vectors = model(ids, mask)
        if architecture == "late":
            scores = model.pairwise_scores(query_vectors, candidate_vectors, query_mask, mask)
        else:
            scores = query_vectors @ candidate_vectors.T
        for row_index, row in enumerate(scores.tolist()):
            rows[row_index].extend(zip(candidates, row))
    return [sorted(row, key=lambda item: item[1], reverse=True)[:k] for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("queries", nargs="*", default=["ayse", "mehmed", "çağla"])
    args = parser.parse_args()
    model, tokenizer, cfg, architecture = load_checkpoint(args.checkpoint)
    names = [line for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line]

    print("\nletter similarities")
    for pair, score in zip(LETTER_PAIRS, pair_scores(model, architecture, tokenizer, cfg, LETTER_PAIRS)):
        print(f"  {pair[0]:>2s} ↔ {pair[1]:<2s}  {score:+.3f}")

    print("\nname similarities")
    for pair, score in zip(NAME_PAIRS, pair_scores(model, architecture, tokenizer, cfg, NAME_PAIRS)):
        print(f"  {pair[0]:>8s} ↔ {pair[1]:<8s}  {score:+.3f}")

    print("\nnearest names")
    for query, results in zip(args.queries, nearest_names(model, architecture, tokenizer, cfg, args.queries, names)):
        rendered = ", ".join(f"{name} ({score:.3f})" for name, score in results)
        print(f"  {query}: {rendered}")


if __name__ == "__main__":
    main()
