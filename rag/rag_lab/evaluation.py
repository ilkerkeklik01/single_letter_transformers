"""Retriever'ı generator'dan bağımsız değerlendiren küçük metrikler."""

from collections.abc import Sequence


def recall_at_k(ranked_ids: Sequence[Sequence[str]], relevant_ids: Sequence[set[str]], k: int) -> float:
    if len(ranked_ids) != len(relevant_ids) or not ranked_ids:
        raise ValueError("sorgu ve relevance listeleri boş olmamalı ve eşleşmeli")
    hits = [bool(set(ranking[:k]) & relevant) for ranking, relevant in zip(ranked_ids, relevant_ids)]
    return sum(hits) / len(hits)


def mean_reciprocal_rank(
    ranked_ids: Sequence[Sequence[str]], relevant_ids: Sequence[set[str]]
) -> float:
    if len(ranked_ids) != len(relevant_ids) or not ranked_ids:
        raise ValueError("sorgu ve relevance listeleri boş olmamalı ve eşleşmeli")
    scores: list[float] = []
    for ranking, relevant in zip(ranked_ids, relevant_ids):
        rank = next((index for index, item in enumerate(ranking, start=1) if item in relevant), None)
        scores.append(0.0 if rank is None else 1.0 / rank)
    return sum(scores) / len(scores)
