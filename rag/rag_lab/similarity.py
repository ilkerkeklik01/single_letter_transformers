"""Cosine benzerliğini kütüphanesiz ve görünür biçimde hesaplar."""

import math
from collections.abc import Sequence


Vector = Sequence[float]


def dot(a: Vector, b: Vector) -> float:
    _same_nonempty_dimension(a, b)
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vector: Vector) -> float:
    if not vector:
        raise ValueError("vektör boş olamaz")
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: Vector) -> list[float]:
    length = l2_norm(vector)
    if length == 0.0:
        raise ValueError("sıfır vektör normalize edilemez")
    return [value / length for value in vector]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Yön benzerliği: -1 zıt, 0 dik, +1 aynı yön."""
    denominator = l2_norm(a) * l2_norm(b)
    _same_nonempty_dimension(a, b)
    if denominator == 0.0:
        raise ValueError("cosine benzerliği sıfır vektör için tanımsızdır")
    return dot(a, b) / denominator


def _same_nonempty_dimension(a: Vector, b: Vector) -> None:
    if not a or not b:
        raise ValueError("vektörler boş olamaz")
    if len(a) != len(b):
        raise ValueError(f"boyutlar eşit değil: {len(a)} != {len(b)}")
