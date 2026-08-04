"""A tiny exact cosine index: enough to expose a vector database's core contract."""

from pathlib import Path

import torch
import torch.nn.functional as F


class VectorIndex:
    def __init__(self, vectors: torch.Tensor, texts: list[str]):
        if vectors.ndim != 2 or vectors.size(0) != len(texts):
            raise ValueError("vectors must be [items, dim] and align with texts")
        self.vectors = F.normalize(vectors.detach().cpu().float(), dim=-1)
        self.texts = list(texts)

    def search(self, query: torch.Tensor, k: int = 5) -> list[list[tuple[str, float]]]:
        query = F.normalize(query.detach().cpu().float(), dim=-1)
        scores = query @ self.vectors.T
        values, indices = scores.topk(min(k, len(self.texts)), dim=-1)
        return [
            [(self.texts[index], float(score)) for index, score in zip(row_indices, row_values)]
            for row_indices, row_values in zip(indices.tolist(), values.tolist())
        ]

    def save(self, path: str | Path) -> None:
        torch.save({"vectors": self.vectors, "texts": self.texts}, path)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(payload["vectors"], payload["texts"])
