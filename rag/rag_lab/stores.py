"""Vektör veritabanının temel sözleşmesini gösteren exact-cosine depolar."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .similarity import cosine_similarity, normalize
from .types import Chunk, SearchResult


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self, query_vector: Sequence[float], k: int = 4, filters: dict | None = None
    ) -> list[SearchResult]: ...


class InMemoryVectorStore:
    def __init__(self):
        self._items: dict[str, tuple[Chunk, list[float]]] = {}
        self.dimension: int | None = None

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk ve vektör sayıları eşit olmalı")
        for chunk, vector in zip(chunks, vectors):
            clean = self._validate_vector(vector)
            self._items[chunk.id] = (chunk, clean)

    def search(
        self, query_vector: Sequence[float], k: int = 4, filters: dict | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k pozitif olmalı")
        query = self._validate_vector(query_vector, set_dimension=False)
        results = [
            SearchResult(chunk=chunk, score=cosine_similarity(query, vector))
            for chunk, vector in self._items.values()
            if _metadata_matches(chunk.metadata, filters)
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]

    def save(self, path: str | Path, model_name: str) -> None:
        payload = {
            "model_name": model_name,
            "dimension": self.dimension,
            "metric": "cosine",
            "items": [
                {"chunk": _chunk_to_dict(chunk), "vector": vector}
                for chunk, vector in self._items.values()
            ],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, expected_model_name: str | None = None) -> "InMemoryVectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_model_name and payload["model_name"] != expected_model_name:
            raise ValueError(
                f"indeks modeli {payload['model_name']!r}, sorgu modeli {expected_model_name!r}"
            )
        store = cls()
        chunks = [_chunk_from_dict(item["chunk"]) for item in payload["items"]]
        store.add(chunks, [item["vector"] for item in payload["items"]])
        return store

    def _validate_vector(self, vector: Sequence[float], set_dimension: bool = True) -> list[float]:
        clean = normalize([float(value) for value in vector])
        if self.dimension is None and set_dimension:
            self.dimension = len(clean)
        if self.dimension is not None and len(clean) != self.dimension:
            raise ValueError(f"vektör boyutu {len(clean)}, indeks boyutu {self.dimension}")
        return clean


class SQLiteVectorStore:
    """Kalıcılığı öğretir; tarama exact'tir ve büyük ölçek için ANN alternatifi değildir."""

    def __init__(self, path: str | Path = ":memory:"):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, text TEXT NOT NULL,
                metadata TEXT NOT NULL, vector TEXT NOT NULL
            )"""
        )

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk ve vektör sayıları eşit olmalı")
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.text,
                json.dumps(chunk.metadata, ensure_ascii=False),
                json.dumps(normalize([float(value) for value in vector])),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)", rows
            )

    def search(
        self, query_vector: Sequence[float], k: int = 4, filters: dict | None = None
    ) -> list[SearchResult]:
        query = normalize([float(value) for value in query_vector])
        results: list[SearchResult] = []
        for row in self.connection.execute(
            "SELECT id, document_id, text, metadata, vector FROM chunks"
        ):
            metadata = json.loads(row[3])
            if _metadata_matches(metadata, filters):
                chunk = Chunk(id=row[0], document_id=row[1], text=row[2], metadata=metadata)
                results.append(SearchResult(chunk, cosine_similarity(query, json.loads(row[4]))))
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]

    def close(self) -> None:
        self.connection.close()


def _metadata_matches(metadata: dict, filters: dict | None) -> bool:
    return not filters or all(metadata.get(key) == value for key, value in filters.items())


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(payload: dict) -> Chunk:
    return Chunk(**payload)
