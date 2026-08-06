"""Örnek veri ve çıktı yardımcıları."""

import json
from pathlib import Path

from rag_lab import Document


ROOT = Path(__file__).resolve().parents[1]


def load_documents() -> list[Document]:
    rows = json.loads((ROOT / "data" / "bilgi_bankasi.json").read_text(encoding="utf-8"))
    return [
        Document(
            id=row["id"],
            text=f"# {row['title']}\n{row['text']}",
            metadata={"title": row["title"], "category": row["category"]},
        )
        for row in rows
    ]


def print_answer(result) -> None:
    print(f"\nCevap:\n{result.answer}\n\nGetirilen parçalar:")
    for source in result.sources:
        print(f"- {source.chunk.id:22} cosine={source.score:+.3f}  {source.chunk.metadata['title']}")
