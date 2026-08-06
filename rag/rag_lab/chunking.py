"""Üç öğretici chunking stratejisi: sabit, cümle ve Markdown bölümü."""

import re
from collections.abc import Callable, Iterable

from .types import Chunk, Document


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def fixed_word_chunks(text: str, size: int = 80, overlap: int = 15) -> list[str]:
    """Kelime pencereleri; ucuz ve öngörülebilir, fakat anlam sınırlarını bilmez."""
    if size <= 0:
        raise ValueError("size pozitif olmalı")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap, 0 <= overlap < size koşulunu sağlamalı")
    words = text.split()
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def sentence_chunks(text: str, max_words: int = 80, overlap_sentences: int = 1) -> list[str]:
    """Cümleleri bölmeden yaklaşık bir kelime bütçesinde pencereler oluşturur."""
    if max_words <= 0:
        raise ValueError("max_words pozitif olmalı")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences negatif olamaz")
    sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(text.strip()) if item.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = start
        word_count = 0
        while end < len(sentences):
            sentence_words = len(sentences[end].split())
            if end > start and word_count + sentence_words > max_words:
                break
            word_count += sentence_words
            end += 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        start = max(start + 1, end - overlap_sentences)
    return chunks


def markdown_chunks(text: str, max_words: int = 120, overlap: int = 0) -> list[str]:
    """Başlıkları korur; uzun bölümleri sabit kelime pencerelerine böler."""
    matches = list(_MARKDOWN_HEADING.finditer(text))
    if not matches:
        return fixed_word_chunks(text, max_words, overlap)

    sections: list[str] = []
    preface = text[: matches[0].start()].strip()
    if preface:
        sections.extend(fixed_word_chunks(preface, max_words, overlap))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(0).strip()
        body = text[match.end() : end].strip()
        body_chunks = fixed_word_chunks(body, max_words, overlap) or [""]
        sections.extend(f"{heading}\n{part}".strip() for part in body_chunks)
    return sections


def chunk_documents(
    documents: Iterable[Document],
    strategy: str = "sentence",
    **kwargs: int,
) -> list[Chunk]:
    strategies: dict[str, Callable[..., list[str]]] = {
        "fixed": fixed_word_chunks,
        "sentence": sentence_chunks,
        "markdown": markdown_chunks,
    }
    try:
        splitter = strategies[strategy]
    except KeyError as exc:
        raise ValueError(f"bilinmeyen strateji: {strategy}; seçenekler={list(strategies)}") from exc

    output: list[Chunk] = []
    for document in documents:
        for position, text in enumerate(splitter(document.text, **kwargs)):
            metadata = {**document.metadata, "chunk_index": position, "strategy": strategy}
            output.append(
                Chunk(
                    id=f"{document.id}::chunk-{position}",
                    document_id=document.id,
                    text=text,
                    metadata=metadata,
                )
            )
    return output
