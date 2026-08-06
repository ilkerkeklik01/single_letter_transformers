"""İnternet, model indirme ve API olmadan uçtan uca RAG iskeleti."""

import sys

from .common import load_documents, print_answer
from rag_lab import ExtractiveGenerator, HashEmbedding, InMemoryVectorStore, RAGPipeline


question = " ".join(sys.argv[1:]) or "İade için kaç günüm var?"
pipeline = RAGPipeline(
    embedder=HashEmbedding(dimension=256),
    store=InMemoryVectorStore(),
    generator=ExtractiveGenerator(),
)
count = pipeline.index(
    load_documents(), chunk_strategy="sentence", max_words=35, overlap_sentences=1
)
print(f"{count} parça indekslendi. HashEmbedding yalnızca eğitim demosudur.")
print_answer(pipeline.ask(question, k=3))
