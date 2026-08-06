"""Embedding ve üretimin ikisini de yerel Ollama API'siyle yapar."""

import argparse

from .common import load_documents, print_answer
from rag_lab import InMemoryVectorStore, OllamaEmbedding, OllamaGenerator, RAGPipeline


parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="*", default=["İade için kaç günüm var?"])
parser.add_argument("--embedding-model", default="embeddinggemma")
parser.add_argument("--generation-model", default="gemma3")
parser.add_argument("--base-url", default="http://localhost:11434")
args = parser.parse_args()

pipeline = RAGPipeline(
    embedder=OllamaEmbedding(args.embedding_model, args.base_url),
    store=InMemoryVectorStore(),
    generator=OllamaGenerator(args.generation_model, args.base_url),
)
pipeline.index(load_documents(), chunk_strategy="sentence", max_words=55, overlap_sentences=1)
print_answer(pipeline.ask(" ".join(args.question), k=3))
