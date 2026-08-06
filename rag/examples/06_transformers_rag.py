"""Embedding ve üretimi doğrudan Hugging Face Transformers ile belleğe yükler."""

import argparse

from .common import load_documents, print_answer
from rag_lab import InMemoryVectorStore, RAGPipeline, TransformersEmbedding, TransformersGenerator


parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="*", default=["İade için kaç günüm var?"])
parser.add_argument(
    "--embedding-model",
    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
parser.add_argument("--generation-model", required=True)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

pipeline = RAGPipeline(
    embedder=TransformersEmbedding(args.embedding_model, device=args.device),
    store=InMemoryVectorStore(),
    generator=TransformersGenerator(args.generation_model, device=args.device),
)
pipeline.index(load_documents(), chunk_strategy="sentence", max_words=55, overlap_sentences=1)
print_answer(pipeline.ask(" ".join(args.question), k=3))
