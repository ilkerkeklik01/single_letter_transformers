"""Retriever ölçümünü generator değerlendirmesinden ayıran mini benchmark."""

from .common import load_documents
from rag_lab import HashEmbedding, InMemoryVectorStore, chunk_documents
from rag_lab.evaluation import mean_reciprocal_rank, recall_at_k


CASES = [
    ("Ürün iadesi kaç gün içinde yapılır?", {"iade"}),
    ("Kargom İstanbul dışına ne zaman gelir?", {"kargo"}),
    ("Şifremi sıfırlayan bağlantı ne kadar geçerli?", {"uyelik"}),
    ("Destek konuşmaları kaç gün saklanıyor?", {"gizlilik"}),
]

chunks = chunk_documents(
    load_documents(), strategy="sentence", max_words=35, overlap_sentences=1
)
embedder = HashEmbedding(256)
store = InMemoryVectorStore()
store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))

rankings = []
for question, _ in CASES:
    results = store.search(embedder.embed([question])[0], k=4)
    ranking = [result.chunk.document_id for result in results]
    rankings.append(ranking)
    print(f"{question}\n  -> {ranking}")

relevant = [expected for _, expected in CASES]
print(f"\nRecall@1 = {recall_at_k(rankings, relevant, k=1):.2f}")
print(f"Recall@3 = {recall_at_k(rankings, relevant, k=3):.2f}")
print(f"MRR      = {mean_reciprocal_rank(rankings, relevant):.2f}")
