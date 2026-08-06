"""In-memory ve SQLite depolarında aynı exact-cosine sorgusu."""

from tempfile import TemporaryDirectory

from .common import load_documents
from rag_lab import HashEmbedding, InMemoryVectorStore, SQLiteVectorStore, chunk_documents


documents = load_documents()
chunks = chunk_documents(documents, strategy="sentence", max_words=35, overlap_sentences=0)
embedder = HashEmbedding(dimension=256)
vectors = embedder.embed([chunk.text for chunk in chunks])
query = embedder.embed(["İade için kaç günüm var?"])[0]

memory = InMemoryVectorStore()
memory.add(chunks, vectors)
print("InMemory sonuçları:")
for result in memory.search(query, k=3):
    print(f"{result.score:+.3f}  {result.chunk.id}")

print("\nMetadata filtresi (category=hesap):")
for result in memory.search(query, k=3, filters={"category": "hesap"}):
    print(f"{result.score:+.3f}  {result.chunk.id}")

with TemporaryDirectory() as directory:
    sqlite = SQLiteVectorStore(f"{directory}/vectors.sqlite3")
    sqlite.add(chunks, vectors)
    print("\nSQLite sonuçları:")
    for result in sqlite.search(query, k=3):
        print(f"{result.score:+.3f}  {result.chunk.id}")
    sqlite.close()
