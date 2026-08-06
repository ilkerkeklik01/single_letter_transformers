import tempfile
import unittest
from pathlib import Path

from rag_lab import Chunk, InMemoryVectorStore, SQLiteVectorStore


CHUNKS = [
    Chunk("a", "doc-a", "iade", {"category": "order"}),
    Chunk("b", "doc-b", "parola", {"category": "account"}),
]
VECTORS = [[1.0, 0.0], [0.0, 1.0]]


class StoreTests(unittest.TestCase):
    def test_memory_search_filter_and_roundtrip(self):
        store = InMemoryVectorStore()
        store.add(CHUNKS, VECTORS)
        self.assertEqual(store.search([0.9, 0.1], k=1)[0].chunk.id, "a")
        filtered = store.search([1, 0], filters={"category": "account"})
        self.assertEqual([item.chunk.id for item in filtered], ["b"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            store.save(path, model_name="demo-v1")
            loaded = InMemoryVectorStore.load(path, expected_model_name="demo-v1")
            self.assertEqual(loaded.search([0, 1], k=1)[0].chunk.id, "b")
            with self.assertRaises(ValueError):
                InMemoryVectorStore.load(path, expected_model_name="demo-v2")

    def test_sqlite_store(self):
        store = SQLiteVectorStore()
        store.add(CHUNKS, VECTORS)
        self.assertEqual(store.search([0, 1], k=1)[0].chunk.id, "b")
        store.close()


if __name__ == "__main__":
    unittest.main()
