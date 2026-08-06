import unittest
from unittest.mock import patch

from rag_lab import OllamaEmbedding, OllamaGenerator


class OllamaBackendTests(unittest.TestCase):
    @patch("rag_lab.embeddings._post_json")
    def test_embedding_uses_batch_embed_endpoint(self, post_json):
        post_json.return_value = {"embeddings": [[1, 0], [0, 1]]}
        backend = OllamaEmbedding("embed-model", "http://ollama.test")
        result = backend.embed(["bir", "iki"])

        self.assertEqual(result, [[1.0, 0.0], [0.0, 1.0]])
        url, payload, _ = post_json.call_args.args
        self.assertEqual(url, "http://ollama.test/api/embed")
        self.assertEqual(payload["input"], ["bir", "iki"])
        self.assertFalse(payload["truncate"])

    @patch("rag_lab.generators._post_json")
    def test_generator_disables_streaming(self, post_json):
        post_json.return_value = {"response": "  cevap  "}
        backend = OllamaGenerator("generate-model", "http://ollama.test")
        self.assertEqual(backend.generate("prompt"), "cevap")

        url, payload, _ = post_json.call_args.args
        self.assertEqual(url, "http://ollama.test/api/generate")
        self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
