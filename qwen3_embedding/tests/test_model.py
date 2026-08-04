import unittest

import torch

from config import ModelConfig
from model import TinyQwen3Embedding
from tokenizer import CharTokenizer


class QwenEmbeddingTests(unittest.TestCase):
    def test_eos_pooling_and_matryoshka_shapes(self):
        tokenizer = CharTokenizer("abcçdefgğhıijklmnoöprsştuüvyz")
        cfg = ModelConfig(tokenizer.vocab_size, tokenizer.pad_id, tokenizer.eos_id)
        model = TinyQwen3Embedding(cfg).eval()
        ids, mask = tokenizer.batch_encode(["ali", "ayşe"], "query")
        self.assertTrue(torch.equal(ids[torch.arange(2), mask.sum(1) - 1], torch.full((2,), tokenizer.eos_id)))
        for dimension in cfg.matryoshka_dims:
            vectors = model(ids, mask, dimension)
            self.assertEqual(tuple(vectors.shape), (2, dimension))
            self.assertTrue(torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
