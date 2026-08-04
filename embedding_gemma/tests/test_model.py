import unittest

import torch

from config import ModelConfig
from losses import global_orthogonal_regularizer
from model import TinyEmbeddingGemma
from tokenizer import CharTokenizer


class EmbeddingGemmaTests(unittest.TestCase):
    def test_mean_pooling_output_shapes(self):
        tokenizer = CharTokenizer("abcçdefgğhıijklmnoöprsştuüvyz")
        cfg = ModelConfig(tokenizer.vocab_size, tokenizer.pad_id)
        model = TinyEmbeddingGemma(cfg).eval()
        ids, mask = tokenizer.batch_encode(["ali", "ayşe"], "query")
        for dimension in cfg.matryoshka_dims:
            vectors = model(ids, mask, dimension)
            self.assertEqual(tuple(vectors.shape), (2, dimension))
            self.assertTrue(torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_gor_prefers_spread_vectors(self):
        basis = torch.cat((torch.eye(4), -torch.eye(4)))
        collapsed = torch.ones(8, 4)
        self.assertLess(global_orthogonal_regularizer(basis), global_orthogonal_regularizer(collapsed))


if __name__ == "__main__":
    unittest.main()
