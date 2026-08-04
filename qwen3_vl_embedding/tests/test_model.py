import unittest

import torch

from config import ModelConfig
from model import TinyQwen3VLEmbedding
from render import render_batch
from tokenizer import CharTokenizer


class QwenVLEmbeddingTests(unittest.TestCase):
    def test_text_image_and_fused_outputs(self):
        tokenizer = CharTokenizer("abcçdefgğhıijklmnoöprsştuüvyz")
        cfg = ModelConfig(tokenizer.vocab_size, tokenizer.pad_id)
        model = TinyQwen3VLEmbedding(cfg).eval()
        ids, mask = tokenizer.batch_encode(["ali", "ayşe"], "query")
        images = render_batch(["ali", "ayşe"], cfg.image_height, cfg.image_width)
        self.assertEqual(tuple(images.shape), (2, 1, cfg.image_height, cfg.image_width))
        for vectors in (
            model.encode_text(ids, mask),
            model.encode_image(images),
            model.encode_fused(ids, mask, images),
        ):
            self.assertEqual(tuple(vectors.shape), (2, cfg.embedding_dim))
            self.assertTrue(torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
