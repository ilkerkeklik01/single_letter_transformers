"""Shape, normalization, loss and retrieval invariants."""

import unittest

import torch

from config import EmbeddingConfig
from late_interaction import LateInteractionEncoder
from losses import matryoshka_info_nce
from mean_encoder import MeanEncoder
from tokenizer import CharTokenizer
from transformer_encoder import TransformerEncoder
from vector_store import VectorIndex


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tokenizer = CharTokenizer(list("abcçdefgğhıijklmnoöprsştuüvyz"))
        self.cfg = EmbeddingConfig(vocab_size=self.tokenizer.vocab_size)
        self.ids, self.mask = self.tokenizer.batch_encode(["ali", "ayşe", "o"])

    def test_dense_encoders_return_unit_vectors(self):
        for encoder_type in (MeanEncoder, TransformerEncoder):
            with self.subTest(encoder=encoder_type.__name__):
                output = encoder_type(self.cfg)(self.ids, self.mask)
                self.assertEqual(tuple(output.shape), (3, self.cfg.embedding_dim))
                self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(3), atol=1e-5))

    def test_padding_does_not_change_mean_embedding(self):
        model = MeanEncoder(self.cfg).eval()
        ids, mask = self.tokenizer.batch_encode(["ali"])
        padded_ids = torch.cat((ids, torch.zeros(1, 3, dtype=torch.long)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(1, 3, dtype=torch.bool)), dim=1)
        self.assertTrue(torch.allclose(model(ids, mask), model(padded_ids, padded_mask)))

    def test_unicode_and_turkish_lowercase_are_normalized(self):
        self.assertEqual(self.tokenizer.encode("ŞİI"), self.tokenizer.encode("s\u0327iı"))

    def test_late_interaction_shapes_and_identity(self):
        model = LateInteractionEncoder(self.cfg).eval()
        tokens = model(self.ids, self.mask)
        scores = model.pairwise_scores(tokens, tokens, self.mask, self.mask)
        self.assertEqual(tuple(scores.shape), (3, 3))
        self.assertTrue(torch.equal(scores.argmax(dim=1), torch.arange(3)))

    def test_matryoshka_loss_backpropagates(self):
        model = MeanEncoder(self.cfg)
        left = model(self.ids, self.mask)
        right = model(self.ids, self.mask)
        loss = matryoshka_info_nce(left, right, self.cfg.matryoshka_dims, self.cfg.temperature)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.projection[-1].weight.grad)

    def test_vector_index_returns_exact_item_first(self):
        vectors = torch.eye(3)
        index = VectorIndex(vectors, ["a", "b", "c"])
        self.assertEqual(index.search(vectors[1:2], k=2)[0][0][0], "b")


if __name__ == "__main__":
    unittest.main()
