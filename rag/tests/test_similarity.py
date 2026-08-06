import math
import unittest

from rag_lab.similarity import cosine_similarity, normalize


class SimilarityTests(unittest.TestCase):
    def test_same_orthogonal_and_opposite_directions(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [10, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 2]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 1], [-1, -1]), -1.0)

    def test_normalization(self):
        result = normalize([3, 4])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in result)), 1.0)

    def test_zero_and_dimension_errors(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
