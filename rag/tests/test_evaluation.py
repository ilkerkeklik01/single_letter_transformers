import unittest

from rag_lab.evaluation import mean_reciprocal_rank, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        rankings = [["a", "b"], ["x", "c"]]
        relevant = [{"a"}, {"c"}]
        self.assertEqual(recall_at_k(rankings, relevant, 1), 0.5)
        self.assertEqual(recall_at_k(rankings, relevant, 2), 1.0)
        self.assertEqual(mean_reciprocal_rank(rankings, relevant), 0.75)


if __name__ == "__main__":
    unittest.main()
