import unittest

from rag_lab.chunking import fixed_word_chunks, markdown_chunks, sentence_chunks


class ChunkingTests(unittest.TestCase):
    def test_fixed_overlap(self):
        chunks = fixed_word_chunks("bir iki üç dört beş altı", size=4, overlap=2)
        self.assertEqual(chunks[0], "bir iki üç dört")
        self.assertEqual(chunks[1], "üç dört beş altı")

    def test_sentence_boundaries_survive(self):
        chunks = sentence_chunks("Bir iki. Üç dört. Beş altı.", max_words=4, overlap_sentences=1)
        self.assertEqual(chunks, ["Bir iki. Üç dört.", "Üç dört. Beş altı."])

    def test_markdown_headings_survive(self):
        chunks = markdown_chunks("# A\nBir iki.\n# B\nÜç dört.", max_words=10)
        self.assertEqual(chunks, ["# A\nBir iki.", "# B\nÜç dört."])


if __name__ == "__main__":
    unittest.main()
