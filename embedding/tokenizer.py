"""Character tokenizer with explicit PAD and UNK tokens for name embeddings."""

from pathlib import Path
import unicodedata

import torch


PAD = "<pad>"
UNK = "<unk>"
TURKISH_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def normalize_text(text: str) -> str:
    """Use canonical Unicode and Turkish-aware lowercase before tokenization."""
    return unicodedata.normalize("NFC", text).translate(TURKISH_LOWER).lower()


class CharTokenizer:
    def __init__(self, chars: list[str]):
        chars = [char for char in chars if char not in {PAD, UNK, "\n"}]
        self.tokens = [PAD, UNK, *sorted(set(chars))]
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = dict(enumerate(self.tokens))
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        text = Path(path).read_text(encoding="utf-8")
        return cls(list(set(text.replace("\n", ""))))

    def encode(self, text: str, max_length: int = 24) -> list[int]:
        text = normalize_text(text)
        encoded = [self.stoi.get(char, self.unk_id) for char in text[:max_length]]
        return encoded or [self.unk_id]

    def batch_encode(
        self, texts: list[str], max_length: int = 24, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text, max_length) for text in texts]
        width = max(1, max(len(row) for row in encoded))
        ids = torch.full((len(texts), width), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(texts), width), dtype=torch.bool, device=device)
        for row_index, row in enumerate(encoded):
            if row:
                ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                mask[row_index, : len(row)] = True
        return ids, mask
