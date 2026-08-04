"""Character tokenizer with short query/document prompts."""

from pathlib import Path
import unicodedata

import torch


SPECIALS = ("<pad>", "<unk>", "<query>", "<document>")
LOWER = str.maketrans({"I": "ı", "İ": "i"})


class CharTokenizer:
    def __init__(self, chars):
        chars = sorted(set(chars) - set("\n"))
        self.tokens = [*SPECIALS, *[char for char in chars if char not in SPECIALS]]
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id, self.unk_id, self.query_id, self.document_id = range(4)

    @property
    def vocab_size(self):
        return len(self.tokens)

    @classmethod
    def from_file(cls, path):
        return cls(Path(path).read_text(encoding="utf-8"))

    def batch_encode(self, texts, role="document", max_length=32, device="cpu"):
        role_id = self.query_id if role == "query" else self.document_id
        rows = []
        for text in texts:
            text = unicodedata.normalize("NFC", text).translate(LOWER).lower()
            rows.append([role_id, *[self.stoi.get(c, self.unk_id) for c in text[: max_length - 1]]])
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row, device=device)
            mask[index, : len(row)] = True
        return ids, mask
