"""Pencil-friendly checks for cosine, InfoNCE, MaxSim and Matryoshka slicing."""

import math

import torch
import torch.nn.functional as F


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


section("1. Cosine similarity")
a = torch.tensor([1.0, 1.0])
b = torch.tensor([2.0, 0.0])
cosine = (a @ b) / (a.norm() * b.norm())
expected = 1.0 / math.sqrt(2.0)
print(f"cos([1,1], [2,0]) = 2 / (sqrt(2)*2) = {cosine:.4f}")
assert torch.allclose(cosine, torch.tensor(expected))


section("2. One-way InfoNCE")
queries = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
documents = F.normalize(torch.tensor([[0.9, 0.1], [0.1, 0.9]]), dim=-1)
temperature = 0.5
logits = queries @ documents.T / temperature
loss = F.cross_entropy(logits, torch.arange(2))
print("rows=query, columns=candidate; diagonal cells are positives")
print(logits)
print(f"cross_entropy = {loss:.4f}")
assert logits.diag().min() > logits[0, 1]


section("3. Late interaction / MaxSim")
query_tokens = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
document_tokens = F.normalize(torch.tensor([[0.8, 0.2], [0.2, 0.8], [-1.0, 0.0]]), dim=-1)
token_scores = query_tokens @ document_tokens.T
maxsim = token_scores.max(dim=1).values.mean()
print("token-to-token cosine matrix")
print(token_scores)
print(f"mean(max per query token) = {maxsim:.4f}")
assert maxsim > 0.9


section("4. Matryoshka prefixes")
embedding = F.normalize(torch.tensor([3.0, 4.0, 1.0, 2.0]), dim=0)
for dimension in (2, 4):
    prefix = F.normalize(embedding[:dimension], dim=0)
    print(f"first {dimension} dims -> {prefix.tolist()}  norm={prefix.norm():.1f}")
    assert torch.allclose(prefix.norm(), torch.tensor(1.0))

print("\nall hand checks passed")
