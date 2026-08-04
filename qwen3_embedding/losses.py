import torch
import torch.nn.functional as F


def matryoshka_loss(left, right, dimensions, temperature):
    labels = torch.arange(left.size(0), device=left.device)
    losses = []
    for dimension in dimensions:
        a, b = F.normalize(left[:, :dimension], dim=-1), F.normalize(right[:, :dimension], dim=-1)
        logits = a @ b.T / temperature
        losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    return torch.stack(losses).mean()
