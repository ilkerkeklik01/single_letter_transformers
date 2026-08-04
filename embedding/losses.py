"""In-batch contrastive and Matryoshka losses."""

import torch
import torch.nn.functional as F


def symmetric_info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def matryoshka_info_nce(
    left: torch.Tensor, right: torch.Tensor, dimensions: tuple[int, ...], temperature: float
) -> torch.Tensor:
    losses = [symmetric_info_nce(left[:, :dim], right[:, :dim], temperature) for dim in dimensions]
    return torch.stack(losses).mean()


def late_interaction_info_nce(scores_ab: torch.Tensor, scores_ba: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(scores_ab.size(0), device=scores_ab.device)
    return 0.5 * (
        F.cross_entropy(scores_ab / temperature, labels)
        + F.cross_entropy(scores_ba / temperature, labels)
    )
