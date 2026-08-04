"""NCE + a small second-moment Global Orthogonal Regularizer."""

import torch
import torch.nn.functional as F


def global_orthogonal_regularizer(vectors):
    vectors = F.normalize(vectors, dim=-1)
    dimension = vectors.size(-1)
    second_moment = vectors.T @ vectors / vectors.size(0)
    target = torch.eye(dimension, device=vectors.device) / dimension
    mean_penalty = vectors.mean(0).pow(2).mean()
    return (second_moment - target).pow(2).mean() + mean_penalty


def embedding_gemma_loss(left, right, cfg):
    labels = torch.arange(left.size(0), device=left.device)
    nce_losses, gor_losses = [], []
    for dimension in cfg.matryoshka_dims:
        a, b = F.normalize(left[:, :dimension], dim=-1), F.normalize(right[:, :dimension], dim=-1)
        logits = a @ b.T / cfg.temperature
        nce_losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
        gor_losses.extend((global_orthogonal_regularizer(a), global_orthogonal_regularizer(b)))
    return torch.stack(nce_losses).mean() + cfg.gor_weight * torch.stack(gor_losses).mean()
