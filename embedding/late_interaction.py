"""ColBERT-style multi-vector encoder and MaxSim late-interaction score."""

import torch
from torch import nn
import torch.nn.functional as F

from config import EmbeddingConfig
from transformer_encoder import TransformerBackbone


class LateInteractionEncoder(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerBackbone(cfg)
        self.token_projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(ids, mask)
        return F.normalize(self.token_projection(tokens), dim=-1) * mask.unsqueeze(-1)

    @staticmethod
    def pairwise_scores(
        queries: torch.Tensor,
        documents: torch.Tensor,
        query_mask: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score every query against every document with sum-of-MaxSim."""
        similarities = torch.einsum("bqd,ckd->bcqk", queries, documents)
        similarities = similarities.masked_fill(~document_mask[None, :, None, :], -torch.inf)
        best_document_token = similarities.max(dim=-1).values
        best_document_token = best_document_token.masked_fill(~query_mask[:, None, :], 0.0)
        lengths = query_mask.sum(dim=-1).clamp_min(1)[:, None]
        return best_document_token.sum(dim=-1) / lengths
