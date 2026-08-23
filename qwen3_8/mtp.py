"""Multi-Token Prediction - the part of Qwen3.8 that `transformers` throws away.

Both released checkpoints carry an `mtp.*` block: a small extra stack that
predicts the token *after* the one the main model is predicting. During training
that second target is an extra supervision signal; at inference it is the draft
model for self-speculative decoding, which is where the speed-up comes from - the
main model can verify several drafted tokens in one pass.

The wiring is a fusion of two views of position `i`:

    what the main model computed at i    ->  pre_fc_norm_hidden
    the embedding of the token AT i+1    ->  pre_fc_norm_embedding
    concatenate, project back down       ->  fc  (2*hidden -> hidden)
    run a normal decoder layer, norm     ->  reuse the main lm_head

so it costs one extra layer, not another model.

**A caveat this folder will not paper over.** `transformers` never loads these
weights - `Qwen3_5PreTrainedModel` lists
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` - and no released model in the
library implements this module, so unlike everything else here it CANNOT be
checked against a reference. The shapes and names are taken from the real
checkpoint and do match it exactly; what is unverified is the **order of the
concatenation** (embedding-first, as `concat_embedding_first` defaults to, versus
hidden-first). Both orders fit `fc`'s `(hidden, 2*hidden)` weight, and only one is
right. Flip the flag if you have a reference decode to check against.
"""

import dataclasses

import torch
from torch import nn

from config import TextConfig
from rms_norm import RMSNorm
from block import DecoderLayer


class MTPHead(nn.Module):
    """One extra decoder stack that predicts the next-next token."""

    def __init__(self, cfg: TextConfig, concat_embedding_first: bool = True):
        super().__init__()
        self.concat_embedding_first = concat_embedding_first
        # MTP layers are always FULL attention - the checkpoint has
        # `mtp.layers.N.self_attn.*` and no `linear_attn`.
        mtp_cfg = dataclasses.replace(
            cfg, num_layers=cfg.mtp_num_hidden_layers,
            layer_types=["full_attention"] * cfg.mtp_num_hidden_layers)

        self.pre_fc_norm_embedding = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [DecoderLayer(mtp_cfg, i) for i in range(cfg.mtp_num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, next_embeds: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden_states, next_embeds: [B, T, hidden] -> [B, T, hidden].

        `next_embeds` should be the embeddings shifted one position left, i.e.
        the token that follows each position.
        """
        embed = self.pre_fc_norm_embedding(next_embeds)
        hidden = self.pre_fc_norm_hidden(hidden_states)
        pair = (embed, hidden) if self.concat_embedding_first else (hidden, embed)

        x = self.fc(torch.cat(pair, dim=-1))
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)
