"""Load HuggingFace Qwen3.8 checkpoints into our from-scratch models.

Qwen3.8's `config.json` declares `model_type = "qwen3_5"`, so the released repos
lay their tensors out exactly the way `transformers` expects for Qwen3.5. Two
layouts exist:

  * text-only (`Qwen3_5MoeForCausalLM`, the 2.4T release):
    `model.*` + `lm_head.weight` + `mtp.*`
  * multimodal (`Qwen3_5ForConditionalGeneration`, the 27B release):
    `model.language_model.*`, `model.visual.*`, `lm_head.weight`, `mtp.*`

The `mtp.*` block is Qwen3.8's multi-token-prediction head. `transformers`
**drops it** - both `Qwen3_5PreTrainedModel` and its MoE sibling list
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` - so if you load one of these
checkpoints with the library you silently get a model without it. This folder
builds it, which is the one place where we keep more of the checkpoint than the
reference does.

Usage:
    from config import TextConfig
    from model import Qwen38Text
    from hf_weights import load_hf_checkpoint

    model = Qwen38Text(TextConfig.qwen3_8_27b())
    load_hf_checkpoint(model, "Qwen/Qwen3.8-27B")     # hub repo id...
    load_hf_checkpoint(model, "path/to/checkpoint")   # ...or local dir
"""

import glob
import json
import os

TEXT_PREFIXES = ("model.language_model.", "model.")


def map_hf_key(hf_key: str, multimodal: bool = False):
    """HF checkpoint key -> our state_dict key, or None if it is not ours."""
    if multimodal:
        # The complete model keeps every part, under its own submodule names.
        if hf_key.startswith("model.language_model."):
            return "language_model." + hf_key[len("model.language_model."):]
        if hf_key.startswith("model.visual."):
            return "visual." + hf_key[len("model.visual."):]
        if hf_key == "lm_head.weight":
            return "language_model.lm_head.weight"
        if hf_key.startswith("mtp."):
            # The MTP head lives inside the text model here, since it predicts
            # text tokens and reuses the text lm_head.
            return "language_model." + hf_key
        return None

    # Text-only: keep the language model and its head, drop the vision tower.
    if hf_key == "lm_head.weight":
        return "lm_head.weight"
    if hf_key.startswith("mtp."):
        return hf_key
    for prefix in TEXT_PREFIXES:
        if hf_key.startswith(prefix):
            rest = hf_key[len(prefix):]
            if prefix == "model." and rest.split(".")[0] in ("visual", "language_model"):
                return None
            return rest
    return None


def _filter(model, items):
    """Map checkpoint keys to ours, keeping only tensors this model builds."""
    multimodal = hasattr(model, "visual")
    wanted = set(model.state_dict())
    mapped = {}
    for hf_key, tensor in items:
        our_key = map_hf_key(hf_key, multimodal)
        if our_key is not None and our_key in wanted:
            mapped[our_key] = tensor
    return mapped


def _load(model, mapped: dict, warn_missing_mtp: bool = True) -> None:
    """Load mapped tensors, tolerating a checkpoint that carries no MTP weights.

    Anything saved by `transformers` has the MTP head stripped, so those keys can
    legitimately be absent - the head then keeps its initialization. Any OTHER
    missing key is a real error.
    """
    missing, _ = model.load_state_dict(mapped, strict=False)
    not_mtp = [k for k in missing if not (k.startswith("mtp.") or ".mtp." in k)]
    if not_mtp:
        raise RuntimeError(f"missing keys in checkpoint: {not_mtp[:10]}")
    if missing and warn_missing_mtp:
        print(f"note: checkpoint has no MTP weights ({len(missing)} tensors); "
              "the head keeps its initialization")


def load_hf_state_dict(model, hf_state_dict: dict, warn_missing_mtp: bool = True) -> None:
    """Load an in-memory HF state dict (e.g. `hf_model.state_dict()`)."""
    _load(model, _filter(model, hf_state_dict.items()), warn_missing_mtp)


def load_hf_checkpoint(model, source: str) -> None:
    """Load from a local checkpoint directory or a HuggingFace Hub repo id.

    Handles single-file and sharded checkpoints, and pulls only the shards that
    hold tensors this model actually wants.
    """
    from safetensors.torch import load_file

    multimodal = hasattr(model, "visual")

    def fetch(filename: str) -> str:
        """Return a local path for `filename`, downloading from the Hub if needed."""
        if os.path.isdir(source):
            path = os.path.join(source, filename)
            return path if os.path.exists(path) else None
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(source, filename)
        except Exception:
            return None

    wanted = set(model.state_dict())
    mapped = {}
    index_path = fetch("model.safetensors.index.json")
    if index_path is not None:
        weight_map = json.load(open(index_path))["weight_map"]
        shards = {}
        for hf_key, shard in weight_map.items():
            our_key = map_hf_key(hf_key, multimodal)
            if our_key is not None and our_key in wanted:
                shards.setdefault(shard, []).append(hf_key)
        for shard, keys in sorted(shards.items()):
            tensors = load_file(fetch(shard))
            for hf_key in keys:
                mapped[map_hf_key(hf_key, multimodal)] = tensors[hf_key]
    else:
        single_path = fetch("model.safetensors")
        if single_path is None:
            files = glob.glob(os.path.join(source, "*.safetensors")) if os.path.isdir(source) else []
            if not files:
                raise FileNotFoundError(f"No safetensors checkpoint found at {source!r}")
            single_path = files[0]
        mapped = _filter(model, load_file(single_path).items())

    _load(model, mapped)
