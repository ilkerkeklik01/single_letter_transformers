"""Load HuggingFace Gemma 4 checkpoints into our from-scratch text model.

Two checkpoint layouts exist, and the mapping handles both:

  * text-only (`Gemma4ForCausalLM`):        `model.*`  + `lm_head.weight`
  * multimodal (`Gemma4ForConditionalGeneration`, which is what the released
    e2b / 26B repos are): `model.language_model.*`, plus `model.vision_tower.*`,
    `model.audio_tower.*`, `model.embed_vision.*`, `model.embed_audio.*` -
    all of which this loader skips, since this folder implements the text stack.

Two Gemma-4-specific wrinkles:

  * `lm_head.weight` is TIED to `embed_tokens.weight`, so the released
    checkpoints do not store it separately. We drop it if present and let the
    tie recreate it.
  * KV-shared layers **do carry k_proj / v_proj / k_norm / v_norm in the
    checkpoint**, even though the model never uses them - the reference lists
    them in `_keys_to_ignore_on_load_unexpected`. We skip them the same way.
    `unexpected_kv_keys()` reports exactly which ones, so the verification
    script can prove they are ignored on purpose rather than lost by accident.

Usage:
    from config import ModelConfig
    from model import TinyGemma
    from hf_weights import load_hf_checkpoint

    model = TinyGemma(ModelConfig.gemma4_e2b())
    load_hf_checkpoint(model, "google/gemma-4-e2b-it")   # hub repo id...
    load_hf_checkpoint(model, "path/to/local/checkpoint")   # ...or local dir
"""

import glob
import json
import os

TEXT_PREFIXES = ("model.language_model.", "model.")
OTHER_TOWERS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")


def map_hf_key(hf_key: str, multimodal: bool = False):
    """HF checkpoint key -> our state_dict key, or None if it is not ours."""
    if hf_key == "lm_head.weight":
        return None            # tied to embed_tokens; recreated by the tie
    if multimodal:
        # The complete model keeps every tower, so "model." is the only prefix.
        return hf_key[len("model."):] if hf_key.startswith("model.") else None
    for prefix in TEXT_PREFIXES:
        if hf_key.startswith(prefix):
            rest = hf_key[len(prefix):]
            # Everything else under "model." is a different tower.
            if prefix == "model." and rest.split(".")[0] in OTHER_TOWERS + ("language_model",):
                return None
            return rest
    return None


def _text_config(model):
    """The text ModelConfig, whether `model` is TinyGemma or the full Gemma4."""
    return getattr(model.cfg, "text", model.cfg)


def unexpected_kv_keys(model, hf_keys) -> list:
    """The k/v tensors a checkpoint stores for KV-shared layers, which the
    architecture deliberately ignores (the reference does the same)."""
    cfg = _text_config(model)
    multimodal = hasattr(model, "vision_tower")
    prefix = "language_model.layers." if multimodal else "layers."
    shared = {i for i in range(cfg.num_layers) if cfg.is_kv_shared(i)}
    dropped = []
    for hf_key in hf_keys:
        our_key = map_hf_key(hf_key, multimodal)
        if our_key is None or not our_key.startswith(prefix):
            continue
        parts = our_key[len(prefix):].split(".")
        if (int(parts[0]) in shared and parts[1] == "self_attn"
                and parts[2] in ("k_proj", "v_proj", "k_norm", "v_norm")):
            dropped.append(hf_key)
    return dropped


def _filter(model, items):
    """Map checkpoint keys to ours, dropping other towers and ignored k/v."""
    multimodal = hasattr(model, "vision_tower")
    wanted = set(model.state_dict())
    mapped = {}
    for hf_key, tensor in items:
        our_key = map_hf_key(hf_key, multimodal)
        if our_key is not None and our_key in wanted:
            mapped[our_key] = tensor
    return mapped


def _tied_head_key(model) -> str:
    return ("language_model.lm_head.weight" if hasattr(model, "vision_tower")
            else "lm_head.weight")


def load_hf_state_dict(model, hf_state_dict: dict) -> None:
    """Load an in-memory HF state dict (e.g. `hf_model.state_dict()`)."""
    mapped = _filter(model, hf_state_dict.items())
    # strict=False only because lm_head.weight is tied, so it is never in `mapped`.
    missing, _ = model.load_state_dict(mapped, strict=False)
    assert missing == [_tied_head_key(model)], f"unexpected missing keys: {missing}"


def load_hf_checkpoint(model, source: str) -> None:
    """Load from a local checkpoint directory or a HuggingFace Hub repo id.

    Handles both single-file (model.safetensors) and sharded
    (model.safetensors.index.json + shards) checkpoints, and only pulls the
    shards that actually contain text-stack tensors - so loading the text model
    from a multimodal repo never downloads the vision or audio shards.
    """
    from safetensors.torch import load_file

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

    multimodal = hasattr(model, "vision_tower")
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

    missing, _ = model.load_state_dict(mapped, strict=False)
    assert missing == [_tied_head_key(model)], f"unexpected missing keys: {missing}"
