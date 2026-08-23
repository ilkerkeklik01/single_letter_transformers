"""Load HuggingFace Muse Glimmer checkpoints into our from-scratch models.

The released checkpoint keeps everything under a "model." prefix except the
head:

    lm_head.weight
    model.language_model.*      (the text stack)
    model.vision_tower.*        (the ViT)
    model.vision_adapter.*      (fc1/fc2 bridge)
    model.vision_projection.*   (into text hidden size)

Because our submodules carry the same names, the mapping is just prefix
surgery, and it works for both classes in this folder:

  * MuseGlimmerVL (the complete model): strip "model.", and tuck the
    top-level lm_head into the language model (which owns its own head here).
  * MuseGlimmer (text-only): additionally strip "language_model." and skip
    every vision tensor.

Works for any Muse Glimmer checkpoint - the tiny HF one the verification
script creates, or the real `unsloth/Muse-Glimmer-30B` (build with
`VLConfig.muse_glimmer_30b()` / `ModelConfig.muse_glimmer_30b()` first;
you'll want ~60 GB of RAM for the real one).

Usage:
    from config import VLConfig
    from vl_model import MuseGlimmerVL
    from hf_weights import load_hf_checkpoint

    model = MuseGlimmerVL(VLConfig.muse_glimmer_30b())
    load_hf_checkpoint(model, "unsloth/Muse-Glimmer-30B")   # hub repo id...
    load_hf_checkpoint(model, "path/to/local/checkpoint")   # ...or local dir
"""

import glob
import json
import os

TEXT_PREFIX = "model.language_model."


def map_hf_key(hf_key: str, multimodal: bool = False):
    """HF checkpoint key -> our state_dict key, or None if not part of the target."""
    if multimodal:
        if hf_key == "lm_head.weight":
            return "language_model.lm_head.weight"
        if hf_key.startswith("model."):
            return hf_key[len("model."):]
        return None
    if hf_key == "lm_head.weight":
        return hf_key
    if hf_key.startswith(TEXT_PREFIX):
        return hf_key[len(TEXT_PREFIX):]
    return None   # vision tensors don't belong in the text-only model


def load_hf_state_dict(model, hf_state_dict: dict) -> None:
    """Load an in-memory HF state dict (e.g. `hf_model.state_dict()`)."""
    multimodal = hasattr(model, "vision_tower")
    mapped = {}
    for hf_key, tensor in hf_state_dict.items():
        our_key = map_hf_key(hf_key, multimodal)
        if our_key is not None:
            mapped[our_key] = tensor
    # strict=True: every one of our parameters must be found, nothing extra.
    model.load_state_dict(mapped, strict=True)


def load_hf_checkpoint(model, source: str) -> None:
    """Load from a local checkpoint directory or a HuggingFace Hub repo id.

    Handles both single-file (model.safetensors) and sharded
    (model.safetensors.index.json + shards) checkpoints, and only pulls the
    shards that contain tensors the target model actually wants.
    """
    from safetensors.torch import load_file

    multimodal = hasattr(model, "vision_tower")

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

    mapped = {}
    index_path = fetch("model.safetensors.index.json")
    if index_path is not None:
        # Sharded: group the wanted keys by shard, load each shard once.
        weight_map = json.load(open(index_path))["weight_map"]
        shards = {}
        for hf_key, shard in weight_map.items():
            if map_hf_key(hf_key, multimodal) is not None:
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
        for hf_key, tensor in load_file(single_path).items():
            our_key = map_hf_key(hf_key, multimodal)
            if our_key is not None:
                mapped[our_key] = tensor

    model.load_state_dict(mapped, strict=True)
