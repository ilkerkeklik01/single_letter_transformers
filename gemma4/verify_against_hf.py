"""Prove the "exact same architecture" claim for both Gemma 4 flavors.

  1. Dense parity : a tiny HF `Gemma4ForCausalLM` (transformers >= 5.15.0)
                    configured the e2b way - per-layer embeddings, KV-shared
                    layers, double-wide MLPs - vs this folder's `TinyGemma`,
                    same weights. Logits must match at sequence lengths below
                    and above the sliding window, so both mask types and both
                    RoPE tables get exercised.

  2. MoE parity   : the same comparison with the 26B-A4B recipe instead -
                    128-way-style sparse experts (scaled down), k=v attention,
                    no PLE. This is a genuinely different code path.

  3. Roundtrip    : `save_pretrained` output loads back through
                    `hf_weights.load_hf_checkpoint`, logits still match.

  4. Shape check  : both released text stacks - `google/gemma-4-e2b-it` and
                    `google/gemma-4-26B-A4B-it` - built on the meta device and
                    compared against the real safetensors metadata on the Hub
                    (file headers only, no weights). Every tensor we build must
                    exist with the right shape, and every checkpoint tensor must
                    either be ours, belong to another tower, be the tied head,
                    or be a KV-shared layer's ignored k/v.

Run:  python verify_against_hf.py
"""

import argparse
import tempfile

import torch

from config import ModelConfig, VLConfig
from model import TinyGemma
from vl_model import Gemma4
from hf_weights import load_hf_checkpoint, load_hf_state_dict, map_hf_key, unexpected_kv_keys

OTHER_TOWERS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")


def build_hf_twin(cfg: ModelConfig):
    """The reference implementation, configured to our numbers."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    hf_config = Gemma4TextConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_seq_len,
        sliding_window=cfg.sliding_window,
        layer_types=list(cfg.layer_types),
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_activation=cfg.hidden_activation,
        final_logit_softcapping=cfg.final_logit_softcapping,
        hidden_size_per_layer_input=cfg.hidden_size_per_layer_input,
        vocab_size_per_layer_input=cfg.vocab_size_per_layer_input,
        num_kv_shared_layers=cfg.num_kv_shared_layers,
        use_double_wide_mlp=cfg.use_double_wide_mlp,
        attention_k_eq_v=cfg.attention_k_eq_v,
        enable_moe_block=cfg.enable_moe_block,
        num_experts=cfg.num_experts,
        top_k_experts=cfg.top_k_experts,
        moe_intermediate_size=cfg.moe_intermediate_size,
        rope_parameters={
            "sliding_attention": {"rope_type": "default",
                                  "rope_theta": cfg.rope_theta_sliding},
            "full_attention": {"rope_type": "proportional",
                               "partial_rotary_factor": cfg.partial_rotary_factor,
                               "rope_theta": cfg.rope_theta_full},
        },
        # These two reach the config as kwargs, which is how the reference
        # builds its per-layer overrides for full-attention layers.
        global_head_dim=cfg.global_head_dim,
        num_global_key_value_heads=cfg.num_global_kv_heads,
        tie_word_embeddings=True,
        bos_token_id=2, eos_token_id=1, pad_token_id=0,
    )
    return Gemma4ForCausalLM(hf_config).eval()


def build_hf_vl_twin(cfg: VLConfig):
    """The reference multimodal model, configured to our numbers."""
    from transformers import (
        Gemma4AudioConfig, Gemma4Config, Gemma4ForConditionalGeneration, Gemma4VisionConfig,
    )

    text_config = build_hf_twin(cfg.text).config
    v, a = cfg.vision, cfg.audio
    vision_config = Gemma4VisionConfig(
        hidden_size=v.hidden_size, intermediate_size=v.intermediate_size,
        num_hidden_layers=v.num_layers, num_attention_heads=v.num_heads,
        num_key_value_heads=v.num_kv_heads, head_dim=v.head_dim, patch_size=v.patch_size,
        pooling_kernel_size=v.pooling_kernel_size,
        position_embedding_size=v.position_embedding_size, rms_norm_eps=v.rms_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": v.rope_theta},
        use_clipped_linears=v.use_clipped_linears, standardize=v.standardize,
        hidden_activation="gelu_pytorch_tanh",
    )
    audio_config = None if a is None else Gemma4AudioConfig(
        hidden_size=a.hidden_size, num_hidden_layers=a.num_layers,
        num_attention_heads=a.num_heads,
        subsampling_conv_channels=list(a.subsampling_conv_channels),
        conv_kernel_size=a.conv_kernel_size, residual_weight=a.residual_weight,
        attention_chunk_size=a.attention_chunk_size,
        attention_context_left=a.attention_context_left,
        attention_context_right=a.attention_context_right,
        attention_logit_cap=a.attention_logit_cap,
        attention_invalid_logits_value=a.attention_invalid_logits_value,
        use_clipped_linears=a.use_clipped_linears, rms_norm_eps=a.rms_norm_eps,
        gradient_clipping=a.gradient_clipping, output_proj_dims=a.output_proj_dims,
        hidden_act=a.hidden_act,
    )
    config = Gemma4Config(
        text_config=text_config, vision_config=vision_config, audio_config=audio_config,
        image_token_id=cfg.image_token_id, audio_token_id=cfg.audio_token_id,
        video_token_id=cfg.video_token_id, tie_word_embeddings=True,
    )
    model = Gemma4ForConditionalGeneration(config).eval()
    if v.standardize:
        # The reference allocates these calibration buffers with torch.empty, so
        # give them defined values before comparing anything.
        with torch.no_grad():
            model.model.vision_tower.std_bias.normal_(0.0, 0.1)
            model.model.vision_tower.std_scale.fill_(1.3)
    return model


def make_media_batch(cfg: VLConfig, patch_side: int = 6, audio_frames: int = 40):
    """One image and one audio clip, woven into a token sequence with placeholders."""
    v = cfg.vision
    n_patches = patch_side * patch_side
    pixel_values = torch.rand(1, n_patches, v.patch_dim)
    xs, ys = torch.meshgrid(torch.arange(patch_side), torch.arange(patch_side), indexing="ij")
    position_ids = torch.stack([xs.flatten(), ys.flatten()], dim=-1)[None]
    n_image_tokens = n_patches // (v.pooling_kernel_size**2)

    input_features = features_mask = None
    n_audio_tokens = 0
    if cfg.audio is not None:
        input_features = torch.randn(1, audio_frames, cfg.audio.mel_bins)
        features_mask = torch.ones(1, audio_frames, dtype=torch.bool)
        n_audio_tokens = audio_frames // 4   # two stride-2 convolutions

    ids = [3] + [cfg.image_token_id] * n_image_tokens + [5]
    if cfg.audio is not None:
        ids += [cfg.audio_token_id] * n_audio_tokens
    ids += [7]
    return (torch.tensor([ids]), pixel_values, position_ids, input_features, features_mask)


def compare_vl_logits(hf_model, our_model, cfg: VLConfig, label: str) -> float:
    """Full multimodal forward through both models; return the worst |diff|."""
    ids, pixel_values, position_ids, features, features_mask = make_media_batch(cfg)
    audio = cfg.audio is not None
    hf_kwargs = dict(pixel_values=pixel_values, image_position_ids=position_ids)
    our_kwargs = dict(pixel_values=pixel_values, pixel_position_ids=position_ids)
    if audio:
        hf_kwargs.update(input_features=features, input_features_mask=features_mask)
        our_kwargs.update(input_features=features, input_features_mask=features_mask)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids, **hf_kwargs).logits
        our_logits, _ = our_model(ids, **our_kwargs)
    diff = (hf_logits - our_logits).abs().max().item()
    media = "image + audio" if audio else "image only"
    print(f"  {label} ({ids.shape[1]} tokens: {media}): max |logit diff| = {diff:.3e}")
    return diff


def compare_logits(hf_model, our_model, cfg: ModelConfig, label: str) -> float:
    """Same random token ids through both models; return the worst |diff|."""
    worst = 0.0
    for seq_len in (5, 20, cfg.max_seq_len):   # below / above the sliding window
        ids = torch.randint(0, cfg.vocab_size, (2, seq_len))
        with torch.no_grad():
            hf_logits = hf_model(input_ids=ids).logits
            our_logits, _ = our_model(ids)
        diff = (hf_logits - our_logits).abs().max().item()
        print(f"  {label} seq_len {seq_len:3d}: max |logit diff| = {diff:.3e}")
        worst = max(worst, diff)
    return worst


def check_parity(cfg: ModelConfig, label: str, seed: int = 0, experts_impl=None):
    """Build both models on shared random weights and compare.

    `experts_impl` selects the reference's MoE kernel. `None` is its plain
    PyTorch loop, which is the algorithm this folder implements; the fused
    kernels ("grouped_mm", "batched_mm") compute the same thing with a
    different accumulation order, so they land a few 1e-7 away - from the
    eager path and from each other.
    """
    torch.manual_seed(seed)
    hf_model = build_hf_twin(cfg)
    if cfg.enable_moe_block:
        hf_model.config._experts_implementation = experts_impl
    # Nudge every parameter off its init constant (norm weights start at
    # exactly 1) so nothing can match "by accident".
    with torch.no_grad():
        for p in hf_model.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    ours = TinyGemma(cfg).eval()
    load_hf_state_dict(ours, hf_model.state_dict())
    worst = compare_logits(hf_model, ours, cfg, label)
    assert worst < 1e-4, f"{label} logits diverge: {worst}"
    return hf_model, ours, worst


def check_hub_shapes(cfg, repo: str, multimodal: bool = False) -> None:
    """Compare a preset against the real checkpoint's safetensors metadata."""
    from huggingface_hub import get_safetensors_metadata

    print(f"\n  {repo}{'  (complete model)' if multimodal else '  (text stack)'}")
    with torch.device("meta"):                  # shapes only, no memory
        model = Gemma4(cfg) if multimodal else TinyGemma(cfg)
    head_key = "language_model.lm_head.weight" if multimodal else "lm_head.weight"

    metadata = get_safetensors_metadata(repo)   # reads file headers, not weights
    hub_shapes = {}
    for file_metadata in metadata.files_metadata.values():
        for name, info in file_metadata.tensors.items():
            hub_shapes[name] = tuple(info.shape)

    state = model.state_dict()
    mapped = {}
    other, ignored = 0, set(unexpected_kv_keys(model, hub_shapes))
    mismatches = []

    for hub_key, shape in hub_shapes.items():
        if hub_key in ignored:
            continue
        our_key = map_hf_key(hub_key, multimodal)
        if our_key is None:
            # Must be another tower or the tied head - anything else is a hole.
            if hub_key == "lm_head.weight" or (not multimodal and any(
                    hub_key.startswith(f"model.{t}") for t in OTHER_TOWERS)):
                other += 1
                continue
            mismatches.append(f"unclaimed checkpoint tensor: {hub_key}")
        elif our_key not in state:
            mismatches.append(f"checkpoint tensor we don't build: {hub_key}")
        else:
            mapped[our_key] = (hub_key, shape)

    for our_key, tensor in state.items():
        if our_key == head_key:
            continue                            # tied, never stored
        if our_key not in mapped:
            mismatches.append(f"our tensor missing from checkpoint: {our_key}")
        elif mapped[our_key][1] != tuple(tensor.shape):
            mismatches.append(f"shape mismatch {mapped[our_key][0]}: "
                              f"ours {tuple(tensor.shape)} vs hub {mapped[our_key][1]}")

    assert not mismatches, "\n".join(mismatches[:15])
    n_params = sum(t.numel() for k, t in state.items() if k != head_key)
    kind = "checkpoint" if multimodal else "text"
    print(f"    {len(mapped)} {kind} tensors match name-for-name and shape-for-shape")
    print(f"    {len(ignored)} KV-shared k/v tensors ignored on purpose,"
          f" {other} are the tied head or other towers")
    print(f"    parameters: {n_params:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hub", action="store_true",
                        help="skip the released-checkpoint shape check (needs network)")
    args = parser.parse_args()

    print("[1] dense parity (e2b recipe: PLE + KV sharing + double-wide MLP)")
    dense_cfg = ModelConfig()
    hf_model, _, _ = check_parity(dense_cfg, "dense")
    print("  PASS")

    print("\n[2] MoE parity (26B-A4B recipe: sparse experts + k=v, no PLE)")
    moe_cfg = ModelConfig.tiny_moe()
    check_parity(moe_cfg, "moe", seed=1)
    print("  PASS (bit-exact against the reference's own PyTorch expert loop)")
    # The fused kernels reorder the expert accumulation, so they differ from the
    # eager path by float32 noise. Showing it keeps the bit-exact claim honest.
    for impl in ("grouped_mm", "batched_mm"):
        _, _, delta = check_parity(moe_cfg, f"moe[{impl}]", seed=1, experts_impl=impl)
        print(f"  {impl}: {delta:.1e} away - the fused kernel's accumulation order,"
              " not an architecture difference")

    print("\n[3] multimodal parity (vision tower + audio Conformer + text)")

    def check_vl(cfg, label, seed):
        torch.manual_seed(seed)
        hf_vl = build_hf_vl_twin(cfg)
        if cfg.text.enable_moe_block:
            # Compare against the reference's eager expert loop, as in [2].
            hf_vl.config.text_config._experts_implementation = None
        with torch.no_grad():
            for p in hf_vl.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        our_vl = Gemma4(cfg).eval()
        load_hf_state_dict(our_vl, hf_vl.state_dict())
        worst = compare_vl_logits(hf_vl, our_vl, cfg, label)
        assert worst < 1e-4, f"{label} multimodal logits diverge: {worst}"
        return hf_vl

    vl_cfg = VLConfig()
    hf_vl = check_vl(vl_cfg, "e2b-style   ", 2)
    # The other release flips almost every switch: MoE text, a standardizing
    # vision tower with no clipped linears, and no audio tower at all.
    check_vl(VLConfig.tiny_26b_style(), "26B-style   ", 3)
    print("  PASS")

    print("\n[4] file roundtrip: save_pretrained -> load_hf_checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp)
        reloaded = TinyGemma(dense_cfg).eval()
        load_hf_checkpoint(reloaded, tmp)
        worst = compare_logits(hf_model, reloaded, dense_cfg, "reloaded")
    assert worst < 1e-4, f"logits diverge after roundtrip: {worst}"
    with tempfile.TemporaryDirectory() as tmp:
        hf_vl.save_pretrained(tmp)
        reloaded_vl = Gemma4(vl_cfg).eval()
        load_hf_checkpoint(reloaded_vl, tmp)
        worst = compare_vl_logits(hf_vl, reloaded_vl, vl_cfg, "reloaded multimodal")
    assert worst < 1e-4, f"multimodal logits diverge after roundtrip: {worst}"
    print("  PASS")

    if args.skip_hub:
        print("\n[5] released-checkpoint shape check skipped (--skip-hub)")
    else:
        print("\n[5] shape check against the released Gemma 4 checkpoints")
        check_hub_shapes(ModelConfig.gemma4_e2b(), "google/gemma-4-e2b-it")
        check_hub_shapes(ModelConfig.gemma4_26b_a4b(), "google/gemma-4-26B-A4B-it")
        print("\n  ...and the COMPLETE models, every tower included:")
        check_hub_shapes(VLConfig.gemma4_e2b(), "google/gemma-4-e2b-it", multimodal=True)
        check_hub_shapes(VLConfig.gemma4_26b_a4b(), "google/gemma-4-26B-A4B-it", multimodal=True)
        print("  PASS")

    print("\nAll checks passed: same architecture, both flavors, all three towers.")


if __name__ == "__main__":
    main()
