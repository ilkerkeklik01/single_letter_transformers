"""Prove the "exact same architecture" claim - text AND vision - four ways.

  1. Text parity    : a tiny HF `MuseGlimmerForConditionalGeneration`
                      (transformers >= 5.15.0) vs this folder's text-only
                      `MuseGlimmer`, same weights, text-only inputs: logits
                      must match at sequence lengths below and above the
                      sliding window.

  2. Vision parity  : the same twin vs our complete `MuseGlimmerVL`, fed two
                      images of DIFFERENT sizes in one batch (so window
                      attention gets ragged edge windows) with their features
                      scattered between text tokens - full-forward logits must
                      match. A 2-frame video is checked at feature level too
                      (per-frame attention segments, temporal pixel shuffle).

  3. File roundtrip : save the HF twin with `save_pretrained`, load the files
                      into a fresh `MuseGlimmerVL` via
                      hf_weights.load_hf_checkpoint, logits must still match.

  4. 30B shape check: build `MuseGlimmerVL(VLConfig.muse_glimmer_30b())` on
                      the meta device (no memory needed) and compare EVERY
                      tensor - text stack, vision tower, adapter, projection -
                      against the real `unsloth/Muse-Glimmer-30B` safetensors
                      metadata on the Hub, in both directions. Only file
                      headers are fetched, no weights. Skip with --skip-hub
                      if you are offline.

Run:  python verify_against_hf.py
"""

import argparse
import tempfile

import torch

from config import ModelConfig, VLConfig
from model import MuseGlimmer
from vl_model import MuseGlimmerVL
from hf_weights import load_hf_checkpoint, load_hf_state_dict, map_hf_key


def build_hf_twin(cfg: VLConfig):
    """The reference implementation, configured to our tiny numbers."""
    from transformers import (
        MuseGlimmerConfig,
        MuseGlimmerForConditionalGeneration,
        MuseGlimmerTextConfig,
        MuseGlimmerVisionConfig,
    )

    t, v = cfg.text, cfg.vision
    text_config = MuseGlimmerTextConfig(
        vocab_size=t.vocab_size,
        hidden_size=t.hidden_size,
        intermediate_size=t.intermediate_size,
        num_hidden_layers=t.num_layers,
        num_attention_heads=t.num_heads,
        num_key_value_heads=t.num_kv_heads,
        head_dim=t.head_dim,
        max_position_embeddings=t.max_seq_len,
        rope_parameters={"rope_type": "default", "rope_theta": t.rope_theta},
        sliding_window=t.sliding_window,
        rms_norm_eps=t.rms_norm_eps,
        post_norm_eps=t.post_norm_eps,
        qk_scale_factor=t.qk_scale_factor,
        final_logit_softcapping=t.final_logit_softcapping,
        output_multiplier=t.output_multiplier,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
    )
    vision_config = MuseGlimmerVisionConfig(
        hidden_size=v.hidden_size,
        intermediate_size=v.intermediate_size,
        num_hidden_layers=v.num_layers,
        num_attention_heads=v.num_heads,
        patch_size=v.patch_size,
        patch_temporal=v.patch_temporal,
        pos_emb_height=v.pos_emb_side,
        pos_emb_width=v.pos_emb_side,
        max_position_embeddings=v.pos_emb_side**2,
        merge_size=v.merge_size,
        layer_norm_eps=v.layer_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": v.rope_theta},
        hidden_act="gelu",
    )
    config = MuseGlimmerConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=cfg.image_token_id,
        video_token_id=cfg.video_token_id,
        out_hidden_size=cfg.out_hidden_size,
        projector_hidden_size=cfg.projector_hidden_size,
        projector_hidden_act="gelu",
    )
    return MuseGlimmerForConditionalGeneration(config).eval()


def make_image_batch(cfg: VLConfig):
    """Two differently-sized images woven into one 24-token text sequence."""
    v = cfg.vision
    # 8x8 patches (multiple whole windows) and 4x6 (ragged right-edge window).
    grid_thw = torch.tensor([[1, 8, 8], [1, 4, 6]])
    pixel_values = torch.randn(int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum()),
                               v.patch_dim)
    tokens_per_image = (grid_thw.prod(-1) // v.merge_size**2).tolist()   # [16, 6]
    ids = ([3] + [cfg.image_token_id] * tokens_per_image[0]
           + [5] + [cfg.image_token_id] * tokens_per_image[1])
    return torch.tensor([ids]), pixel_values, grid_thw


def compare_text_logits(hf_model, our_model, cfg: ModelConfig) -> float:
    """Same random token ids through both models; return the worst |diff|."""
    worst = 0.0
    for seq_len in (5, 20, cfg.max_seq_len):   # below / above the sliding window
        ids = torch.randint(0, cfg.vocab_size, (2, seq_len))
        with torch.no_grad():
            hf_logits = hf_model(input_ids=ids).logits
            our_logits, _ = our_model(ids)
        diff = (hf_logits - our_logits).abs().max().item()
        print(f"  seq_len {seq_len:3d}: max |logit diff| = {diff:.3e}")
        worst = max(worst, diff)
    return worst


def compare_vision_logits(hf_model, our_model, cfg: VLConfig) -> float:
    """Full multimodal forward through both models; return the worst |diff|."""
    ids, pixel_values, grid_thw = make_image_batch(cfg)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids, pixel_values=pixel_values,
                             image_grid_thw=grid_thw).logits
        our_logits, _ = our_model(ids, pixel_values=pixel_values,
                                  image_grid_thw=grid_thw)
    diff = (hf_logits - our_logits).abs().max().item()
    print(f"  2 images ({ids.shape[1]} tokens): max |logit diff| = {diff:.3e}")
    return diff


def compare_video_features(hf_model, our_model, cfg: VLConfig) -> float:
    """A 2-frame clip through both vision paths; return the worst |diff|."""
    grid_thw = torch.tensor([[2, 4, 4]])
    pixel_values = torch.randn(2 * 4 * 4, cfg.vision.patch_dim)
    with torch.no_grad():
        hf_features = torch.cat(
            hf_model.get_video_features(pixel_values, grid_thw).pooler_output
        )
        our_features = our_model.get_image_features(pixel_values, grid_thw)
    diff = (hf_features - our_features).abs().max().item()
    print(f"  2-frame video ({our_features.shape[0]} tokens): max |feature diff| = {diff:.3e}")
    return diff


def check_30b_shapes() -> None:
    """Compare our 30B-config module against the Hub checkpoint's metadata."""
    from huggingface_hub import get_safetensors_metadata

    repo = "unsloth/Muse-Glimmer-30B"
    print(f"\n[4] 30B shape check against {repo}")
    with torch.device("meta"):                  # shapes only, no memory
        big = MuseGlimmerVL(VLConfig.muse_glimmer_30b())

    metadata = get_safetensors_metadata(repo)   # reads file headers, not weights
    hub_shapes = {}
    for file_metadata in metadata.files_metadata.values():
        for name, info in file_metadata.tensors.items():
            hub_shapes[name] = tuple(info.shape)

    mismatches = []
    state = big.state_dict()
    mapped_hub = {}
    for hub_key, shape in hub_shapes.items():
        our_key = map_hf_key(hub_key, multimodal=True)
        if our_key is None:
            mismatches.append(f"hub tensor with no home in our model: {hub_key}")
        else:
            mapped_hub[our_key] = (hub_key, shape)

    for our_key, tensor in state.items():
        if our_key not in mapped_hub:
            mismatches.append(f"our tensor missing on hub: {our_key}")
        elif mapped_hub[our_key][1] != tuple(tensor.shape):
            mismatches.append(
                f"shape mismatch {mapped_hub[our_key][0]}: "
                f"ours {tuple(tensor.shape)} vs hub {mapped_hub[our_key][1]}"
            )
    extra = set(mapped_hub) - set(state)
    mismatches.extend(f"hub tensor we don't build: {mapped_hub[k][0]}" for k in extra)

    assert not mismatches, "\n".join(mismatches)
    print(f"  all {len(hub_shapes)} checkpoint tensors match our complete model,")
    print("  name-for-name and shape-for-shape, in both directions")
    n_params = sum(t.numel() for t in state.values())
    print(f"  30B-config parameter count: {n_params:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hub", action="store_true",
                        help="skip the 30B shape check (needs network)")
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = VLConfig()   # tiny defaults, text and vision both

    hf_model = build_hf_twin(cfg)
    # Nudge every parameter off its init constant (norm weights start at
    # exactly 1 or 0) so nothing matches "by accident".
    with torch.no_grad():
        for p in hf_model.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    print("[1] text parity: HF reference vs from-scratch, shared random weights")
    our_text = MuseGlimmer(cfg.text).eval()
    load_hf_state_dict(our_text, hf_model.state_dict())   # skips the vision keys
    worst = compare_text_logits(hf_model, our_text, cfg.text)
    assert worst < 1e-4, f"text logits diverge: {worst}"
    print("  PASS")

    print("\n[2] vision parity: images in the token stream, and a video clip")
    our_vl = MuseGlimmerVL(cfg).eval()
    load_hf_state_dict(our_vl, hf_model.state_dict())     # loads everything
    worst = compare_vision_logits(hf_model, our_vl, cfg)
    worst = max(worst, compare_video_features(hf_model, our_vl, cfg))
    assert worst < 1e-4, f"vision path diverges: {worst}"
    print("  PASS")

    print("\n[3] file roundtrip: save_pretrained -> load_hf_checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp)
        reloaded = MuseGlimmerVL(cfg).eval()
        load_hf_checkpoint(reloaded, tmp)
        worst = compare_vision_logits(hf_model, reloaded, cfg)
    assert worst < 1e-4, f"logits diverge after roundtrip: {worst}"
    print("  PASS")

    if args.skip_hub:
        print("\n[4] 30B shape check skipped (--skip-hub)")
    else:
        check_30b_shapes()
        print("  PASS")

    print("\nAll checks passed: same architecture - text and vision - only the config differs.")


if __name__ == "__main__":
    main()
