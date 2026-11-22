#!/usr/bin/env python
"""The vendored NeuroRVQ-EEG FM v1 encoder (src/ni26/fm/neurorvq.py) vs the OFFICIAL repo code, CPU, < 30 s. Exact parity is
required (max |d| <= 1e-5 relative; measured 0.0):
  1. the released weights (sha256 pinned) minus the pretraining-only keys load STRICT into NrEnc.net; the reference's own
     load (strict=False, as its load_neurorqv_fm) reports only those keys + its unused heads / fc_norms
  2. the electrode list == the reference's; Muse TP9 / AF7 / AF8 / TP10 -> rows 103 / 4 / 5 / 100
  3. tokens after blocks[:L] (L 4 / 12), the 4 branches concatenated, == the reference's (heads / fc_norms -> Identity,
     return_patch_tokens=True, its blocks truncated to L) on raw-uV randn and structured inputs
  4. prep: (B, 4, 640) @ 128 Hz -> (B, 4, 1000) @ 200 Hz, |x| <= 500
Reference: the clone at ~/workspace/datasets/pretrained_eeg_fms/neurorvq/repo_src/NeuroRVQ (commit pinned in the module;
einops from the local env).
  python tests/test_neurorvq.py  ->  "OK"
"""

import copy
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26.fm import neurorvq as V  # noqa: E402

REPO = Path.home() / "workspace/datasets/pretrained_eeg_fms/neurorvq/repo_src/NeuroRVQ"

OK = True


def check(cond, msg):
    global OK
    OK &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}", flush=True)


def main():
    torch.manual_seed(0)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    check(rev == V.REPO_COMMIT, f"0. reference clone at the pinned commit {rev[:8]}")
    import hashlib
    check(hashlib.sha256(V.WEIGHTS.read_bytes()).hexdigest() == V.SHA256, f"0. weights sha256 {V.SHA256[:12]} (pinned)")
    sd = torch.load(V.WEIGHTS, map_location="cpu", weights_only=True)
    enc_sd = V.encoder_state(sd)
    enc = V.NrEnc({}).eval()
    try:
        enc.net.load_state_dict(enc_sd, strict=True)
        strict = ""
    except RuntimeError as e:
        strict = str(e)[:800]
    n_enc = sum(v.numel() for v in enc_sd.values())
    check(not strict, f"1. released weights minus {V.NR_DROP} -> NrEnc.net STRICT ({len(enc_sd)} tensors, {n_enc / 1e6:.2f}M params)"
          + (f": {strict}" if strict else ""))
    # ---- the reference, as load_neurorqv_fm builds it
    sys.path.insert(0, str(REPO / "NeuroRVQ_EEG"))
    sys.path.insert(0, str(REPO))
    import yaml
    from NeuroRVQ import NeuroRVQFM
    from inference.modules.NeuroRVQ_EEG_FM_inference_modules import ch_names_global, create_embedding_ix
    args = yaml.safe_load(open(REPO / "flags" / "NeuroRVQ_EEG_v1.yml"))
    ref = NeuroRVQFM(n_patches=args["n_patches"], patch_size=args["patch_size"], in_chans=args["in_chans_second_stage"],
                     out_chans=args["out_chans_second_stage"], num_classes=0, embed_dim=args["embed_dim_second_stage"],
                     depth=args["depth_second_stage"], num_heads=args["num_heads_second_stage"],
                     mlp_ratio=args["mlp_ratio_second_stage"], qkv_bias=args["qkv_bias_second_stage"],
                     qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=args["drop_rate_second_stage"],
                     attn_drop_rate=args["attn_drop_rate_second_stage"], drop_path_rate=args["drop_path_rate_second_stage"],
                     init_values=args["init_values_second_stage"], init_scale=args["init_scale_second_stage"],
                     n_global_electrodes=len(ch_names_global), use_as_encoder=True, vocab_size=args["n_code"],
                     use_for_pretraining=False)
    miss, unexp = ref.load_state_dict(sd, strict=False)
    bad_miss = [k for k in miss if not k.startswith(("fc_norm_", "head_"))]
    bad_unexp = [k for k in unexp if not k.startswith(V.NR_DROP)]
    check(not bad_miss and not bad_unexp, f"1. the reference's own load: missing {len(miss)} (all fc_norm_* / head_*: {not bad_miss}), "
          f"unexpected {len(unexp)} (all {V.NR_DROP}: {not bad_unexp})")
    check(list(np.char.decode(ch_names_global)) == V.NR_CH, f"2. the vendored electrode list == the reference's ({len(V.NR_CH)})")
    muse = [b"tp9", b"af7", b"af8", b"tp10"]
    idx = [int(np.where(ch_names_global == c)[0][0]) for c in muse]
    check(idx == [103, 4, 5, 100] and enc.spat.tolist() == idx, f"2. Muse TP9 / AF7 / AF8 / TP10 -> rows {idx} (vendored {enc.spat.tolist()})")
    ref.eval()
    for nm in ("head_1", "head_2", "head_3", "head_4", "fc_norm_1", "fc_norm_2", "fc_norm_3", "fc_norm_4"):
        setattr(ref, nm, nn.Identity())
    B, C, S = 3, 4, 5
    g = torch.Generator().manual_seed(1)
    t = torch.arange(S * 200) / 200.0
    xs = sum((30.0 / f ** 0.6) * torch.sin(2 * torch.pi * f * t + 6.28 * torch.rand(B, C, 1, generator=g)) for f in (1, 3, 7, 11, 19, 31))
    inputs = {"randn_uV": 40.0 * torch.randn(B, C, S * 200, generator=g), "structured_uV": xs.clamp(-500, 500)}
    temp_ix, spat_ix = create_embedding_ix(S, args["n_patches"], np.array(muse), ch_names_global)
    for name, x in inputs.items():
        for L in (4, 12):
            r = copy.deepcopy(ref)
            r.blocks = nn.ModuleList(list(r.blocks)[:L])
            with torch.no_grad():
                x1, x2, x3, x4, _ = r(x.reshape(B, C, S, 200), temp_ix.expand(B, -1), spat_ix.expand(B, -1), return_patch_tokens=True)
                want = torch.cat([x1, x2, x3, x4], dim=-1)
                got = enc(x, L).reshape(B, C * S, -1)
            d = (want - got).abs().max().item()
            gm = want.abs().max().item()
            check(got.shape == want.shape and d <= 1e-5 * gm, f"3. {name} L{L}: tokens {tuple(got.shape)} max|d| {d:.2e} "
                  f"(rel {d / gm:.1e})")
    # prep contract (descriptive): 640 @ 128 Hz -> 1000 @ 200 Hz, |x| <= 500
    xp = V.NrEnc.prep(100.0 * torch.randn(2, 4, 640, generator=g))
    check(tuple(xp.shape) == (2, 4, 1000) and float(xp.abs().max()) <= 500.0, f"prep: {tuple(xp.shape)}, max|x| {float(xp.abs().max()):.1f}")
    assert V.load_pretrained().net.pos_embed.shape == (105, 200)
    print("OK" if OK else "FAILED", flush=True)
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
