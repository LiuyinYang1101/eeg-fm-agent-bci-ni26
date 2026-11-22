#!/usr/bin/env python
"""Parity of the pure-torch ST-EEGFormer (src/ni26/fm/steeg_pt.py) with the timm harness path (t2p_fm.FMClf(STEEG4)).

Builds the harness classifier (pinned checkpoint, or seeded random init with --random), perturbs EVERY parameter (so
zero-initialised tables such as the channel embedding and the head matter), loads its state dict into
steeg_pt.STClassifier with strict=True, and compares logits on random windows (41 kit channels, 4 s at 120 Hz):
fp32 (max |dlogit|, bitwise flag) and, on CUDA, bf16 autocast (the training / packaged inference precision).

  python src/ni26/fm/steeg_pt_check.py --fm st_large [--random] [--device cuda] [--n 8]
Exit 1 if fp32 max |dlogit| > --tol (default 1e-4) or any argmax differs in fp32.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))


def convert(sd):
    """FMClf(STEEG4) state dict -> STClassifier state dict (same names; returned as a float32 copy)."""
    return {k: v.detach().float().clone() if v.is_floating_point() else v.clone() for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default="st_large")
    ap.add_argument("--random", action="store_true", help="seeded random init instead of the pinned checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--calib", default="gain")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    args = ap.parse_args()
    import t2p_fm
    from ni26.fm import steeg_pt as P
    names = json.loads((Path(args.export) / "meta.json").read_text())["ch_names"]
    torch.manual_seed(0)
    enc = t2p_fm.make_encoder(("rand_" if args.random else "") + args.fm, names, print, 0.0, args.calib)
    ref = t2p_fm.FMClf(enc, enc.n_layers, "mean", 3).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.02 * torch.randn_like(p))
    sd = ref.state_dict()
    new = P.STClassifier(args.fm, enc.chan_idx.tolist(), int(round(P.ST_FS * 4.0)), args.calib, -1, 3).eval()
    r = new.load_state_dict(convert(sd), strict=True)
    print(f"strict load ok ({len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M values); {r}")
    assert enc.chan_idx.tolist() == new.chan_idx.tolist()
    dev = torch.device(args.device)
    ref, new = ref.to(dev), new.to(dev)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(args.n, len(names), 480, generator=g).to(dev)
    res = {}
    with torch.no_grad():
        a, b = ref(x).float(), new(x).float()
        res["fp32_max_abs"] = float((a - b).abs().max())
        res["fp32_bitwise"] = bool(torch.equal(a, b))
        res["fp32_argmax_agree"] = float((a.argmax(1) == b.argmax(1)).float().mean())
        res["logit_scale"] = float(a.abs().mean())
        if dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a16, b16 = ref(x).float(), new(x).float()
            res["bf16_max_abs"] = float((a16 - b16).abs().max())
            res["bf16_bitwise"] = bool(torch.equal(a16, b16))
            res["bf16_vs_fp32_max_abs"] = float((b16 - b).abs().max())
            pa, pb = torch.softmax(a16, 1), torch.softmax(b16, 1)
            res["bf16_max_abs_prob"] = float((pa - pb).abs().max())
    print(json.dumps(dict(fm=args.fm, random=args.random, device=args.device, calib=args.calib, **res), indent=1))
    if res["fp32_max_abs"] > args.tol or res["fp32_argmax_agree"] < 1.0:
        raise SystemExit("!! pure-torch ST-EEGFormer != timm path")


if __name__ == "__main__":
    main()
