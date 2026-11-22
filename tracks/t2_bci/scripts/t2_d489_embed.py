#!/usr/bin/env python
"""T2-D489 step 0 (registered 2026-11-14 16:36, before any number): ST-EEGFormer-large ENCODER embeddings of every public PROTEUS window,
as the ST-L members see their input (causal sz alignment per stream, calib gain, 128 Hz x 4 s). Feature per window = LayerNorm (no affine)
of the last layer's tokens, mean over tokens: the input of the members' 'mean' head. Row i = export index row i (t2_r6_louo.py's contract).
  E0: the public checkpoint (never saw PROTEUS)          E1: --fm-ckpt a t2p_cpt checkpoint (d111-all saw every public window; descriptive)
  gpu-submit python tracks/t2_bci/scripts/t2_d489_embed.py --out ~/workspace/outputs/neural_interface_challenge/t2_d489/fp_e0.npy [--fm-ckpt P]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def embed(enc, XA, dev, bs):
    L, out = enc.n_layers, []
    with torch.no_grad():
        for s in range(0, len(XA), bs):
            g = enc(enc.prep(torch.as_tensor(XA[s:s + bs], device=dev)), L)          # (B, C, P, D), v.norm'd at the last layer
            out.append(F.layer_norm(g.float(), (g.shape[-1],)).mean((1, 2)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(EXP))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fm-ckpt", default="")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: embed the first N windows")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[d489] {s}", flush=True)  # noqa: E731
    X, idx, meta = B.load_export(a.export)
    assert (idx["i"].to_numpy() == np.arange(len(idx))).all()
    scale = float(np.median(np.asarray(X[::7]).std(-1)))                              # global constant (the bench: train windows[::7])
    XA = B.aligned_all(X, idx, "sz", np.ones(len(idx), bool), 0, 0.05, scale)         # n0 0 / shrink 0.05 = the members' __sz__n00
    if a.limit:
        XA = XA[:a.limit]
    if a.fm_ckpt:
        t2p_fm.use_cpt("st_large", a.fm_ckpt, log)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = t2p_fm.make_encoder("st_large", list(meta["ch_names"]), log, 0.0, "gain").to(dev).eval()
    Fp = embed(enc, XA, dev, a.bs)
    assert np.isfinite(Fp).all() and Fp.shape[0] == len(XA)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, Fp)
    log(f"{Fp.shape} -> {a.out} in {time.time() - t0:.0f}s; ckpt {a.fm_ckpt or 'public'}; provenance {getattr(enc, 'provenance', None)}")


if __name__ == "__main__":
    main()
