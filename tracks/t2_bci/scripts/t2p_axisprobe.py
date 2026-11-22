#!/usr/bin/env python
"""Track 2 T2-D126 (FM program §0d item 6): do FROZEN ST-L / MOR features use time order? Linear-probe cell-BA of pooled
frozen features of the kit windows under four inputs: original, TIME-shuffled (32 segments of 15 samples = 0.125 s = one
patch of both FMs, permuted, the same permutation for every channel), CHANNEL-shuffled (rows permuted; each slot keeps
its channel name / embedding), both. Pretraining contract: align wz, calib zwin, pinned checkpoints (no CPT, no FT).

  python t2p_axisprobe.py --export <export_proteus_v1> --out probe.json [--fms st_large,mor_p1v2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402
from ni26 import metrics  # noqa: E402

SEG = 15                                   # 0.125 s at 120 Hz = one ST patch (16 @ 128 Hz) = one MOR patch (32 @ 256 Hz)


def transform(X, cond, rng):
    X = X.copy()
    N, C, T = X.shape
    if cond in ("time", "both"):
        nseg = T // SEG
        for i in range(N):
            p = rng.permutation(nseg)
            X[i, :, :nseg * SEG] = X[i, :, :nseg * SEG].reshape(C, nseg, SEG)[:, p].reshape(C, nseg * SEG)
    if cond in ("chan", "both"):
        for i in range(N):
            X[i] = X[i, rng.permutation(C)]
    return X


@torch.no_grad()
def features(enc, X, layers, dev, bs=64):
    out = {L: [] for L in layers}
    for a in range(0, len(X), bs):
        x = torch.as_tensor(X[a:a + bs], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            xp = enc.prep(x)
            for L in layers:
                g = enc(xp, L)                                  # (B, C, P, D)
                out[L].append(g.float().mean((1, 2)).cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fms", default="st_large,mor_p1v2")
    ap.add_argument("--ckpt", default="", help="a CPT checkpoint for the (single) --fms family (t2p_fm.use_cpt; manipulation check)")
    ap.add_argument("--probe", default="logreg", choices=("logreg", "lda"),
                    help="lda = shrinkage LDA (whitened, the lead's inhouse_fm A56 transfer); logreg = StandardScaler + logistic")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, idx, meta = B.load_export(args.export)
    (_, trm, vam, tem), = B.folds(idx, "kit")
    tr, va, te = map(np.flatnonzero, (trm, vam, tem))
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    scale = float(np.median(np.asarray(X[tr[::7]]).std(-1)))
    XA = B.aligned_all(X, idx, "wz", trm, 0.0, 0.05, scale)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    for fm in args.fms.split(","):
        if args.ckpt:
            assert len(args.fms.split(",")) == 1, "--ckpt: one family"
            t2p_fm.use_cpt(fm, args.ckpt, print)
        enc = t2p_fm.make_encoder(fm, meta["ch_names"], lambda m: print(f"[{fm}] {m}", flush=True), 0.0, "zwin").to(dev).eval()
        Lf = enc.n_layers
        layers = sorted({Lf // 2, Lf})
        res[fm] = {}
        for cond in ("orig", "time", "chan", "both"):
            Xc = transform(XA, cond, np.random.default_rng(0)) if cond != "orig" else XA
            F = features(enc, Xc, layers, dev)
            for L, f in F.items():
                clf = (make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")) if args.probe == "lda"
                       else make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))).fit(f[tr], y[tr])
                r = {}
                for name, rr in (("test", te), ("val", va)):
                    r[name] = float(metrics.cell_balanced_accuracy(y[rr], clf.predict(f[rr]), cells[rr])[0])
                res[fm][f"L{L}:{cond}"] = r
                print(f"[probe] {fm} layer {L}/{Lf} {cond:5s}: test {r['test']:.4f} val {r['val']:.4f}", flush=True)
        del enc
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
