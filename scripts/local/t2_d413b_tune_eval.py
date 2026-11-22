#!/usr/bin/env python
"""T2-D413b PHASE B tuning metric (pre-registered 2026-11-09 11:23): TRAINING WINDOWS ONLY (no val / test window is read).
For one generator stem (<gen-dir>/{own,interp}_<proto>_<tag>.npz): the gate's (a) AUC (own; interp reported), (b) copy share (max-corr, as
amended 10:17), (c) per-class band-power / covariance distances vs the user spread, and TSTR-tr = FBCNet r1 seed 0 trained on the
interpolated synthetic windows, cell BA on the REAL training windows (the bench's align-none scale). Appends one JSON line to
<gen-dir>/tune.jsonl. The config pick ranks by (a) AUC subject to (b) <= 0.05 and (c) on all classes.
    gpu-submit ... python scripts/local/t2_d413b_tune_eval.py --export E --gen-dir outputs/t2_d413b --protocol kit --tag <tag>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "local"))
import t2_d413_gate as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    D = Path(a.gen_dir)
    X, idx, meta = G.load_export(a.export)
    (_, trm, _, _), = G.folds(idx, a.protocol)
    tr = np.flatnonzero(trm)
    users = sorted(idx.iloc[tr]["subject"].unique())
    u = idx.iloc[tr]["subject"].map({s: k for k, s in enumerate(users)}).to_numpy()
    y = idx["y"].to_numpy()[tr]
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()[tr]
    Xr = np.asarray(X[tr], dtype=np.float32)
    own, itp = np.load(D / f"own_{a.protocol}_{a.tag}.npz"), np.load(D / f"interp_{a.protocol}_{a.tag}.npz")
    assert np.array_equal(own["train_idx"], tr) and np.array_equal(own["y"], y) and np.array_equal(own["u"], u)
    Xo, Xi, yi = own["X"], itp["X"], itp["y"]
    R = dict(protocol=a.protocol, tag=a.tag, note=a.note)
    Fr, Fo, Fi = G.logbp(Xr), G.logbp(Xo), G.logbp(Xi)
    cv = G.StratifiedKFold(5, shuffle=True, random_state=0)
    clf = G.make_pipeline(G.StandardScaler(), G.LogisticRegression(C=1.0, max_iter=5000))
    R["auc_own"] = float(G.cross_val_score(clf, np.r_[Fr, Fo], np.r_[np.zeros(len(Fr)), np.ones(len(Fo))], cv=cv, scoring="roc_auc").mean())
    R["auc_interp"] = float(G.cross_val_score(clf, np.r_[Fr, Fi], np.r_[np.zeros(len(Fr)), np.ones(len(Fi))], cv=cv, scoring="roc_auc").mean())
    mu, sd = Xr.mean((0, 2), keepdims=True), Xr.std((0, 2), keepdims=True)

    def unit(Z):
        Z = torch.as_tensor(((Z - mu) / sd).reshape(len(Z), -1), dtype=torch.float64)
        Z = Z - Z.mean(1, keepdim=True)
        return Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Ur, Uo = unit(Xr), unit(Xo)
    K = Ur @ Ur.T; K.fill_diagonal_(-float("inf"))
    thr = float(np.percentile(K.max(1).values.numpy(), 99))
    R["copy_share"] = float(((Uo @ Ur.T).max(1).values.numpy() > thr).mean())
    Cr, Co = G.covs(Xr), G.covs(Xo)
    cls = {}
    for c in range(3):
        k = y == c
        mr, Cm = Fr[k].mean(0), Cr[k].mean(0)
        sb = float(np.median([np.linalg.norm(Fr[k & (u == j)].mean(0) - mr) for j in range(len(users))]))
        sc = float(np.median([G.airm(Cr[k & (u == j)].mean(0), Cm) for j in range(len(users))]))
        ko = own["y"] == c
        cls[c] = dict(bp=round(float(np.linalg.norm(Fo[ko].mean(0) - mr)), 3), bp_spread=round(sb, 3), cov=round(G.airm(Co[ko].mean(0), Cm), 3), cov_spread=round(sc, 3))
        cls[c]["pass"] = cls[c]["bp"] <= sb and cls[c]["cov"] <= sc
    R["class"] = cls
    s_ = float(np.median(np.asarray(X[tr[::7]]).std(-1)))
    ev = (Xr / s_, y, cells, Xr / s_, y, cells)
    R["tstr_tr"] = G.tstr(Xi / s_, yi, itp["u"], *ev)
    R["ok_b_c"] = bool(R["copy_share"] <= 0.05 and all(v["pass"] for v in cls.values()))
    with open(D / "tune.jsonl", "a") as fh:
        fh.write(json.dumps(R) + "\n")
    print(f"== {a.protocol} {a.tag}: AUC own {R['auc_own']:.3f} (interp {R['auc_interp']:.3f}) | copy {R['copy_share']:.3f} | "
          + "; ".join(f"c{c} bp {v['bp']:.2f}/{v['bp_spread']:.2f} cov {v['cov']:.2f}/{v['cov_spread']:.2f}" for c, v in cls.items())
          + f" | TSTR-tr {R['tstr_tr']:.3f} | (b)+(c) {'ok' if R['ok_b_c'] else 'no'}  [{a.note}]", flush=True)


if __name__ == "__main__":
    main()
