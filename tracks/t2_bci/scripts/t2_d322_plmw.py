#!/usr/bin/env python
"""T2-D322: online pseudo-label self-training of per-stream member weights (V1 'PL-MW'), zero GPU, on the saved member
probabilities of T2-D310 (fbc_tv8c / sh_tv8c / stl_rcll = the X4 package, w0 = 0.4 / 0.3 / 0.3). Constants fixed in the row.
  python t2_d322_plmw.py --export E --out DIR [--jobs 6]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

TAU, ETA, BETA, FLOOR, WARM, LAM, ITERS, N0 = 0.6, 0.5, 0.02, 0.05, 10, 0.95, 10, 3
MEM = ("fbc_tv8c", "sh_tv8c", "stl_rcll")
W0 = np.array([0.4, 0.3, 0.3])
SPEC = "~/workspace/outputs/neural_interface_challenge/t2_d310/spec.json"


def sk_last(H, lam=LAM, iters=ITERS, n0=N0):
    """The shipped decoder's row t: SK10 with forgetting lam over [n0 uniform pseudo-rows; stored mixes 1..t]
    (t2p_report.causal_sinkhorn's inner loop, lam < 1 branch)."""
    K = H.shape[1]
    M = np.vstack([np.full((n0, K), 1.0 / K), H])
    w = lam ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
    for _ in range(iters):
        M = M * (w.sum() / K) / (M * w[:, None]).sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
    return M[-1]


def run_stream(P, adapt):
    """P: (n, n_members, K) member probabilities of one stream in time order. Returns the predictions (n,)."""
    n = P.shape[0]
    w = W0.copy()
    H = np.empty((n, P.shape[2]))
    pred = np.empty(n, int)
    for t in range(n):
        H[t] = np.tensordot(w, P[t], 1)
        q = sk_last(H[:t + 1])
        pred[t] = int(q.argmax())
        if adapt:
            if t >= WARM and q.max() >= TAU:
                w = w * np.power(np.clip(P[t][:, pred[t]], 1e-6, None), ETA)
                w = w / w.sum()
            w = (1 - BETA) * w + BETA * W0
            w = np.maximum(w, FLOOR)
            w = w / w.sum()
    return pred


def bal(y, p):
    return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])


def job(args):
    export, proto, split, i = args
    d = pd.read_csv(f"{export}/index.csv")
    J = json.load(open(SPEC))
    probs, ix = [], None
    for m in MEM:
        M = J["members"][m]
        z = np.load(M[proto].format(s=M["offset"] + i))
        ix_ = z["test_idx"] if split == "test" else z["val_idx"]
        assert ix is None or (ix == ix_).all()
        ix = ix_
        probs.append(z["prob_last"] if split == "test" else z["prob_val_last"])
    P = np.stack(probs, 1).astype(np.float64)                  # (n, 3, K)
    g = d.iloc[ix].reset_index(drop=True)
    out = []
    for arm, adapt in (("base", False), ("plmw", True)):
        pred = np.empty(len(g), int)
        for _, gs in g.groupby("stream"):
            rows = gs.sort_values("order").index.to_numpy()
            pred[rows] = run_stream(P[rows], adapt)
        y = g.y.to_numpy()
        for c in g.cell.unique():
            mk = (g.cell == c).to_numpy()
            out.append(dict(proto=proto, split=split, i=i, arm=arm, cell=c, subj=c.split("|")[0][-3:], ba=bal(y[mk], pred[mk])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(a.export, p, s, i) for p in ("kit", "seen1") for s in ("test", "val") for i in range(12)]
    with Pool(a.jobs) as pool:
        R = pd.DataFrame([r for rr in pool.imap_unordered(job, tasks) for r in rr])
    R.to_csv(f"{a.out}/rows.csv", index=False)
    R["grp"] = np.select([R.subj == "P04", R.subj == "P10"], ["P04", "P10"], "newT")
    res = {}
    for name, mk in (("typical", R.grp != "P04"), ("P04", R.grp == "P04"), ("P10", R.grp == "P10"), ("newT", R.grp == "newT")):
        T = R[mk].groupby(["arm", "i"]).ba.mean().unstack("arm")
        dd = T["plmw"] - T["base"]
        res[name] = dict(d=float(dd.mean()), up=int((dd > 0).sum()), base=float(T["base"].mean()))
    ok = res["typical"]["d"] > 0.003 and res["typical"]["up"] >= 9 and res["P04"]["d"] >= -0.010
    res["bar"] = "PASS" if ok else "not met"
    json.dump(res, open(f"{a.out}/read.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
