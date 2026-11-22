#!/usr/bin/env python
"""T2-D472 (written before any number): a two-timescale stream decoder on top of M7-B3's. Shape C (T2-D459: M7's weights, fresh CNN
draws D419 / D429, D442 control FM); every arm uses the shipped start vector. Base B3 = causal Sinkhorn 10 / n0 3 / lambda 0.95 + start.
PRIMARY: TT = the normalised geometric mean of the lambda 0.90 and lambda 1.0 decoder posteriors (fast + slow memory). Descriptive:
TT_ar (arithmetic mean of the same two), TT2 (geometric mean of 0.95 and 1.0). Typical (non-P04), kit + seen1, test + val, 12 draws.
BAR (primary): typical >= +0.002, >= 10 / 12, no user < -0.002, vs B3.
    mem-guard python scripts/local/t2_d472_twotime.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = S[src]
W = J["arms"]["M7f"]
START = np.array([0.4073, 0.2541, 0.3386])
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()


def dec(P, lam, n0=3, iters=10):
    K, out = P.shape[1], np.empty_like(P)
    for t in range(len(P)):
        Mx = np.vstack([np.tile(START, (n0, 1)), P[:t + 1]])
        w = lam ** np.arange(Mx.shape[0] - 1, -1, -1, dtype=np.float64)
        for _ in range(iters):
            Mx = Mx * (w.sum() / K) / (Mx * w[:, None]).sum(0, keepdims=True)
            Mx = Mx / Mx.sum(1, keepdims=True)
        out[t] = Mx[-1]
    return out


def geo(a, b):
    g = np.sqrt(np.maximum(a, 1e-12) * np.maximum(b, 1e-12))
    return g / g.sum(1, keepdims=True)


def task(job):
    proto, i, split = job
    mix, ix = 0.0, None
    for m, w in W.items():
        z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
        ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
        mix = mix + w * p.astype(np.float64)
    preds = {k: np.empty(len(ix), int) for k in ("B3", "TT", "TT_ar", "TT2")}
    for s_ in np.unique(ST[ix]):
        r = np.flatnonzero(ST[ix] == s_)
        r = r[np.argsort(OD[ix][r], kind="stable")]
        d90, d95, d100 = dec(mix[r], 0.90), dec(mix[r], 0.95), dec(mix[r], 1.0)
        preds["B3"][r] = d95.argmax(1)
        preds["TT"][r] = geo(d90, d100).argmax(1)
        preds["TT_ar"][r] = (d90 + d100).argmax(1)
        preds["TT2"][r] = geo(d95, d100).argmax(1)
    out = []
    for arm, pr in preds.items():
        for c in np.unique(CELL[ix]):
            mk = CELL[ix] == c
            out.append((arm, proto, i, split, c, U[ix][mk][0], np.mean([np.mean(pr[mk][Y[ix][mk] == v] == v) for v in np.unique(Y[ix][mk])])))
    return out


if __name__ == "__main__":
    with Pool(12) as pool:
        rows = sum(pool.map(task, [(p, i, s) for p in ("kit", "seen1") for i in range(12) for s in ("test", "val")]), [])
    D = pd.DataFrame(rows, columns=["arm", "proto", "i", "split", "cell", "u", "ba"])
    key = ["proto", "i", "split", "cell", "u"]
    base = D[D.arm == "B3"].set_index(key).ba
    print(f"B3 typical level {D[(D.arm == 'B3') & (D.u != 'P04')].ba.mean():.4f}")
    for arm in ("TT", "TT_ar", "TT2"):
        d = (D[D.arm == arm].set_index(key).ba - base).rename("d").reset_index()
        t = d[d.u != "P04"]
        dd = t.groupby("i").d.mean()
        print(f"{arm:6s} vs B3: typical {dd.mean():+.4f} ({int((dd > 0).sum())}/12)  " + " ".join(f"{u} {v:+.4f}" for u, v in d.groupby("u").d.mean().items())
              + "  | " + " ".join(f"{s} {v:+.4f}" for s, v in t.groupby("split").d.mean().items()), flush=True)
