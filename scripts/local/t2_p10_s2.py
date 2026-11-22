#!/usr/bin/env python
"""Panel P10 seed S2 (descriptive, zero GPU): what does doubling the CNN nets per family do, with the decoder start vector?
Shape C (T2-D459: M7's weights, fresh CNN draws D419 / D429, D442 control FM) vs C2 = the same with each CNN family's slot split
equally over TWO nets (C's fresh draw + spec_m7's draw of the same recipe, independent seeds), both decoded with the shipped start vector.
Local draws hold ONE net per family; the ZIP holds 15, so 1 -> 2 here is not 15 -> 30 there (a 1/n model scales the gain by 1/15).
    mem-guard python scripts/local/t2_p10_s2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
M7 = json.load(open(O / "t2_d408/spec_m7.json"))["members"]
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = S[src]
        M[dst + "_b"] = M7[dst]
C = J["arms"]["M7f"]
C2 = {}
for k, v in C.items():
    if k in ("fbc_tv8c", "sh_tv8c", "ifn"):
        C2[k], C2[k + "_b"] = v / 2, v / 2
    else:
        C2[k] = v
START = np.array([0.4073, 0.2541, 0.3386])
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()


def dec(P, n0=3, lam=0.95, iters=10):
    K, out = P.shape[1], np.empty_like(P)
    for t in range(len(P)):
        Mx = np.vstack([np.tile(START, (n0, 1)), P[:t + 1]])
        w = lam ** np.arange(Mx.shape[0] - 1, -1, -1, dtype=np.float64)
        for _ in range(iters):
            Mx = Mx * (w.sum() / K) / (Mx * w[:, None]).sum(0, keepdims=True)
            Mx = Mx / Mx.sum(1, keepdims=True)
        out[t] = Mx[-1]
    return out


rows = []
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            for name, arm in (("C", C), ("C2", C2)):
                mix, ix = 0.0, None
                for m, w in arm.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or (ix == ix_).all()
                    ix, mix = ix_, mix + w * p.astype(np.float64)
                pred = np.empty(len(ix), int)
                for s_ in np.unique(ST[ix]):
                    r = np.flatnonzero(ST[ix] == s_)
                    r = r[np.argsort(OD[ix][r], kind="stable")]
                    pred[r] = dec(mix[r]).argmax(1)
                for c in np.unique(CELL[ix]):
                    mk = CELL[ix] == c
                    rows.append((name, proto, i, split, c, U[ix][mk][0], np.mean([np.mean(pred[mk][Y[ix][mk] == v] == v) for v in np.unique(Y[ix][mk])])))
D = pd.DataFrame(rows, columns=["arm", "proto", "i", "split", "cell", "u", "ba"])
key = ["proto", "i", "split", "cell", "u"]
d = (D[D.arm == "C2"].set_index(key).ba - D[D.arm == "C"].set_index(key).ba).rename("d").reset_index()
t = d[d.u != "P04"]
dd = t.groupby("i").d.mean()
print(f"C2 (two nets per CNN family) vs C (one), start vector on both: typical {dd.mean():+.4f} ({int((dd > 0).sum())}/12); level C {D[(D.arm == 'C') & (D.u != 'P04')].ba.mean():.4f}; "
      + " ".join(f"{u} {v:+.4f}" for u, v in d.groupby("u").d.mean().items()))
