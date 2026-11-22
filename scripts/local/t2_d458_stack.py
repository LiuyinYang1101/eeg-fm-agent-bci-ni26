#!/usr/bin/env python
"""T2-D458 (written before any number): the stack M8f (tilt members) + B3 (biased decode start) vs M7f + U3 on T2-D442's draws; parts beside it.
Derived from t2_d456_seed.py, whose header follows. T2-D456: the stream decode started from the ensemble's class bias instead of uniform pseudo-rows.
M7's stored members, 12 draws, kit + seen1, test + val. Arms: U3 (shipped: 3 uniform pseudo-rows), U6, U10, B3, B6, B10 (pseudo-rows =
the mean raw mix row of the OTHER users in the same draw / protocol / split). Typical = non-P04; first third = by position in the stream.
    mem-guard python scripts/local/t2_d456_seed.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()


def decode(P, seed, n0, lam=0.95, iters=10):   # t2p_report.causal_sinkhorn for ONE stream, pseudo-rows = seed (K,)
    K, out = P.shape[1], np.empty_like(P)
    for t in range(len(P)):
        M = np.vstack([np.tile(seed, (n0, 1)), P[:t + 1]])
        w = lam ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
        for _ in range(iters):
            M = M * (w.sum() / K) / (M * w[:, None]).sum(0, keepdims=True)
            M = M / M.sum(1, keepdims=True)
        out[t] = M[-1]
    return out


rows = []
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
          for shape in ("M7f", "M8f"):
            mix, ix = 0.0, None
            for m, w in J["arms"][shape].items():
                M = J["members"][m]
                z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                mix = mix + w * p.astype(np.float64)
            mix = mix / mix.sum(1, keepdims=True)
            u = U[ix]
            for s_ in np.unique(ST[ix]):
                r = np.flatnonzero(ST[ix] == s_)
                r = r[np.argsort(OD[ix][r], kind="stable")]
                bias = mix[u != u[r[0]]].mean(0)
                third = np.minimum(2, 3 * np.arange(len(r)) // len(r))
                for arm, seed, n0 in [(shape + "_U3", np.full(3, 1 / 3), 3), (shape + "_B3", bias, 3)]:
                    pred = decode(mix[r], seed, n0).argmax(1)
                    rows.append(pd.DataFrame(dict(proto=proto, i=i, split=split, arm=arm, u=u[r], cell=CELL[ix][r], y=Y[ix][r], third=third, ok=pred == Y[ix][r])))
D = pd.concat(rows, ignore_index=True)
ba = D.groupby(["arm", "split", "proto", "i", "u", "cell", "y"]).ok.mean().groupby(["arm", "split", "proto", "i", "u", "cell"]).mean().reset_index(name="ba")
for split in ("val", "test", "both"):
    b = ba if split == "both" else ba[ba.split == split]
    ref = b[b.arm == "M7f_U3"].set_index(["split", "proto", "i", "u", "cell"]).ba
    for arm in ("M8f_B3", "M7f_B3", "M8f_U3"):
        a = b[b.arm == arm].set_index(["split", "proto", "i", "u", "cell"]).ba
        d = (a - ref).reset_index()
        t = d[d.u != "P04"]
        dd = t.groupby("i").ba.mean()
        f = D[(D.u != "P04") & (D.third == 0) & ((D.split == split) | (split == "both"))]
        f1 = f[f.arm == arm].ok.mean() - f[f.arm == "M7f_U3"].ok.mean()
        print(f"{split:5s} {arm:7s} vs M7f_U3: typical {dd.mean():+.4f} ({int((dd > 0).sum()):2d}/12)  first third {f1:+.4f}  "
              + " ".join(f"{k} {v:+.4f}" for k, v in d.groupby("u").ba.mean().items()), flush=True)
b = ba.set_index(["arm", "split", "proto", "i", "u", "cell"]).ba
d = (b["M8f_B3"] - b["M7f_U3"]).reset_index()
d = d[d.u != "P04"]
print("stack by protocol:", {k: (round(v.groupby(v.i).ba.mean().mean(), 4), int((v.groupby(v.i).ba.mean() > 0).sum())) for k, v in d.groupby("proto")})
