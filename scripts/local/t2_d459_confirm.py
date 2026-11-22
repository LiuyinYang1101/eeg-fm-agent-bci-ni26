#!/usr/bin/env python
"""T2-D459 (written before any number): T2-D458 on fresh CNN draws with ONE constant, cross-fitted start vector. See the row.
Shapes: C = M7's weights, CNNs from D419 / D429 control draws, FM = D442 control draws; T = the same with D442's tilt FM draws.
Start vector for a split = mean raw mix row of shape C (or T for T_B3) over the OTHER split (disjoint users), all draws and protocols.
    mem-guard python scripts/local/t2_d459_confirm.py
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
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        assert d in S[src]["kit"] and S[src]["offset"] in (324, 348), S[src]
        M[dst] = S[src]
SHAPES = {"C": J["arms"]["M7f"], "T": J["arms"]["M8f"]}
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()


def decode(P, seed, n0=3, lam=0.95, iters=10):
    K, out = P.shape[1], np.empty_like(P)
    for t in range(len(P)):
        Mx = np.vstack([np.tile(seed, (n0, 1)), P[:t + 1]])
        w = lam ** np.arange(Mx.shape[0] - 1, -1, -1, dtype=np.float64)
        for _ in range(iters):
            Mx = Mx * (w.sum() / K) / (Mx * w[:, None]).sum(0, keepdims=True)
            Mx = Mx / Mx.sum(1, keepdims=True)
        out[t] = Mx[-1]
    return out


MIX = {}
for sh, arm in SHAPES.items():
    for proto in ("kit", "seen1"):
        for i in range(12):
            for split in ("test", "val"):
                mix, ix = 0.0, None
                for m, w in arm.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or (ix == ix_).all()
                    ix, mix = ix_, mix + w * p.astype(np.float64)
                MIX[(sh, proto, i, split)] = (ix, mix / mix.sum(1, keepdims=True))
VEC = {(sh, split): np.concatenate([v[1] for k, v in MIX.items() if k[0] == sh and k[3] != split]).mean(0) for sh in SHAPES for split in ("test", "val")}
print("start vectors (mi, sub, word):", {k: np.round(v, 3).tolist() for k, v in VEC.items()})
assert not (set(U[np.concatenate([v[0] for k, v in MIX.items() if k[3] == "test"])]) & set(U[np.concatenate([v[0] for k, v in MIX.items() if k[3] == "val"])])), "users overlap"
rows = []
for (sh, proto, i, split), (ix, mix) in MIX.items():
    u = U[ix]
    for s_ in np.unique(ST[ix]):
        r = np.flatnonzero(ST[ix] == s_)
        r = r[np.argsort(OD[ix][r], kind="stable")]
        third = np.minimum(2, 3 * np.arange(len(r)) // len(r))
        for arm, seed in ((sh + "_U3", np.full(3, 1 / 3)), (sh + "_B3", VEC[(sh, split)])):
            pred = decode(mix[r], seed).argmax(1)
            rows.append(pd.DataFrame(dict(proto=proto, i=i, split=split, arm=arm, u=u[r], cell=CELL[ix][r], y=Y[ix][r], third=third, ok=pred == Y[ix][r])))
D = pd.concat(rows, ignore_index=True)
key = ["split", "proto", "i", "u", "cell"]
ba = D.groupby(["arm"] + key + ["y"]).ok.mean().groupby(["arm"] + key).mean()
print("levels, typical:", {a: round(ba[a].reset_index().query("u != 'P04'").ba.mean() if False else ba[a][ba[a].index.get_level_values("u") != "P04"].mean(), 4) for a in ("C_U3", "C_B3", "T_U3", "T_B3")})
for arm in ("C_B3", "T_B3", "T_U3"):
    d = (ba[arm] - ba["C_U3"]).rename("d").reset_index()
    t = d[d.u != "P04"]
    dd = t.groupby("i").d.mean()
    f = D[(D.u != "P04") & (D.third == 0)]
    f1 = f[f.arm == arm].ok.mean() - f[f.arm == "C_U3"].ok.mean()
    print(f"{arm} vs C_U3: typical {dd.mean():+.4f} ({int((dd > 0).sum()):2d}/12)  first third {f1:+.4f}  " + " ".join(f"{k} {v:+.4f}" for k, v in d.groupby("u").d.mean().items())
          + "  | " + " ".join(f"{k} {v.groupby(v.i).d.mean().mean():+.4f} ({int((v.groupby(v.i).d.mean() > 0).sum())})" for k, v in t.groupby("proto"))
          + "  | " + " ".join(f"{k} {v.groupby(v.i).d.mean().mean():+.4f} ({int((v.groupby(v.i).d.mean() > 0).sum())})" for k, v in t.groupby("split")), flush=True)
