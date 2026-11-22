#!/usr/bin/env python
"""T2-D422 (pre-registered in docs/DECISIONS_T2.md before any code or D422 number; zero GPU; backlog 111): D417's dose-a ST-L member at a re-picked share,
cross-fitted on the metric (D416b's machinery).
  members / seeds = outputs/t2_d417/spec.json; anchor M7c (c0 at 0.07); candidates M7a(t): lp10 -> stl_a at t in T, e100f0-f3 at (0.35 - t) / 4 each;
  objective on a protocol = mean over the 12 draws of the cell-mean BA over all its cells (test + val) under causal SKλ (10 it, n0 3, λ 0.95); ties -> t nearest 0.07;
  cross-fit: kit cells at the seen1-argmax t, seen1 cells at the kit-argmax t; vs M7c on the same cells.
  BAR: typical >= +0.003 with >= 9 / 12 AND W50 >= 0 AND no user below -0.020 -> T2-D422r (fresh seeds 120-131) before any build.
    mem-guard python scripts/local/t2_d422_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
R = O / "t2_d422"
R.mkdir(exist_ok=True)
idx = pd.read_csv(E / "index.csv")
STRM, ORDR, Y, CELL, SUBJ = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy()
N = 12
T = (0.07, 0.10, 0.14, 0.20)
S = json.load(open(O / "t2_d417/spec.json"))
W0 = S["arms"]["M7c"]
E100 = ["e100f0", "e100f1", "e100f2", "e100f3"]
assert abs(W0["stl_c0"] - 0.07) < 1e-9 and all(abs(W0[m] - 0.07) < 1e-9 for m in E100) and abs(sum(W0.values()) - 1) < 1e-9


def arm(t):
    w = {k: v for k, v in W0.items() if k != "stl_c0"}
    w["stl_a"] = t
    for m in E100:
        w[m] = (0.35 - t) / 4
    assert abs(sum(w.values()) - 1) < 1e-9
    return w


MEM = sorted(set(W0) | {"stl_a"})
D = {}
for proto in ("kit", "seen1"):
    for i in range(N):
        for split in ("test", "val"):
            ps, ix = {}, None
            for m in MEM:
                M = S["members"][m]
                z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all()), (m, proto, i, split)
                ix, ps[m] = ix_, p.astype(np.float64)
            D[(proto, i, split)] = (ix, ps)


def cell_ba(key, w):
    ix, ps = D[key]
    P = causal_sinkhorn(sum(v * ps[m] for m, v in w.items()), (STRM[ix], ORDR[ix]), 10, 3, 0.95).argmax(1)
    out = []
    for c in np.unique(CELL[ix]):
        mk = CELL[ix] == c
        yy = Y[ix][mk]
        out.append((c, SUBJ[ix][mk][0], float(np.mean([np.mean(P[mk][yy == k] == k) for k in np.unique(yy)]))))
    return out


def objective(w, proto):
    return float(np.mean([np.mean([b for s in ("test", "val") for (_, _, b) in cell_ba((proto, i, s), w)]) for i in range(N)]))


def cells_df(w_by_proto):
    return pd.DataFrame([dict(proto=p, split=s, i=i, cell=c, subj=u, ba=b) for (p, i, s) in D for c, u, b in cell_ba((p, i, s), w_by_proto[p])])


def compare(A, B):
    Dd = A.merge(B, on=["proto", "split", "i", "cell", "subj"], suffixes=("_a", "_b"))
    Dd["d"] = Dd.ba_b - Dd.ba_a
    out = {}
    typ = Dd[Dd.subj.isin(["P05", "P08", "P09", "P10"])].groupby("i").d.mean()
    out["typical"], out["typical_up"] = float(typ.mean()), int((typ > 0).sum())
    for g, mk in (("P04", Dd.subj == "P04"), ("P10", Dd.subj == "P10"), ("newT", Dd.subj.isin(["P05", "P08", "P09"]))):
        out[g] = float(Dd[mk].groupby("i").d.mean().mean())
    out["users"] = {u: float(v) for u, v in Dd.groupby(["subj", "i"]).d.mean().groupby("subj").mean().items()}
    ret = Dd[(Dd.subj == "P10") | ((Dd.subj == "P04") & (Dd.proto == "seen1"))]
    new = Dd[Dd.subj.isin(["P05", "P08", "P09"]) | ((Dd.subj == "P04") & (Dd.proto == "kit"))]
    w50 = 0.5 * ret.groupby("i").d.mean() + 0.5 * new.groupby("i").d.mean()
    out.update(W50=float(w50.mean()), W50_up=int((w50 > 0).sum()))
    return out


if __name__ == "__main__":
    obj = {p: {t: objective(arm(t), p) for t in T} for p in ("kit", "seen1")}
    obj0 = {p: objective(W0, p) for p in ("kit", "seen1")}
    for p in ("kit", "seen1"):
        print(f"[D422] {p} objective: M7c {obj0[p]:.4f} · " + " · ".join(f"t {t:.2f} {obj[p][t]:.4f}" for t in T))

    def argmax(p):
        best = max(obj[p].values())
        return min((t for t in T if obj[p][t] >= best - 1e-12), key=lambda t: abs(t - 0.07))
    tk, ts_ = argmax("kit"), argmax("seen1")
    both = {t: (obj["kit"][t] + obj["seen1"][t]) / 2 for t in T}
    tb = min((t for t in T if both[t] >= max(both.values()) - 1e-12), key=lambda t: abs(t - 0.07))
    print(f"[D422] argmax t: kit {tk:.2f}, seen1 {ts_:.2f}, both {tb:.2f}")
    base = cells_df({"kit": W0, "seen1": W0})
    r = compare(base, cells_df({"kit": arm(ts_), "seen1": arm(tk)}))
    print(f"== CROSS-FIT vs M7c: typical {r['typical']:+.4f} ({r['typical_up']} / 12), P04 {r['P04']:+.4f}, P10 {r['P10']:+.4f}, newT {r['newT']:+.4f}, "
          f"W50 {r['W50']:+.4f} ({r['W50_up']} / 12); users {', '.join(f'{u} {v:+.4f}' for u, v in sorted(r['users'].items()))}")
    ok = r["typical"] >= 0.003 and r["typical_up"] >= 9 and r["W50"] >= 0 and min(r["users"].values()) >= -0.020
    json.dump(dict(T=T, obj=obj, obj_M7c=obj0, t_kit=tk, t_seen1=ts_, t_both=tb, xfit=r, passed=bool(ok)), open(R / "read.json", "w"), indent=1, default=str)
    print(f"== VERDICT (pre-registered): {'PASS -> T2-D422r (fresh seeds 120-131) at t ' + f'{tb:.2f}' if ok else 'NOT MET'}")
