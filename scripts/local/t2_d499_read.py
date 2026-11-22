#!/usr/bin/env python
"""T2-D499 read (written 2026-11-14 before any D499 number): the same-day returning panel (sd46: session 1 runs 1-4 train, runs 5-6
scored; every user returning at session distance 0). CNN mix = FBCNet 0.5714 + ShallowNet 0.4286 (M7's ratio), each family decoded by the
causal Sinkhorn (10 it, n0 3, lambda 0.95, uniform start) then pooled; per user = the mean of its cells over seeds 770-772.
Arms: r0 (the pooled net = the pu twin's prob_last), the same r in both families at 1/3, 1/2, 2/3, 1 (the routed fb3 / fb2 / fb6 / fb1 files),
'shipped' (FBCNet 1/3 + ShallowNet 1/2) and D462's 'm2' (FBCNet 2/3 + ShallowNet 1).
VALIDITY (Track1's kill): (a) P04: r1 > r1/3 > r0 AND P10: |r1 - r1/3| <= 0.005; (b) NOT every user above 0.90 at 'shipped'.
If valid: the ten-user dose curve (mean, typical = non-P04), each user's best same-r arm, the share of users whose best r >= 1.
    mem-guard python scripts/local/t2_d499_read.py
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
R = O / "t2_d499"
idx = pd.read_csv(EXP / "index.csv")
U, STREAM, ORD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8c", 0.5714), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vrtv8c", 0.4286)}
TAG = {"r1/3": "3", "r1/2": "2", "r2/3": "6", "r1": "1"}
SEEDS = (770, 771, 772)


def load(f, key="prob_last"):
    z = np.load(f)
    p = z[key].astype(np.float64)
    return z["test_idx"], p / p.sum(1, keepdims=True)


rows = []
for s in SEEDS:
    P = {}
    for fam, (stem, _) in FAM.items():
        ix, P[(fam, "r0")] = load(O / f"sofia_s1/ni26-t2p-d499-{fam}-sd46-5090/runs/sd46__{stem}pu__s{s}.npz")
        for a, tg in TAG.items():
            ix_, P[(fam, a)] = load(R / f"route_{fam}/sd46__{stem}fb{tg}__s{s}.npz")
            assert (ix_ == ix).all()
    key = (STREAM[ix], ORD[ix])
    D = {k: causal_sinkhorn(v, key, 10, 3, 0.95) for k, v in P.items()}
    arms = {a: {"fbc": a, "sh": a} for a in ("r0", *TAG)}
    arms["shipped"], arms["m2"] = {"fbc": "r1/3", "sh": "r1/2"}, {"fbc": "r2/3", "sh": "r1"}
    for a, pick in arms.items():
        pr = sum(FAM[f][1] * D[(f, pick[f])] for f in FAM).argmax(1)
        for c in np.unique(CELL[ix]):
            mk = CELL[ix] == c
            rows.append(dict(seed=s, arm=a, cell=c, user=U[ix][mk][0],
                             ba=float(np.mean([np.mean(pr[mk][Y[ix][mk] == q] == q) for q in np.unique(Y[ix][mk])]))))
C = pd.DataFrame(rows)
C.to_csv(R / "cells.csv", index=False)
pu = C.groupby(["user", "arm"]).ba.mean().unstack()
order = ["r0", "r1/3", "r1/2", "r2/3", "r1", "shipped", "m2"]
print("D499 per user (CNN mix, same-day returning):", flush=True)
for u in pu.index:
    print(f"   {u}: " + " ".join(f"{a} {pu.loc[u, a]:.3f}" for a in order), flush=True)
p4, p10 = pu.loc["P04"], pu.loc["P10"]
va = bool(p4["r1"] > p4["r1/3"] > p4["r0"]) and bool(abs(p10["r1"] - p10["r1/3"]) <= 0.005)
vb = not bool((pu["shipped"] > 0.90).all())
print(f"D499 VALIDITY (a) P04 r1 {p4['r1']:.3f} > r1/3 {p4['r1/3']:.3f} > r0 {p4['r0']:.3f} and P10 |r1 - r1/3| {abs(p10['r1'] - p10['r1/3']):.4f} <= 0.005 -> {va}; "
      f"(b) not every user > 0.90 at shipped ({int((pu['shipped'] > 0.90).sum())} / {len(pu)} above) -> {vb}", flush=True)
res = dict(per_user={u: {a: round(float(pu.loc[u, a]), 4) for a in order} for u in pu.index}, valid_a=va, valid_b=vb)
if va and vb:
    typ = pu.drop("P04", errors="ignore")
    curve = {a: round(float(typ[a].mean()), 4) for a in order}
    same = ["r0", "r1/3", "r1/2", "r2/3", "r1"]
    best = pu[same].idxmax(1)
    share = float(np.mean([same.index(b) >= same.index("r1") for b in best]))
    res.update(curve_typical=curve, best_r=best.to_dict(), share_best_r1=share)
    print("D499 VALID. dose curve, typical (non-P04): " + " ".join(f"{a} {v:.4f}" for a, v in curve.items())
          + f"; m2 - shipped {curve['m2'] - curve['shipped']:+.4f}; best r per user " + " ".join(f"{u}:{b}" for u, b in best.items())
          + f"; share with best r = 1: {share:.2f}", flush=True)
else:
    print("D499 NOT VALID as a returning-side instrument (Track1's kill): nothing else is read", flush=True)
json.dump(res, open(R / "read.json", "w"), indent=1)
