#!/usr/bin/env python
"""T2-D506 (A0; registered 2026-11-15 08:52 before any number; zero GPU): each user's level as a STRANGER beside its level with its OWN
labels, on the same windows and the same model shape. CNN mix = FBCNet 0.5714 + ShallowNet 0.4286 (M7's ratio), shipped doses (FBCNet
r 1/3 = fb3, ShallowNet r 1/2 = fb2) and r 0 (the pu twin's prob_last); each family decoded by the causal Sinkhorn (10 / 3 / 0.95,
uniform start) then pooled; D498's K = 9 files, draws 750-752.
  L_str     = held-out stranger, session 1 runs 5-6, decoded as their own streams (as T2-D499 scores them)
  L_str_all = held-out stranger, all its windows, full streams
  L_own     = T2-D499 same-day (sd46) at the shipped dose (outputs/t2_d499/cells.csv)
  L_xday    = P04 / P10 sessions 2-3 in the nine folds that trained their session 1
Rule: weak signal if L_own < 0.75; not covered if L_own >= 0.75 and L_own - L_str >= +0.08; else covered.
Prize = mean over the ten of max(0, L_own - L_str) for the not-covered users x 0.57 (sealed new-user share); < +0.012 -> row B first.
    mem-guard python scripts/local/t2_d506_coverage.py
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
S, R = O / "sofia_s1", O / "t2_d498"
idx = pd.read_csv(EXP / "index.csv")
U, SES, RUN = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.run.to_numpy()
STREAM, ORD, Y, CELL = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8c", 0.5714, "3", "fbc"),
       "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vrtv8c", 0.4286, "2", "sh")}
DRAWS = (750, 751, 752)


def load(f, key="prob_last"):
    z = np.load(f)
    p = z[key].astype(np.float64)
    return z["test_idx"], p / p.sum(1, keepdims=True)


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in np.unique(Y[rows][m])]))
    return out


def mix(P, arm, m, rows):
    key = (STREAM[rows], ORD[rows])
    return sum(w * causal_sinkhorn(P[(f, arm)][m], key, 10, 3, 0.95) for f, (_, w, _, _) in FAM.items()).argmax(1)


recs = []
for f in range(10):
    for d in DRAWS:
        P, ix = {}, None
        for fam, (stem, _, tg, rd) in FAM.items():
            ix0, P[(fam, "r0")] = load(S / f"ni26-t2p-d498-{fam}-lc9-5090/runs/lc9_{f}__{stem}pu__s{d}.npz")
            ix1, P[(fam, "shipped")] = load(R / f"route_{rd}_lc9/lc9_{f}__{stem}fb{tg}__s{d}.npz")
            assert (ix0 == ix1).all() and (ix is None or (ix == ix0).all())
            ix = ix0
        held = [u for u in set(U[ix]) if (U[ix] == u).sum() == (U == u).sum()][0]
        for arm in ("r0", "shipped"):
            for col, sel in (("L_str", (U[ix] == held) & (SES[ix] == "1") & (RUN[ix] >= 5)), ("L_str_all", U[ix] == held)):
                m = np.flatnonzero(sel)
                m = m[np.lexsort((ORD[ix][m], STREAM[ix][m]))]
                for c, v in cells(mix(P, arm, m, ix[m]), ix[m]).items():
                    recs.append(dict(fold=f, draw=d, user=held, col=col, arm=arm, cell=c, ba=v))
            for u in ("P04", "P10"):
                if u == held:
                    continue
                m = np.flatnonzero((U[ix] == u) & np.isin(SES[ix], ("2", "3")))
                m = m[np.lexsort((ORD[ix][m], STREAM[ix][m]))]
                for c, v in cells(mix(P, arm, m, ix[m]), ix[m]).items():
                    recs.append(dict(fold=f, draw=d, user=u, col="L_xday", arm=arm, cell=c, ba=v))
    print(f"[d506] fold {f} done", flush=True)
C = pd.DataFrame(recs)
out = O / "t2_d506"
out.mkdir(exist_ok=True)
C.to_csv(out / "cells.csv", index=False)
T = C.groupby(["user", "col", "arm"]).ba.mean().unstack(["col", "arm"])
own = pd.read_csv(O / "t2_d499/cells.csv").groupby(["user", "arm"]).ba.mean().unstack()
st = json.load(open(R / "read.json"))["i"]["stranger_per_user"]
tab = pd.DataFrame({"stack_str": {u: st[u]["stack"] for u in st}, "L_str_all": T[("L_str_all", "shipped")], "L_str": T[("L_str", "shipped")],
                    "L_str_r0": T[("L_str", "r0")], "L_own": own["shipped"], "L_own_r0": own["r0"]})
tab["L_xday"] = T[("L_xday", "shipped")] if ("L_xday", "shipped") in T else np.nan
tab["gap"] = tab.L_own - tab.L_str
tab["class"] = np.where(tab.L_own < 0.75, "weak signal", np.where(tab.gap >= 0.08, "not covered", "covered"))
prize_new = float(np.where(tab["class"] == "not covered", tab.gap.clip(lower=0), 0).mean())
res = dict(table=tab.round(4).reset_index().rename(columns={"index": "user"}).to_dict("records"), mean_L_str=round(float(tab.L_str.mean()), 4),
           mean_L_own=round(float(tab.L_own.mean()), 4), prize_new_users=round(prize_new, 4), prize_sealed=round(prize_new * 0.57, 4))
res["order"] = "B before A1" if res["prize_sealed"] < 0.012 else "A1 before B"
print("D506 per user (CNN mix; shipped dose unless r0):\n" + tab.round(3).to_string(), flush=True)
print(f"D506 ten-user mean: L_str {res['mean_L_str']:.4f} | L_own {res['mean_L_own']:.4f} | all ten at own level {tab.gap.mean():+.4f}; "
      f"coverage prize (not-covered users only) {prize_new:+.4f} on new users = {res['prize_sealed']:+.4f} sealed -> {res['order']}", flush=True)
json.dump(res, open(out / "read.json", "w"), indent=1)
