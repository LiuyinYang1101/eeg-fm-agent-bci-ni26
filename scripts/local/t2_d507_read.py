#!/usr/bin/env python
"""T2-D507 read (written 2026-11-15 before any D507 number): Scherer2015 in the like-for-like form, ShallowNet pooled net alone, K = 9 panel.
Per arm x fold x seed (780-785): the held-out stranger's windows decoded whole by the causal Sinkhorn (10 / 3 / 0.95, uniform start), cell BA;
per user = the mean over its fold's six seeds; Delta = arm - C (paired by seed). Returning (descriptive): P04 / P10 sessions 2-3 in the nine
folds that trained their session 1. BAR (amended before any number): NOT-COVERED six (P01 P03 P05 P07 P09 P10) mean Delta >= +0.020 AND the
other four (P02 P04 P06 P08) >= -0.003; hard four (P01 P03 P04 P10) descriptive.
    mem-guard python scripts/local/t2_d507_read.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
idx = pd.read_csv(EXP / "index.csv")
U, SES, STREAM, ORD, Y, CELL = (idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy(), idx.cell.to_numpy())
REC = "shallow__ea__n00__r1_pulx_vp2_e180_vr"
XS = "__xscherer2015_v1"
ARMS = {"c0": "", "c": "__car", "s": f"{XS}__car", "h1": f"{XS}__car__xdh", "h25": f"{XS}__car__xdhw025", "h25b": f"{XS}__car__xdhw025b"}
SEEDS = range(780, 786)
NC6, OTH4, HARD4 = ("P01", "P03", "P05", "P07", "P09", "P10"), ("P02", "P04", "P06", "P08"), ("P01", "P03", "P04", "P10")


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in np.unique(Y[rows][m])]))
    return out


def dec(p, rows):
    p = p / p.sum(1, keepdims=True)
    return causal_sinkhorn(p, (STREAM[rows], ORD[rows]), 10, 3, 0.95).argmax(1)


recs = []
for arm, xt in ARMS.items():
    d = glob.glob(str(S / f"ni26-t2p-d507-{arm}-[0-9]*"))
    assert len(d) == 1, (arm, d)
    for f in range(10):
        for s in SEEDS:
            z = np.load(f"{d[0]}/runs/lc9_{f}__{REC}{xt}__s{s}.npz")
            te, p = z["test_idx"], z["prob_last"].astype(np.float64)
            held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
            for kind, sel in [("stranger", U[te] == held)] + [("returning", (U[te] == u) & np.isin(SES[te], ("2", "3"))) for u in ("P04", "P10") if u != held]:
                m = np.flatnonzero(sel)
                m = m[np.lexsort((ORD[te][m], STREAM[te][m]))]
                for c, v in cells(dec(p[m], te[m]), te[m]).items():
                    recs.append(dict(arm=arm, fold=f, seed=s, kind=kind, user=U[te[m]][0], cell=c, ba=v))
    print(f"[d507] arm {arm} read", flush=True)
R = pd.DataFrame(recs)
out = O / "t2_d507"
out.mkdir(exist_ok=True)
R.to_csv(out / "cells.csv", index=False)
us = R.groupby(["kind", "arm", "seed", "user"]).ba.mean().unstack("arm")          # per kind x seed x user
res = {"stranger": {}, "returning": {}}
st = us.loc["stranger"]
lev = st.groupby("user").mean()
print("D507 stranger level per user (ShallowNet pooled, causal SK):\n" + lev.round(3).to_string(), flush=True)
for arm in ARMS:
    if arm == "c":
        continue
    dl = (st[arm] - st["c"]).groupby("user").mean()
    by_seed = (st[arm] - st["c"]).unstack("user")
    g = {k: float(dl[list(v)].mean()) for k, v in (("nc6", NC6), ("oth4", OTH4), ("hard4", HARD4))}
    up = int((by_seed[list(NC6)].mean(1) > 0).sum())
    passed = bool(arm not in ("c0",) and g["nc6"] >= 0.020 and g["oth4"] >= -0.003)
    rt = us.loc["returning"]
    rd = (rt[arm] - rt["c"]).groupby("user").mean()
    res["stranger"][arm] = dict(per_user={u: round(float(v), 4) for u, v in dl.items()}, **{k: round(v, 4) for k, v in g.items()},
                                nc6_seeds_up=up, passed=passed)
    res["returning"][arm] = {u: round(float(v), 4) for u, v in rd.items()}
    print(f"D507 {arm:4s} vs C: NOT-COVERED six {g['nc6']:+.4f} ({up} / 6 seeds up) | other four {g['oth4']:+.4f} | hard four (descr.) {g['hard4']:+.4f} | "
          f"returning P04 {rd.get('P04', np.nan):+.4f} P10 {rd.get('P10', np.nan):+.4f} | " + ("PASS" if passed else "no pass") + "\n   per user: "
          + " ".join(f"{u} {v:+.3f}" for u, v in dl.items()), flush=True)
best = max((a for a in res["stranger"] if a != "c0"), key=lambda a: res["stranger"][a]["nc6"])
res["verdict"] = dict(best=best, any_pass=any(v["passed"] for a, v in res["stranger"].items() if a != "c0"))
print(f"D507 VERDICT: best arm {best} (NOT-COVERED six {res['stranger'][best]['nc6']:+.4f}); "
      + ("an arm PASSES -> replicate on seeds 786-791" if res["verdict"]["any_pass"] else "NO arm passes the bar"), flush=True)
json.dump(res, open(out / "read.json", "w"), indent=1)
