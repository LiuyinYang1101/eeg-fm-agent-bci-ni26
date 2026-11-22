#!/usr/bin/env python
"""T2-D505 (registered 2026-11-15 before any number; zero GPU): the own-session gain by session, per family, for the POOLED nets, on
T2-D498's K = 9 panel. For P04 / P10 sessions 2 and 3: each family alone as a pooled net (CNN: the pu twin's prob_last = no copies, seeds
750-752; FM: the mean of the six public-checkpoint ST-L members), decoded by the causal Sinkhorn (10 / 3 / 0.95, uniform start);
'in' = the folds whose training includes the user's session 1, 'out' = the fold that held the user out; gain = in - out.
Stale family: session-3 gain <= half its session-2 gain for both users.
    mem-guard python scripts/local/t2_d505_ownsess.py
"""
import glob
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
CNN = {"fbc": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cpu", "sh": "shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cpu",
       "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cpu"}


def ba(pred, rows):
    out = []
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        out.append(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in np.unique(Y[rows][m])]))
    return float(np.mean(out))


def norm(p):
    return p / p.sum(1, keepdims=True)


recs = []
for f in range(10):
    fam = {}
    for k, stem in CNN.items():
        for s in (750, 751, 752):
            z = np.load(S / f"ni26-t2p-d498-{k}-lc9-5090/runs/lc9_{f}__{stem}__s{s}.npz")
            fam.setdefault(k, []).append((z["test_idx"], norm(z["prob_last"].astype(np.float64))))
    fm = []
    for a, seeds in (("stl", (760, 761, 762)), ("stlcar", (763, 764, 765))):
        for s in seeds:
            fs = glob.glob(f"{S}/ni26-t2p-d498-{a}_s{s}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
            assert len(fs) == 1
            z = np.load(fs[0])
            fm.append((z["test_idx"], norm(z["prob_last"].astype(np.float64))))
    fam["fm"] = [(fm[0][0], np.mean([p for _, p in fm], 0))]
    te = fm[0][0]
    held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
    for u in ("P04", "P10"):
        for ses in ("2", "3"):
            m = np.flatnonzero((U[te] == u) & (SES[te] == ses))
            if not len(m):
                continue
            m = m[np.argsort(ORD[te][m])]
            rows = te[m]
            key = (STREAM[rows], ORD[rows])
            for k, lst in fam.items():
                for ix, p in lst:
                    assert (ix == te).all()
                    recs.append(dict(fold=f, user=u, session=ses, fam=k, side="out" if u == held else "in",
                                     ba=ba(causal_sinkhorn(norm(p[m]), key, 10, 3, 0.95).argmax(1), rows)))
R = pd.DataFrame(recs)
out = O / "t2_d505"
out.mkdir(exist_ok=True)
R.to_csv(out / "cells.csv", index=False)
T = R.groupby(["fam", "user", "session", "side"]).ba.mean().unstack()
T["gain"] = T["in"] - T["out"]
print("D505 own-session gain (in - out), pooled nets, by family x user x session:", flush=True)
for (k, u, ses), r in T.iterrows():
    print(f"   {k:3s} {u} s{ses}: out {r['out']:.3f} in {r['in']:.3f} gain {r['gain']:+.4f}", flush=True)
G = T["gain"].unstack("session")
stale = {k: bool(all(G.loc[(k, u), "3"] <= 0.5 * G.loc[(k, u), "2"] for u in ("P04", "P10"))) for k in G.index.get_level_values(0).unique()}
print("D505 stale (session-3 gain <= half the session-2 gain for both users): " + " ".join(f"{k} {v}" for k, v in stale.items()), flush=True)
T.to_csv(out / "table.csv")
