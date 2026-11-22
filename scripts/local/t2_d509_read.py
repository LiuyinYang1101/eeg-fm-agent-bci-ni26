#!/usr/bin/env python
"""T2-D509 read (written 2026-11-15 before any D509 number): the in-stream true-label fine-tune ceiling. Per fold x seed (790-792) x N in
(30, 60, 100): the held-out user's streams decoded whole by the causal Sinkhorn (10 / 3 / 0.95, uniform start), is<N> (adapted on the stream's
first N windows) vs is0 (unadapted), cell BA on the SAME scored windows (is<N>'s mask); per user = the mean over seeds.
RULE (NOT-COVERED six P01 P03 P05 P07 P09 P10, N = 100): < +0.03 -> in-stream adaptation closes; >= +0.05 -> one registered
confident-window decoded-teacher row; else descriptive.
    mem-guard python scripts/local/t2_d509_read.py
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
D = O / "t2_d509"
idx = pd.read_csv(EXP / "index.csv")
U, STREAM, ORD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
REC = "shallow__ea__n00__r1_pulx_vp2_e180_vr"
NS, SEEDS = (30, 60, 100), (790, 791, 792)
NC6, OTH4 = ("P01", "P03", "P05", "P07", "P09", "P10"), ("P02", "P04", "P06", "P08")


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        ks = np.unique(Y[rows][m])
        if len(ks) < 2:
            continue
        out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
    return out


recs = []
for f in range(10):
    for s in SEEDS:
        z0 = np.load(D / f"runs/lc9_{f}__{REC}is0__s{s}.npz")
        te, p0 = z0["test_idx"], z0["prob_last"].astype(np.float64)
        held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
        hm = np.flatnonzero(U[te] == held)
        hm = hm[np.lexsort((ORD[te][hm], STREAM[te][hm]))]
        key = (STREAM[te][hm], ORD[te][hm])
        d0 = causal_sinkhorn(p0[hm] / p0[hm].sum(1, keepdims=True), key, 10, 3, 0.95).argmax(1)
        for n in NS:
            z = np.load(D / f"runs/lc9_{f}__{REC}is{n}__s{s}.npz")
            assert (z["test_idx"] == te).all()
            pn, sc = z["prob_last"].astype(np.float64), z["scored"]
            assert not sc[np.setdiff1d(np.arange(len(te)), hm)].any(), "only the held-out user's rows are adapted"
            dn = causal_sinkhorn(pn[hm] / pn[hm].sum(1, keepdims=True), key, 10, 3, 0.95).argmax(1)
            k = sc[hm]
            if not k.any():
                continue
            rows = te[hm][k]
            a, b = cells(d0[k], rows), cells(dn[k], rows)
            for c in a:
                recs.append(dict(fold=f, seed=s, user=held, n=n, cell=c, base=a[c], adapt=b[c], n_scored=int(k.sum())))
    print(f"[d509] fold {f} read", flush=True)
R = pd.DataFrame(recs)
R.to_csv(D / "cells.csv", index=False)
pu = R.groupby(["n", "user"])[["base", "adapt"]].mean()
pu["gain"] = pu.adapt - pu.base
res = {}
for n in NS:
    t = pu.loc[n]
    g6, g4, g10 = float(t.loc[[u for u in NC6 if u in t.index], "gain"].mean()), float(t.loc[[u for u in OTH4 if u in t.index], "gain"].mean()), float(t.gain.mean())
    res[n] = dict(nc6=round(g6, 4), oth4=round(g4, 4), all10=round(g10, 4), per_user={u: [round(float(r.base), 3), round(float(r.adapt), 3)] for u, r in t.iterrows()})
    print(f"D509 N {n:3d}: NOT-COVERED six {g6:+.4f} | other four {g4:+.4f} | all ten {g10:+.4f}   per user base->adapt: "
          + " ".join(f"{u} {r.base:.3f}->{r.adapt:.3f}" for u, r in t.iterrows()), flush=True)
g = res[100]["nc6"]
verdict = "CLOSES (< +0.03)" if g < 0.03 else ("OPENS one row (>= +0.05)" if g >= 0.05 else "descriptive (between +0.03 and +0.05)")
res["verdict"] = verdict
print(f"D509 RULE at N = 100 on the six: {g:+.4f} -> in-stream adaptation {verdict}", flush=True)
json.dump({str(k): v for k, v in res.items()}, open(D / "read.json", "w"), indent=1)
