#!/usr/bin/env python
"""T2-D448 (zero GPU; the organisers removed 'strictly chronological within each run' from the T2 FAQ on Nov 10): which DELIVERED ORDER
fits our re-run drops, and which decode setting is best under each order? Decode only, on stored member probabilities (the members'
own causal aligners ran in recording order and are not re-run here). Arms: M7 and S1 (the uploaded shapes). Typical = non-P04 cells;
W50-style 'all' = all cells; 12 draws, kit + seen1, test + val; shuffles averaged over 3 random repetitions.
Orders: chrono | runs_perm (runs of a session in random order, windows in time order inside) | in_run (runs in order, windows shuffled
inside each run) | in_session (all windows of a session shuffled).
    mem-guard python scripts/local/t2_d448_orderfit.py
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
J = json.load(open(O / "t2_d408/spec_m7.json"))
ARMS = {"M7": J["arms"]["M7"], "S1": {"fbc_pvtv8c": 4 / 9, "sh_pvtv8c": 3 / 9, "stl_rcll": 2 / 9}}
idx = pd.read_csv(EXP / "index.csv")
ST, OD, RUN, Y, CELL, SUBJ = (idx.stream.to_numpy(), idx.order.to_numpy(), idx.run.to_numpy().astype(int), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy())
LAMS = (0.90, 0.95, 0.98, 1.0)
REPS = 3


def keys(ix, kind, rng):
    st, od, run = ST[ix], OD[ix], RUN[ix]
    if kind == "chrono":
        return od.astype(float)
    k = np.empty(len(ix))
    for s in np.unique(st):
        m = np.flatnonzero(st == s)
        if kind == "in_session":
            k[m] = rng.permutation(len(m))
            continue
        runs = np.unique(run[m])
        pos = dict(zip(runs, rng.permutation(len(runs)) if kind == "runs_perm" else np.arange(len(runs))))
        for r in runs:
            mr = m[run[m] == r]
            inner = rng.permutation(len(mr)) if kind == "in_run" else np.argsort(np.argsort(od[mr]))
            k[mr] = pos[r] * 1000 + inner
    return k


def typ(P, ix):
    y, cells, sj = Y[ix], CELL[ix], SUBJ[ix]
    t, a = [], []
    for c in np.unique(cells):
        mk = cells == c
        b = np.mean([np.mean(P[mk][y[mk] == q] == q) for q in np.unique(y[mk])])
        a.append(b)
        if sj[mk][0] != "P04":
            t.append(b)
    return np.mean(t), np.mean(a)


res = {}
for arm, W in ARMS.items():
    for proto in ("kit", "seen1"):
        for i in range(12):
            for split in ("test", "val"):
                mix, ix = 0.0, None
                for m, w in W.items():
                    M = J["members"][m]
                    z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                    ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    mix = mix + w * p.astype(np.float64)
                for kind in ("chrono", "runs_perm", "in_run", "in_session"):
                    for rep in range(1 if kind == "chrono" else REPS):
                        k = keys(ix, kind, np.random.default_rng(1000 * i + rep))
                        for lam in LAMS:
                            t, a = typ(causal_sinkhorn(mix, (ST[ix], k), 10, 3, lam).argmax(1), ix)
                            res.setdefault((arm, kind, lam), {}).setdefault(i, []).append((t, a))
for arm in ARMS:
    ref = np.array([np.mean([x[0] for x in res[(arm, "chrono", 0.95)][i]]) for i in range(12)])
    refa = np.array([np.mean([x[1] for x in res[(arm, "chrono", 0.95)][i]]) for i in range(12)])
    print(f"\n== {arm}: typical (all cells) vs chronological at the shipped lam 0.95 [{ref.mean():.4f} ({refa.mean():.4f})]; draws up of 12")
    for kind in ("chrono", "runs_perm", "in_run", "in_session"):
        out = f"  {kind:11s}"
        for lam in LAMS:
            v = np.array([np.mean([x[0] for x in res[(arm, kind, lam)][i]]) for i in range(12)])
            va = np.array([np.mean([x[1] for x in res[(arm, kind, lam)][i]]) for i in range(12)])
            out += f" | lam {lam:.2f}: {np.mean(v - ref):+.4f} ({np.mean(va - refa):+.4f}) {int((v > ref).sum()):2d}"
        print(out, flush=True)
