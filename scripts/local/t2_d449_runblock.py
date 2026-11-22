#!/usr/bin/env python
"""T2-D449 (panel P8 bet B2; bars pre-registered in docs/DECISIONS_T2.md before any number): a RUN-BLOCK decode on top of today's
SK-lambda. 71 of 84 public runs are exactly class-balanced, so balance the classes inside each run. Stored M7 mix, 12 draws,
kit + seen1, test + val; typical = non-P04 cells.
  A     = SK-lambda over the session (shipped: 10 iters, n0 3, lam 0.95)
  RB2t  = PRIMARY (i): causal Sinkhorn (lam 1, n0 3) inside each TRUE run, applied to A's output
  RB1t  = the same on the raw mix (descriptive)
  RB2c  = (ii): as RB2t with runs inferred from the window count, schedule [69, 30, 30, ...] (T2-D285's QI schedule)
  RB2w  = (ii) with the wrong schedule [69, 36, 36, ...] (allocation risk, descriptive)
    mem-guard python scripts/local/t2_d449_runblock.py
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
idx = pd.read_csv(EXP / "index.csv")
ST, OD, RUN, Y, CELL, SUBJ = (idx.stream.to_numpy(), idx.order.to_numpy(), idx.run.to_numpy().astype(int), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy())


def sched(st, od, first, rest):
    seg = np.empty(len(st), dtype=object)
    for s in np.unique(st):
        m = np.flatnonzero(st == s)
        m = m[np.argsort(od[m], kind="stable")]
        pos = np.arange(len(m))
        seg[m] = [f"{s}#{0 if p < first else 1 + (p - first) // rest}" for p in pos]
    return seg


res = {}
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            mix, ix = 0.0, None
            for m, w in J["arms"]["M7"].items():
                M = J["members"][m]
                z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                mix = mix + w * p.astype(np.float64)
            st, od = ST[ix], OD[ix]
            true = np.array([f"{a}#{b}" for a, b in zip(st, RUN[ix])], dtype=object)
            A = causal_sinkhorn(mix, (st, od), 10, 3, 0.95)
            arms = {"A": A, "RB2t": causal_sinkhorn(A, (true, od), 10, 3, 1.0), "RB1t": causal_sinkhorn(mix, (true, od), 10, 3, 1.0),
                    "RB2c": causal_sinkhorn(A, (sched(st, od, 69, 30), od), 10, 3, 1.0), "RB2w": causal_sinkhorn(A, (sched(st, od, 69, 36), od), 10, 3, 1.0)}
            y, cells, sj = Y[ix], CELL[ix], SUBJ[ix]
            for k, q in arms.items():
                P = q.argmax(1)
                for c in np.unique(cells):
                    mk = cells == c
                    res.setdefault(k, []).append((i, sj[mk][0], np.mean([np.mean(P[mk][y[mk] == v] == v) for v in np.unique(y[mk])])))
D = {k: pd.DataFrame(v, columns=["i", "subj", "ba"]) for k, v in res.items()}
ty = lambda d: d[d.subj != "P04"].groupby("i").ba.mean()   # noqa: E731
for k in ("RB2t", "RB1t", "RB2c", "RB2w"):
    d = ty(D[k]) - ty(D["A"])
    u = (D[k].groupby(["subj", "i"]).ba.mean() - D["A"].groupby(["subj", "i"]).ba.mean()).groupby("subj").mean()
    print(f"{k:5s} vs A: typical {d.mean():+.4f} ({int((d > 0).sum())} / 12)  " + " ".join(f"{s} {x:+.4f}" for s, x in u.items()), flush=True)
print(f"A typical level {ty(D['A']).mean():.4f}")
