#!/usr/bin/env python
"""Order sensitivity of the causal Sinkhorn decode on M7's stored mix (the manager's ask 2026-11-10 after the sealed re-runs; zero GPU).
Only the DECODE is simulated (member probabilities are the stored ones, computed in recording order); the template's causal aligners and
per-user routing also depend on order and are not covered here. Typical = non-P04 cells; 12 draws, kit + seen1, test + val.
    mem-guard python scripts/local/t2_order_sim.py
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
rng = np.random.default_rng(0)
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
            st, od = idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]
            y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
            one = np.zeros(len(ix), int)
            glob = np.lexsort((od, st)).argsort()
            V = {"recording order (as gated)": (st, od), "no decode (argmax of the mix)": None,
                 "shuffled inside each stream": (st, rng.permutation(len(ix))),
                 "streams reversed in time": (st, -od),
                 "all streams as ONE stream, in order": (one, glob),
                 "ONE stream, fully shuffled": (one, rng.permutation(len(ix)))}
            for k, so in V.items():
                P = mix.argmax(1) if so is None else causal_sinkhorn(mix, so, 10, 3, 0.95).argmax(1)
                for c in np.unique(cells):
                    mk = cells == c
                    if subj[mk][0] == "P04":
                        continue
                    res.setdefault(k, {}).setdefault(i, []).append(np.mean([np.mean(P[mk][y[mk] == q] == q) for q in np.unique(y[mk])]))
base = np.array([np.mean(res["recording order (as gated)"][i]) for i in range(12)])
for k, r in res.items():
    v = np.array([np.mean(r[i]) for i in range(12)])
    print(f"{k:38s} typical {v.mean():.4f}  vs recording order {np.mean(v - base):+.4f} ({int((v > base).sum())} / 12 up)")
