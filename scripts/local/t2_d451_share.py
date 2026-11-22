#!/usr/bin/env python
"""T2-D451 (pre-registered before any number): M7's group shares re-read on the SEALED-LIKE metric (typical = non-P04 cells; T2-D450),
because T2-D416b's 'shares are at the optimum' was fitted on ALL cells, where half of the kit cells are P04's.
Grid: FM share f in 0.25 .. 0.75 (five ST-L members, equal), IFNet share 0.0 / 0.1 / 0.2, FBCNet : ShallowNet = 4 : 3 of the rest.
SELECT on M7's own ST-L seeds (argmax typical); JUDGE that point on the FRESH ST-L seeds of T2-D442 (M7f), vs the shipped shares there.
    mem-guard python scripts/local/t2_d451_share.py
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
J = json.load(open(O / "t2_d442/spec.json"))
MEM = J["members"]
SETS = {"own": ["e100f0", "e100f1", "e100f2", "e100f3", "lp10"], "fresh": ["n_e100f0", "n_e100f1", "n_e100f2", "n_e100f3", "n_lp10"]}
idx = pd.read_csv(EXP / "index.csv")
ST, OD, Y, CELL, SUBJ = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy()
P = {}


def grp(names, proto, i, split):
    out, ix = 0.0, None
    for m in names:
        z = np.load(MEM[m][proto].format(s=MEM[m].get("offset", 0) + i))
        ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
        out = out + p.astype(np.float64) / len(names)
    return ix, out


GRID = [(f, n) for f in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75) for n in (0.0, 0.1, 0.2) if f + n < 0.95]
res = {}
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            ix, fb = grp(["fbc_tv8c"], proto, i, split)
            _, sh = grp(["sh_tv8c"], proto, i, split)
            _, ifn = grp(["ifn"], proto, i, split)
            y, cells, sj = Y[ix], CELL[ix], SUBJ[ix]
            for sname, names in SETS.items():
                _, st = grp(names, proto, i, split)
                for f, n in GRID:
                    r = 1 - f - n
                    pred = causal_sinkhorn(r * 4 / 7 * fb + r * 3 / 7 * sh + n * ifn + f * st, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1)
                    for c in np.unique(cells):
                        mk = cells == c
                        res.setdefault((sname, f, n), []).append((i, sj[mk][0], np.mean([np.mean(pred[mk][y[mk] == q] == q) for q in np.unique(y[mk])])))
D = {k: pd.DataFrame(v, columns=["i", "subj", "ba"]) for k, v in res.items()}
ty = lambda d: d[d.subj != "P04"].groupby("i").ba.mean()   # noqa: E731
for sname in SETS:
    ref = D[(sname, 0.35, 0.1)]
    print(f"\n== ST-L seeds: {sname}; typical delta vs the shipped shares (FM 0.35, IFNet 0.10) [level {ty(ref).mean():.4f}]; draws up; then P05 P08 P09 P10 | P04")
    for f, n in GRID:
        d = D[(sname, f, n)]
        dd = ty(d) - ty(ref)
        u = (d.groupby(["subj", "i"]).ba.mean() - ref.groupby(["subj", "i"]).ba.mean()).groupby("subj").mean()
        print(f"  FM {f:.2f} IFNet {n:.1f} (FBC {(1 - f - n) * 4 / 7:.3f} SH {(1 - f - n) * 3 / 7:.3f}): {dd.mean():+.4f} ({int((dd > 0).sum()):2d}/12)  "
              + " ".join(f"{u[s]:+.4f}" for s in ("P05", "P08", "P09", "P10")) + f" | {u['P04']:+.4f}", flush=True)
best = max(GRID, key=lambda g: ty(D[("own", *g)]).mean())
d, ref = D[("fresh", *best)], D[("fresh", 0.35, 0.1)]
dd = ty(d) - ty(ref)
print(f"\nSELECTED on own seeds: FM {best[0]} IFNet {best[1]}  ->  JUDGED on fresh seeds: typical {dd.mean():+.4f} ({int((dd > 0).sum())}/12)")
