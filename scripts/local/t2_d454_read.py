#!/usr/bin/env python
"""T2-D454 read (written before any number). Kit protocol only (held-out P04 P05 P08 P09; typical = P05 P08 P09), seeds 460-471.
Members: ctl / mi / math / word first stage x FT (shipped 30-epoch fine-tune) / LP (head only). Votes = equal means:
  V_ft = the three experts (FT);  V_lp = the three experts (LP);  C3 = three control restarts (draws i, i+1, i+2: the same net count).
Reported under causal SK: each member alone; V_ft and V_lp vs C3; and M7's shape on the kit cells with three of its five ST-L members
(e100f0-2) replaced by the vote vs by C3.  BAR: V_ft or V_lp vs C3 alone typical >= +0.010 with >= 9 / 12, and in M7's shape >= +0.003 (>= 9 / 12).
    mem-guard python scripts/local/t2_d454_read.py
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
J = json.load(open(O / "t2_d408/spec_m7.json"))
idx = pd.read_csv(EXP / "index.csv")
ST, OD, Y, CELL, SUBJ = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy()
F = {}
for e in ("ctl", "mi", "math", "word"):
    ds = glob.glob(f"{O}/sofia_s1/ni26-t2p-d454-{e}-[0-9]*")
    assert len(ds) == 1, (e, ds)
    for f in glob.glob(f"{ds[0]}/runs/kit__fm-*__s*.npz"):
        F[(e, "LP" if "lp29" in Path(f).name else "FT", int(f.rsplit("__s", 1)[1][:-4]) - 460)] = f


def P(key, i, split):
    z = np.load(F[key + (i % 12,)]) if isinstance(key, tuple) else None
    return (z["test_idx"], z["prob_last"].astype(np.float64)) if split == "test" else (z["val_idx"], z["prob_val_last"].astype(np.float64))


def m7part(m, i, split):
    M = J["members"][m]
    z = np.load(M["kit"].format(s=M.get("offset", 0) + i))
    return z["prob_last"].astype(np.float64) if split == "test" else z["prob_val_last"].astype(np.float64)


ARMS = {f"{e}_{k}": [((e, k), 0)] for e in ("ctl", "mi", "math", "word") for k in ("FT", "LP")}
ARMS.update({"V_ft": [((e, "FT"), 0) for e in ("mi", "math", "word")], "V_lp": [((e, "LP"), 0) for e in ("mi", "math", "word")],
             "C3": [(("ctl", "FT"), d) for d in (0, 1, 2)], "C3_lp": [(("ctl", "LP"), d) for d in (0, 1, 2)]})
rows = {}
for i in range(12):
    for split in ("test", "val"):
        base = sum(w * m7part(m, i, split) for m, w in J["arms"]["M7"].items() if m not in ("e100f0", "e100f1", "e100f2"))
        for a, parts in ARMS.items():
            ix, p = None, 0.0
            for key, d in parts:
                ix, q = P(key, i + d, split)
                p = p + q / len(parts)
            for shape, mix in (("alone", p), ("M7", base + 0.21 * p)):
                pred = causal_sinkhorn(mix, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1)
                y, cells, sj = Y[ix], CELL[ix], SUBJ[ix]
                for c in np.unique(cells):
                    mk = cells == c
                    rows.setdefault((shape, a), []).append((i, sj[mk][0], np.mean([np.mean(pred[mk][y[mk] == v] == v) for v in np.unique(y[mk])])))
D = {k: pd.DataFrame(v, columns=["i", "subj", "ba"]) for k, v in rows.items()}
ty = lambda d: d[d.subj != "P04"].groupby("i").ba.mean()   # noqa: E731
for a in ARMS:
    print(f"alone {a:8s} typical {ty(D[('alone', a)]).mean():.4f}  " + " ".join(f"{s} {v:.3f}" for s, v in D[("alone", a)].groupby("subj").ba.mean().items()))
for shape in ("alone", "M7"):
    for a, b in (("V_ft", "C3"), ("V_lp", "C3"), ("V_lp", "C3_lp"), ("mi_FT", "ctl_FT"), ("math_FT", "ctl_FT"), ("word_FT", "ctl_FT")):
        d = ty(D[(shape, a)]) - ty(D[(shape, b)])
        u = (D[(shape, a)].groupby(["subj", "i"]).ba.mean() - D[(shape, b)].groupby(["subj", "i"]).ba.mean()).groupby("subj").mean()
        print(f"{shape:5s} {a:8s} vs {b:6s}: typical {d.mean():+.4f} ({int((d > 0).sum())}/12)  " + " ".join(f"{s} {v:+.4f}" for s, v in u.items()), flush=True)
