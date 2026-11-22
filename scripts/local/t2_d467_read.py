#!/usr/bin/env python
"""T2-D467 read (written before any number). Base = T2-D459's fresh shape C (M7's weights; fresh CNN draws D419 / D429; D442 control FM).
(a) each classical member alone (decoded, typical) and its error overlap with C: share of C's wrong windows it gets right, and the
    correlation of the two correctness indicators (typical cells);
(b) the BLOCK = the equal mean of fbcsp, mdm, xdts, fbts, as ONE member at w 0.10 (primary) / 0.05 / 0.15 (descriptive): C x (1 - w) + w block.
BAR (w 0.10): typical >= +0.003, >= 9 / 12, no typical user < -0.005; P09 apart.
    mem-guard python scripts/local/t2_d467_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = S[src]
CL = {"fbcsp": O / "t2_d467/runs", "mdm": O / "t2_d467/runs", "xdts": O / "t2_d467/runs", "fbts": O / "t2_d328/runs"}
for k, d in CL.items():
    M[k] = {"offset": 0, **{p: str(d / f"{p}__{k}__s{{s}}.npz") for p in ("kit", "seen1")}}
C = J["arms"]["M7f"]
blk = {k: 0.25 for k in CL}
arms = {k: {k: 1.0} for k in CL}
arms["block"] = blk
arms["C"] = C
for w in (0.05, 0.10, 0.15):
    arms[f"C+block{w:.2f}"] = {**{k: v * (1 - w) for k, v in C.items()}, **{k: w / 4 for k in CL}}
spec = {"members": M, "arms": arms}
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy()
R = {a: cell_rows(spec, a, 12, idx) for a in arms}
for a in list(CL) + ["block", "C"]:
    t = R[a][R[a].subj != "P04"]
    print(f"alone {a:6s}: typical {t.ba.mean():.4f}  " + " ".join(f"{s} {v:.3f}" for s, v in R[a].groupby("subj").ba.mean().items()), flush=True)


def correct(arm, proto, i, split):
    mix, ix = 0.0, None
    for m, w in arms[arm].items():
        z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
        ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
        mix = mix + w * p.astype(np.float64)
    pred = causal_sinkhorn(mix, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1)
    keep = U[ix] != "P04"
    return (pred == Y[ix])[keep]


for a in list(CL) + ["block"]:
    rescued, corr = [], []
    for proto in ("kit", "seen1"):
        for i in range(12):
            for split in ("test", "val"):
                c0, c1 = correct("C", proto, i, split), correct(a, proto, i, split)
                rescued.append(c1[~c0].mean())
                corr.append(np.corrcoef(c0.astype(float), c1.astype(float))[0, 1])
    print(f"errors {a:6s}: gets right {np.mean(rescued):.3f} of C's wrong windows; correctness correlation with C {np.mean(corr):+.3f}", flush=True)
key = ["proto", "split", "i", "cell"]
for w in (0.10, 0.05, 0.15):
    a = f"C+block{w:.2f}"
    m = R[a].merge(R["C"][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    print(f"{a} vs C: typical {d.mean():+.4f} ({int((d > 0).sum()):2d}/12)  " + " ".join(f"{s} {v:+.4f}" for s, v in (m.ba - m.ba_b).groupby(m.subj).mean().items())
          + "  | " + " ".join(f"{s} {v:+.4f}" for s, v in (t.ba - t.ba_b).groupby(t.proto).mean().items()), flush=True)
