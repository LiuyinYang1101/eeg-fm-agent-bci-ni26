#!/usr/bin/env python
"""Stream-position descriptive (handover item; zero GPU, no bar): where in a stream do our stacks lose windows?
Windows of each scored stream are ranked by `order`; bins = first 20 / next 60 / the rest. Per bin: class-balanced accuracy pooled over
the windows of that bin (per draw, then mean over 12 draws), for the raw argmax of the mix and for the shipped causal SKλ decode
(10 iters, n0 3, λ 0.95). Groups as t2_p7_celltype.py (new_s1 / later sessions P04 / P10).
    mem-guard python scripts/local/t2_streampos.py
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
BINS = [(0, 20, "first 20"), (20, 80, "next 60"), (80, 10**9, "rest")]


def ba(p, y):
    return np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]) if len(y) else np.nan


rows = []
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
                d = idx.iloc[ix].reset_index(drop=True)
                d["raw"] = mix.argmax(1)
                d["sk"] = causal_sinkhorn(mix, (d.stream.to_numpy(), d.order.to_numpy()), 10, 3, 0.95).argmax(1)
                d["pos"] = d.groupby("stream").order.rank(method="first").astype(int) - 1
                d["n"] = d.groupby("stream").order.transform("size")
                late = d.session.astype(float).astype(int) > 1
                d["g"] = np.where(~late, "new_s1", np.where(d.subject.str[-3:] == "P04", "late_P04", "late_P10"))
                for g, s in d.groupby("g"):
                    for lo, hi, nm in BINS:
                        b = s[(s.pos >= lo) & (s.pos < hi)]
                        rows.append(dict(arm=arm, g=g, bin=nm, i=i, n=len(b), raw=ba(b.raw.to_numpy(), b.y.to_numpy()), sk=ba(b.sk.to_numpy(), b.y.to_numpy())))
                if arm == "M7" and i == 0:
                    rows.append(dict(arm="_len", g=proto + "/" + split, bin="", i=0, n=int(d.groupby("stream").size().median()), raw=d.groupby("stream").size().min(), sk=d.groupby("stream").size().max()))
R = pd.DataFrame(rows)
print("stream length (windows) median / min / max:", {r.g: (r.n, int(r.raw), int(r.sk)) for r in R[R.arm == "_len"].itertuples()})
R = R[R.arm != "_len"]
T = R.groupby(["arm", "g", "bin"], sort=False).agg(n=("n", "mean"), raw=("raw", "mean"), sk=("sk", "mean")).reset_index()
T["sk_gain"] = T.sk - T.raw
print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
