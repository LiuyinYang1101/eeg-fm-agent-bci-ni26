#!/usr/bin/env python
"""Where M7's remaining error sits (descriptive, zero GPU, no bar; for the panel of 2026-11-12). Stored M7 members (outputs/t2_d408/spec_m7.json),
12 draws, kit + seen1, test + val, causal SK (10 / n0 3 / lambda 0.95) = the cells of t2_cluster_ci.py. Per window: correct or not, then grouped
by user x protocol, by true class -> predicted class, by third of the session, by third of the run, by returning vs new, and by cell.
    mem-guard python scripts/local/t2_m7_anatomy.py
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
idx["u"] = idx.subject.str[-3:]
idx["pos_s"] = idx.groupby("stream").order.transform(lambda s: np.minimum(2, (3 * s.rank(method="first").sub(1) / len(s)).astype(int)))
idx["pos_r"] = idx.groupby(["stream", "interface", "run"]).order.transform(lambda s: np.minimum(2, (3 * s.rank(method="first").sub(1) / len(s)).astype(int)))
rows = []
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            mix, ix = 0.0, None
            for m, w in J["arms"]["M7"].items():
                M = J["members"][m]
                z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                mix = mix + w * p.astype(np.float64)
            pred = causal_sinkhorn(mix, (idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]), 10, 3, 0.95).argmax(1)
            d = idx.iloc[ix][["u", "session", "interface", "cell", "y", "pos_s", "pos_r", "run"]].copy()
            d["proto"], d["i"], d["pred"], d["raw"] = proto, i, pred, mix.argmax(1)
            rows.append(d)
D = pd.concat(rows, ignore_index=True)
D["ok"], D["okraw"] = D.pred == D.y, D.raw == D.y
D["grp"] = np.where(D.session.astype(int) == 1, "new_s1", np.where(D.proto == "kit", "new_late", "ret_late"))
N = {0: "mi", 1: "sub", 2: "word"}


def ba(d, col="ok"):  # cell-averaged balanced accuracy, as scored
    return d.groupby(["proto", "i", "cell", "y"])[col].mean().groupby(["proto", "i", "cell"]).mean().mean()


T = D[D.u != "P04"]
print(f"M7 typical BA {ba(T):.4f} (before the stream decode {ba(T, 'okraw'):.4f});  P04 {ba(D[D.u == 'P04']):.4f};  all {ba(D):.4f}")
print("by user x protocol (BA, cells):")
for (u, pr), d in D.groupby(["u", "proto"]):
    print(f"  {u} {pr:5s} {ba(d):.3f}  before decode {ba(d, 'okraw'):.3f}  cells {d.cell.nunique()}  sessions {sorted(d.session.unique())}")
print("by group:", {g: round(ba(d), 3) for g, d in D.groupby("grp")}, " without P04:", {g: round(ba(d), 3) for g, d in T.groupby("grp")})
for name, d in (("typical", T), ("P04", D[D.u == "P04"])):
    c = pd.crosstab(d.y.map(N), d.pred.map(N), normalize="index").round(3)
    print(f"confusion, {name} (rows = true):\n{c.to_string()}")
    r = pd.crosstab(d.y.map(N), d.raw.map(N), normalize="index").round(3)
    print(f"  before the decode: recall " + " ".join(f"{k} {r.loc[k, k]:.3f}" for k in ("mi", "sub", "word")))
print("per typical user, recall by class:")
print(T.groupby(["u", T.y.map(N)]).ok.mean().unstack().round(3).to_string())
print("by third of the session (typical): decoded", T.groupby("pos_s").ok.mean().round(3).to_dict(), " before decode", T.groupby("pos_s").okraw.mean().round(3).to_dict())
print("by third of the run (typical):     decoded", T.groupby("pos_r").ok.mean().round(3).to_dict(), " before decode", T.groupby("pos_r").okraw.mean().round(3).to_dict())
print("by run number (typical):", T.groupby("run").ok.mean().round(3).to_dict())
print("by interface (typical):", {k: round(ba(d), 3) for k, d in T.groupby("interface")})
C = D.groupby(["proto", "i", "cell", "y"]).ok.mean().groupby(["proto", "i", "cell"]).mean().groupby(["proto", "cell"]).mean().sort_values()
print("lowest cells:\n" + C.head(6).round(3).to_string())
print("highest cells:\n" + C.tail(3).round(3).to_string())
tot = (1 - C).sum()
print("share of the total cell error by user:", ((1 - C).groupby(C.index.get_level_values("cell").str.split("/").str[1].str[:3]).sum() / tot).round(3).to_dict())
