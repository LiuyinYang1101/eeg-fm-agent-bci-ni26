#!/usr/bin/env python
"""T2-D432 read (pre-registered in docs/DECISIONS_T2.md before any D432 number; descriptive levels, zero GPU): the STOCK braindecode ShallowFBCSPNet and
EEGNet (base recipe r1, causal EA, no copies / views; seeds 360-371, scripts/local/t2_d432_local.sh) beside M7's ShallowNet member (sh_tv8c), S1 and M7,
cell-mean BA under SKλ on the cells of t2_cluster_ci.py. BAR (the manager's, written first): stock ShallowNet >= 0.75 on non-P04 cells -> our member is
the weak one (its own row); <= 0.71 -> the question closes; between -> reported, no action.
    mem-guard python scripts/local/t2_d432_read.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d408/spec_m7.json"))
for name, dn, model in (("sh_stock", "sh-stock", "shallow"), ("eegnet_stock", "eegnet-stock", "eegnet")):
    d = O / f"sofia_s1/ni26-t2p-d432-{dn}-5090/runs"
    J["members"][name] = {"offset": 360, **{p: str(d / f"{p}__{model}__ea__n00__r1__s{{s}}.npz") for p in ("kit", "seen1")}}
    assert all(Path(J["members"][name][p].format(s=360 + i)).exists() for p in ("kit", "seen1") for i in range(12)), sorted(x.name for x in d.glob("*"))[:5]
J["arms"] = {"stock ShallowNet + EA": {"sh_stock": 1.0}, "stock EEGNet + EA": {"eegnet_stock": 1.0}, "our ShallowNet member (sh_tv8c)": {"sh_tv8c": 1.0},
             "S1": {"fbc_pvtv8c": 4 / 9, "sh_pvtv8c": 3 / 9, "stl_rcll": 2 / 9}, "M7": J["arms"]["M7"]}
idx = pd.read_csv(EXP / "index.csv")
print(f"{'arm':34s} non-P04 (draw sd)   P05    P08    P09    P10 s2-3   P04    all cells")
for arm in J["arms"]:
    d = cell_rows(J, arm, 12, idx)
    n = d[d.subj != "P04"]
    u = d.groupby("subj").ba.mean()
    print(f"{arm:34s} {n.ba.mean():.3f} ({n.groupby('i').ba.mean().std():.3f})     " + "  ".join(f"{u[k]:.3f}" for k in ("P05", "P08", "P09", "P10", "P04")) + f"    {d.ba.mean():.3f}")
