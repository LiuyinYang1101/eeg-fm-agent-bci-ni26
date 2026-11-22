#!/usr/bin/env python
"""T2-D434 read (pre-registered before any D434 number; zero GPU): fine-tuned MantisV2 members (outputs/t2_d434/runs_{sp,sp0,ch}, seeds 0-5) under SKλ on
t2_cluster_ci's cells, 6 draws. Reports per arm: the member alone (non-P04, all cells, per-draw sd), the error phi with M7, and M7 + the arm at 0.05.
BAR (written first): an arm is ALIVE iff member alone non-P04 >= 0.70, OR (sp - sp0 non-P04 >= +0.03 AND phi(sp, M7) < 0.3 AND sp non-P04 >= 0.62 = the
stock ShallowNet of T2-D432). Otherwise the time-series line closes (frozen + fine-tuned both tested).
    mem-guard python scripts/local/t2_d434_read.py
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
R = O / "t2_d434"
N = 6
J = json.load(open(O / "t2_d408/spec_m7.json"))
M7 = J["arms"]["M7"]
ARMS = [a for a in ("sp", "sp0", "ch") if all((R / f"runs_{a}/{p}__mantisft__s{i}.npz").exists() for p in ("kit", "seen1") for i in range(N))]
J["arms"] = {"M7": M7}
for a in ARMS:
    J["members"][a] = {"offset": 0, **{p: str(R / f"runs_{a}/{p}__mantisft__s{{s}}.npz") for p in ("kit", "seen1")}}
    J["arms"][a] = {a: 1.0}
    J["arms"][f"M7+{a}"] = {**{k: 0.95 * v for k, v in M7.items()}, a: 0.05}
idx = pd.read_csv(EXP / "index.csv")


def errors(arm):
    out = []
    for proto in ("kit", "seen1"):
        for i in range(N):
            for split in ("test", "val"):
                mix = 0.0
                for m, w in J["arms"][arm].items():
                    z = np.load(J["members"][m][proto].format(s=J["members"][m].get("offset", 0) + i))
                    ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    mix = mix + w * p.astype(np.float64)
                P = causal_sinkhorn(mix, (idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]), 10, 3, 0.95).argmax(1)
                out.append(P != idx.y.to_numpy()[ix])
    return np.concatenate(out).astype(float)


C = {a: cell_rows(J, a, N, idx) for a in J["arms"]}
e7 = errors("M7")
key = ["proto", "split", "i", "cell"]
lev, phi = {}, {}
print(f"{'arm':6s} non-P04 (sd)   all    P05    P08    P09    P10    P04   phi(M7)   M7+0.05: typical (up/6)  P04      all cells")
for a in ARMS:
    d = C[a]
    n = d[d.subj != "P04"]
    u = d.groupby("subj").ba.mean()
    lev[a], phi[a] = n.ba.mean(), float(np.corrcoef(errors(a), e7)[0, 1])
    m = C[f"M7+{a}"].merge(C["M7"][key + ["ba"]], on=key, suffixes=("", "_m7"))
    m["d"] = m.ba - m.ba_m7
    t = m[m.subj.isin(["P05", "P08", "P09", "P10"])].groupby("i").d.mean()
    print(f"{a:6s} {lev[a]:.3f} ({n.groupby('i').ba.mean().std():.3f})  {d.ba.mean():.3f}  " + "  ".join(f"{u[k]:.3f}" for k in ("P05", "P08", "P09", "P10", "P04"))
          + f"   {phi[a]:.3f}     {t.mean():+.4f} ({int((t > 0).sum())}/6)        {m[m.subj == 'P04'].d.mean():+.4f}   {m.d.mean():+.4f}")
alive = [a for a in ARMS if lev[a] >= 0.70]
if "sp" in lev and "sp0" in lev and lev["sp"] - lev["sp0"] >= 0.03 and phi["sp"] < 0.3 and lev["sp"] >= 0.62 and "sp" not in alive:
    alive.append("sp")
print(f"== VERDICT (pre-registered): {'ALIVE: ' + ', '.join(alive) if alive else 'CLOSED (no arm reaches the bar)'}"
      + (f"  [sp - sp0 = {lev['sp'] - lev['sp0']:+.3f}]" if "sp0" in lev and "sp" in lev else ""))
