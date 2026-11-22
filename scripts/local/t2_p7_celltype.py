#!/usr/bin/env python
"""Panel P7 ask A (the manager, 2026-11-10; descriptive, zero GPU, no bar): commodity member vs our stacks BY CELL TYPE, on existing files.
Cells as t2_cluster_ci.py (kit + seen1, test + val, SKλ 10 / n0 3 / λ 0.95, 12 seed-paired draws), grouped by what the sealed set holds:
  new_s1   = a user absent from training, session 1            (kit P04 s1 P05 P08 P09; seen1 P05 P08)
  new_late = a user absent from training, sessions 2-3          (kit P04 s2-3)                      [sealed share ~ 47 %]
  ret_late = a user whose session 1 trained, sessions 2-3       (seen1 P04 s2-3, P10 s2-3)          [sealed share ~ 43 %]
Members and M7 from outputs/t2_d408/spec_m7.json. Only 2 users carry later sessions, so no user-cluster interval is meaningful there:
P04 and P10 are printed apart instead.
    mem-guard python scripts/local/t2_p7_celltype.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d408/spec_m7.json"))
M7 = J["arms"]["M7"]
cnn = {k: M7[k] for k in ("fbc_tv8c", "sh_tv8c", "ifn")}
fm = {k: v for k, v in M7.items() if k not in cnn}
norm = lambda d: {k: v / sum(d.values()) for k, v in d.items()}  # noqa: E731
pv = {{"fbc_tv8c": "fbc_pvtv8c", "sh_tv8c": "sh_pvtv8c"}.get(k, k): v for k, v in M7.items()}
J["arms"] = {
    "SH (M7's ShallowNet-EA member alone)": {"sh_tv8c": 1.0},
    "FBC (M7's FBCNet member alone)": {"fbc_tv8c": 1.0},
    "CNN-only (FBCNet + ShallowNet + IFNet)": norm(cnn),
    "FM-only (5 ST-L)": norm(fm),
    "S1 (uploaded, 0.808)": {"fbc_pvtv8c": 4 / 9, "sh_pvtv8c": 3 / 9, "stl_rcll": 2 / 9},
    "M7 (slot 4)": M7,
    "M7pv (M7 + S1's pv10 copies)": pv,
    "M7pvF (M7pv + FBTS 0.05)": {**{k: 0.95 * v for k, v in pv.items()}, "fbts": 0.05},
}
idx = pd.read_csv(EXP / "index.csv")
later = lambda d: d.ses.astype(float).astype(int) > 1  # noqa: E731


def group(d):
    g = pd.Series("new_s1", index=d.index)
    g[later(d) & (d.proto == "kit")] = "new_late"
    g[later(d) & (d.proto == "seen1")] = "ret_late"
    return g


R = {}
for arm in J["arms"]:
    d = cell_rows(J, arm, 12, idx)
    d["g"] = group(d)
    R[arm] = d
ref = R["M7 (slot 4)"]
print("cells per draw:", ref[ref.i == 0].groupby(["g", "subj"]).size().to_dict())
cols = [("new_s1", None), ("new_late", "P04"), ("ret_late", "P04"), ("ret_late", "P10"), ("ret_late", None)]
print(f"{'arm':44s}" + "".join(f"{g + ('/' + u if u else ''):>16s}" for g, u in cols) + "   later(all)")
key = ["proto", "split", "i", "cell"]
for arm, d in R.items():
    out = f"{arm:44s}"
    m = d.merge(ref[key + ["ba"]], on=key, suffixes=("", "_m7"))
    for g, u in cols:
        s = m[(m.g == g) & ((m.subj == u) if u else True)]
        dd = (s.ba - s.ba_m7).groupby(s.i).mean()
        out += f"  {s.ba.mean():.3f} {dd.mean():+.3f}{int((dd > 0).sum()):>3d}"
    s = m[m.g != "new_s1"]
    dd = (s.ba - s.ba_m7).groupby(s.i).mean()
    print(out + f"   {s.ba.mean():.3f} {dd.mean():+.4f} ({int((dd > 0).sum())} / 12)")
print("each column: cell-mean BA, then the paired difference to M7 and the draws (of 12) in which the arm beats M7")
