#!/usr/bin/env python
"""T2-D450 (the user, 2026-11-11: "think reversely": why did M7 gain so much on the sealed set, and what does that say about the data?).
Descriptive, zero GPU. (1) The four ZIPs with sealed scores (A1, MOR, S1, M7) on local cells BY CELL TYPE and user: which local slice
reproduces the sealed spacing (A1 ~ MOR ~ S1, M7 + 0.020)? (2) The ladder S1 -> M7 one change at a time: which change carries the gain,
and in which slice? Cells as t2_cluster_ci (kit + seen1, test + val, causal SK, 12 draws).
  new_s1 = a user absent from training, session 1; new_late = such a user's sessions 2-3 (P04 only); ret = a user whose session 1 trained.
    mem-guard python scripts/local/t2_d450_reverse.py
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
J["members"].update({k: v for k, v in json.load(open(O / "t2_d310/spec.json"))["members"].items() if k not in J["members"]})
M7 = J["arms"]["M7"]
ST5 = ["r0", "r12", "r24", "vp12", "lp10"]
A = {
    "A1  (sealed 0.798)": {"fbc_c": 4 / 9, "sh_c": 3 / 9, "stl_rcll": 2 / 9},
    "MOR (sealed 0.798)": {"fbc_c": 0.4, "sh_c": 0.3, "stl_rcll": 0.2, "mor": 0.1},
    "S1  (sealed 0.799)": {"fbc_pvtv8c": 4 / 9, "sh_pvtv8c": 3 / 9, "stl_rcll": 2 / 9},
    "M7  (sealed 0.819)": M7,
    "L1 S1 with weaker copies (tv8c)": {"fbc_tv8c": 4 / 9, "sh_tv8c": 3 / 9, "stl_rcll": 2 / 9},
    "L2 + FM share 0.22 -> 0.35": {"fbc_tv8c": 0.65 * 4 / 7, "sh_tv8c": 0.65 * 3 / 7, "stl_rcll": 0.35},
    "L3 + five ST-L members (30 ep)": {"fbc_tv8c": 0.65 * 4 / 7, "sh_tv8c": 0.65 * 3 / 7, **{m: 0.07 for m in ST5}},
    "L4 + IFNet 0.10 (= M5)": {"fbc_tv8c": M7["fbc_tv8c"], "sh_tv8c": M7["sh_tv8c"], "ifn": 0.1, **{m: 0.07 for m in ST5}},
    "L5 + 100-epoch ST-L (= M7)": M7,
}
J["arms"] = A
idx = pd.read_csv(EXP / "index.csv")
R = {}
for a in A:
    d = cell_rows(J, a, 12, idx)
    late = d.ses.astype(float).astype(int) > 1
    d["g"] = "new_s1"
    d.loc[late & (d.proto == "kit"), "g"] = "new_late(P04)"
    d.loc[late & (d.proto == "seen1"), "g"] = "ret_late"
    d["gu"] = d.g + "/" + d.subj
    R[a] = d
cols = ["new_s1/P05", "new_s1/P08", "new_s1/P09", "new_s1/P04", "new_late(P04)/P04", "ret_late/P04", "ret_late/P10"]
names = list(A)
print("LEVELS (cell-mean BA)        " + "".join(f"{c.replace('new_', 'n_').replace('ret_late', 'ret'):>14s}" for c in cols) + "   new_s1(noP04)  typical")
for a in names:
    d = R[a]
    print(f"{a:30s}" + "".join(f"{d[d.gu == c].ba.mean():14.3f}" for c in cols) + f"{d[(d.g == 'new_s1') & (d.subj != 'P04')].ba.mean():14.3f}{d[d.subj != 'P04'].ba.mean():10.3f}")
key = ["proto", "split", "i", "cell"]


def delta(a, b, sel):
    m = R[a].merge(R[b][key + ["ba"]], on=key, suffixes=("", "_b"))
    m = m[sel(m)]
    dd = (m.ba - m.ba_b).groupby(m.i).mean()
    return dd.mean(), int((dd > 0).sum())


S = {"n_s1 P05": lambda m: m.gu == "new_s1/P05", "n_s1 P08": lambda m: m.gu == "new_s1/P08", "n_s1 P09": lambda m: m.gu == "new_s1/P09",
     "n_s1 P04": lambda m: m.gu == "new_s1/P04", "n_late P04": lambda m: m.g == "new_late(P04)", "ret P04": lambda m: m.gu == "ret_late/P04",
     "ret P10": lambda m: m.gu == "ret_late/P10", "new_s1 noP04": lambda m: (m.g == "new_s1") & (m.subj != "P04"), "typical": lambda m: m.subj != "P04", "all": lambda m: m.subj != ""}
print("\nPAIRED DIFFERENCES (mean over 12 draws, draws up)     " + "".join(f"{k:>15s}" for k in S))
pairs = [(names[1], names[0], "MOR - A1   [sealed  0.000]"), (names[2], names[0], "S1 - A1    [sealed +0.001]"), (names[3], names[2], "M7 - S1    [sealed +0.020]"),
         (names[4], names[2], "step 1: weaker copies"), (names[5], names[4], "step 2: FM share 0.35"), (names[6], names[5], "step 3: five ST-L members"),
         (names[7], names[6], "step 4: + IFNet"), (names[8], names[7], "step 5: 100-epoch ST-L")]
for a, b, nm in pairs:
    print(f"{nm:54s}" + "".join("{:+9.4f} {:>2d}   ".format(*delta(a, b, f)) for f in S.values()), flush=True)
