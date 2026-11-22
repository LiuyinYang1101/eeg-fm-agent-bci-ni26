#!/usr/bin/env python
"""T2-D477 read (written before any number): IFNet with 1-4 Hz (ifnb3) and + 40-58 Hz (ifnb4) input bands, ShallowNet with a 61-sample
temporal filter (sht61), each paired with T2-D429's control on seeds 348-359 (same GPU, same recipe, copies + tv8c, routed r 0.5 tag 2).
 alone : the routed member vs its control (12 draws), decoded (causal SKλ as every shape-C read).
 in C  : T2-D459's shape C (M7's weights; D419 / D429 / D442 control draws) with the family's control files replaced by the arm's.
BAR per arm, in C: typical >= +0.006 with >= 9 / 12, no typical user below -0.005 (T2-D475's). Alone reported; a fresh replicate if alone
>= +0.010.     mem-guard python scripts/local/t2_d477_read.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    Sp = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = Sp[src]
C = J["arms"]["M7f"]
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
ARMS = {"sht61": ("shallow_t61", "sh_tv8c"), "ifnb3": ("ifnet_b3", "ifn"), "ifnb4": ("ifnet_b4", "ifn")}


def compare(spec, a, b, n=12):
    R = {x: cell_rows(spec, x, n, idx) for x in (a, b)}
    m = R[a].merge(R[b][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    return f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{n}) level {t.ba.mean():.3f} vs {t.ba_b.mean():.3f}  " + " ".join(f"{s} {v:+.4f}" for s, v in u.items())


for arm in sys.argv[1:] or list(ARMS):
    model, slot = ARMS[arm]
    r = O / f"t2_d477/route_{arm}"
    if not r.exists():
        print(f"{arm}: not run yet")
        continue
    M[arm] = {"offset": 348, **{p: str(r / f"{p}__{model}__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{{s}}.npz") for p in ("kit", "seen1")}}
    spec = {"members": M, "arms": {"alone": {arm: 1.0}, "ctl": {slot: 1.0},
                                    "C_arm": {(arm if k == slot else k): v for k, v in C.items()}, "C_ctl": C}}
    print(f"{arm} alone vs control:", compare(spec, "alone", "ctl"), flush=True)
    print(f"{arm} in C vs control:", compare(spec, "C_arm", "C_ctl"), flush=True)
