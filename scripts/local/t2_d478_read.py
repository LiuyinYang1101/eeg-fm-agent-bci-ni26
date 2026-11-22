#!/usr/bin/env python
"""T2-D478 read (written before any number): the CNN families trained with stream-balanced batches + the per-stream prior-blind loss (sb16),
each paired with its control on the same seeds and GPU (FBCNet: T2-D419 e180, 324-335; ShallowNet / IFNet: T2-D429 ctl, 348-359), copies + tv8c,
routed as shipped. Read DECODED only (causal SKλ, as every shape-C read).
 alone : the routed member vs its control (12 draws).   in C : T2-D459's shape C with the family's control files replaced by the arm's.
 all   : all three families replaced at once (reported, not a bar).
BAR per family, in C: typical >= +0.006 with >= 9 / 12, no typical user below -0.005. KILL: all three families <= +0.005 alone.
    mem-guard python scripts/local/t2_d478_read.py
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
# arm -> (routed stem pattern, slot in C, seed offset)
ARMS = {"fbc": ("{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_sb16tv8cfb3__s{{s}}.npz", "fbc_tv8c", 324),
        "sh": ("{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_sb16tv8cfb2__s{{s}}.npz", "sh_tv8c", 348),
        "ifn": ("{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vr_sb16tv8cfb2__s{{s}}.npz", "ifn", 348)}


def compare(spec, a, b, n=12):
    R = {x: cell_rows(spec, x, n, idx) for x in (a, b)}
    m = R[a].merge(R[b][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    return f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{n}) level {t.ba.mean():.3f} vs {t.ba_b.mean():.3f}  " + " ".join(f"{s} {v:+.4f}" for s, v in u.items())


done = []
for arm, (pat, slot, s0) in ARMS.items():
    r = O / f"t2_d478/route_{arm}"
    if not r.exists():
        print(f"{arm}: not run yet")
        continue
    M[f"sb_{arm}"] = {"offset": s0, **{p: str(r / pat.format(p=p)) for p in ("kit", "seen1")}}
    spec = {"members": M, "arms": {"alone": {f"sb_{arm}": 1.0}, "ctl": {slot: 1.0},
                                    "C_arm": {(f"sb_{arm}" if k == slot else k): v for k, v in C.items()}, "C_ctl": C}}
    print(f"{arm} alone vs control:", compare(spec, "alone", "ctl"), flush=True)
    print(f"{arm} in C vs control:", compare(spec, "C_arm", "C_ctl"), flush=True)
    done.append((arm, slot))
if len(done) == 3:
    sw = {slot: f"sb_{arm}" for arm, slot in done}
    spec = {"members": M, "arms": {"C_all": {sw.get(k, k): v for k, v in C.items()}, "C_ctl": C}}
    print("all three in C vs control:", compare(spec, "C_all", "C_ctl"), flush=True)
