#!/usr/bin/env python
"""T2-D465 read (written before any number): EEGConformer as a 4th CNN family. Routed files are built first with t2p_fpcalib blend
(r 1/2, as ShallowNet / IFNet ship) into outputs/t2_d465/route_conf. Base shape C = M7's weights with fresh CNN draws (FBCNet D419 s324+,
ShallowNet / IFNet D429 ctl s348+) and D442's control FM draws (as T2-D459). Arms, typical (non-P04), kit + seen1, test + val, 12 draws, causal SK:
  (a) alone: Conformer vs ShallowNet vs IFNet (one routed tv8c net each);
  (b) swap: Conformer in IFNet's 0.10 slot vs C;
  (c) added at 0.10 (all weights x 1/1.1) vs C + a SECOND IFNet draw (spec_m7's ifn, seeds 168+) at 0.10 (equal net count).
Stop rule: (a) Conformer decoded < 0.68 -> (b) / (c) not candidates. Bar (b) or (c): >= +0.003, >= 9 / 12, no typical user < -0.005; P09 apart.
    mem-guard python scripts/local/t2_d465_read.py
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
M7 = json.load(open(O / "t2_d408/spec_m7.json"))
M["ifn_b"] = M7["members"]["ifn"]
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = S[src]
M["conf"] = {"offset": 528, **{p: str(O / f"t2_d465/route_conf/{p}__conformer__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{{s}}.npz") for p in ("kit", "seen1")}}
C = J["arms"]["M7f"]
arms = {"conf": {"conf": 1.0}, "sh": {"sh_tv8c": 1.0}, "ifn": {"ifn": 1.0}, "C": C,
        "swap": {("conf" if k == "ifn" else k): v for k, v in C.items()},
        "add_conf": {**{k: v / 1.1 for k, v in C.items()}, "conf": 0.1 / 1.1},
        "add_ifn": {**{k: v / 1.1 for k, v in C.items()}, "ifn_b": 0.1 / 1.1}}
spec = {"members": M, "arms": arms}
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
R = {a: cell_rows(spec, a, 12, idx) for a in arms}
for a in ("conf", "sh", "ifn"):
    t = R[a][R[a].subj != "P04"]
    print(f"alone {a:5s}: typical {t.ba.mean():.4f}  " + " ".join(f"{s} {v:.3f}" for s, v in R[a].groupby("subj").ba.mean().items()), flush=True)
for a, b in (("swap", "C"), ("add_conf", "add_ifn"), ("add_conf", "C")):
    m = R[a].merge(R[b][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    print(f"{a} vs {b}: typical {d.mean():+.4f} ({int((d > 0).sum()):2d}/12)  " + " ".join(f"{s} {v:+.4f}" for s, v in (m.ba - m.ba_b).groupby(m.subj).mean().items())
          + "  | " + " ".join(f"{s} {v:+.4f}" for s, v in (t.ba - t.ba_b).groupby(t.proto).mean().items()), flush=True)
