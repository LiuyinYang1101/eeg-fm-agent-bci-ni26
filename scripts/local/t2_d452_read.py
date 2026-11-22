#!/usr/bin/env python
"""T2-D452 read (written before any number): REVE-base (align none / sz) and LaBraM-base under the current fine-tune recipe, seeds 424-435.
Per arm: the member alone (typical = non-P04, causal SK), and M7 with the member ADDED as a sixth FM member (FM slot 0.35 over six, equal)
vs M7. BARS: alone >= 0.70 (kill < 0.60); sixth member typical >= +0.003 with >= 9 / 12 and no non-P04 user < -0.01.
    mem-guard python scripts/local/t2_d452_read.py
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d452"
R.mkdir(exist_ok=True)
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
ST = ["e100f0", "e100f1", "e100f2", "e100f3", "lp10"]
arms = {"M7": M7}
for r in ("reve-none", "reve-sz", "labram-sz"):
    t = {"offset": 424}
    for p in ("kit", "seen1"):
        ds = glob.glob(f"{O}/sofia_s1/ni26-t2p-d452-{r}-{p}-[0-9]*")
        if len(ds) != 1:
            break
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-*__s*.npz")})
        assert len(st) == 1, (r, p, st)
        t[p] = st[0] + "__s{s}.npz"
    else:
        M[r] = t
        arms[f"alone_{r}"] = {r: 1.0}
        arms[f"M7+{r}"] = {**{k: v for k, v in M7.items() if k not in ST}, **{k: 0.35 / 6 for k in ST + [r]}}
json.dump({"members": M, "arms": arms}, open(R / "spec.json", "w"), indent=1)
idx = pd.read_csv(EXP / "index.csv")
for a in arms:
    if a.startswith("alone_"):
        d = cell_rows({"members": M, "arms": arms}, a, 12, idx)
        print(f"{a}: typical level {d[d.subj != 'P04'].ba.mean():.4f}  " + " ".join(f"{s} {v:.3f}" for s, v in d.groupby("subj").ba.mean().items()), flush=True)
    elif a != "M7":
        print(f"\n== {a} vs M7", flush=True)
        out = subprocess.run([sys.executable, str(HERE / "t2_cluster_ci.py"), str(R / "spec.json"), "M7", a], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr[-1500:], flush=True)
