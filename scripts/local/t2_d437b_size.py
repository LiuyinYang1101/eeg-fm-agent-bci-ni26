#!/usr/bin/env python
"""T2-D437b (pre-registered in the D437 row before any pooled number): does a DIVERSE bag overtake the seed bag as the bag grows?
Each result file is one bag's mean probability (K nets). A bag of m K nets = the mean of m seed files of the same arm
(draw i = files 372 + (i + j) mod 12, j < m: overlapping draws at m > 1). M7's shape, ShallowNet slot = the pooled bag.
BAR at m = 2, 4: arm vs the same-size seed bag typical >= +0.003, >= 9 / 12, no non-P04 user < -0.01, typical without P05 >= 0.
    mem-guard python scripts/local/t2_d437b_size.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d437/pool"
HERE = Path(__file__).resolve().parent
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
SUF = {"seeds": "bags", "uw1": "bagw1", "uw4": "bagw4", "runs": "bagr"}
arms = {}
for a, suf in SUF.items():
    d = O / f"sofia_s1/ni26-t2p-d437-sh-{a}-5090/runs"
    for m in (1, 2, 4):
        out = R / f"{a}_x{m}"
        out.mkdir(parents=True, exist_ok=True)
        for p in ("kit", "seen1"):
            Z = [np.load(d / f"{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr{suf}__s{372 + i}.npz") for i in range(12)]
            for i in range(12):
                g = [Z[(i + j) % 12] for j in range(m)]
                assert all((z["test_idx"] == g[0]["test_idx"]).all() and (z["val_idx"] == g[0]["val_idx"]).all() for z in g)
                np.savez(out / f"{p}__s{i}.npz", test_idx=g[0]["test_idx"], val_idx=g[0]["val_idx"],
                         prob_last=np.mean([z["prob_last"] for z in g], 0), prob_val_last=np.mean([z["prob_val_last"] for z in g], 0))
        M[f"{a}_x{m}"] = {"offset": 0, **{p: f"{out}/{p}__s{{s}}.npz" for p in ("kit", "seen1")}}
        arms[f"{a}_x{m}"] = {(f"{a}_x{m}" if k == "sh_tv8c" else k): v for k, v in M7.items()}
json.dump({"members": M, "arms": arms}, open(R / "spec.json", "w"), indent=1)
pairs = [(f"seeds_x{m}", f"{a}_x{m}") for m in (1, 2, 4) for a in ("uw1", "uw4", "runs")]
pairs += [(f"{a}_x1", f"{a}_x{m}") for a in SUF for m in (2, 4)] + [("seeds_x1", "uw1_x4"), ("seeds_x1", "runs_x4")]
for anc, arm in pairs:
    out = subprocess.run([sys.executable, str(HERE / "t2_cluster_ci.py"), str(R / "spec.json"), anc, arm], capture_output=True, text=True)
    t = [x.strip() for x in out.stdout.splitlines() if x.strip().startswith(("typical:", "P0", "P10", "LOPO", "W50"))]
    if not t:
        print(arm, anc, out.stderr[-800:])
        continue
    us = " ".join(x.replace(" / 12)", ")").replace("(", "") for x in t if x[0] == "P")
    print(f"{arm:9s} vs {anc:9s} | {[x for x in t if x.startswith('typical')][0].split(';')[0]} | {us} | {[x for x in t if x.startswith('LOPO')][0][:60]} | {[x for x in t if x.startswith('W50')][0][-28:]}", flush=True)
