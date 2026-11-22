#!/usr/bin/env python
"""T2-D437 read (pre-registered in docs/DECISIONS_T2.md before any D437 number): bagging WITHOUT removing a user, ShallowNet, seeds 372-383.
Arms = K-net bags per seed: seeds (the ANCHOR, plain restarts), uw1 / uw4 (Dirichlet user shares), runs (run-level bootstrap).
Prints per arm: the bag's mean single-net accuracy and mean pairwise error phi INSIDE the bag (kit / seen1 test, from the result files),
the bag alone, and M7's shape with its ShallowNet slot = the bag, each vs the seed bag (typical = non-P04, W50, per user; causal SKλ).
BAR: M7 shape typical >= +0.003 with >= 9 / 12 draws up and no non-P04 user < -0.01 -> the FBCNet arm + a fresh-seed replicate; else closed.
    mem-guard python scripts/local/t2_d437_read.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d437"
R.mkdir(exist_ok=True)
HERE = Path(__file__).resolve().parent
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
SUF = {"seeds": "bags", "uw1": "bagw1", "uw4": "bagw4", "runs": "bagr", "ub2": "bagb2", "ub5": "bagb5"}   # ub2 / ub5 = T2-D438 (user-level boosting)
SUF = {a: v for a, v in SUF.items()   # complete arms only (24 files)
       if len(list((O / f"sofia_s1/ni26-t2p-d437-sh-{a}-5090/runs").glob(f"*{v}__s*.npz"))) == 24}
arms = {}
for a, suf in SUF.items():
    d = O / f"sofia_s1/ni26-t2p-d437-sh-{a}-5090/runs"
    M[f"bag_{a}"] = {"offset": 372, **{p: f"{d}/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr{suf}__s{{s}}.npz" for p in ("kit", "seen1")}}
    st = {p: [json.loads(str(np.load(M[f"bag_{a}"][p].format(s=372 + i))["res"])) for i in range(12)] for p in ("kit", "seen1")}
    print(f"{a:6s} " + "  ".join(f"{p}: K {st[p][0]['bag_k']}, single-net acc {np.mean([r['bag_acc'] for r in st[p]]):.4f}, "
                                  f"phi inside the bag {np.mean([r['bag_phi'] for r in st[p]]):.3f}" for p in st))
    if "boost_err" in st["kit"][0]:
        print("       kit LOUO user errors (mean over seeds):", np.round(np.mean([r["boost_err"] for r in st["kit"]], 0), 3).tolist(),
              "multipliers:", np.round(np.mean([r["boost_mult"] for r in st["kit"]], 0), 2).tolist())
    arms[f"alone_{a}"] = {f"bag_{a}": 1.0}
    arms[f"M7_{a}"] = {(f"bag_{a}" if k == "sh_tv8c" else k): v for k, v in M7.items()}
json.dump({"members": M, "arms": arms}, open(R / "spec.json", "w"), indent=1)
for a in [a for a in SUF if a != "seeds"]:
    for shape in ("alone", "M7"):
        print(f"\n== {shape}: {a} vs seeds", flush=True)
        out = subprocess.run([sys.executable, str(HERE / "t2_cluster_ci.py"), str(R / "spec.json"), f"{shape}_seeds", f"{shape}_{a}"],
                             capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr[-2000:], flush=True)
