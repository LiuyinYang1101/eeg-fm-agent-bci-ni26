#!/usr/bin/env python
"""T2-D415 (pre-registered in docs/DECISIONS_T2.md before any D415 number; upgrade-all step 0, zero GPU): M7 with its FBCNet slot swapped to M6's
TILT FBCNet (fbc_tv8c -> fbc_tl15f, the same weight 0.3143; every other member as M7). Existing files only: M7's members from
outputs/t2_d408/spec_m7.json, fbc_tl15f (offset 204) from outputs/t2_d378/spec.json. Causal SKλ via t2_d320_rescreen; W50 + user-cluster CI via
t2_cluster_ci. BAR (standard adoption bar with the user's R2 guard, typed 2026-11-09): M7t vs M7 typical >= +0.003 with >= 9 / 12 AND W50 >= 0
AND no single user (P04 P05 P08 P09 P10) below -0.020; newT reported.
    mem-guard python scripts/local/t2_d415_read.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
R = O / "t2_d415"
R.mkdir(exist_ok=True)
PY = sys.executable
m7 = json.load(open(O / "t2_d408/spec_m7.json"))
M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])
M["fbc_tl15f"] = json.load(open(O / "t2_d378/spec.json"))["members"]["fbc_tl15f"]
for k in ("kit", "seen1"):
    assert all(os.path.exists(M["fbc_tl15f"][k].format(s=M["fbc_tl15f"]["offset"] + i)) for i in range(12))
A = {"M7": A7, "M7t": {("fbc_tl15f" if k == "fbc_tv8c" else k): v for k, v in A7.items()}}
assert abs(sum(A["M7t"].values()) - sum(A7.values())) < 1e-12 and "fbc_tv8c" not in A["M7t"]
json.dump({"n": 12, "anchor": "M7", "members": M, "arms": A}, open(R / "spec.json", "w"), indent=1)
json.dump({str(R / "spec.json"): list(A)}, open(R / "sk.json", "w"))
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with open(R / "run.log", "w") as fh:
    subprocess.run([PY, "-I", "tracks/t2_bci/scripts/t2_d320_rescreen.py", "--export", str(E), "--out", str(R), "--jobs", "4",
                    "--specs", str(R / "spec.json"), "--sk-list", str(R / "sk.json")], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
d = pd.read_csv(R / "rescreen_sk.csv")
d = d[(d.status == "ok") & (d.dec == "sk")]
print(d[["arm", "anchor", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
out = subprocess.run([PY, "-I", "scripts/local/t2_cluster_ci.py", str(R / "spec.json"), "M7", "M7t"], cwd=REPO, env=dict(env, OMP_NUM_THREADS="2"),
                     capture_output=True, text=True, check=True).stdout
(R / "cluster_M7t.txt").write_text(out)
print(out[out.index("per user"):])
users = {u: float(v) for u, v in re.findall(r"^\s+(P\d\d): ([+-]\d\.\d+)", out, re.M)}
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", out).group(1))
r = d[d.arm == "M7t"].iloc[0]
ok = r.typical >= 0.003 and r.typical_up >= 9 and w50 >= 0 and min(users.values()) >= -0.020
print(f"== VERDICT (pre-registered: typical >= +0.003 on >= 9 / 12, W50 >= 0, no user < -0.020): {'PASS -> build M7t + gate' if ok else 'NOT MET'}  "
      f"[typical {r.typical:+.4f} ({int(r.typical_up)} / 12), W50 {w50:+.4f}, worst user {min(users, key=users.get)} {min(users.values()):+.4f}, newT {r.newT:+.4f}]")
