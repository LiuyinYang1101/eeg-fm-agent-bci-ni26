#!/usr/bin/env python
"""T2-D417 read (pre-registered in docs/DECISIONS_T2.md before any D417 number): the AUD re-dose ST-L members in M7's shape.
Members: the 12 fetched t2p_multi dirs outputs/sofia_s1/ni26-t2p-d417-<dose>-<proto>-<jobid>/runs (one stem per dir, seeds 108-119).
Arms on outputs/t2_d408/spec_m7.json's M7: M7c = lp10 -> c0 (the paired control, M7's own recipe on seeds 108-119) = the ANCHOR;
M7a / M7b / M7c_ = lp10 -> a / b / c (AUD 1: lp10 / lp15 / lp20 + lh3 + vp2); M7l1 / M7l2 = lp10 -> l1 / l2 (AUD 4 LoRA). Causal SKλ (t2_d320_rescreen),
W50 + user-cluster CI (t2_cluster_ci) per dose vs M7c. Member alone (each dose vs c0, kit + seen1 test cells, SKλ) reported.
BAR per dose (standard, with the user's R2 guard): typical >= +0.003 with >= 9 / 12 AND W50 >= 0 AND no single user below -0.020.
A pass (the best passing dose if several) -> T2-D417r: that dose + c0 on fresh seeds 120-131 before any build.
    mem-guard python scripts/local/t2_d417_read.py
"""
import glob
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
R = O / "t2_d417"
R.mkdir(exist_ok=True)
PY = sys.executable
DOSES = ("c0", "a", "b", "c", "l1", "l2")
m7 = json.load(open(O / "t2_d408/spec_m7.json"))
M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])


def member(dose):
    out = {"offset": 108}
    for p in ("kit", "seen1"):
        d = glob.glob(str(O / f"sofia_s1/ni26-t2p-d417-{dose}-{p}-[0-9]*/runs"))
        assert len(d) == 1, (dose, p, d)
        f = glob.glob(f"{d[0]}/{p}__fm-st_large-mean__*__s108.npz")
        assert len(f) == 1, (dose, p, f)
        out[p] = f[0].replace("__s108.npz", "__s{s}.npz")
        assert all(os.path.exists(out[p].format(s=108 + i)) for i in range(12)), (dose, p)
    return out


for dose in DOSES:
    M[f"stl_{dose}"] = member(dose)
assert "lp10" in A7
A = {}
for dose in DOSES:
    A["M7c" if dose == "c0" else ("M7c_" if dose == "c" else f"M7{dose}")] = {(f"stl_{dose}" if k == "lp10" else k): v for k, v in A7.items()}
json.dump({"n": 12, "anchor": "M7c", "members": M, "arms": A}, open(R / "spec.json", "w"), indent=1)
json.dump({str(R / "spec.json"): list(A)}, open(R / "sk.json", "w"))
# the member alone: each dose vs c0 as one-member arms
Ms = {"n": 12, "anchor": "c0", "members": {f"stl_{d}": M[f"stl_{d}"] for d in DOSES}, "arms": {d: {f"stl_{d}": 1.0} for d in DOSES}}
json.dump(Ms, open(R / "spec_member.json", "w"), indent=1)
json.dump({str(R / "spec_member.json"): list(Ms["arms"])}, open(R / "sk_member.json", "w"))
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
for sp, sk, tag in (("spec.json", "sk.json", "pkg"), ("spec_member.json", "sk_member.json", "member")):
    (R / tag).mkdir(exist_ok=True)
    with open(R / f"run_{tag}.log", "w") as fh:
        subprocess.run([PY, "-I", "tracks/t2_bci/scripts/t2_d320_rescreen.py", "--export", str(E), "--out", str(R / tag), "--jobs", "4",
                        "--specs", str(R / sp), "--sk-list", str(R / sk)], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
cols = ["arm", "anchor", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]
fmt = lambda v: f"{v:+.4f}"  # noqa: E731
dm = pd.read_csv(R / "member/rescreen_sk.csv"); dm = dm[(dm.status == "ok") & (dm.dec == "sk")]
print("== MEMBER ALONE (each dose vs c0):"); print(dm[cols].to_string(index=False, float_format=fmt))
d = pd.read_csv(R / "pkg/rescreen_sk.csv"); d = d[(d.status == "ok") & (d.dec == "sk")]
print("== IN M7's SHAPE (lp10 slot; vs M7c):"); print(d[cols].to_string(index=False, float_format=fmt))
verdict = {}
for arm in [a for a in A if a != "M7c"]:
    out = subprocess.run([PY, "-I", "scripts/local/t2_cluster_ci.py", str(R / "spec.json"), "M7c", arm], cwd=REPO, env=dict(env, OMP_NUM_THREADS="2"),
                         capture_output=True, text=True, check=True).stdout
    (R / f"cluster_{arm}.txt").write_text(out)
    users = {u: float(v) for u, v in re.findall(r"^\s+(P\d\d): ([+-]\d\.\d+)", out, re.M)}
    w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", out).group(1))
    r = d[d.arm == arm].iloc[0]
    ok = bool(r.typical >= 0.003 and r.typical_up >= 9 and w50 >= 0 and min(users.values()) >= -0.020)
    verdict[arm] = dict(typical=float(r.typical), up=int(r.typical_up), W50=w50, worst=min(users.values()), newT=float(r.newT), P04=float(r.P04), passed=ok)
    print(f"   {arm}: typical {r.typical:+.4f} ({int(r.typical_up)} / 12), W50 {w50:+.4f}, worst user {min(users, key=users.get)} {min(users.values()):+.4f} -> "
          f"{'PASS' if ok else 'not met'}")
json.dump(verdict, open(R / "verdict.json", "w"), indent=1)
best = max((a for a in verdict if verdict[a]["passed"]), key=lambda a: verdict[a]["typical"], default=None)
print(f"== VERDICT (pre-registered, per dose): {'PASS: ' + best + ' -> T2-D417r (seeds 120-131) before any build' if best else 'NOT MET (no dose)'}")
