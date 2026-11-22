#!/usr/bin/env python
"""T2-D413d (pre-registered in docs/DECISIONS_T2.md before any D413d number): the tuned-generator FBCNet (T2-D413c's mf arm) as an ADDED member
beside the control FBCNet, CPU-only, from EXISTING files (local job 536's two arms + T2-D412's control arm). Weights fixed a priori:
M6's FBCNet slot (S3f's fbc_tl15f, 0.3143) split EQUALLY between two FBCNets, all other members as shipped:
  SPmf  = slot -> 1/2 control (s300+i) + 1/2 mf (s300+i)          THE DECISION ARM
  SPctl = slot -> 1/2 control (s300+i) + 1/2 control (s276+i)     the ANCHOR: the same two-net slot with a second control seed, so the
                                                                   bagging gain of a second FBCNet is not credited to the synthetic data
  SWctl = slot -> control (s300+i) alone (M6's shape; reported: SPctl vs SWctl = the bagging gain, SPmf vs SWctl descriptive)
Causal SKλ (10 it, n0 3, λ 0.95) via t2_d320_rescreen; bar = T2-D413's on SPmf vs SPctl: typical >= +0.003 with >= 9 / 12, P04 >= -0.010,
newT >= 0; W50 reported beside, never judged.
    mem-guard python scripts/local/t2_d413d_read.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
R = O / "t2_d413d"
R.mkdir(exist_ok=True)
PY = sys.executable
d378 = json.load(open(O / "t2_d378/spec.json"))
M, S3f = dict(d378["members"]), d378["arms"]["S3f"]


def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m


M["fbc_mff"] = mem(300, str(O / "t2_d413c/read/route_fbc_mff/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15_mf50tv8cfb3"))
M["fbc_ctlf"] = mem(300, str(O / "t2_d413c/read/route_fbc_ctlf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3"))
M["fbc_ctl276"] = mem(276, str(O / "t2_d412/route_fbc_ctlf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3"))
w = S3f["fbc_tl15f"]


def arm(parts):
    a = {k: v for k, v in S3f.items() if k != "fbc_tl15f"}
    a.update(parts)
    assert abs(sum(a.values()) - sum(S3f.values())) < 1e-9
    return a


A = {"SPctl": arm({"fbc_ctlf": w / 2, "fbc_ctl276": w / 2}), "SPmf": arm({"fbc_ctlf": w / 2, "fbc_mff": w / 2}), "SWctl": arm({"fbc_ctlf": w})}
json.dump({"n": 12, "anchor": "SPctl", "members": M, "arms": A}, open(R / "spec.json", "w"), indent=1)
json.dump({str(R / "spec.json"): list(A)}, open(R / "sk.json", "w"))
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with open(R / "run.log", "w") as fh:
    subprocess.run([PY, "-I", "tracks/t2_bci/scripts/t2_d320_rescreen.py", "--export", str(E), "--out", str(R), "--jobs", "4",
                    "--specs", str(R / "spec.json"), "--sk-list", str(R / "sk.json")], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
d = pd.read_csv(R / "rescreen_sk.csv")
d = d[(d.status == "ok") & (d.dec == "sk")]
print(d[["arm", "anchor", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
for a_, b_ in (("SPctl", "SPmf"), ("SWctl", "SPctl"), ("SWctl", "SPmf")):
    out = subprocess.run([PY, "-I", "scripts/local/t2_cluster_ci.py", str(R / "spec.json"), a_, b_], cwd=REPO, env=dict(env, OMP_NUM_THREADS="2"),
                         capture_output=True, text=True, check=True).stdout
    (R / f"cluster_{b_}_vs_{a_}.txt").write_text(out)
    print(f"== {b_} vs {a_}:")
    print(out[out.index("per user"):] if "per user" in out else out[-1500:])
r = d[d.arm == "SPmf"].iloc[0]
ok = r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.010 and r.newT >= 0
print(f"== VERDICT (T2-D413's bar, SPmf vs SPctl): {'PASS -> a runtime-priced template decision (one more FBCNet forward), then D413r' if ok else 'NOT MET'}  "
      f"[typical {r.typical:+.4f} ({int(r.typical_up)} / 12), P04 {r.P04:+.4f}, new {r.newT:+.4f}]")
