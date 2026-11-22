#!/usr/bin/env python
"""T2-D442 / T2-D443 read (pre-registered in docs/DECISIONS_T2.md before any number; this script written before any number).
Fresh ST-L seeds 400-411, M7's two member recipes (plain 100 epochs; lp10 at 30), each with a tilt-0.15 twin on the same seeds.
Members as T2-D379's restarts: plain member j of draw i = seed 400 + (i + 3 j) mod 12 (j = 0..3); the lp10 member = seed 400 + i.
  M7f  = M7 with its five ST-L members -> the fresh controls            (the paired anchor; M7f vs M7 = a redraw's size, descriptive)
  M8f  = M7f with all five -> their tilt twins                          (T2-D442's DECISION arm, vs M7f)
  T_e100 / T_lp = only the four plain / only the lp10 member tilted     (descriptive)
  ST10_35 / ST10_45 = M7's five + the fresh five, FM share 0.35 / 0.45  (T2-D443, vs M7f and vs M7)
BAR (both rows): typical (non-P04) >= +0.003 with >= 9 / 12 draws up and no non-P04 user < -0.01.
    mem-guard python scripts/local/t2_d442_read.py
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d442"
HERE = Path(__file__).resolve().parent
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
SEEDS = list(range(400, 412))


def src(rec, p, s):
    tag = {"e100": "fmE100", "e100tl": "fmE100", "lp10": "fmE30", "lp10tl": "fmE30"}[rec]
    f = [x for d in glob.glob(f"{O}/sofia_s1/ni26-t2p-d442-{rec}-[abx]-{p}-[0-9]*") for x in glob.glob(f"{d}/runs/{p}__fm-st_large-mean__sz__*__s{s}.npz")]
    assert len(f) == 1, (rec, p, s, f)
    b = os.path.basename(f[0])
    assert tag in b and (("tl0.15" in b) == rec.endswith("tl")) and (("lp10" in b) == rec.startswith("lp10")), (rec, b)
    return f[0]


def member(name, rec, j):
    d = R / "restarts" / name
    d.mkdir(parents=True, exist_ok=True)
    for p in ("kit", "seen1"):
        for i in range(12):
            t = d / f"{p}__s{i}.npz"
            if t.is_symlink() or t.exists():
                t.unlink()
            os.symlink(src(rec, p, SEEDS[(i + 3 * j) % 12]), t)
    M[name] = {"offset": 0, **{p: f"{d}/{p}__s{{s}}.npz" for p in ("kit", "seen1")}}


for tl in ("", "tl"):
    for j in range(4):
        member(f"n_e100{tl}f{j}", "e100" + tl, j)
    member(f"n_lp10{tl}", "lp10" + tl, 0)
old = [f"e100f{j}" for j in range(4)] + ["lp10"]
newc = [f"n_e100f{j}" for j in range(4)] + ["n_lp10"]
newt = [f"n_e100tlf{j}" for j in range(4)] + ["n_lp10tl"]
assert all(abs(M7[m] - 0.07) < 1e-9 for m in old)
cnn = {k: v for k, v in M7.items() if k not in old}


def swap(m):
    return {**cnn, **{k: 0.07 for k in m}}


def st10(share):
    c = {k: v * (1 - share) / 0.65 for k, v in cnn.items()}
    return {**c, **{k: share / 10 for k in old + newc}}


A = {"M7": M7, "M7f": swap(newc), "M8f": swap(newt), "T_e100": swap(newt[:4] + newc[4:]), "T_lp": swap(newc[:4] + newt[4:]),
     "ST10_35": st10(0.35), "ST10_45": st10(0.45), "alone_c": {k: 0.2 for k in newc}, "alone_t": {k: 0.2 for k in newt}}
assert all(abs(sum(a.values()) - 1) < 1e-9 for a in A.values())
json.dump({"members": M, "arms": A}, open(R / "spec.json", "w"), indent=1)
for anc, arm in [("M7f", "M8f"), ("M7", "M7f"), ("M7f", "T_e100"), ("M7f", "T_lp"), ("alone_c", "alone_t"),
                 ("M7f", "ST10_35"), ("M7f", "ST10_45"), ("M7", "ST10_35"), ("M7", "ST10_45")]:
    print(f"\n== {arm} vs {anc}", flush=True)
    out = subprocess.run([sys.executable, str(HERE / "t2_cluster_ci.py"), str(R / "spec.json"), anc, arm], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr[-2000:], flush=True)
