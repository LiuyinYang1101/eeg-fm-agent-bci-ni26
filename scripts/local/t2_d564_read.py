#!/usr/bin/env python
"""T2-D564 read (registered 2026-11-17 before any number): W20's mixes on one set computed TWICE by the lab (`t2_lab.py mix --base w20`):
REF with the shipped teacher, ARM with NI26_TEACH_DIR = outputs/t2_d564_w<wt>_<d0>/teacher_p1 (heads only: the P1 files are the set's own);
--combo adds D563's knn_both to the ARM (the combination read on C). Judged with the lab's rule: guards = ten > 0, >= 2 / 3 draws > 0, no
stranger < -0.020, returning >= -0.003; stack bar = guards and ten >= +0.001 and sealed mix > 0.
    mem-guard env OMP_NUM_THREADS=4 ... python scripts/local/t2_d564_read.py --set A --wt 0.075 [--combo]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=("A", "B", "C"))
ap.add_argument("--wt", type=float, required=True, choices=(0.0, 0.075, 0.15))
ap.add_argument("--combo", action="store_true")
a = ap.parse_args()
O = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
D0 = {"A": 825, "B": 828, "C": 831}[a.set]
TD = f"{O}/t2_d564_w{int(round(a.wt * 1000)):03d}_{D0}/teacher_p1"
assert os.path.isdir(TD), TD
tag = f"{a.set}_w{int(round(a.wt * 1000)):03d}" + ("_combo" if a.combo else "")
paths = {"ref": f"{O}/t2_lab/d564_mix_{a.set}_ref.npz", "arm": f"{O}/t2_lab/d564_mix_{tag}.npz"}
for side in ("ref", "arm"):
    if side == "ref" and os.path.exists(paths["ref"]):
        continue
    env = {k: v for k, v in os.environ.items() if k != "NI26_TEACH_DIR"}
    cmd = [sys.executable, "-u", "scripts/local/t2_lab.py", "mix", "--set", a.set, "--base", "w20", "--out", paths[side]]
    if side == "arm":
        env["NI26_TEACH_DIR"] = TD
        if a.combo:
            cmd += ["--idea", "knn_both"]
    subprocess.run(cmd, env=env, check=True)
ITEMS = {825: "D539b,D543,D546b,D553,D555", 828: "D539b,D543,D546b,D553,D557", 831: "D539b,D543,D546b,D553,D555"}[D0]
sys.argv = [sys.argv[0], "--draws", str(D0), "--items", ITEMS]
src = open("scripts/local/t2_d540a_stack_read.py").read()
g = {"__name__": "d564_head", "__file__": os.path.abspath("scripts/local/t2_d540a_stack_read.py")}
exec(compile(src.split("\nBASE = read([])\n")[0], "t2_d540a_stack_read_head(d564)", "exec"), g)


def loadmix(p):
    z = np.load(p)
    return {(int(k.split("_")[1]), int(k.split("_")[2])): (z[k], z["mx_" + k[3:]]) for k in z.files if k.startswith("ix_")}, z


REF, zr = loadmix(paths["ref"])
ARM, za = loadmix(paths["arm"])
assert str(zr["teach"]) == "" and str(za["teach"]) == TD and str(zr["base"]) == str(za["base"]) == "w20", (zr["teach"], za["teach"])
assert REF.keys() == ARM.keys() and all(np.array_equal(REF[k][0], ARM[k][0]) for k in REF)
sc = g["judge"](ARM, REF)
sc["guards"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
sc["stack_bar"] = bool(sc["ten"] >= 0.001 and sc["guards"] and sc["sealed_mix"] > 0)
sc["max_abs_dp"] = float(max(np.abs(ARM[k][1] - REF[k][1]).max() for k in REF))
res = {"set": a.set, "wt": a.wt, "combo": a.combo, "level": {"W20": g["level"](REF), "arm": g["level"](ARM)}, "arm vs W20": sc}
print(f"D564 {tag}: {res}", flush=True)
json.dump(res, open(f"{O}/t2_lab/d564_{tag}.json", "w"), indent=1, default=str)
