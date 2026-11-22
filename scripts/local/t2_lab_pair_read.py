#!/usr/bin/env python
"""A 2-process lab read ON TOP OF W20 (generalises t2_d564_read.py): W20's mixes on one set computed by the lab (`t2_lab.py mix --base w20`)
twice: REF in the plain environment, ARM with the given env overrides (e.g. a D559 schedule root: NI26_D559_ROOT / NI26_D559_CKS; a D564
teacher: NI26_TEACH_DIR) and optionally one registered lab idea (--idea). Judged with the lab's rule: guards = ten > 0, >= 2 / 3 draws > 0,
no stranger < -0.020, returning >= -0.003; stack bar = guards and ten >= +0.001 and sealed mix > 0.
    mem-guard env OMP_NUM_THREADS=4 ... python scripts/local/t2_lab_pair_read.py --set A --tag d559_d8 \
        --env NI26_D559_ROOT=t2_d559_d8_825 --env NI26_D559_CKS=40,60,80,100,120,140,160,175 [--idea knn_both]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=("A", "B", "C", "D", "E"))
ap.add_argument("--tag", required=True)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--idea", default="")
ap.add_argument("--ref-idea", default="", help="the same registered idea applied to REF too (e.g. w20_fb: both sides under the feedback teacher)")
ap.add_argument("--base-env", action="append", default=[], help="T2-D567: an env applied to BOTH sides (e.g. NI26_LAB_D557=1: REF = W20+D557 on C; NI26_LAB_PRE_R=1: the template-faithful pre-refit P1 part, its own cached REF)")
ap.add_argument("--ref-env", action="append", default=[], help="T2-D569: an env applied to REF only (e.g. the identity P1 root: NI26_D559_ROOT=t2_d569_id_825)")
a = ap.parse_args()
O = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
D0 = {"A": 825, "B": 828, "C": 831, "D": 834, "E": 860}[a.set]   # T2-D615: E
ENV = dict(kv.split("=", 1) for kv in a.env)
KEYS = ("NI26_LAB_P1REPS", "NI26_LAB_D611", "NI26_LAB_D611B", "NI26_LAB_D618", "NI26_LAB_D618B", "NI26_LAB_D607", "NI26_LAB_D606", "NI26_LAB_PRE_R", "NI26_D559_ROOT", "NI26_D559_CKS", "NI26_TEACH_DIR", "NI26_LAB_D557", "NI26_LAB_D567", "NI26_D567_LOCAL", "NI26_LAB_D568", "NI26_LAB_D568B", "NI26_LAB_STARTS", "NI26_LAB_D582", "NI26_LAB_D582B", "NI26_LAB_D583", "NI26_LAB_D584", "NI26_LAB_D586", "NI26_LAB_D583B", "NI26_LAB_D586B", "NI26_LAB_D588", "NI26_LAB_D588B", "NI26_LAB_D596", "NI26_LAB_D596B")
BENV = dict(kv.split("=", 1) for kv in a.base_env)
RENV = dict(kv.split("=", 1) for kv in a.ref_env)
assert set(ENV) <= set(KEYS) and set(BENV) <= {"NI26_LAB_D557", "NI26_D567_LOCAL", "NI26_LAB_STARTS", "NI26_LAB_PRE_R"} and not set(ENV) & set(BENV), (ENV, BENV)
assert set(RENV) <= {"NI26_D559_ROOT", "NI26_D559_CKS", "NI26_LAB_D568B", "NI26_LAB_STARTS", "NI26_LAB_D582B"} and not set(RENV) & set(BENV), (RENV, BENV)
assert "NI26_LAB_PRE_R" not in ENV, "NI26_LAB_PRE_R is both-sides only (--base-env): a PRE_R ARM is never read against a non-PRE_R REF"   # Track3 LOW on d2a2c3df
BTAG = "".join(f"_{k.lower()}-{v}" for k, v in sorted(BENV.items()))   # a REF built under a base env gets its own cached file
BTAG += "".join(f"_{v}" if k == "NI26_D559_ROOT" else f"_{k.lower()}-{v}" for k, v in sorted(RENV.items()) if k != "NI26_D559_CKS")   # T2-D569 / D568 B: a REF-only env gets its own cached file
paths = {"ref": f"{O}/t2_lab/pair_mix_{a.set}_ref{('_' + a.ref_idea) if a.ref_idea else ''}{BTAG}.npz", "arm": f"{O}/t2_lab/pair_mix_{a.set}_{a.tag}.npz"}
for side in ("ref", "arm"):
    if side == "ref" and os.path.exists(paths["ref"]):
        continue
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(BENV)
    if side == "ref":
        env.update(RENV)
    cmd = [sys.executable, "-u", "scripts/local/t2_lab.py", "mix", "--set", a.set, "--base", "w20", "--out", paths[side]]
    if side == "ref" and a.ref_idea:
        cmd += ["--idea", a.ref_idea]
    if side == "arm":
        env.update(ENV)
        if a.idea:
            cmd += ["--idea", a.idea]
    subprocess.run(cmd, env=env, check=True)
ITEMS = {825: "D539b,D543,D546b,D553,D555", 828: "D539b,D543,D546b,D553,D557", 831: "D539b,D543,D546b,D553,D555", 834: "D539b,D543,D546b,D553,D557", 860: "D539b,D543,D546b,D553,D557"}[D0]
sys.argv = [sys.argv[0], "--draws", str(D0), "--items", ITEMS]
src = open("scripts/local/t2_d540a_stack_read.py").read()
g = {"__name__": "pair_head", "__file__": os.path.abspath("scripts/local/t2_d540a_stack_read.py")}
exec(compile(src.split("\nBASE = read([])\n")[0], "t2_d540a_stack_read_head(pair)", "exec"), g)


def loadmix(p):
    z = np.load(p)
    return {(int(k.split("_")[1]), int(k.split("_")[2])): (z[k], z["mx_" + k[3:]]) for k in z.files if k.startswith("ix_")}, z


REF, zr = loadmix(paths["ref"])
ARM, za = loadmix(paths["arm"])
assert str(zr["teach"]) == "" and str(zr["base"]) == str(za["base"]) == "w20" and str(zr["idea"]) == a.ref_idea
assert REF.keys() == ARM.keys() and all(np.array_equal(REF[k][0], ARM[k][0]) for k in REF)
sc = g["judge"](ARM, REF)
sc["guards"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
sc["stack_bar"] = bool(sc["ten"] >= 0.001 and sc["guards"] and sc["sealed_mix"] > 0)
res = {"set": a.set, "tag": a.tag, "env": ENV, "base_env": BENV, "ref_env": RENV, "ref_file": paths["ref"], "idea": a.idea, "level": {"W20": g["level"](REF), "arm": g["level"](ARM)}, "arm vs W20": sc}
print(f"PAIR {a.set} {a.tag}: {res}", flush=True)
json.dump(res, open(f"{O}/t2_lab/pair_{a.set}_{a.tag}.json", "w"), indent=1, default=str)
