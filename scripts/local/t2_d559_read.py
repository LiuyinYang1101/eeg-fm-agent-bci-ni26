#!/usr/bin/env python
"""T2-D559 read (the refit-schedule study; registered 2026-11-17 before any number). REF = W19 (read(D539b, D543, D546b, D553)) on the draw
set's chain at the shipped refit points 60 / 100 / 150; ARM = the SAME read with the schedule's root (outputs/t2_d559_<tag>_<d0>: the same CNN
members, its own teacher at the schedule's points, its own in-stream students) and the reader's CKS = the schedule (every in-stream refit of
W19 moves with it: P1 students, D539b side heads, D553 FM side heads). Each side is computed in its own process (the reader's globals), the
mixes saved, then judged in the standard head with the lab's rule (= D540a's): guards = ten > 0, >= 2 / 3 draws > 0, no stranger < -0.020,
returning >= -0.003; stack bar = guards and ten >= +0.001 and sealed mix > 0.
    mem-guard env OMP_NUM_THREADS=4 ... python scripts/local/t2_d559_read.py --draws 825 --tag s6 --cks 40,60,80,100,125,150
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, required=True, choices=(825, 828, 831))
ap.add_argument("--tag", required=True)
ap.add_argument("--cks", required=True)
ap.add_argument("--mix-out", default="")
a = ap.parse_args()
ITEMS = {825: "D539b,D543,D546b,D553,D555", 828: "D539b,D543,D546b,D553,D557", 831: "D539b,D543,D546b,D553,D555"}[a.draws]
OUTD = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
ROOT = f"t2_d559_{a.tag}_{a.draws}"


def head():
    sys.argv = [sys.argv[0], "--draws", str(a.draws), "--items", ITEMS]
    src = open("scripts/local/t2_d540a_stack_read.py").read()
    cut = "\nBASE = read([])\n"
    assert src.count(cut) == 1, "the stack reader's tail moved"
    g = {"__name__": "t2_d559_head", "__file__": os.path.abspath("scripts/local/t2_d540a_stack_read.py")}
    exec(compile(src.split(cut)[0], "t2_d540a_stack_read_head(d559)", "exec"), g)
    return g


if a.mix_out:                                            # one side: W19's mixes in this process's environment
    g = head()
    M = g["read"](g["W19C"] + ["D553"])
    np.savez(a.mix_out, **{f"ix_{f}_{d}": ix for (f, d), (ix, mx) in M.items()}, **{f"mx_{f}_{d}": mx for (f, d), (ix, mx) in M.items()},
             cks=np.array(g["CKS"]), chk=json.dumps({k: v for k, v in g["CHK"].items() if not isinstance(v, list)}, default=str))
    print(f"[d559] side saved: {a.mix_out} cks {g['CKS']} ({time.time() - T0:.0f} s)", flush=True)
    raise SystemExit(0)

assert os.path.isdir(f"{OUTD}/{ROOT}/p1_sh/runs") and os.path.isdir(f"{OUTD}/{ROOT}/teacher_p1"), ROOT
paths = {}
for side in ("ref", "arm"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NI26_D559_")}
    if side == "arm":
        env.update(NI26_D559_ROOT=ROOT, NI26_D559_CKS=a.cks)
    paths[side] = f"{OUTD}/{ROOT}/mix_{side}.npz"
    subprocess.run([sys.executable, "-u", __file__, "--draws", str(a.draws), "--tag", a.tag, "--cks", a.cks, "--mix-out", paths[side]],
                   env=env, check=True)
g = head()


def loadmix(p):
    z = np.load(p)
    return {(int(k.split("_")[1]), int(k.split("_")[2])): (z[k], z["mx_" + k[3:]]) for k in z.files if k.startswith("ix_")}, z


REF, zr = loadmix(paths["ref"])
ARM, za = loadmix(paths["arm"])
assert tuple(zr["cks"]) == (60, 100, 150) and tuple(za["cks"]) == tuple(int(x) for x in a.cks.split(",")), (zr["cks"], za["cks"])
assert REF.keys() == ARM.keys() and all(np.array_equal(REF[k][0], ARM[k][0]) for k in REF)
sc = g["judge"](ARM, REF)
sc["guards"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
sc["stack_bar"] = bool(sc["ten"] >= 0.001 and sc["guards"] and sc["sealed_mix"] > 0)
res = {"draws": list(g["DRAWS"]), "tag": a.tag, "cks": a.cks, "level": {"W19": g["level"](REF), f"W19 @ {a.cks}": g["level"](ARM)},
       "schedule vs W19": sc, "checks": {"ref": json.loads(str(zr["chk"])), "arm": json.loads(str(za["chk"]))}}
for k, v in res.items():
    print(f"D559 {k}: {v}", flush=True)
print(f"D559 {a.tag} @ {a.draws}: {'PASS (stack bar)' if sc['stack_bar'] else 'FAIL'} ({time.time() - T0:.0f} s)", flush=True)
json.dump(res, open(f"{OUTD}/t2_d559_{a.tag}_{a.draws}_read.json", "w"), indent=1, default=str)
