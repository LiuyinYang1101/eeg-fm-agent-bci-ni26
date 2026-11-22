#!/usr/bin/env python
"""Per-RETURNING-user deltas of a saved pair read (the astra R3 protection, T2-D594b: the pooled returning gain must be >= 0 for P04 AND P10
separately). = scripts/local/t2_d532_read.py score()'s 'returning' part (cells of sessions 2-3 of non-held users, per user: the mean cell
delta over folds x draws), without the final mean over users. Reads the reader head exactly as t2_lab_pair_read.py does.
    python scripts/local/t2_ret_users.py --set A|B|C|D --ref <pair_mix ref npz> --arm <pair_mix arm npz>  ->  one 'RET {...}' line
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=("A", "B", "C", "D", "E"))
ap.add_argument("--ref", required=True)
ap.add_argument("--arm", required=True)
a = ap.parse_args()
D0 = {"A": 825, "B": 828, "C": 831, "D": 834, "E": 860}[a.set]   # T2-D615: E
ITEMS = {825: "D539b,D543,D546b,D553,D555", 828: "D539b,D543,D546b,D553,D557", 831: "D539b,D543,D546b,D553,D555", 834: "D539b,D543,D546b,D553,D557", 860: "D539b,D543,D546b,D553,D557"}[D0]
for k in [k for k in os.environ if k.startswith("NI26_")]:
    del os.environ[k]
sys.argv = [sys.argv[0], "--draws", str(D0), "--items", ITEMS]
src = open("scripts/local/t2_d540a_stack_read.py").read()
g = {"__name__": "ret_head", "__file__": os.path.abspath("scripts/local/t2_d540a_stack_read.py")}
exec(compile(src.split("\nBASE = read([])\n")[0], "t2_d540a_stack_read_head(ret)", "exec"), g)


def loadmix(p):
    z = np.load(p)
    return {(int(k.split("_")[1]), int(k.split("_")[2])): (z[k], z["mx_" + k[3:]]) for k in z.files if k.startswith("ix_")}


A_, B_ = loadmix(a.arm), loadmix(a.ref)
assert A_.keys() == B_.keys()
U, SES, TRU, cells = g["U"], g["SES"], g["TRU"], g["cells"]
recs = []
for (f, d), (ix, ma) in A_.items():
    mb = B_[(f, d)][1]
    ret = (U[ix] != TRU[f]) & np.isin(SES[ix], ("2", "3"))
    for u in np.unique(U[ix][ret]):
        m = ret & (U[ix] == u)
        ca, cb = cells(ma.argmax(1)[m], ix[m]), cells(mb.argmax(1)[m], ix[m])
        for c in ca:
            recs.append(dict(user=str(u), d=ca[c] - cb[c]))
R = pd.DataFrame(recs)
out = {u: round(float(v), 4) for u, v in R.groupby("user").d.mean().items()}
print("RET " + json.dumps(dict(set=a.set, per_user=out, mean=round(float(np.mean(list(out.values()))), 4))), flush=True)
