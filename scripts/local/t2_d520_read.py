#!/usr/bin/env python
"""T2-D520 read (written 2026-11-15 before any D520 number): the grid64d ST-L (G, seeds 830-832) vs a 7th plain mean ST-L (C7, seeds 833-835) as an
ADDED FM member, on D512's K = 9 draws 810-812 (draw d takes seed index d - 810). The stack = FINAL + DC + side (W17 'thirds'), the FM slot = the
mean of 7 (D512's six tilt members 820-825 + the 7th) vs the shipped 6. Reads: G7 vs C7 (the decision), G7 vs 6 and C7 vs 6; the member alone
(the FM family decoded by its own causal Sinkhorn, strangers: G vs C). Fetch first: s1 fetch ni26 ni26-t2p-d520-<arm>-<jobid> (each run dir).
    mem-guard python scripts/local/t2_d520_read.py
"""
import glob
import json

import numpy as np

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head (S, load, realign, fm_mean ...), d517 / d518 helpers, side_files, mix pieces

SEEDS = {"G": (830, 831, 832), "C7": (833, 834, 835)}


def member(arm, s, f):
    tag = "grid" if arm == "G" else "c7"
    ro = "grid64d" if arm == "G" else "mean"
    fs = glob.glob(f"{S}/ni26-t2p-d520-{tag}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/runs/lc9_{f}__fm-st_large-{ro}__sz__n00__*__s{s}.npz")
    assert len(fs) == 1, (arm, s, f, fs)
    return load(fs[0])


def mix7(arm, w_dc=0.075):
    """FINAL + DC + side ('thirds', D517's single side net) with the FM slot = (6 * the 6-member mean + the 7th) / 7; arm None = the shipped 6."""
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        if arm is not None:
            ixm, pm = member(arm, SEEDS[arm][d - 810], f)
            Pp["fm"] = (6.0 * Pp["fm"] + realign(ixm, pm, ix)) / 7.0
        for fam in ("sh", "ifn"):
            Sn = np.mean([pooled(p, ix) for p in side_files("one", fam, f, d)], 0)
            Pp[fam] = (2.0 / 3.0) * Pp[fam] + (1.0 / 3.0) * Sn
        P[(f, d)] = Pp
    vec = {}
    for d in DRAWS:
        for fam in W:
            per = {f: P[(f, d)][fam][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
            for f in range(10):
                v = np.mean([per[g] for g in range(10) if g != f], 0)
                vec[(f, d, fam)] = v / v.sum()
    out = {}
    for (f, d), (ix, q) in RAW.items():
        key = (STREAM[ix], ORD[ix])
        mx = sum(W[fam] * causal_sinkhorn(P[(f, d)][fam], key, 10, 3, 0.95, start=vec[(f, d, fam)]) for fam in W)
        te, p = DC[f]
        pos = {int(r): k for k, r in enumerate(te)}
        q_ = p[np.array([pos[int(r)] for r in ix])]
        mx = (1 - w_dc) * mx + w_dc * causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCVEC[f])
        out[(f, d)] = (ix, mx)
    return out


def alone(arm):
    """the member alone, decoded (its own causal Sinkhorn, start = the other folds' stranger mean of the member)."""
    P = {}
    for (f, d), (ix, q) in RAW.items():
        ixm, pm = member(arm, SEEDS[arm][d - 810], f)
        P[(f, d)] = realign(ixm, pm, ix)
    out = {}
    for d in DRAWS:
        per = {f: P[(f, d)][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
        for f in range(10):
            ix = RAW[(f, d)][0]
            v = np.mean([per[g] for g in range(10) if g != f], 0)
            out[(f, d)] = (ix, causal_sinkhorn(P[(f, d)], (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=v / v.sum()))
    return out


M = {a: mix7(a) for a in (None, "G", "C7")}
res = {}
for x, ref in (("G", "C7"), ("G", None), ("C7", None)):
    e = score(M[x], M[ref])
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    res[f"{x} vs {ref or '6'}"] = e
    print(f"D520 FINAL+DC+side: {x}7 vs {(ref + '7') if ref else 'the shipped 6'}: {e}", flush=True)
e = score(alone("G"), alone("C7"))
res["alone G vs C7"] = e
print(f"D520 member alone (decoded): G vs C7: {e}", flush=True)
json.dump(res, open(O_ / "t2_d520.json", "w"), indent=1, default=str)
