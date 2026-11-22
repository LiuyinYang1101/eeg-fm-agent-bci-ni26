#!/usr/bin/env python
"""T2-D524 EARLY LOOK (registered 2026-11-16 before any D524 number; descriptive, not the decision): 21 of 24 jobs ended, so SD vs CTL on
the SAME available seeds per fold half (a = folds 0-4: 820, 821, 823, 824, 825; b = folds 5-9: 820, 823, 824, 825), the FM slot = the mean
of those members, in W17 (FINAL + DC 0.075 + S5 side nets 'thirds'). The registered read (t2_d524_read.py, all six) decides.
    mem-guard python scripts/local/t2_d524_early.py
"""
import glob
import json

import numpy as np

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head (S, load, realign, parts_of, FIN), side_files, score, DC, DCVEC, mix_arm

SEEDS = {"a": (820, 821, 823, 824, 825), "b": (820, 823, 824, 825)}


def fm_arm(arm, f):
    h = "a" if f < 5 else "b"
    ps, ixs = [], None
    for s in SEEDS[h]:
        base = "stlcar" if s >= 823 else "stl"
        fs = glob.glob(f"{S}/ni26-t2p-d524-{base}{arm}_s{s}_{h}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
        assert len(fs) == 1, (arm, s, f, fs)
        ix_, p = load(fs[0])
        assert ixs is None or (ix_ == ixs).all()
        ixs = ix_
        ps.append(p)
    return ixs, np.mean(ps, 0)


FMA = {arm: {f: fm_arm(arm, f) for f in range(10)} for arm in ("sd", "ctl")}


def mix(fm):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        Pp["fm"] = realign(*FMA[fm][f], ix)
        for fam in ("sh", "ifn"):
            S_ = np.mean([pooled(p, ix) for p in side_files("S5", fam, f, d)], 0)
            Pp[fam] = (2.0 / 3.0) * Pp[fam] + (1.0 / 3.0) * S_
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
        mx = 0.925 * mx + 0.075 * causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCVEC[f])
        out[(f, d)] = (ix, mx)
    return out


M = {fm: mix(fm) for fm in ("sd", "ctl")}
REF = mix_arm("S5")
res = {}
for a, b in (("sd", "ctl"),):
    e = score(M[a], M[b])
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    res["EARLY W17: FM sd vs FM ctl (same available seeds)"] = e
    print(f"D524 EARLY W17: FM sd vs ctl (same seeds): {e}", flush=True)
for fm in ("sd", "ctl"):
    e = score(M[fm], REF)
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    res[f"EARLY W17: FM {fm} (subset) vs W17 (ORIG six), descriptive"] = e
    print(f"D524 EARLY W17: FM {fm} (subset) vs W17 ORIG six (descriptive): {e}", flush=True)
json.dump(res, open(O_ / "t2_d524_early.json", "w"), indent=1, default=str)
