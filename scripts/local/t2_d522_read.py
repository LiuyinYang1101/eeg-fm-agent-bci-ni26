#!/usr/bin/env python
"""T2-D522 / D523 read (written 2026-11-15 before any D522 / D523 number), in the W17 context on D512's K = 9 draws 810-812: FINAL + the side nets
(S5, 'thirds') + a window-mean tree family through its own causal Sinkhorn (cross-fitted start).
  D522: the dc8 bag (outputs/t2_d515/pred_s1-2-3_dc8.npz, the 8 half-second means per channel) IN PLACE of the 82-mean bag at share 0.075, vs W17;
  D523: the 82-mean bag at share 0.05 and 0.10, vs the shipped 0.075.
Part rule; sealed mix 0.57 x strangers + 0.43 x returning.
    mem-guard python scripts/local/t2_d522_read.py
"""
import json

import numpy as np

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head, side_files, score, parts_of, FIN, DC (82-mean bag), DCVEC


def bag(path):
    z = np.load(path)
    D_ = {f: (z[f"te_{f}"], z[f"bag_{f}"]) for f in range(10)}
    V = {}
    for f in range(10):
        v = np.mean([D_[g][1][U[D_[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
        V[f] = v / v.sum()
    return D_, V


def mix(arm, w_dc, DCs, DCV):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        for fam in ("sh", "ifn"):
            S_ = np.mean([pooled(p, ix) for p in side_files(arm, fam, f, d)], 0)
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
        te, p = DCs[f]
        pos = {int(r): k for k, r in enumerate(te)}
        q_ = p[np.array([pos[int(r)] for r in ix])]
        mx = (1 - w_dc) * mx + w_dc * causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCV[f])
        out[(f, d)] = (ix, mx)
    return out


DC8, DC8V = bag(O_ / "t2_d515/pred_s1-2-3_dc8.npz")
REF = mix("S5", 0.075, DC, DCVEC)                                  # W17 (the shipped form: 5 side seeds)
res = {}
for name, A in (("D522 dc8 bag @0.075 vs W17 (82-mean @0.075)", mix("S5", 0.075, DC8, DC8V)),
                ("D523 82-mean @0.05 vs @0.075", mix("S5", 0.05, DC, DCVEC)),
                ("D523 82-mean @0.10 vs @0.075", mix("S5", 0.10, DC, DCVEC))):
    e = score(A, REF)
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    res[name] = e
    print(f"{name}: {e}", flush=True)
json.dump(res, open(O_ / "t2_d522_d523.json", "w"), indent=1, default=str)
