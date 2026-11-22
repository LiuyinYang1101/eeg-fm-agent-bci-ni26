#!/usr/bin/env python
"""T2-D518 / D519 read (written 2026-11-15 before any D518 / D519 number): the W17 side-net slot filled by 5 pooled side nets per family, inside
FINAL+DC ('thirds': part = 2/3 FINAL part + 1/3 the mean of the 5), on D512's K = 9 draws 810-812 (ten strangers every window, six, returning).
  one = D517's single shipped-config side net (seed d)            S5 = shipped config, seeds d, d + 1000 ... d + 4000
  C5  = shipped (seed d) + 4 config variants (seed d)              V5 = D519's views (none, ma6, ema3, dt37 at seed d + a second none at d + 1000)
Decisions: C5 vs S5 (D518), V5 vs S5 (D519); also S5 vs one and C5 vs one; part rule.
    mem-guard python scripts/local/t2_d518_read.py [--arms C5,V5]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

src = open("scripts/local/t2_d517_final.py").read().split("\nres = {}\n")[0]
exec(compile(src, "t2_d517_final_head", "exec"))   # D512 head, pooled(), score(), DC / DCVEC, FIN, REC, D7

D8, D9, D21 = O_ / "t2_d518", O_ / "t2_d519", O_ / "t2_d521"
VAR = {"sh": ("shallow_t13", "shallow_t41", "shallow_p50", "shallow_f60"), "ifn": ("ifnet_k127", "ifnet_k31", "ifnet_s96", "ifnet_st4")}
VIEWS = ("ma6", "ema3", "dt37")


def side_files(arm, fam, f, d):
    m, rec = REC[fam]
    one = D7 / f"side_dcmean_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__sidedcmean__s{d}.npz"
    if arm == "one":
        return [one]
    if arm == "S5":
        return [one] + [D8 / f"S5_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__sidedcmean__s{d + k * 1000}.npz" for k in range(1, 5)]
    if arm == "C5":
        return [one] + [D8 / f"C5_{fam}/runs/lc9_{f}__{v}__ea__n00__{rec}__sidedcmean__s{d}.npz" for v in VAR[fam]]
    if arm == "V5":
        return [one, D8 / f"S5_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__sidedcmean__s{d + 1000}.npz"] + \
            [D9 / f"V_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__sidedcmean__view{v}__s{d}.npz" for v in VIEWS]
    if arm in ("R5", "R5d"):                          # T2-D521: the raw nets (R5 = raw window + level, R5d = de-meaned window), sofia fetches;
        rec = REC[fam][1]                              # the same 5 enter both slots
        tag, km = ("raw", "__km") if arm == "R5" else ("dm", "")
        out = []
        for k in range(5):
            fs = glob.glob(f"{S}/ni26-t2p-d521-{tag}_{'a' if f < 5 else 'b'}-[0-9]*/runs/lc9_{f}__rawnet__none__n00__{rec}{km}__s{d + k * 1000}.npz")
            assert len(fs) == 1, (arm, f, d, k, fs)
            out.append(Path(fs[0]))
        return out
    if arm in ("S5R5", "S5R5d"):
        return side_files("S5", fam, f, d) + side_files(arm[2:], fam, f, d)
    raise ValueError(arm)


def mix_arm(arm, w_dc=0.075):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        for fam in ("sh", "ifn"):
            S = np.mean([pooled(p, ix) for p in side_files(arm, fam, f, d)], 0)
            Pp[fam] = (2.0 / 3.0) * Pp[fam] + (1.0 / 3.0) * S
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


def alone(arm, fam="sh"):
    """T2-D521: the arm's 5 nets alone, decoded (own causal Sinkhorn, start = the other folds' stranger mean) -> {(f, d): (ix, posterior)}."""
    P = {(f, d): np.mean([pooled(p, ix) for p in side_files(arm, fam, f, d)], 0) for (f, d), (ix, q) in RAW.items()}
    out = {}
    for d in DRAWS:
        per = {f: P[(f, d)][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
        for f in range(10):
            ix = RAW[(f, d)][0]
            v = np.mean([per[g] for g in range(10) if g != f], 0)
            out[(f, d)] = (ix, causal_sinkhorn(P[(f, d)], (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=v / v.sum()))
    return out


def level(A):
    """the ten strangers' cell-mean balanced accuracy (per user, then the mean)."""
    per = {}
    for (f, d), (ix, mx) in A.items():
        m = U[ix] == TRU[f]
        c = cells(mx.argmax(1)[m], ix[m])
        per.setdefault(TRU[f], []).append(np.mean(list(c.values())))
    return round(float(np.mean([np.mean(v) for v in per.values()])), 4)


def err_corr(A, B):
    """strangers' windows: phi between A-correct and B-correct, and P(A right | B wrong)."""
    a, b = [], []
    for (f, d), (ix, ma) in A.items():
        m = U[ix] == TRU[f]
        a.append(ma.argmax(1)[m] == Y[ix][m])
        b.append(B[(f, d)][1].argmax(1)[m] == Y[ix][m])
    a, b = np.concatenate(a).astype(float), np.concatenate(b).astype(float)
    return dict(phi=round(float(np.corrcoef(a, b)[0, 1]), 3), p_right_given_stack_wrong=round(float(a[b == 0].mean()), 3),
                acc_a=round(float(a.mean()), 3), acc_stack=round(float(b.mean()), 3))


ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="C5")
a = ap.parse_args()
arms = a.arms.split(",")
need = ["one", "S5"] + arms
for arm in need:
    miss = [p for (f, d) in RAW for fam in ("sh", "ifn") for p in side_files(arm, fam, f, d) if not p.exists()]
    assert not miss, f"{arm}: {len(miss)} files missing, e.g. {miss[0]}"
M = {arm: mix_arm(arm) for arm in need}
res = {}
for x, ref in [("S5", "one")] + [(arm, "S5") for arm in arms] + [(arm, "one") for arm in arms]:
    e = score(M[x], M[ref])
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    res[f"{x} vs {ref}"] = e
    print(f"D518 FINAL+DC thirds: {x:3s} vs {ref:3s}: {e}", flush=True)
raw_arms = [r for r in ("R5", "R5d") if r in arms or ("S5" + r) in arms]
if raw_arms:                                            # T2-D521: the raw nets alone and their error correlation with the stack
    AS = alone("S5")
    res["S5 (sh side nets) alone level"] = level(AS)
    res["S5 alone err vs the stack (FINAL+DC+S5)"] = err_corr(AS, M["S5"])
    for r in raw_arms:
        AR = alone(r)
        res[f"{r} alone level"], res[f"{r} alone vs S5 alone"] = level(AR), score(AR, AS)
        res[f"{r} alone err vs the stack (FINAL+DC+S5)"] = err_corr(AR, M["S5"])
    if len(raw_arms) == 2:
        res["R5 alone vs R5d alone (the level)"] = score(alone("R5"), alone("R5d"))
    for k in [k for k in res if k.startswith(("R5", "S5 (", "S5 alone"))]:
        print(f"D521 {k}: {res[k]}", flush=True)
json.dump(res, open(O_ / f"t2_d518_{'_'.join(arms)}.json", "w"), indent=1, default=str)
