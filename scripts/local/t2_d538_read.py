#!/usr/bin/env python
"""T2-D538 (W19-F; registered 2026-11-16 19:36:59 before any number; committed before its inputs exist): the FBCNet window-mean side nets, read
on D536's draws 816-818 (a NOMINATION read; the manager's condition 1: a nominated item is confirmed on fresh draws 819-821 before any W19 line).

D536's reader head is exec'd unchanged (D532's head at DRAWS 816-818: W18's form = FM SD six + MI at 826, side nets UNC, DC bag 7-9).
BASE = W18's form; its level is asserted equal to D536's recorded BASE level.
W19-F = BASE with the FBCNet part (2/3) * routed + (1/3) * mean(the 5 FBCNet side nets, seeds d + k * 1000) BEFORE its corrector (the template's
single side_r = 1/3 applied to cnn1 as to cnn2 / cnn3; FBCNet has no in-stream part). Side outputs recomputed from the dumps (side_prob 'unc':
checked vs prob_last <= 1e-5 and the dumped z vs side_z <= 1e-6).
Rule (D536's): ten > 0, >= 2 / 3 draws, no stranger < -0.020, returning >= -0.003, AND ten >= +0.001 -> NOMINATED.
Descriptive: the FBCNet side nets' mean alone vs the FBCNet routed part alone (argmax, no corrector; ten strangers' cell BA).
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d538_read.py
"""
import json
import time

import numpy as np

T0 = time.time()
SRC36 = open("scripts/local/t2_d536_read.py").read()
CUT = '\nW18 = ["D524", "D529"]\n'
assert SRC36.count(CUT) == 1, "the D536 reader's tail moved"
exec(compile(SRC36.split(CUT)[0], "t2_d536_read_head(d538)", "exec"))
assert DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")

D536 = json.load(open(O / "t2_d536.json"))
FBC_SIDE = O / "t2_d538/side_fbc/runs"
FBC_STEM = "lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr__sidedcmean__s{s}.npz"
WITH_FBC = False
_parts_of_d536 = parts_of


def fbc_side(f, d):
    ix = IX[f]
    return np.mean([realign(*side_prob(FBC_SIDE / FBC_STEM.format(f=f, s=d + k * 1000), f, "unc"), ix) for k in range(5)], 0)


def parts_of(f, d, parts):                       # D536's parts_of; with WITH_FBC the FBCNet part gets its side nets at r 1/3
    P = _parts_of_d536(f, d, parts)
    if WITH_FBC:
        P = dict(P)
        P["fbc"] = (2.0 / 3.0) * P["fbc"] + (1.0 / 3.0) * fbc_side(f, d)
    return P


def alone(fn):
    per = {}
    for f in range(10):
        ix = IX[f]
        m = U[ix] == TRU[f]
        for d in DRAWS:
            c = cells(fn(f, d).argmax(1)[m], ix[m])
            per.setdefault(TRU[f], []).append(np.mean(list(c.values())))
    pu = {u: float(np.mean(v)) for u, v in per.items()}
    return round(float(np.mean(list(pu.values()))), 4), {u: round(v, 4) for u, v in sorted(pu.items())}


W18 = ["D524", "D529"]
SIDE_MODE = "unc"
BASE = mix(W18)
lv = level(BASE)
assert abs(lv - D536["level"]["BASE (W18 form)"]) < 5e-5, (lv, D536["level"])
n_before = CHK["files"]
WITH_FBC = True
F = mix(W18)
n_f = CHK["files"] - n_before
print(f"D538 checks: {CHK}; side files read in the W19-F mix: {n_f} (450 expected: 300 ShallowNet / IFNet + 150 FBCNet)", flush=True)
assert n_f == 450, n_f
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
res = {"draws": list(DRAWS), "checks": dict(CHK), "level": {"BASE (W18 form)": lv, "W19F (FBCNet side nets)": level(F)}}
head = score(F, BASE)
head = dict(per_stranger=per_stranger(F, BASE), **head)
head["nominated"] = bool(head["part"] and head["ten"] >= 0.001)
res["HEADLINE W19F vs BASE (W18 form), draws 816-818"] = head
a_side = alone(fbc_side)
a_rt = alone(lambda f, d: np.mean([realign(*load(CNN / RT["fbc"].format(f=f, s=d)), IX[f])], 0))
res["descriptive: FBCNet side nets alone vs FBCNet routed alone (argmax, ten strangers)"] = dict(side=a_side, routed=a_rt,
                                                                                                    delta=round(a_side[0] - a_rt[0], 4))
for k, v in res.items():
    print(f"D538 {k}: {v}", flush=True)
print(f"D538 verdict: {'NOMINATED (confirm on fresh draws 819-821 before any W19 line)' if head['nominated'] else 'NOT NOMINATED (closed)'}"
      f" ({time.time() - T0:.0f} s)", flush=True)
json.dump(res, open(O / "t2_d538.json", "w"), indent=1, default=str)
