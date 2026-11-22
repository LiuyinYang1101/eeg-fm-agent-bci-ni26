#!/usr/bin/env python
"""T2-D551 step 1 (registered 2026-11-17 before this pick existed): D526's REGISTERED SINGLE FORM = the ALL-TEN pick on draws 810-812 in W17's
context, with t2_d525_read.py's own head, 80-point grid, objective (stranger_delta) and tie-break (closest to the shipped shares), all ten
folds instead of D526's halves. The halves' picks are recomputed first and asserted equal to D526's recorded picks (the code path is D526's).
The pick's in-sample delta is SELECTION on 810-812, never a result; the pick enters the 819-821 stack read (D551) as fixed shares.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d551_d526_pick.py
"""
import json
import sys

import numpy as np

sys.argv = [sys.argv[0], "--rows", "D526"]
SRC = open("scripts/local/t2_d525_read.py").read()
CUT = '\nif "D525" in ROWS:'
assert SRC.count(CUT) == 1, "the D525 reader's tail moved"
exec(compile(SRC.split(CUT)[0], "t2_d525_read_head(d551)", "exec"))    # REF == W17 asserted inside; combine, stranger_delta, W, D0
import itertools  # noqa: E402

ship = np.array([W["fbc"], W["sh"], W["ifn"], W["fm"]])
GRID = []
for fm, ifn, shf in itertools.product((0.25, 0.30, 0.35, 0.40, 0.45), (0.05, 0.10, 0.15, 0.20), (0.35, W["sh"] / (W["fbc"] + W["sh"]), 0.50, 0.57)):
    rest = 1.0 - fm - ifn
    GRID.append({"fbc": rest * (1 - shf), "sh": rest * shf, "ifn": ifn, "fm": fm})
assert len(GRID) == 80


def pick(folds):
    sc = []
    for g in GRID:
        A = combine(D0, g, folds=set(folds))
        dist = float(np.abs(np.array([g["fbc"], g["sh"], g["ifn"], g["fm"]]) - ship).sum())
        sc.append((stranger_delta(A, folds), -dist, g))
    sc.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return sc[0][2], [dict(delta=round(s, 4), **{k: round(v, 4) for k, v in g.items()}) for s, _, g in sc[:5]]


rec = json.load(open(O_ / "t2_d525_D525_D526_D527_D528.json"))["D526 picks"]
for h, fl in (("A", range(0, 5)), ("B", range(5, 10))):
    g, _ = pick(fl)
    assert {k: round(v, 4) for k, v in g.items()} == rec[h], (h, g, rec[h])
print(f"D551 check: the half picks reproduce D526's recorded picks {rec}", flush=True)
g10, top = pick(range(10))
out = {"D526 all-ten pick (810-812, W17 context; the registered single form)": {k: float(v) for k, v in g10.items()},
       "top5 in-sample (SELECTION, not a result)": top, "shipped": {k: float(v) for k, v in W.items()}, "W_DC": W_DC}
json.dump(out, open(O_ / "t2_d551_d526_pick.json", "w"), indent=1)
print(f"D551 D526 all-ten pick: {out}", flush=True)
