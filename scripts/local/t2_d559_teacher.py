#!/usr/bin/env python
"""T2-D559 (the refit-schedule study; registered 2026-11-17 before any number; zero GPU): the P1 teacher files for one schedule = scripts/local/
t2_d512_teacher.py EXACTLY with four constants redirected (each asserted to occur once): R = OUT's root (outputs/t2_d559_<tag>_<d0>, whose
route_* are symlinks to the draw set's chain), OUT = R/teacher_p1, DRAWS = (d0, d0 + 1, d0 + 2), CKS = the schedule (post_r<c> for every c).
    python scripts/local/t2_d559_teacher.py --root t2_d559_s6_825 --draws 825 --cks 40,60,80,100,125,150
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--draws", type=int, required=True, choices=(825, 828, 831))
ap.add_argument("--cks", required=True)
a = ap.parse_args()
assert a.root.startswith("t2_d559_") and a.root.endswith(f"_{a.draws}"), a.root
cks = tuple(int(x) for x in a.cks.split(","))
assert list(cks) == sorted(set(cks)) and cks[0] >= 2, cks
SRC = Path(__file__).resolve().parent / "t2_d512_teacher.py"
src = SRC.read_text()
for x, y in (('R = O / "t2_d512"\n', f'R = O / "{a.root}"\n'),
             ('OUT = O / "t2_d512/teacher_p1"\n', f'OUT = O / "{a.root}/teacher_p1"\n'),
             ("DRAWS = (810, 811, 812)\n", f"DRAWS = {(a.draws, a.draws + 1, a.draws + 2)}\n"),
             ("CKS = (60, 100, 150)\n", f"CKS = {cks}\n")):
    assert src.count(x) == 1, (x, src.count(x))
    src = src.replace(x, y)
exec(compile(src, str(SRC), "exec"), {"__file__": str(SRC), "__name__": "__main__"})
