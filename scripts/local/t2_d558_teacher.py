#!/usr/bin/env python
"""T2-D558 (the second-read set on FRESH draws 831-833; zero GPU): the P1 teacher files = scripts/local/t2_d512_teacher.py EXACTLY (as
t2_d536_teacher.py), with three constants redirected: R = outputs/t2_d558, OUT = outputs/t2_d558/teacher_p1, DRAWS = (831, 832, 833).
Each redirect is asserted to occur exactly once in the source.
    mem-guard python scripts/local/t2_d558_teacher.py
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "t2_d512_teacher.py"
src = SRC.read_text()
for a, b in (('R = O / "t2_d512"\n', 'R = O / "t2_d558"\n'),
             ('OUT = O / "t2_d512/teacher_p1"\n', 'OUT = O / "t2_d558/teacher_p1"\n'),
             ("DRAWS = (810, 811, 812)\n", "DRAWS = (831, 832, 833)\n")):
    assert src.count(a) == 1, (a, src.count(a))
    src = src.replace(a, b)
exec(compile(src, str(SRC), "exec"), {"__file__": str(SRC), "__name__": "__main__"})
