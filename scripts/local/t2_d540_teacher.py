#!/usr/bin/env python
"""T2-D540 (the W19 confirm chain on FRESH draws 819-821; zero GPU): the P1 teacher files = scripts/local/t2_d512_teacher.py EXACTLY (as
t2_d536_teacher.py), with three constants redirected: R = outputs/t2_d540, OUT = outputs/t2_d540/teacher_p1, DRAWS = (819, 820, 821).
Each redirect is asserted to occur exactly once in the source.
    mem-guard python scripts/local/t2_d540_teacher.py
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "t2_d512_teacher.py"
src = SRC.read_text()
for a, b in (('R = O / "t2_d512"\n', 'R = O / "t2_d540"\n'),
             ('OUT = O / "t2_d512/teacher_p1"\n', 'OUT = O / "t2_d540/teacher_p1"\n'),
             ("DRAWS = (810, 811, 812)\n", "DRAWS = (819, 820, 821)\n")):
    assert src.count(a) == 1, (a, src.count(a))
    src = src.replace(a, b)
exec(compile(src, str(SRC), "exec"), {"__file__": str(SRC), "__name__": "__main__"})
