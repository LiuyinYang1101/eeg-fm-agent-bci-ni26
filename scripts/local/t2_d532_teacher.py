#!/usr/bin/env python
"""T2-D532 (registered 2026-11-16 before any number; zero GPU): the P1 teacher files for the W18 read's FRESH CNN draws 813-815 =
scripts/local/t2_d512_teacher.py EXACTLY (its D511 construction: the fresh CNN base + T2-D498's six FM members, causal, every test row; the
prefix re-decodes post_r60 / r100 / r150), with three constants redirected: R = outputs/t2_d532 (the fresh routed members, t2_d532_cnn_local.sh),
OUT = outputs/t2_d532/teacher_p1, DRAWS = (813, 814, 815). Each redirect is asserted to occur exactly once in the source.
    mem-guard python scripts/local/t2_d532_teacher.py
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "t2_d512_teacher.py"
src = SRC.read_text()
for a, b in (('R = O / "t2_d512"\n', 'R = O / "t2_d532"\n'),
             ('OUT = O / "t2_d512/teacher_p1"\n', 'OUT = O / "t2_d532/teacher_p1"\n'),
             ("DRAWS = (810, 811, 812)\n", "DRAWS = (813, 814, 815)\n")):
    assert src.count(a) == 1, (a, src.count(a))
    src = src.replace(a, b)
exec(compile(src, str(SRC), "exec"), {"__file__": str(SRC), "__name__": "__main__"})
