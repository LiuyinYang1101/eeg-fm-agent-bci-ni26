#!/usr/bin/env python
"""T2-D454 POST-HOC, descriptive (written AFTER the pre-registered read, which failed its bar; no bar here, no adoption from this).
Question raised by the read: the MI expert alone in M7's shape read +0.0110 vs one control net. Is that the MI first stage or the comparison
(1 net vs 1 net at a 3-net weight)?  Same harness, extra arms at EQUAL net count: MI3 = three MI-expert restarts, MIX = 2 control + 1 MI,
MIX2 = 1 control + 2 MI, each vs C3 (three control restarts).
    mem-guard python scripts/local/t2_d454_post.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "t2_d454_read.py").read_text()
a = 'rows = {}\n'
assert src.count(a) == 1
src = src.replace(a, 'ARMS = {"C3": ARMS["C3"], "MI3": [(("mi", "FT"), d) for d in (0, 1, 2)], "MIX": [(("ctl", "FT"), 0), (("ctl", "FT"), 1), (("mi", "FT"), 2)],'
                     ' "MIX2": [(("ctl", "FT"), 0), (("mi", "FT"), 1), (("mi", "FT"), 2)]}\n' + a)
b = '(("V_ft", "C3"), ("V_lp", "C3"), ("V_lp", "C3_lp"), ("mi_FT", "ctl_FT"), ("math_FT", "ctl_FT"), ("word_FT", "ctl_FT"))'
assert src.count(b) == 1
src = src.replace(b, '(("MI3", "C3"), ("MIX", "C3"), ("MIX2", "C3"))').replace("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(HERE)!r})")
sys.path.insert(0, str(HERE))
exec(compile(src, "t2_d454_read.py", "exec"))
