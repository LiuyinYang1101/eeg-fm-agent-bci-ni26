#!/usr/bin/env python
"""T2-D467 POST HOC (descriptive, written after the read; no adoption): the two strongest classical members only (fbts, fbcsp), and fbts
alone, at w 0.05 / 0.10 on the same shape C, so the near-chance members (mdm, xdts) do not decide the answer.
    mem-guard python scripts/local/t2_d467_post.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "t2_d467_read.py").read_text()
cut = src.index("R = {a: cell_rows(spec, a, 12, idx) for a in arms}")
src = src[:cut] + '''arms = {"C": C}
for name, mem in (("fbts", ["fbts"]), ("fbts+fbcsp", ["fbts", "fbcsp"])):
    for w in (0.05, 0.10):
        arms[f"C+{name}@{w:.2f}"] = {**{k: v * (1 - w) for k, v in C.items()}, **{k: w / len(mem) for k in mem}}
spec = {"members": M, "arms": arms}
R = {a: cell_rows(spec, a, 12, idx) for a in arms}
key = ["proto", "split", "i", "cell"]
for a in arms:
    if a == "C":
        continue
    m = R[a].merge(R["C"][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    print(f"{a} vs C: typical {d.mean():+.4f} ({int((d > 0).sum()):2d}/12)  " + " ".join(f"{s} {v:+.4f}" for s, v in (m.ba - m.ba_b).groupby(m.subj).mean().items()), flush=True)
'''
sys.path.insert(0, str(HERE))
exec(compile(src, "t2_d467_read.py", "exec"))
