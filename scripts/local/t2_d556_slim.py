#!/usr/bin/env python
"""T2-D556 (DESCRIPTIVE, no decision; the manager's runtime-insurance ask after the field analysis): how much of W19 survives a SLIMMER FM slot?
On fresh draws 822-824 (the D553 read's files), W19 = BASE + D539b + D543 + D546b + D553 (the stack reader's head exec'd unchanged).
The FM slot (six ST-L members, ~4 H200 ms each, ~45 % of the per-window time) is the one component whose size the stored outputs can vary
(the CNN families are one seed per draw locally; their 15-seed bag and TTA views are fixed in the package, not in these files).
  (1) each FM member's leave-one-out (W19 vs W19 without it), ten / sealed mix;
  (2) a greedy drop order, least valuable first (selected ON these draws: descriptive), and W19 with 5 / 4 / 3 members vs the full W19.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d556_slim.py
"""
import json
import sys
import time

T0 = time.time()
sys.argv = [sys.argv[0], "--draws", "822", "--items", "D539b,D543,D546b,D553"]
SRC = open("scripts/local/t2_d540a_stack_read.py").read()
CUT = "\nBASE = read([])\n"
assert SRC.count(CUT) == 1
exec(compile(SRC.split(CUT)[0], "t2_d540a_stack_read_head(d556)", "exec"))
ON = ["D539b", "D543", "D546b", "D553"]
DROP = set()
_fm_forms0 = fm_forms


def fm_forms(parts):
    return [(s, f_) for s, f_ in _fm_forms0(parts) if s not in DROP]


FULL = read(ON)
seeds = [s for s, _ in _fm_forms0(W18)]
res = {"members": seeds, "level_full": level(FULL), "loo": {}}
for s in seeds:
    DROP = {s}
    res["loo"][s] = judge(FULL, read(ON))
    DROP = set()
    print(f"D556 LOO member {s}: {res['loo'][s]}", flush=True)
order = sorted(seeds, key=lambda s: res["loo"][s]["sealed_mix"])
res["greedy_drop_order"] = order
res["slim"] = {}
for k in (1, 2, 3):
    DROP = set(order[:k])
    sc = judge(read(ON), FULL)
    res["slim"][6 - k] = dict(dropped=sorted(DROP), **sc)
    print(f"D556 W19 with {6 - k} FM members (dropped {sorted(DROP)}) vs full: {sc}", flush=True)
DROP = set()
json.dump(res, open(O / "t2_d556_slim.json", "w"), indent=1, default=str)
print(f"D556 done ({time.time() - T0:.0f} s)", flush=True)
