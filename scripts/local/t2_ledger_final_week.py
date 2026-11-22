#!/usr/bin/env python
"""T2 final-week ledger (2026-11-15, the manager's request for the method notes): every shipped step read on ONE panel, T2-D512's fresh K = 9
draws 810-812 (ten strangers every window, the six non-calibrated, the worst stranger, returning P04 / P10 sessions 2-3), start vectors
cross-fitted per variant, M7's family shares, per-family causal Sinkhorn (10 / 3 / 0.95):
  base      = D498's stack shape (D512's BASE)
  FINAL     = base + P1 (in-stream adaptation, W15) + P2 (tilt FM donors)
  +DC       = FINAL + the window-mean tree family at 0.075 (W16)
  +side     = +DC + the window-mean side nets, 'thirds' (W17)
Each step vs the previous one, and each vs base (cumulative); the part rule; the sealed-mix expectation 0.57 x strangers + 0.43 x returning.
    mem-guard python scripts/local/t2_ledger_final_week.py
"""
import json
import sys

src = open("scripts/local/t2_d517_final.py").read().split("\nres = {}\n")[0]
exec(compile(src, "t2_d517_final_head", "exec"))   # t2_d517_read head (D512 head, mixtures, score) + form_parts / mix

V = {"base": mixtures("base", {}, 0.0), "FINAL": mix("final", None, 0.0), "+DC": mix("final", None, 0.075), "+side": mix("thirds", "dcmean", 0.075)}
names = list(V)
res = {}
for k in range(1, len(names)):
    a, b = names[k], names[k - 1]
    for ref in (b, "base") if b != "base" else (b,):
        e = score(V[a], V[ref])
        e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
        res[f"{a} vs {ref}"] = e
        print(f"LEDGER {a:6s} vs {ref:6s}: {e}", flush=True)
json.dump(res, open(O_ / "t2_ledger_final_week.json", "w"), indent=1, default=str)
