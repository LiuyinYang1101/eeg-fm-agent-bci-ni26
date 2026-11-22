#!/usr/bin/env python
"""T2-D649 (W29): the lambda-max pick from the smoke by the ENGAGEMENT curve ONLY (registered + pinned before the smoke): per task (lambda 0 = the
twin, 0.1, 0.3, 1), the adversary's held-out training-user accuracy at the END OF THE RAMP (the first epoch whose step >= 0.3 x the total steps),
averaged over the task's folds; lambda max = the SMALLEST lambda > 0 whose accuracy there has fallen >= 50 % of the way from the twin's to chance
(1 / n_users); none -> "not engaged" (D371: not tested, not failed). Never reads a score.
  python scripts/local/t2_d649_lambda_pick.py <the fetched smoke task dirs ...>  -> outputs/t2_d649/lambda_pick.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"
acc = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/runs/*.npz")):
        if "grl" not in Path(f).name:
            continue
        res = json.loads(str(np.load(f)["res"]))
        lam = float(res["fm_args"]["grl"])
        cv, end = res["grl_curve"], res["grl_ramp_end_step"]
        at = next(c for c in cv if c["step"] >= end)
        acc.setdefault(lam, []).append((at["adv_hold_acc"], res["grl_chance"], at["epoch"], Path(f).name[:40]))
assert 0.0 in acc and len(acc) >= 2, sorted(acc)
twin = float(np.mean([a for a, _, _, _ in acc[0.0]]))
chance = float(np.mean([c for _, c, _, _ in acc[0.0]]))
rows, pick = {}, None
for lam in sorted(acc):
    a = float(np.mean([x for x, _, _, _ in acc[lam]]))
    frac = (twin - a) / max(twin - chance, 1e-9)
    rows[lam] = dict(acc=a, frac_to_chance=frac, n=len(acc[lam]), epoch=acc[lam][0][2])
    if lam > 0 and pick is None and frac >= 0.5:
        pick = lam
out = dict(twin_acc=twin, chance=chance, rows=rows, lambda_max=pick, engaged=pick is not None)
(O / "t2_d649").mkdir(parents=True, exist_ok=True)
(O / "t2_d649" / "lambda_pick.json").write_text(json.dumps(out, indent=1, default=float))
print(f"T2-D649 smoke: twin acc {twin:.3f}, chance {chance:.3f}; " + "; ".join(f"lambda {k:g}: acc {v['acc']:.3f} ({v['frac_to_chance']:+.0%} of the way, n {v['n']})"
                                                                     for k, v in rows.items()) + f" -> lambda max {pick if pick is not None else 'NONE (not engaged)'}")
