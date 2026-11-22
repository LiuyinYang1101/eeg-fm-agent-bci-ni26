#!/usr/bin/env python
"""T2-D209 (zero compute), as pre-registered in docs/DECISIONS_T2.md (the panel's T2 item 2, user-approved 2026-10-23): w4321 vs w5221,
leave-one-participant-out, on the vp2-180 package's D203 + D204 route files (= T2-D206's read_pick / read_conf rows: 24 replicates).
Participants in the mix (t2p_pkgshape groups): P04 (kit test + seen1 seen), P09 (kit test), P05 / P08 (seen1 val), P10 (seen1 seen).
  delta_p  = w4321 - w5221 on p's cells in the mix components it belongs to (the mean of its kit-test and seen1-seen columns for P04;
             kit for P09; seen1 for P05 / P08 / P10), averaged over the 24 replicates;
  fold p   : pick w4321 iff mean_{q != p} delta_q > 0 (in-sample on the other four), else w5221; held-out score_p = delta_p if w4321 picked, else 0.
  adopt w4321 iff mean_p score_p > 0 AND #{p : score_p > 0} >= 4 (of 5); otherwise w5221.
  python scripts/local/t2_d209_lopo.py
"""
import json
from pathlib import Path

import numpy as np

D = Path.home() / "workspace/outputs/neural_interface_challenge/t2_d206"
COLS = {"P04": [("kit", "P04"), ("seen1", "P04")], "P09": [("kit", "P09")], "P05": [("seen1", "P05")], "P08": [("seen1", "P08")], "P10": [("seen1", "P10")]}


def main():
    deltas = {p: [] for p in COLS}
    for f in ("read_pick.json", "read_conf.json"):
        rows = {(x["proto"], x["i"], x["arm"]): x for x in json.load(open(D / f))["rows"]}
        for i in range(12):
            for p, cs in COLS.items():
                deltas[p].append(np.mean([rows[(pr, i, "w4321")][c] - rows[(pr, i, "w5221")][c] for pr, c in cs]))
    d = {p: float(np.mean(v)) for p, v in deltas.items()}
    up = {p: int(sum(x > 0 for x in v)) for p, v in deltas.items()}
    score = {}
    for p in COLS:
        others = np.mean([d[q] for q in COLS if q != p])
        score[p] = d[p] if others > 0 else 0.0
        print(f"fold {p}: others mean {others:+.4f} -> pick {'w4321' if others > 0 else 'w5221'}; held-out delta {d[p]:+.4f} "
              f"({up[p]} / 24 replicates up) -> score {score[p]:+.4f}")
    m, k = float(np.mean(list(score.values()))), sum(v > 0 for v in score.values())
    ok = m > 0 and k >= 4
    print(f"mean held-out score {m:+.4f}, folds > 0: {k} / 5 -> {'ADOPT w4321' if ok else 'w5221 stays'}")
    (D.parent / "t2_d209").mkdir(exist_ok=True)
    (D.parent / "t2_d209/read.json").write_text(json.dumps(dict(delta=d, up=up, score=score, mean=m, folds_pos=k, adopt=ok), indent=1) + "\n")


if __name__ == "__main__":
    main()
