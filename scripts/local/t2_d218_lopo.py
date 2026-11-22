#!/usr/bin/env python
"""T2-D218 (zero compute), as pre-registered in docs/DECISIONS_T2.md: does the side-input MOR (T2-X1) earn a larger package weight?
x1_w4312 = FBCNet 0.4 / ShallowNet 0.3 / ST-L 0.1 / MOR-with-side 0.2 vs the shipped c_c = w4321 with the wz MOR, on T2-X1's route files
(12 replicates), by T2-D209's leave-one-participant-out rule:
  delta_p = x1_w4312 - c_c on p's cells in its mix components (P04: mean of kit test and seen1 seen; P09: kit test; P05 / P08: seen1 val;
            P10: seen1 seen), averaged over the 12 replicates;
  fold p: pick x1_w4312 iff mean_{q != p} delta_q > 0; held-out score_p = delta_p if picked, else 0;
  candidate iff mean_p score_p > 0 AND >= 4 / 5 folds have score_p > 0.  Descriptive beside it: c_w4312 (MOR 0.2, no side) and c_x1 (w4321, side).
  mem-guard python scripts/local/t2_d218_lopo.py   (runs t2p_pkgshape on outputs/t2_d218/spec.json first)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
COLS = {"P04": [("kit", "P04"), ("seen1", "P04")], "P09": [("kit", "P09")], "P05": [("seen1", "P05")], "P08": [("seen1", "P08")], "P10": [("seen1", "P10")]}


def lopo(rows, arm, base="c_c", n=12):
    deltas = {p: [float(np.mean([rows[(pr, i, arm)][c] - rows[(pr, i, base)][c] for pr, c in cs])) for i in range(n)] for p, cs in COLS.items()}
    d = {p: float(np.mean(v)) for p, v in deltas.items()}
    up = {p: int(sum(x > 0 for x in v)) for p, v in deltas.items()}
    score = {p: (d[p] if np.mean([d[q] for q in COLS if q != p]) > 0 else 0.0) for p in COLS}
    return d, up, score


def main():
    out = O / "t2_d218"
    with open(out / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(out / "spec.json"),
                             "--out", str(out / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"t2p_pkgshape rc {rc} (see {out / 'read.txt'})"
    r = json.load(open(out / "read.json"))
    rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
    res = {}
    for arm in ("x1_w4312", "c_w4312", "c_x1"):
        d, up, score = lopo(rows, arm)
        m, k = float(np.mean(list(score.values()))), sum(v > 0 for v in score.values())
        res[arm] = dict(delta=d, up=up, score=score, mean=m, folds_pos=k)
        tag = "DECIDING" if arm == "x1_w4312" else "descriptive"
        print(f"{arm} vs c_c ({tag}): held-out " + ", ".join(f"{p} {score[p]:+.4f} (d {d[p]:+.4f}, {up[p]} / 12 up)" for p in COLS)
              + f" -> mean {m:+.4f}, folds > 0: {k} / 5; package mix {r['arms'][arm]['mix']['d']:+.4f}, new {r['arms'][arm]['new']['d']:+.4f}")
    x = res["x1_w4312"]
    ok = x["mean"] > 0 and x["folds_pos"] >= 4
    print("VERDICT:", "CANDIDATE for the user (template work + MOR members rebuilt + a fresh-seed confirm; beyond the conservative GO count)" if ok
          else "NOT A CANDIDATE (MOR stays wz at 0.1)")
    (out / "verdict.json").write_text(json.dumps(dict(res, candidate=ok), indent=1) + "\n")


if __name__ == "__main__":
    main()
