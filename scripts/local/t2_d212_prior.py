#!/usr/bin/env python
"""T2-D212 (zero compute), as pre-registered in docs/DECISIONS_T2.md (the panel's T2 item 4b, user-approved 2026-10-23): the causal output prior
(alpha, n0) for the w4321 vp2-180 package, cross-fitted by user. Grid: alpha 0 (none) + alpha {0.5, 1, 1.5} x n0 {1, 3, 10}; anchor (1, 3) = the package.
Members = T2-D206's route files (D203 seeds 156-167 / 120-131 = "pick", D204 seeds 168-179 / 132-143 = "conf"; 24 replicates), weights w4321.
  delta_p(g)  = g - anchor on participant p's mix cells (as T2-D209: P04 kit + seen1, P09 kit, P05 / P08 / P10 seen1), mean over the 24 replicates;
  candidate   = the grid point with the largest mean_p delta_p (in-sample, all five);
  cross-fit   : fold p picks the best g on the other four (the anchor if none > 0); score_p = delta_p(pick_p);
  pooled read : the candidate's arm-level deltas, the mean of the two reads (up-counts summed over 24).
  adopt the candidate iff mean_p score_p > 0 AND >= 4 / 5 folds > 0 AND pooled mix >= +0.005 with >= 18 / 24 up AND kit test >= -0.005 AND new >= 0.
  mem-guard python scripts/local/t2_d212_prior.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
D = O / "t2_d212"
COLS = {"P04": [("kit", "P04"), ("seen1", "P04")], "P09": [("kit", "P09")], "P05": [("seen1", "P05")], "P08": [("seen1", "P08")], "P10": [("seen1", "P10")]}
GRID = [(0.0, 0.0)] + [(a, n) for a in (0.5, 1.0, 1.5) for n in (1.0, 3.0, 10.0)]
W = {"fbc": 0.4, "sh": 0.3, "stl": 0.2, "mor": 0.1}
name = lambda g: "none" if g[0] == 0 else f"a{g[0]:g}_n{g[1]:g}"  # noqa: E731


def main():
    D.mkdir(parents=True, exist_ok=True)
    reads = {}
    for st in ("pick", "conf"):
        sp = json.load(open(O / f"t2_d206/spec_{st}.json"))
        spec = {"n": 12, "anchor": "a1_n3", "bar": "d141", "members": sp["members"], "arms": {name(g): dict(W) for g in GRID},
                "arm_prior": {name(g): list(g) for g in GRID}}
        (D / f"spec_{st}.json").write_text(json.dumps(spec, indent=2) + "\n")
        with open(D / f"read_{st}.txt", "w") as fh:
            rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(D / f"spec_{st}.json"),
                                 "--out", str(D / f"read_{st}.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
        assert rc == 0, (st, rc)
        reads[st] = json.load(open(D / f"read_{st}.json"))
    delta = {}
    for g in GRID:
        if name(g) == "a1_n3":
            continue
        d = {p: [] for p in COLS}
        for st in ("pick", "conf"):
            rows = {(x["proto"], x["i"], x["arm"]): x for x in reads[st]["rows"]}
            for i in range(12):
                for p, cs in COLS.items():
                    d[p].append(np.mean([rows[(pr, i, name(g))][c] - rows[(pr, i, "a1_n3")][c] for pr, c in cs]))
        delta[name(g)] = {p: float(np.mean(v)) for p, v in d.items()}
    allm = {g: float(np.mean(list(v.values()))) for g, v in delta.items()}
    cand = max(allm, key=allm.get)
    print("in-sample mean delta per grid point:", {g: round(v, 4) for g, v in sorted(allm.items(), key=lambda t: -t[1])})
    score = {}
    for p in COLS:
        oth = {g: np.mean([delta[g][q] for q in COLS if q != p]) for g in delta}
        pk = max(oth, key=oth.get)
        pk = pk if oth[pk] > 0 else "a1_n3"
        score[p] = delta[pk][p] if pk != "a1_n3" else 0.0
        print(f"fold {p}: pick {pk} (others {oth.get(pk, 0):+.4f}); held-out {score[p]:+.4f}")
    m, k = float(np.mean(list(score.values()))), sum(v > 0 for v in score.values())
    a0, a1 = reads["pick"]["arms"][cand], reads["conf"]["arms"][cand]
    P = {x: (a0[x]["d"] + a1[x]["d"]) / 2 for x in ("mix", "kit_test", "seen", "new")}; up = a0["mix"]["up"] + a1["mix"]["up"]
    ok = m > 0 and k >= 4 and P["mix"] >= 0.005 and up >= 18 and P["kit_test"] >= -0.005 and P["new"] >= 0
    print(f"candidate {cand}: cross-fit mean {m:+.4f}, folds > 0 {k} / 5; pooled 24: mix {P['mix']:+.4f} ({up} / 24), kit test {P['kit_test']:+.4f}, "
          f"seen {P['seen']:+.4f}, new {P['new']:+.4f}")
    print("VERDICT:", f"ADOPT {cand}" if ok else "NOT ADOPTED (the prior stays alpha 1, n0 3)")
    (D / "read.json").write_text(json.dumps(dict(delta=delta, in_sample=allm, candidate=cand, cross_fit=score, pooled=P, up=up, adopt=ok), indent=1) + "\n")


if __name__ == "__main__":
    main()
