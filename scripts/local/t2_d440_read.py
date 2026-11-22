#!/usr/bin/env python
"""T2-D440 read (pre-registered in docs/DECISIONS_T2.md before any D440 number): class experts on ShallowNet, three nets per seed, seeds 384-395.
Arms: s3 (ANCHOR: three plain 3-class nets), ovr (one-vs-rest), pair (pairwise + coupling), ovc (ovr on class-specific channel sets).
Per arm (member alone, causal SKλ, non-P04 rows): recall per class (mi / sub / word), the three pair-confusion shares of its errors, error phi
vs the anchor; then t2_cluster_ci: alone and M7's shape with its ShallowNet slot = the arm, each vs the anchor.
BAR: M7 shape typical >= +0.003 with >= 9 / 12 draws up and no non-P04 user < -0.01.
    mem-guard python scripts/local/t2_d440_read.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d440"
R.mkdir(exist_ok=True)
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
SUF = {"s3": "bags3", "ovr": "bagovr", "pair": "bagpw", "ovc": "bagovc"}
idx = pd.read_csv(EXP / "index.csv")
arms, wrong = {}, {}
print("member alone, SKλ, non-P04 rows: recall mi / sub / word | share of errors mi-sub / mi-word / sub-word | error phi vs s3")
for a, suf in SUF.items():
    d = O / f"sofia_s1/ni26-t2p-d440-sh-{a}-5090/runs"
    M[f"x_{a}"] = {"offset": 384, **{p: f"{d}/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr{suf}__s{{s}}.npz" for p in ("kit", "seen1")}}
    arms[f"alone_{a}"] = {f"x_{a}": 1.0}
    arms[f"M7_{a}"] = {(f"x_{a}" if k == "sh_tv8c" else k): v for k, v in M7.items()}
    rec, conf, w = [], np.zeros((3, 3)), []
    for p in ("kit", "seen1"):
        for i in range(12):
            z = np.load(M[f"x_{a}"][p].format(s=384 + i))
            for ix, pr in ((z["test_idx"], z["prob_last"]), (z["val_idx"], z["prob_val_last"])):
                dd = idx.iloc[ix]
                pred = causal_sinkhorn(pr.astype(np.float64), (dd.stream.to_numpy(), dd.order.to_numpy()), 10, 3, 0.95).argmax(1)
                nt = (dd.subject.str[-3:] != "P04").to_numpy()
                y = dd.y.to_numpy()
                rec.append([np.mean(pred[nt & (y == k)] == k) for k in range(3)])
                for t in range(3):
                    for q in range(3):
                        conf[t, q] += np.sum(nt & (y == t) & (pred == q))
                w.append((pred != y)[nt])
    wrong[a] = np.concatenate(w).astype(float)
    e = conf.sum() - np.trace(conf)
    sh = [(conf[i, j] + conf[j, i]) / e for i, j in ((0, 1), (0, 2), (1, 2))]
    print(f"  {a:4s} {np.round(np.mean(rec, 0), 3).tolist()} mean {np.mean(rec):.4f} | {np.round(sh, 3).tolist()} | "
          f"{np.corrcoef(wrong[a], wrong['s3'])[0, 1]:.3f}")
json.dump({"members": M, "arms": arms}, open(R / "spec.json", "w"), indent=1)
for a in ("ovr", "pair", "ovc"):
    for shape in ("alone", "M7"):
        print(f"\n== {shape}: {a} vs s3", flush=True)
        out = subprocess.run([sys.executable, str(HERE / "t2_cluster_ci.py"), str(R / "spec.json"), f"{shape}_s3", f"{shape}_{a}"], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr[-2000:], flush=True)
