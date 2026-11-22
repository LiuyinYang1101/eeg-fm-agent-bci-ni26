#!/usr/bin/env python
"""T2 learning-curve diagnosis of the SHIPPED member lines (the owner's directive 2026-11-16 ≈ 20:2x, relayed by the manager; zero GPU,
descriptive: it decides nothing by itself). Source = the curves the runs already log (res['curves']): train loss (+ train acc for CNNs) every
epoch, and test cell BA every 5 epochs on the fold's test windows (the held-out stranger + the returning sessions of lc9_f), member ALONE, argmax.
Per family: the mean curve over runs (fold x seed), its best epoch vs the shipped final epoch, the final - best gap, the slope over the last
20 % of epochs, the train / held-out gap, and across-run noise. Plus a CROSS-FITTED epoch check: the best epoch picked on folds 0-4's mean curve
is scored on folds 5-9 (and the reverse); 'cf_gain' = the held-out half's BA at the picked epoch minus at the final epoch (what an
early stop would buy without a re-pick on the scored folds).
    OMP_NUM_THREADS=1 python scripts/local/t2_curves_diag.py
"""
import glob
import json
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
FAM = {
    "FM SD (6 seeds, W18's members; d531 sd)": [p for s in range(826, 832) for p in glob.glob(str(S / f"ni26-t2p-d531-stl*sd_s{s}_*-*/runs/*.npz"))],
    "FM CTL (6 seeds, head-less twins; d531 ctl)": [p for s in range(826, 832) for p in glob.glob(str(S / f"ni26-t2p-d531-stl*ctl_s{s}_*-*/runs/*.npz"))],
    "FM MI (s826; d532)": glob.glob(str(S / "ni26-t2p-d532-mi_s826_*-*/runs/*.npz")),
    "FBCNet pooled (d536)": glob.glob(str(O / "t2_d536/cnn_fbc/runs/*__s81[678].npz")),
    "ShallowNet pooled (d536)": glob.glob(str(O / "t2_d536/cnn_sh/runs/*__s81[678].npz")),
    "IFNet pooled (d536)": glob.glob(str(O / "t2_d536/cnn_ifn/runs/*__s81[678].npz")),
    "side ShallowNet S5 (d536)": glob.glob(str(O / "t2_d536/side_sh/runs/*.npz")),
    "side IFNet S5 (d536)": glob.glob(str(O / "t2_d536/side_ifn/runs/*.npz")),
}


def curves(path):
    d = json.loads(str(np.load(path)["res"]))
    c = d.get("curves") or []
    ep = [r["epoch"] for r in c if "test_cell_bacc" in r]
    te = [r["test_cell_bacc"] for r in c if "test_cell_bacc" in r]
    tl = {r["epoch"]: r["train_loss"] for r in c}
    ta = {r["epoch"]: r["train_acc"] for r in c if "train_acc" in r}
    return d.get("fold", d.get("protocol")), np.array(ep), np.array(te), tl, ta, len(c)


out = {}
for fam, files in FAM.items():
    if not files:
        out[fam] = "no run files found locally"
        print(f"== {fam}: no files", flush=True)
        continue
    rows = [curves(p) for p in sorted(files)]
    E = rows[0][1]
    rows = [r for r in rows if len(r[1]) == len(E) and (r[1] == E).all()]
    T = np.stack([r[2] for r in rows])                       # runs x eval epochs
    n_ep = rows[0][5]
    m = T.mean(0)
    b = int(np.argmax(m))
    last20 = E >= 0.8 * n_ep
    slope = float(np.polyfit(E[last20], m[last20], 1)[0] * (0.2 * n_ep)) if last20.sum() >= 2 else float("nan")
    fold = np.array([int(str(r[0]).split("_")[-1]) for r in rows])
    cf = []
    for pick, score in ((fold < 5, fold >= 5), (fold >= 5, fold < 5)):
        if pick.any() and score.any():
            bp = int(np.argmax(T[pick].mean(0)))
            cf.append(float(T[score].mean(0)[bp] - T[score].mean(0)[-1]))
    fin_tl = np.mean([r[3][n_ep] for r in rows])
    gap = None
    if rows[0][4]:
        gap = round(float(np.mean([r[4][n_ep] for r in rows]) - m[-1]), 4)
    res = dict(runs=len(rows), epochs=n_ep, eval_epochs=f"{int(E[0])}..{int(E[-1])} step {int(E[1] - E[0]) if len(E) > 1 else 0}",
               final_ba=round(float(m[-1]), 4), best_epoch=int(E[b]), best_ba=round(float(m[b]), 4), final_minus_best=round(float(m[-1] - m[b]), 4),
               last20pct_change=round(slope, 4), cf_earlystop_gain=[round(x, 4) for x in cf],
               run_sd_final=round(float(T[:, -1].std()), 4), run_sd_best_epoch=round(float(np.std([E[int(np.argmax(t))] for t in T])), 1),
               train_loss_final=round(float(fin_tl), 4), train_acc_minus_test_ba=gap,
               mean_curve={int(e): round(float(v), 4) for e, v in list(zip(E, m))[:: max(1, len(E) // 10)] + [(E[-1], m[-1])]})
    v = ("OVERFIT (turned down)" if res["final_minus_best"] < -0.005 and res["last20pct_change"] < 0 else
         "UNDERFIT (still rising)" if res["last20pct_change"] > 0.003 else "FLAT at the end (fine)")
    res["verdict"] = v
    out[fam] = res
    print(f"== {fam}: {res}", flush=True)
json.dump(out, open(O / "t2_curves_diag.json", "w"), indent=1)
