#!/usr/bin/env python
"""T2-D461 (written before any number): shipped vs oracle route on returning users, FBCNet fresh draws (D419), seen1. See the row.
    mem-guard python scripts/local/t2_d461_route.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
R0 = 1 / 3
rows, acc = [], []
for s in range(324, 336):
    a = np.load(glob.glob(f"{O}/sofia_s1/ni26-t2p-d419-fbc_e180_s324-5090/runs/seen1__*tv8cpu__s{s}.npz")[0], allow_pickle=True)
    b = np.load(glob.glob(f"{O}/t2_d419/route_fbc_e180/seen1__*__s{s}.npz")[0], allow_pickle=True)
    users = [str(u)[-3:] for u in a["users"]]
    for ix, sh, Pu, routed in ((a["test_idx"], a["prob_last"], a["prob_users_test"], b["prob_last"]), (a["val_idx"], a["prob_val_last"], a["prob_users_val"], b["prob_val_last"])):
        assert (ix == (b["test_idx"] if len(ix) == len(b["test_idx"]) else b["val_idx"])).all()
        sh, Pu, routed = sh.astype(np.float64), Pu.astype(np.float64), routed.astype(np.float64)
        cand = (1 - R0) * sh[None] + R0 * Pu
        err = np.abs(cand - routed[None]).max(2)
        k = err.argmin(0)
        k[np.abs(routed - sh).max(1) < 1e-7] = -1
        assert (np.where(k >= 0, err.min(0), 0) < 1e-5).all(), "route not recoverable"
        u = U[ix]
        own = np.array([users.index(x) if x in users else -2 for x in u])
        pos = np.zeros(len(ix), int)
        for s_ in np.unique(ST[ix]):
            r = np.flatnonzero(ST[ix] == s_)
            pos[r[np.argsort(OD[ix][r], kind="stable")]] = np.arange(len(r))
        ret = own >= 0
        acc.append(pd.DataFrame(dict(u=u[ret], pos=pos[ret], hit=(k == own)[ret], none=(k == -1)[ret])))
        for route, kk in (("shipped", k), ("oracle", np.where(ret, own, k))):
            for r_ in (0.0, R0, 2 / 3, 1.0):
                mix = np.where((kk >= 0)[:, None], (1 - r_) * sh + r_ * Pu[np.maximum(kk, 0), np.arange(len(ix))], sh)
                pred = causal_sinkhorn(mix, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1)
                rows.append(pd.DataFrame(dict(s=s, route=route, r=round(r_, 2), u=u, ret=ret, pos=pos, cell=CELL[ix], y=Y[ix], ok=pred == Y[ix])))
A, D = pd.concat(acc), pd.concat(rows, ignore_index=True)
bins = lambda p: pd.cut(p, [-1, 19, 49, 10 ** 6], labels=["1-20", "21-50", "51+"])  # noqa: E731
print("users with a copy:", users)
print("routing accuracy on returning users by window bin:\n", A.groupby(["u", bins(A.pos)], observed=True).hit.mean().unstack().round(3).to_string(), "\n no route yet:", round(A.none.mean(), 4))
D = D[D.ret]
ba = D.groupby(["route", "r", "s", "u", "cell", "y"]).ok.mean().groupby(["route", "r", "s", "u", "cell"]).mean().reset_index(name="ba")
print("returning cells, member alone under causal SK (BA, mean of 12 draws):")
print(ba.groupby(["route", "r", "u"]).ba.mean().unstack().assign(both=ba.groupby(["route", "r"]).ba.mean()).round(4).to_string())
t = ba.groupby(["route", "r", "s"]).ba.mean().unstack(["route", "r"])
for a_, b_ in ((("oracle", 0.33), ("shipped", 0.33)), (("oracle", 0.67), ("shipped", 0.33)), (("oracle", 1.0), ("shipped", 0.33)), (("shipped", 0.33), ("shipped", 0.0))):
    d = t[a_] - t[b_]
    print(f"{a_} - {b_}: {d.mean():+.4f} ({int((d > 0).sum())}/12)")
w = D[(D.r == 0.33)].groupby(["route", bins(D[(D.r == 0.33)].pos)], observed=True).ok.mean().unstack().round(4)
print("window accuracy by bin at r 1/3:\n", w.to_string())
