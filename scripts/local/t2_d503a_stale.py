#!/usr/bin/env python
"""T2-D503 (a) (registered 2026-11-15 before any number; zero GPU): the per-user copy's value by SESSION and FAMILY on the seen1 protocol's
stored per-copy files (FBCNet T2-D419 seeds 324-335, shipped r 1/3, unaligned; ShallowNet / IFNet T2-D429 seeds 348-359, shipped r 1/2,
EA-aligned per stream). Each family alone, decoded by the causal Sinkhorn (10 / 3 / 0.95, uniform start); oracle own-copy route.
Per user x session (P04 / P10 sessions 2 and 3): pooled, blend at the shipped r, copy only (r 1); gains vs pooled.
    mem-guard python scripts/local/t2_d503a_stale.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
FAM = {"fbc": ("ni26-t2p-d419-fbc_e180_s324-5090", "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cpu", 324, 1 / 3),
       "sh": ("ni26-t2p-d429-sh-ctl-5090", "shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cpu", 348, 0.5),
       "ifn": ("ni26-t2p-d429-ifn-ctl-5090", "ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cpu", 348, 0.5)}
idx = pd.read_csv(EXP / "index.csv")
U, SES, STREAM, ORD, Y, CELL = (idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy(), idx.cell.to_numpy())


def ba(pred, rows):
    out = []
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        out.append(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in np.unique(Y[rows][m])]))
    return float(np.mean(out))


recs = []
for fam, (d, stem, off, r) in FAM.items():
    for i in range(12):
        z = np.load(S / d / "runs" / f"seen1__{stem}__s{off + i}.npz")
        te, P0, Pu, users = z["test_idx"], z["prob_last"].astype(np.float64), z["prob_users_test"].astype(np.float64), [str(u) for u in z["users"]]
        for u in ("P04", "P10"):
            k = users.index(u)
            for ses in ("2", "3"):
                m = np.flatnonzero((U[te] == u) & (SES[te] == ses))
                m = m[np.argsort(ORD[te][m])]
                rows = te[m]
                key = (STREAM[rows], ORD[rows])
                for arm, p in (("pooled", P0[m]), ("shipped_r", (1 - r) * P0[m] + r * Pu[k][m]), ("copy", Pu[k][m])):
                    recs.append(dict(fam=fam, seed=i, user=u, session=ses, arm=arm,
                                     ba=ba(causal_sinkhorn(p / p.sum(1, keepdims=True), key, 10, 3, 0.95).argmax(1), rows)))
R = pd.DataFrame(recs)
out = O / "t2_d503"
out.mkdir(exist_ok=True)
R.to_csv(out / "a_cells.csv", index=False)
T = R.groupby(["fam", "user", "session", "arm"]).ba.mean().unstack()
T = T.assign(shipped_gain=T["shipped_r"] - T["pooled"], copy_gain=T["copy"] - T["pooled"])
print("D503 (a) per family x user x session (pooled | shipped-r gain | copy-only gain):", flush=True)
for (fam, u, ses), row in T.iterrows():
    print(f"   {fam:3s} {u} s{ses}: pooled {row['pooled']:.3f} | shipped {row['shipped_gain']:+.4f} | copy {row['copy_gain']:+.4f}", flush=True)
dec = T["copy_gain"].unstack("session")
dec = dec.assign(decay=dec["3"] - dec["2"])
print("D503 (a) copy-only gain session 3 - session 2, per family and user:\n" + dec.round(4).to_string(), flush=True)
T.to_csv(out / "a_table.csv")
