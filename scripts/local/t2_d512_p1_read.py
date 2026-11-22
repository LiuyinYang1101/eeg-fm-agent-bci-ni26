#!/usr/bin/env python
"""T2-D512 part P1 read, CNN side only (written 2026-11-15 before any D512 number; the manager's ask 10:1x: the nearer local number for step 1).
Per fold f x draw s (810-812): BASE = D512's fresh CNN base (one ROUTED net per family with copies + tv8c views) + T2-D498's FM part, each family
decoded by the causal Sinkhorn (10 / 3 / 0.95) from its cross-fitted start vector, pooled at M7's shares (outputs/t2_d512/teacher_p1).
For each adapted slot: EXTRA (the control) = (1 - r) * routed member + r * the UNADAPTED pooled net of the same seed (is0); ADAPT = the same with
the adapted copy (the P1 form's arm file); TRUE = with the true-label copy (isS: the ceiling). r = 1/2 (also 1/3, 1).
Credited P1 = ADAPT - EXTRA (the extra-net effect is not adaptation's); also ADAPT - BASE and EXTRA - BASE. Strangers = the fold's held-out user
(every window = the sealed-like set; position >= 30 also reported); returning = P04 / P10 sessions 2-3.
    mem-guard python scripts/local/t2_d512_p1_read.py --slots sh[,ifn,fbc] --arm isSpc|isSpr
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slots", required=True)
ap.add_argument("--arm", required=True, help="the P1 form's arm suffix: isSpc (D510's form) or isSpr (D511's)")
args = ap.parse_args()
SLOTS, ARM = args.slots.split(","), args.arm
O = Path.home() / "workspace/outputs/neural_interface_challenge"
D = O / "t2_d512"
idx = pd.read_csv(EXP / "index.csv")
U, SES, STREAM, ORD, Y, CELL = (idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy(), idx.cell.to_numpy())
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
REC = {"sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr", "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vr", "fbc": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75"}
DRAWS, RS = (810, 811, 812), (1 / 3, 1 / 2, 1.0)
VARIANTS = [ARM, ARM + "r1", ARM + "r2", "isS"]


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        ks = np.unique(Y[rows][m])
        if len(ks) >= 2:
            out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
    return out


def norm(p):
    return p / p.sum(1, keepdims=True)


recs = []
for f in range(10):
    for s in DRAWS:
        t = np.load(D / f"teacher_p1/lc9_{f}__s{s}.npz")
        rows, held = t["rows"], t["held"]
        key = (STREAM[rows], ORD[rows])
        pos = pd.Series(STREAM[rows]).groupby(STREAM[rows]).cumcount().to_numpy()
        kind = np.where(held, "stranger", "returning")
        fixed = W["fm"] * t["D_fm"].astype(np.float64) + sum(W[k] * t[f"D_{k}"].astype(np.float64) for k in ("fbc", "sh", "ifn") if k not in SLOTS)
        Rr = {k: norm(t[f"R_{k}"].astype(np.float64)) for k in SLOTS}
        V = {k: t[f"vec_{k}"] for k in SLOTS}
        base = (fixed + sum(W[k] * t[f"D_{k}"].astype(np.float64) for k in SLOTS)).argmax(1)
        st = {}
        for v in ["is0"] + VARIANTS:
            P = {}
            for k in SLOTS:
                z = np.load(D / f"p1_{k}/runs/lc9_{f}__{REC[k]}{v}__s{s}.npz")
                mp = {int(r): j for j, r in enumerate(z["test_idx"])}
                P[k] = norm(z["prob_last"].astype(np.float64))[np.array([mp[int(r)] for r in rows])]
            st[v] = P
        for r in RS:
            dec = {}
            for v, P in st.items():
                dec[v] = (fixed + sum(W[k] * causal_sinkhorn(norm((1 - r) * Rr[k] + r * P[k]), key, 10, 3, 0.95, start=V[k]) for k in SLOTS)).argmax(1)
            for sel, lab in ((pos >= 0, "all"), (pos >= 30, "pos30")):
                for kd in ("stranger", "returning"):
                    for u in np.unique(U[rows][(kind == kd) & sel]):
                        ku = (kind == kd) & sel & (U[rows] == u)
                        cb = cells(base[ku], rows[ku])
                        cv = {v: cells(dec[v][ku], rows[ku]) for v in dec}
                        for c in cb:
                            recs.append(dict(fold=f, draw=s, r=round(r, 3), set=lab, kind=kd, user=u, cell=c, base=cb[c], **{v: cv[v][c] for v in cv}))
    print(f"[d512 p1] fold {f} read", flush=True)
Rd = pd.DataFrame(recs)
out = D / f"p1_{'+'.join(SLOTS)}_{ARM}"
out.mkdir(exist_ok=True)
Rd.to_csv(out / "cells.csv", index=False)
res = {}
for (r, lab), g in Rd.groupby(["r", "set"]):
    e = {}
    for kd in ("stranger", "returning"):
        gk = g[g.kind == kd]
        pu = gk.groupby(["draw", "user"])[["base", "is0"] + VARIANTS].mean()
        d = {"P1 (adapt - extra)": pu[ARM] - pu["is0"], "adapt - base": pu[ARM] - pu["base"], "extra - base": pu["is0"] - pu["base"],
             "TRUE - extra": pu["isS"] - pu["is0"]}
        e[kd] = {n: dict(mean=round(float(v.groupby("user").mean().mean()), 4), draws=[round(float(x), 4) for x in v.groupby("draw").mean()],
                         min_user=round(float(v.groupby("user").mean().min()), 4), min_who=str(v.groupby("user").mean().idxmin()))
                 for n, v in d.items()}
        if kd == "stranger":
            e["reps"] = [round(float((pu[a] - pu["is0"]).groupby("user").mean().mean()), 4) for a in (ARM, ARM + "r1", ARM + "r2")]
    res[f"r{r}_{lab}"] = e
    s_, rt = e["stranger"], e["returning"]
    print(f"D512 P1 [{'+'.join(SLOTS)}, {ARM}] r {r:.3f} {lab:5s}: strangers P1 {s_['P1 (adapt - extra)']['mean']:+.4f} draws {s_['P1 (adapt - extra)']['draws']} "
          f"min {s_['P1 (adapt - extra)']['min_user']:+.4f} ({s_['P1 (adapt - extra)']['min_who']}) | adapt-base {s_['adapt - base']['mean']:+.4f} | "
          f"extra-base {s_['extra - base']['mean']:+.4f} | TRUE-extra {s_['TRUE - extra']['mean']:+.4f} | reps {e['reps']} | returning P1 "
          f"{rt['P1 (adapt - extra)']['mean']:+.4f}", flush=True)
json.dump(res, open(out / "read.json", "w"), indent=1)
