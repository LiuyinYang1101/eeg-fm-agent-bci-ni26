#!/usr/bin/env python
"""T2-D510 read (written 2026-11-15 before any D510 number): the package as its own in-stream teacher, read AT THE STACK LEVEL.
Per fold f x seed s (790-792; teacher draw s - 40): T2-D498's stack on every test row (outputs/t2_d510/teacher), its ShallowNet slot replaced by
the D510 student: slot = (1 - r) * unadapted (is0) + r * adapted (arm file), r in {1/3, 1/2, 1}; the slot decoded by the causal Sinkhorn
(10 / 3 / 0.95) from its cross-fitted start vector, then pooled with the teacher file's fixed decoded fbc / ifn / fm parts at M7's shares;
vs the same stack at r = 0, cell BA on the arm's SCORED windows. Strangers = the fold's held-out user; returning = P04 / P10 sessions 2-3.
PRIMARY = isSpc (SCHED x conf) at r 1/2. BAR: ten strangers' mean >= +0.010, no stranger < -0.020, returning mean >= -0.003;
+0.005 .. +0.010 -> IFNet next; < +0.005 -> closes.
    mem-guard python scripts/local/t2_d510_read.py
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
D = O / "t2_d510"
idx = pd.read_csv(EXP / "index.csv")
U, SES, STREAM, ORD, Y, CELL = (idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy(), idx.cell.to_numpy())
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
ap = argparse.ArgumentParser()
ap.add_argument("--slots", default="sh", help="T2-D510b: the adapted CNN slots, e.g. sh,ifn (sh = D510's files, ifn = D510b's)")
ap.add_argument("--row", default="d510", choices=("d510", "d511"), help="T2-D511: its own students (seeds 800-802), teacher and score set")
ap.add_argument("--pos-min", type=int, default=-1, help="T2-D511: override the score set's first stream position (0 = every window)")
args_ = ap.parse_args()
SLOTS, ROW = args_.slots.split(","), args_.row
SRC = {"sh": (O / "t2_d510", "shallow__ea__n00__r1_pulx_vp2_e180_vr"), "ifn": (O / "t2_d510b", "ifnet__ea__n00__r1_pulx_vp2_e180_vr")}
SFX = "" if SLOTS == ["sh"] else "_" + "+".join(SLOTS)
SEEDS = (790, 791, 792)
LAB = {"true": "", "all": "pa", "conf": "pc", "soft": "ps"}
ARMS = [f"is{n}{c}" for n in (60, 100) for c in LAB.values()] + [f"isS{c}" for c in LAB.values()] + ["isSpcr1", "isSpcr2"]
RS = (1 / 3, 1 / 2, 1.0)
PRIMARY = ("isSpc", 0.5)
POS_MIN = None                                          # T2-D510: score the arm's own scored windows
if ROW == "d511":                                       # T2-D511: fresh students, the prefix re-decoded labels, score set = stream position >= 30
    D = O / "t2_d511"
    SRC = {"sh": (D / "sh", "shallow__ea__n00__r1_pulx_vp2_e180_vr"), "ifn": (D / "ifn", "ifnet__ea__n00__r1_pulx_vp2_e180_vr"),
           "fbc": (D / "fbc", "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75")}
    SEEDS, ARMS, PRIMARY, POS_MIN = (800, 801, 802), ["isS", "isSpc", "isSpr", "isSprr1", "isSprr2"], ("isSpr", 0.5), 30
if args_.pos_min >= 0:
    POS_MIN = args_.pos_min
    SFX += f"_pos{POS_MIN}"
NC6 = ("P01", "P03", "P05", "P07", "P09", "P10")


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        ks = np.unique(Y[rows][m])
        if len(ks) < 2:
            continue
        out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
    return out


def norm(p):
    return p / p.sum(1, keepdims=True)


recs, posr, alone = [], [], []
for f in range(10):
    for s in SEEDS:
        t = np.load(D / f"teacher/lc9_{f}__s{s}.npz")
        rows, held = t["rows"], t["held"]
        key = (STREAM[rows], ORD[rows])
        fixed = sum(W[k] * t[f"D_{k}"].astype(np.float64) for k in W if k not in SLOTS)
        P0, VS, MP = {}, {}, {}
        for sl in SLOTS:
            z0 = np.load(SRC[sl][0] / f"runs/lc9_{f}__{SRC[sl][1]}is0__s{s}.npz")
            pos = {int(r): k for k, r in enumerate(z0["test_idx"])}
            MP[sl] = np.array([pos[int(r)] for r in rows])
            P0[sl], VS[sl] = norm(z0["prob_last"].astype(np.float64))[MP[sl]], t[f"vec_{sl}"]
        D0 = {sl: causal_sinkhorn(P0[sl], key, 10, 3, 0.95, start=VS[sl]) for sl in SLOTS}
        st0 = (fixed + sum(W[sl] * D0[sl] for sl in SLOTS)).argmax(1)
        al0 = D0[SLOTS[0]].argmax(1)
        kind = np.where(held, "stranger", "returning")
        for arm in ARMS:
            fas = {sl: SRC[sl][0] / f"runs/lc9_{f}__{SRC[sl][1]}{arm}__s{s}.npz" for sl in SLOTS}
            if not all(fa.exists() for fa in fas.values()):
                continue
            ZA = {sl: np.load(fa) for sl, fa in fas.items()}
            PA = {sl: norm(ZA[sl]["prob_last"].astype(np.float64))[MP[sl]] for sl in SLOTS}
            sc = np.logical_and.reduce([ZA[sl]["scored"][MP[sl]] for sl in SLOTS])
            za = ZA[SLOTS[0]]
            if POS_MIN is not None:                     # T2-D511: every window at stream position >= POS_MIN (unadapted ones count, delta 0)
                sc = za["pos"][MP[SLOTS[0]]] >= POS_MIN
            for r in RS:
                DA = {sl: causal_sinkhorn(norm((1 - r) * P0[sl] + r * PA[sl]), key, 10, 3, 0.95, start=VS[sl]) for sl in SLOTS}
                dsh = DA[SLOTS[0]]
                st1 = (fixed + sum(W[sl] * DA[sl] for sl in SLOTS)).argmax(1)
                for kd in ("stranger", "returning"):
                    k = sc & (kind == kd)
                    if not k.any():
                        continue
                    for u in np.unique(U[rows][k]):
                        ku = k & (U[rows] == u)
                        a, b = cells(st0[ku], rows[ku]), cells(st1[ku], rows[ku])
                        for c in a:
                            recs.append(dict(fold=f, seed=s, arm=arm, r=round(r, 3), kind=kd, user=u, cell=c, base=a[c], new=b[c]))
                        if r == 1.0:
                            a2, b2 = cells(al0[ku], rows[ku]), cells(dsh.argmax(1)[ku], rows[ku])
                            for c in a2:
                                alone.append(dict(fold=f, seed=s, arm=arm, kind=kd, user=u, cell=c, base=a2[c], new=b2[c]))
                if arm == PRIMARY[0] and r == PRIMARY[1] and "pos" in za:
                    ps = za["pos"][MP[SLOTS[0]]]
                    for lo, hi in ((30, 60), (60, 100), (100, 150), (150, 10 ** 6)):
                        k = sc & (ps >= lo) & (ps < hi)
                        for kd in ("stranger", "returning"):
                            kk = k & (kind == kd)
                            if kk.any():
                                posr.append(dict(fold=f, seed=s, kind=kd, bin=f"{lo}-{hi if hi < 10 ** 6 else 'end'}",
                                                 d_acc=float(np.mean(st1[kk] == Y[rows][kk]) - np.mean(st0[kk] == Y[rows][kk])), n=int(kk.sum())))
    print(f"[d510] fold {f} read", flush=True)
R = pd.DataFrame(recs)
R.to_csv(D / f"cells{SFX}.csv", index=False)
R["d"] = R.new - R.base
pu = R.groupby(["arm", "r", "kind", "user"]).d.mean()
res = {}
print("D510 stack-level Δ vs the unadapted stack (same scored windows); strangers: ten-user mean | min user | not-covered six; returning P04 / P10:", flush=True)
for arm in ARMS:
    for r in RS:
        r_ = round(r, 3)
        if (arm, r_, "stranger") not in pu.index:
            continue
        sv = pu.loc[(arm, r_, "stranger")]
        rv = pu.loc[(arm, r_, "returning")] if (arm, r_, "returning") in pu.index else pd.Series(dtype=float)
        e = dict(ten=round(float(sv.mean()), 4), min_user=round(float(sv.min()), 4), min_who=str(sv.idxmin()),
                 nc6=round(float(sv[[u for u in NC6 if u in sv.index]].mean()), 4), ret_mean=round(float(rv.mean()), 4) if len(rv) else None,
                 ret={u: round(float(v), 4) for u, v in rv.items()}, per_user={u: round(float(v), 4) for u, v in sv.items()})
        res[f"{arm}@{r_}"] = e
        print(f"   {arm:8s} r {r_:.3f}: ten {e['ten']:+.4f} | min {e['min_user']:+.4f} ({e['min_who']}) | nc6 {e['nc6']:+.4f} | returning "
              + (f"{e['ret_mean']:+.4f} [" + " ".join(f"{u} {v:+.4f}" for u, v in e['ret'].items()) + "]" if len(rv) else "-"), flush=True)
A = pd.DataFrame(alone)
A["d"] = A.new - A.base
al = A[A.kind == "stranger"].groupby(["arm", "user"]).d.mean().groupby("arm").mean()
print("D510 student ALONE (decoded, r 1), strangers ten-user Δ: " + " ".join(f"{a} {v:+.4f}" for a, v in al.items()), flush=True)
res["alone"] = {a: round(float(v), 4) for a, v in al.items()}
reps = []
for arm in ((PRIMARY[0], PRIMARY[0] + "r1", PRIMARY[0] + "r2")):
    if not (R.arm == arm).any():
        continue
    q = R[(R.arm == arm) & (R.r == 0.5) & (R.kind == "stranger")].groupby("user").d.mean().mean()
    reps.append(float(q))
res["primary_reps"] = dict(values=[round(v, 4) for v in reps], mean=round(float(np.mean(reps)), 4), sd=round(float(np.std(reps, ddof=1)), 4))
print(f"D510 primary repeats (fine-tune seeds): {[round(v, 4) for v in reps]} → mean {np.mean(reps):+.4f}, sd {np.std(reps, ddof=1):.4f}", flush=True)
P = pd.DataFrame(posr)
if len(P):
    pb = P.groupby(["kind", "bin"]).apply(lambda g: np.average(g.d_acc, weights=g.n)).unstack()
    print("D510 primary, window-accuracy Δ by position in the stream:\n" + pb.round(4).to_string(), flush=True)
    res["by_position"] = {k: {b: round(float(v), 4) for b, v in row.items()} for k, row in pb.iterrows()}
ft = np.concatenate([np.load(fp)["ft"] for sl in SLOTS for fp in glob.glob(str(SRC[sl][0] / "runs/*isT__s*.npz"))])
res["finetune_seconds"] = dict(n=int(len(ft)), mean=round(float(ft[:, 1].mean()), 3), p95=round(float(np.quantile(ft[:, 1], 0.95)), 3))
print(f"D510 fine-tune time (the 5090): {len(ft)} fits, mean {ft[:, 1].mean():.3f} s, p95 {np.quantile(ft[:, 1], 0.95):.3f} s", flush=True)
pr = res[f"{PRIMARY[0]}@{PRIMARY[1]}"]
ceil = res.get(f"isS@{PRIMARY[1]}", {}).get("ten")
ok = pr["ten"] >= 0.010 and pr["min_user"] >= -0.020 and (pr["ret_mean"] is None or pr["ret_mean"] >= -0.003)
verdict = "PASS" if ok else ("IFNet next (+0.005 .. +0.010)" if pr["ten"] >= 0.005 else "CLOSES (< +0.005)" if pr["ten"] < 0.005 else "guard fails")
if not ok and pr["ten"] >= 0.010:
    verdict = "guard fails (level passes, a guard does not)"
res["verdict"] = dict(primary=pr, ceiling_ten=ceil, share_of_ceiling=round(pr["ten"] / ceil, 3) if ceil else None, verdict=verdict)
print(f"D510 [{'+'.join(SLOTS)}] PRIMARY SCHED x conf x r 1/2: ten {pr['ten']:+.4f}, min {pr['min_user']:+.4f} ({pr['min_who']}), returning {pr['ret_mean']}; "
      f"true-label ceiling (SCHED, r 1/2) {ceil}; -> {verdict}", flush=True)
if ROW == "d511":                                       # T2-D511's bands (the manager's, fixed before any number)
    g = pr["min_user"] >= -0.020 and (pr["ret_mean"] is None or pr["ret_mean"] >= -0.003)
    band = ("guards fail" if not g else ">= +0.010: the owner's line + build plan" if pr["ten"] >= 0.010 else
            "+0.006 .. +0.010: an OPTION for the final stack (audit spread stated)" if pr["ten"] >= 0.006 else "< +0.006: in-stream adaptation closes for good")
    res["d511_band"] = band
    print(f"D511 BAND [{'+'.join(SLOTS)}, pos >= {POS_MIN}]: primary ten {pr['ten']:+.4f} -> {band}", flush=True)
json.dump(res, open(D / f"read{SFX}.json", "w"), indent=1)
