#!/usr/bin/env python
"""T2-D498 read (written 2026-11-14 before any D498 member landed): the full stack on the K = 9 panel (lc9 folds 0-9; fold f holds one
user out entirely = the STRANGER; P04 / P10 sessions 2-3 of a trained user = RETURNING).
Members per fold: CNN families = M7's lines with copies + tv8c routed per fold (t2_d498/route_<fam>_lc9; seeds 750-752 = the 3 draws);
FM family = 3 plain (seeds 760-762) + 3 CAR (763-765) ST-L from the PUBLIC checkpoint (the same six in every draw), equal-weight mean.
Shares = M7's (FBCNet 0.3143, ShallowNet 0.2357, IFNet 0.10, FM 0.35). Decode = W13's: each family through the causal Sinkhorn
(10 it, n0 3, lambda 0.95) from its own start vector, cross-fitted by T2-D459's rule over the OTHER folds' strangers; then the weighted pool.
 (i)  stranger level per user (stack and each family alone), returning level; vs the kit-protocol A13 W14-lite level (k3c3, from
      t2_d475/cells_a13lite.csv) = the registered stop rule; beside it, the same run's kit-like subset (strangers P05 P08 P09) because
      this run's FM has no CPT (a stated confound).
      STOP iff ten-stranger typical (non-P04) >= kit-protocol typical - 0.01.
 (ii) family shares for the STRANGER side: each share x {0.5, 1, 2} (81 sets, renormalised), picked leave-one-user-out (fit on nine
      strangers' cells over the 3 draws, apply to the tenth) vs M7's shares. KILL: LOUO typical < +0.003 or any user < -0.005.
 (iii) per stranger, per-class recall under the shipped shares (one class collapsing vs diffuse).
    mem-guard python scripts/local/t2_d498_read.py
"""
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
R = O / "t2_d498"
idx = pd.read_csv(EXP / "index.csv")
U, STREAM, ORD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
CNNF = {"fbc": "route_fbc_lc9/lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz",
        "sh": "route_sh_lc9/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
        "ifn": "route_ifn_lc9/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz"}
DRAWS = (750, 751, 752)


def fm_files(f):
    out = []
    for a, seeds in (("stl", (760, 761, 762)), ("stlcar", (763, 764, 765))):
        for s in seeds:
            fs = glob.glob(f"{S}/ni26-t2p-d498-{a}_s{s}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
            assert len(fs) == 1, (a, s, f, fs)
            out.append(fs[0])
    return out


def load(path):
    z = np.load(path)
    p = z["prob_last"].astype(np.float64)
    return z["test_idx"], p / p.sum(1, keepdims=True)


MIX = {}
for f in range(10):
    fm = [load(p) for p in fm_files(f)]
    ix = fm[0][0]
    assert all((x[0] == ix).all() for x in fm)
    pfm = np.mean([x[1] for x in fm], 0)
    for d in DRAWS:
        mx = {"fm": pfm}
        for fam, pat in CNNF.items():
            ix_, p = load(R / pat.format(f=f, s=d))
            assert (ix_ == ix).all(), (fam, f, d)
            mx[fam] = p
        MIX[(f, d)] = (ix, mx)
TRU = {}
for f in range(10):
    ix = MIX[(f, DRAWS[0])][0]
    TRU[f] = [u for u in set(U[ix]) if (U[ix] == u).sum() == (U == u).sum()][0]
assert sorted(TRU.values()) == sorted(set(U)), TRU


def vec(f_out, d, fam):
    per = []
    for f in range(10):
        if f == f_out:
            continue
        ix, mx = MIX[(f, d)]
        per.append(mx[fam][U[ix] == TRU[f]].mean(0))
    v = np.mean(per, 0)
    return v / v.sum()


DEC = {}                                     # per (fold, draw): each family's decoded rows (causal, own start), in ix order
for (f, d), (ix, mx) in MIX.items():
    key = (STREAM[ix], ORD[ix])
    DEC[(f, d)] = {fam: causal_sinkhorn(p, key, 10, 3, 0.95, start=vec(f, d, fam)) for fam, p in mx.items()}
    DEC[(f, d)]["J"] = causal_sinkhorn(sum(W[k] * mx[k] for k in W), key, 10, 3, 0.95)    # M7's joint decode (descriptive)


def cells_of(f, d, pred):
    ix = MIX[(f, d)][0]
    out = []
    for c in np.unique(CELL[ix]):
        mk = CELL[ix] == c
        u = U[ix][mk][0]
        out.append(dict(fold=f, draw=d, cell=c, user=u, kind="stranger" if u == TRU[f] else "returning",
                        ba=float(np.mean([np.mean(pred[mk][Y[ix][mk] == q] == q) for q in np.unique(Y[ix][mk])]))))
    return out


def stack_pred(f, d, w):
    return sum(w[k] * DEC[(f, d)][k] for k in w).argmax(1)


rows = []
for (f, d) in MIX:
    for arm, w in (("stack", W), ("fbc", {"fbc": 1}), ("sh", {"sh": 1}), ("ifn", {"ifn": 1}), ("fm", {"fm": 1})):
        rows += [dict(r, arm=arm) for r in cells_of(f, d, stack_pred(f, d, w))]
    rows += [dict(r, arm="J") for r in cells_of(f, d, DEC[(f, d)]["J"].argmax(1))]
C = pd.DataFrame(rows)
C.to_csv(R / "cells.csv", index=False)
res = {}
st = C[C.kind == "stranger"]
pu = st.groupby(["user", "arm"]).ba.mean().unstack()
typ = pu.drop("P04", errors="ignore").mean()
ret = C[C.kind == "returning"].groupby("arm").ba.mean()
kit = pd.read_csv(O / "t2_d475/cells_a13lite.csv")
kit_typ = float(kit[(kit.arm == "k3c3") & (kit.subj != "P04")].ba.mean())
sub3 = float(pu.loc[["P05", "P08", "P09"], "stack"].mean())
res["i"] = dict(stranger_typical={k: round(float(v), 4) for k, v in typ.items()}, stranger_per_user={u: {a: round(float(pu.loc[u, a]), 4) for a in pu.columns} for u in pu.index},
                returning={k: round(float(v), 4) for k, v in ret.items()}, kit_protocol_typical_w14lite=round(kit_typ, 4), same_run_P05_P08_P09=round(sub3, 4))
print(f"D498 (i) ten strangers, stack: typical {typ['stack']:.4f} (J {typ['J']:.4f}); alone fbc {typ['fbc']:.4f} sh {typ['sh']:.4f} ifn {typ['ifn']:.4f} fm {typ['fm']:.4f}", flush=True)
print("   per stranger (stack | fm alone): " + " ".join(f"{u} {pu.loc[u, 'stack']:.3f}|{pu.loc[u, 'fm']:.3f}" for u in pu.index), flush=True)
print(f"   returning (P04 / P10 s2-3) stack {ret['stack']:.4f}; kit-protocol W14-lite typical {kit_typ:.4f}; same run P05 P08 P09 {sub3:.4f}", flush=True)
stop = typ["stack"] >= kit_typ - 0.01
print(f"D498 (i) STOP RULE: ten-stranger typical {typ['stack']:.4f} vs kit-protocol {kit_typ:.4f} - 0.01 -> "
      + ("STOP: the loss is NOT on the stranger side" if stop else "the loss IS on the stranger side: continue to (ii) / (iii)"), flush=True)
if not stop:
    users = sorted(TRU.values())
    sets = [dict(zip(W, (W[k] * m for k, m in zip(W, ms)))) for ms in itertools.product((0.5, 1, 2), repeat=4)]
    sets = [{k: v / sum(w.values()) for k, v in w.items()} for w in sets]
    score = np.zeros((len(sets), len(users)))                 # mean stranger cell BA per (share set, user) over its folds / draws
    for j, w in enumerate(sets):
        rr = []
        for (f, d) in MIX:
            rr += [r for r in cells_of(f, d, stack_pred(f, d, w)) if r["kind"] == "stranger"]
        T = pd.DataFrame(rr).groupby("user").ba.mean()
        score[j] = [T[u] for u in users]
    base = int(np.argmin([sum(abs(w[k] - W[k] / sum(W.values())) for k in W) for w in sets]))   # the shipped set (all x 1)
    louo = {}
    for k, u in enumerate(users):
        others = [m for m in range(len(users)) if m != k and users[m] != "P04"]
        j = int(np.argmax(score[:, others].mean(1)))
        louo[u] = dict(pick={kk: round(v, 4) for kk, v in sets[j].items()}, delta=round(float(score[j, k] - score[base, k]), 4))
    dl = pd.Series({u: v["delta"] for u, v in louo.items()})
    g, mn = float(dl.drop("P04", errors="ignore").mean()), float(dl.drop("P04", errors="ignore").min())
    kill = g < 0.003 or mn < -0.005
    jall = int(np.argmax(score[:, [m for m in range(len(users)) if users[m] != "P04"]].mean(1)))
    res["ii"] = dict(louo=louo, typical_delta=round(g, 4), min_user=round(mn, 4), killed=bool(kill), all_user_pick={kk: round(v, 4) for kk, v in sets[jall].items()})
    print(f"D498 (ii) LOUO stranger-side shares vs M7's: typical {g:+.4f}, per user " + " ".join(f"{u} {v:+.4f}" for u, v in dl.items())
          + f" -> {'KILLED' if kill else 'PASSES'}; the all-ten pick {res['ii']['all_user_pick']}", flush=True)
    rec = {}
    for (f, d) in MIX:
        ix = MIX[(f, d)][0]
        m = U[ix] == TRU[f]
        pr = stack_pred(f, d, W)[m]
        yy = Y[ix][m]
        rec.setdefault(TRU[f], []).append([float(np.mean(pr[yy == q] == q)) for q in range(3)])
    rec = {u: np.mean(v, 0) for u, v in rec.items()}
    res["iii"] = {u: [round(float(x), 3) for x in v] for u, v in rec.items()}
    col = [u for u, v in rec.items() if v.min() < 0.35 and np.sort(v)[1] > 0.6]
    print("D498 (iii) per-class recall (MI, SUB, WORD) under the stack: " + " ".join(f"{u} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for u, v in rec.items())
          + f" | one class collapsing (min < 0.35, others > 0.6): {col}", flush=True)
json.dump(res, open(R / "read.json", "w"), indent=1)
