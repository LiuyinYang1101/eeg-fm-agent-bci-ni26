#!/usr/bin/env python
"""T2-D512 P1 teacher files (zero GPU; written 2026-11-15 before any D512 number): the D511 teacher's construction on D512's FRESH CNN base
(seeds 810-812: one routed net per family with copies + tv8c, scripts/local/t2_d512_cnn_local.sh) and T2-D498's FM part (the six public-checkpoint
ST-L members; the fresh FM members come from sofia later, for the full stack only). The student seed = the CNN draw (810-812). Also saved: R_<fam>,
the RAW routed probabilities of each CNN family on the same rows (the P1 read blends the adapted copy into these). Plus post_r<c> for c in 60 / 100 / 150: the labels of windows 1 .. c RE-DECODED over
the whole delivered prefix (per family the Sinkhorn of [n0 start rows; p_1 .. p_c], 10 it / n0 3 / lambda 0.95, every row kept; pooled at M7's
shares; NaN where a window is past c or its stream is shorter than c). Per fold f and student seed s: T2-D498's shipped stack shape (M7's shares fbc 0.3143 / sh 0.2357 / ifn 0.10 / fm 0.35;
FM = 3 plain + 3 CAR ST-L; each family decoded by the causal Sinkhorn 10 / 3 / 0.95 from its cross-fitted start vector (T2-D459 / D502's
rule: the mean posterior of the OTHER folds' strangers), then pooled), exactly T2-D502's MIX / vec / W. Saved for EVERY test row (the held-out user + the returning P04 / P10 sessions 2-3; `held` marks the first):
rows (global index), post (the stack's decoded posterior: window t uses windows <= t of its stream only), and for the stack-level swap read the
other families' decoded outputs (D_fbc, D_ifn, D_fm) and the ShallowNet slot's start vector (vec_sh).
    mem-guard python scripts/local/t2_d510_teacher.py
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
S = O / "sofia_s1"
R = O / "t2_d512"
OUT = O / "t2_d512/teacher_p1"
CKS = (60, 100, 150)
OUT.mkdir(parents=True, exist_ok=True)
idx = pd.read_csv(EXP / "index.csv")
U, STREAM, ORD, Y = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy()
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
CNNF = {"fbc": "route_fbc/lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz",
        "sh": "route_sh/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
        "ifn": "route_ifn/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz"}
DRAWS = (810, 811, 812)


def load(path):
    z = np.load(path)
    p = z["prob_last"].astype(np.float64)
    return z["test_idx"], p / p.sum(1, keepdims=True)


MIX = {}
for f in range(10):
    fm = []
    for a, seeds in (("stl", (760, 761, 762)), ("stlcar", (763, 764, 765))):
        for s in seeds:
            fs = glob.glob(f"{S}/ni26-t2p-d498-{a}_s{s}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
            assert len(fs) == 1, (a, s, f, fs)
            fm.append(load(fs[0]))
    ix = fm[0][0]
    pfm = np.mean([x[1] for x in fm], 0)
    for d in DRAWS:
        mx = {"fm": pfm}
        for fam, pat in CNNF.items():
            ix_, p = load(R / pat.format(f=f, s=d))
            assert (ix_ == ix).all()
            mx[fam] = p
        MIX[(f, d)] = (ix, mx)
TRU = {f: [u for u in set(U[MIX[(f, DRAWS[0])][0]]) if (U[MIX[(f, DRAWS[0])][0]] == u).sum() == (U == u).sum()][0] for f in range(10)}


def prefix_decode(P, start, iters=10, n0=3, lam=0.95):
    """T2-D511: the Sinkhorn of [n0 start rows; P] with T2-D304's forgetting weights; returns EVERY data row (not only the last)."""
    K = P.shape[1]
    M = np.vstack([np.tile(np.asarray(start, np.float64), (int(n0), 1)), P.astype(np.float64)])
    w = float(lam) ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
    for _ in range(int(iters)):
        M = M * (w.sum() / K) / (M * w[:, None]).sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
    return M[int(n0):]


def vec(f_out, d, fam):
    per = [MIX[(f, d)][1][fam][U[MIX[(f, d)][0]] == TRU[f]].mean(0) for f in range(10) if f != f_out]
    v = np.mean(per, 0)
    return v / v.sum()


acc = []
for (f, d), (ix, mx) in MIX.items():
    h = np.arange(len(ix))                                  # every test row: the held-out user AND the returning P04 / P10 sessions 2-3
    h = h[np.lexsort((ORD[ix][h], STREAM[ix][h]))]
    rows = ix[h]
    key = (STREAM[rows], ORD[rows])
    Dk = {k: causal_sinkhorn(mx[k][h], key, 10, 3, 0.95, start=vec(f, d, k)) for k in W}
    post = sum(W[k] * Dk[k] for k in W)
    post = post / post.sum(1, keepdims=True)
    PR = {c: np.full((len(rows), 3), np.nan) for c in CKS}
    V = {k: vec(f, d, k) for k in W}
    for st_ in np.unique(STREAM[rows]):
        j = np.flatnonzero(STREAM[rows] == st_)                   # already in acquisition order (h is sorted by stream, order)
        for c in CKS:
            if len(j) <= c:
                continue
            q = sum(W[k] * prefix_decode(mx[k][h][j[:c]], V[k]) for k in W)
            PR[c][j[:c]] = q / q.sum(1, keepdims=True)
    np.savez_compressed(OUT / f"lc9_{f}__s{d}.npz", rows=rows, post=post.astype(np.float32), **{f"R_{k}": mx[k][h].astype(np.float32) for k in ("fbc", "sh", "ifn")}, **{f"post_r{c}": PR[c].astype(np.float32) for c in CKS}, D_fbc=Dk["fbc"].astype(np.float32),
                        D_ifn=Dk["ifn"].astype(np.float32), D_fm=Dk["fm"].astype(np.float32), D_sh=Dk["sh"].astype(np.float32), vec_sh=vec(f, d, "sh"),
                        vec_ifn=vec(f, d, "ifn"), vec_fbc=vec(f, d, "fbc"), draw=d, user=TRU[f],
                        held=U[rows] == TRU[f])
    hm = U[rows] == TRU[f]
    acc.append(dict(fold=f, draw=d, user=TRU[f], acc=float(np.mean(post[hm].argmax(1) == Y[rows][hm]))))
A = pd.DataFrame(acc)
print("D512 P1 teacher (fresh CNN base + D498 FM, causal) window accuracy on the held-out user, mean over draws: "
      + " ".join(f"{u} {v:.3f}" for u, v in A.groupby("user").acc.mean().items()), flush=True)
