#!/usr/bin/env python
"""T2-D539 (W19-H, stage 1; registered 2026-11-16 19:36:59 before any number; zero GPU): the in-stream re-fit of the side HEADS, the side nets frozen,
read on D536's draws 816-818 (a NOMINATION read; the manager's condition 1: a nominated item is confirmed on fresh draws 819-821 before any W19 line).

D536's reader head is exec'd unchanged (it execs D532's head at DRAWS 816-818: W18's form = FM SD six + MI at 826, side nets UNC, DC bag 7-9).
BASE = W18's form (SIDE_MODE 'unc'); its level is asserted equal to D536's recorded BASE level.
W19-H = BASE with every ShallowNet / IFNet side net's output recomputed from its dump with re-fitted heads:
  per test stream (every stream, as W18's in-stream), rows sorted by 'order' (= t2p_bench stream_rows_); at c in (60, 100, 150), if the stream has
  more than c windows: labels = confR exactly as t2p_bench labels_('confR') (argmax of the teacher's post_r{c} over positions 0 .. c-1, the confident
  half of each predicted class, ceil(n / 2), stable sort by -max posterior; a non-finite post_r row is dropped); teacher = outputs/t2_d536/teacher_p1
  (the teacher the P1 parts were read with); fit from the TRAINED head (W0, b0) at every re-fit: mean CE(softmax(L + W s + b), y)
  + (lam / 2)(|W - W0|^2 + |b - b0|^2), scipy L-BFGS-B (float64, gtol 1e-8, maxiter 500); the head of re-fit c scores positions c .. next c - 1
  (the last to the stream's end); positions < 60 and streams with <= 60 windows keep the trained head.
Decision arm: lam = 10 ONLY. Descriptive (never picked): lam 1, lam 100, and the true-label ceiling (true labels on every prefix window, lam 10).
Rule (D536's): ten > 0, >= 2 / 3 draws, no stranger < -0.020, returning >= -0.003, AND ten >= +0.001 -> NOMINATED (not a part: 816-818 were read).
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d539_read.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

T0 = time.time()
SRC36 = open("scripts/local/t2_d536_read.py").read()
CUT = '\nW18 = ["D524", "D529"]\n'
assert SRC36.count(CUT) == 1, "the D536 reader's tail moved"
exec(compile(SRC36.split(CUT)[0], "t2_d536_read_head(d539)", "exec"))
assert DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")

D536 = json.load(open(O / "t2_d536.json"))
TEACH = O / "t2_d536/teacher_p1"
CKS = (60, 100, 150)
LAM_MAIN = 10.0
_TCACHE = {}
STATS = {}


def teacher(f, d):
    if (f, d) not in _TCACHE:
        z = np.load(need(TEACH / f"lc9_{f}__s{d}.npz"))
        rows = z["rows"].astype(np.int64)
        _TCACHE[(f, d)] = ({int(r): k for k, r in enumerate(rows)}, {c: z[f"post_r{c}"].astype(np.float64) for c in CKS})
    return _TCACHE[(f, d)]


def conf_half(pp):
    """t2p_bench labels_('confR')'s selection: the confident half of each predicted class (class-balanced; no class vanishes)."""
    ya, cf = pp.argmax(1), pp.max(1)
    keep = np.zeros(len(ya), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return ya, keep


def fit_head(Ls, Ss, ys, W0, b0, lam):
    K, D = W0.shape
    n = len(ys)
    th0 = np.concatenate([W0.ravel(), b0])
    Y1 = np.eye(K)[ys]

    def fg(th):
        W, b = th[:K * D].reshape(K, D), th[K * D:]
        z = Ls + Ss @ W.T + b
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        G = (np.exp(z - lse[:, None]) - Y1) / n
        dth = th - th0
        f = float((lse - z[np.arange(n), ys]).mean() + 0.5 * lam * (dth @ dth))
        return f, np.concatenate([(G.T @ Ss).ravel(), G.sum(0)]) + lam * dth

    r = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(gtol=1e-8, maxiter=500))
    return r.x[:K * D].reshape(K, D), r.x[K * D:], bool(r.success)


def softmax(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def side_prob_h(path, f, arm):
    """one side net's output with the in-stream head re-fit (arm = (label form 'confR' | 'true', lam))."""
    form, lam = arm
    z = np.load(need(path))
    te = z["test_idx"].astype(np.int64)
    L, Wh, bh = (z[k].astype(np.float64) for k in ("side_logit_last", "side_w", "side_b"))
    S0 = z["side_te"].astype(np.float64)
    d = int(re.search(r"__s(\d+)\.npz$", str(path)).group(1)) % 1000
    assert d in DRAWS, (path, d)
    p = softmax(L + S0 @ Wh.T + bh)
    pos, post = teacher(f, d)
    st = STATS.setdefault(arm, dict(fits=0, fail=0, dropped=0, sel=0, sel_right=0, dW=[], streams=0, streams_short=0))
    st_te, od_te = STREAM[te], ORD[te]
    for s in np.unique(st_te):
        j = np.flatnonzero(st_te == s)
        j = j[np.argsort(od_te[j], kind="stable")]
        st["streams"] += 1
        if len(j) <= CKS[0]:
            st["streams_short"] += 1
            continue
        for k, c in enumerate(CKS):
            if len(j) <= c:
                break
            e = CKS[k + 1] if k + 1 < len(CKS) else len(j)
            a = j[:c]
            if form == "true":
                sel, ys = a, Y[te[a]].astype(np.int64)
            else:
                pp = np.stack([post[c][pos[int(r)]] for r in te[a]])
                fin = np.isfinite(pp).all(1)
                st["dropped"] += int((~fin).sum())
                ya, keep = conf_half(pp[fin])
                sel, ys = a[fin][keep], ya[keep].astype(np.int64)
                st["sel"] += len(sel)
                st["sel_right"] += int((ys == Y[te[sel]]).sum())
            W, b, ok = fit_head(L[sel], S0[sel], ys, Wh, bh, lam)
            st["fits"] += 1
            st["fail"] += int(not ok)
            st["dW"].append(float(np.abs(W - Wh).mean()))
            q = j[c:e]
            p[q] = softmax(L[q] + S0[q] @ W.T + b)
    return te, p


ARM = None
_side_prob_unc = side_prob


def side_prob(path, f, mode):                        # the head's parts_of resolves this name at call time
    if mode == "h":
        return side_prob_h(path, f, ARM)
    return _side_prob_unc(path, f, mode)


W18 = ["D524", "D529"]
SIDE_MODE = "unc"
BASE = mix(W18)
lv = level(BASE)
assert abs(lv - D536["level"]["BASE (W18 form)"]) < 5e-5, (lv, D536["level"])
print(f"D539 BASE (W18 form) level {lv} = D536's {D536['level']['BASE (W18 form)']}; checks {CHK}", flush=True)
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK

res = {"draws": list(DRAWS), "lam_main": LAM_MAIN, "checks": dict(CHK), "level": {"BASE (W18 form)": lv}}
SIDE_MODE = "h"
for arm, name in (((("confR", LAM_MAIN)), "W19H lam10 (DECISION)"), (("confR", 1.0), "lam1 (descriptive)"),
                  (("confR", 100.0), "lam100 (descriptive)"), (("true", LAM_MAIN), "true-label ceiling lam10 (descriptive)")):
    ARM = arm
    t_ = time.time()
    H = mix(W18)
    sc = score(H, BASE)
    sc = dict(per_stranger=per_stranger(H, BASE), **sc)
    s_ = STATS[arm]
    sc["fits"] = dict(fits=s_["fits"], fail=s_["fail"], streams=s_["streams"], streams_short=s_["streams_short"], dropped=s_["dropped"],
                      teacher_acc=(round(s_["sel_right"] / s_["sel"], 4) if s_["sel"] else None), sel=s_["sel"],
                      mean_abs_dW=round(float(np.mean(s_["dW"])), 5) if s_["dW"] else None, sec=round(time.time() - t_, 1))
    res["level"][name] = level(H)
    if arm == ("confR", LAM_MAIN):
        sc["nominated"] = bool(sc["part"] and sc["ten"] >= 0.001)
    res[f"{name} vs BASE"] = sc
    print(f"D539 {name} vs BASE: {sc}", flush=True)

head = res["W19H lam10 (DECISION) vs BASE"]
print(f"D539 levels: {res['level']}", flush=True)
print(f"D539 verdict: {'NOMINATED (confirm on fresh draws 819-821 before any W19 line)' if head['nominated'] else 'NOT NOMINATED (closed)'}"
      f" ({time.time() - T0:.0f} s)", flush=True)
json.dump(res, open(O / "t2_d539.json", "w"), indent=1, default=str)
