#!/usr/bin/env python
"""T2-D540 (registered 2026-11-16 19:43:45; committed before the chain's outputs exist): the W19 CONFIRM read on FRESH draws 819-821.
W18's form re-trained at 819-821 by scripts/local/t2_d540_chain_local.sh (LESSONS A43); the FM side reused (SD six 826-831 + MI 826), as D536.

D536's reader head is exec'd with its draw constants moved (816-818 -> 819-821, t2_d536 -> t2_d540, DC bag 7-8-9 -> 10-11-12); each replacement
is asserted to happen exactly once. Arms (every one vs BASE = W18's form at 819-821):
  F   = BASE + FBCNet side nets at r 1/3 (D538's form: FBCNet part = (2/3) routed + (1/3) mean of its 5 side nets, before its corrector);
  H1  = BASE + D539b (D539's in-stream head re-fit, lam 1 the only arm, on the ShallowNet / IFNet side heads);
  FH  = BASE + F + the lam-1 re-fit on ALL three families' side heads.
W17's form at 819-821 is reported beside (descriptive).
Rule (D540): each single item by D536's rule (ten > 0, >= 2 / 3 draws, no stranger < -0.020, returning >= -0.003, AND ten >= +0.001).
Both confirm -> FH by the same rule -> W19 = FH, else W19 = F alone (named before any number). One confirms -> W19 = that item. None -> no W19.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d540_read.py
"""
import json
import re
import sys
import time

import numpy as np
from scipy.optimize import minimize

T0 = time.time()
CHECK816 = "--check816" in sys.argv     # the reader's own test: on D536's draws it must reproduce D539's lam-1 arm and D538's headline exactly
src = open("scripts/local/t2_d536_read.py").read()
CUT = '\nW18 = ["D524", "D529"]\n'
assert src.count(CUT) == 1, "the D536 reader's tail moved"
src = src.split(CUT)[0]
for a, b in (("new = 'DRAWS, CNN, DCF = (816, 817, 818), O / \"t2_d536\", O / \"t2_d536/pred_s7-8-9_dc.npz\"'",
              "new = 'DRAWS, CNN, DCF = (819, 820, 821), O / \"t2_d540\", O / \"t2_d540/pred_s10-11-12_dc.npz\"'"),
             ('assert not A.check and A.d524_form == "six" and DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")',
              'assert not A.check and A.d524_form == "six" and DRAWS == (819, 820, 821) and str(CNN).endswith("t2_d540")')):
    if CHECK816:
        continue
    assert src.count(a) == 1, (a[:60], src.count(a))
    src = src.replace(a, b)
exec(compile(src, "t2_d536_read_head(d540)", "exec"))
if CHECK816:
    assert DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")
else:
    assert DRAWS == (819, 820, 821) and str(CNN).endswith("t2_d540") and str(DCF).endswith("pred_s10-11-12_dc.npz")

# ---- D539's in-stream head re-fit, verbatim (scripts/local/t2_d539_read.py at 3fd109bf), teacher redirected to the chain's own teacher ----
TEACH = CNN / "teacher_p1"
CKS = (60, 100, 150)
_TCACHE = {}
STATS = {}


def teacher(f, d):
    if (f, d) not in _TCACHE:
        z = np.load(need(TEACH / f"lc9_{f}__s{d}.npz"))
        rows = z["rows"].astype(np.int64)
        _TCACHE[(f, d)] = ({int(r): k for k, r in enumerate(rows)}, {c: z[f"post_r{c}"].astype(np.float64) for c in CKS})
    return _TCACHE[(f, d)]


def conf_half(pp):
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
# ---- end of D539's code ----


ARM = ("confR", 1.0)                                  # D539b: lam 1 is the only arm
_side_prob_unc = side_prob


def side_prob(path, f, mode):                         # the head's parts_of resolves this name at call time
    if mode == "h":
        return side_prob_h(path, f, ARM)
    return _side_prob_unc(path, f, mode)


FBC_SIDE = (O / "t2_d538/side_fbc/runs") if CHECK816 else (CNN / "side_fbc/runs")
FBC_STEM = "lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr__sidedcmean__s{s}.npz"
WITH_FBC = False
_parts_of_d536 = parts_of


def parts_of(f, d, parts):                           # D536's parts_of; WITH_FBC adds the FBCNet side nets at r 1/3 (D538's form)
    P = _parts_of_d536(f, d, parts)
    if WITH_FBC:
        ix = IX[f]
        S_ = np.mean([realign(*side_prob(FBC_SIDE / FBC_STEM.format(f=f, s=d + k * 1000), f, SIDE_MODE), ix) for k in range(5)], 0)
        P = dict(P)
        P["fbc"] = (2.0 / 3.0) * P["fbc"] + (1.0 / 3.0) * S_
    return P


def judge(Aa, Bb):
    sc = score(Aa, Bb)
    sc = dict(per_stranger=per_stranger(Aa, Bb), **sc)
    sc["confirmed"] = bool(sc["part"] and sc["ten"] >= 0.001)
    return sc


W18 = ["D524", "D529"]
SIDE_MODE, WITH_FBC = "unc", False
BASE = mix(W18)
W17F = mix([])
SIDE_MODE, WITH_FBC = "unc", True
F = mix(W18)
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
SIDE_MODE, WITH_FBC = "h", False
H1 = mix(W18)
SIDE_MODE, WITH_FBC = "h", True
FH = mix(W18)
print(f"D540 checks: {CHK}; re-fits: { {k: dict(v, dW=round(float(np.mean(v['dW'])), 5)) for k, v in STATS.items()} }", flush=True)
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
res = {"draws": list(DRAWS), "checks": dict(CHK),
       "level": {"W17 form": level(W17F), "BASE (W18 form)": level(BASE), "F": level(F), "H1": level(H1), "FH": level(FH)}}
res["F vs BASE"] = judge(F, BASE)
res["H1 vs BASE"] = judge(H1, BASE)
res["FH vs BASE"] = judge(FH, BASE)
res["BASE (W18 form) vs W17 form (descriptive)"] = dict(score(BASE, W17F), per_stranger=per_stranger(BASE, W17F))
cf, ch, cfh = res["F vs BASE"]["confirmed"], res["H1 vs BASE"]["confirmed"], res["FH vs BASE"]["confirmed"]
if cf and ch:
    w19 = "FH" if cfh else "F"
elif cf or ch:
    w19 = "F" if cf else "H1"
else:
    w19 = None
res["W19"] = w19
for k, v in res.items():
    print(f"D540 {k}: {v}", flush=True)
print(f"D540 verdict: {'W19 = ' + w19 + ' (draft the owner line)' if w19 else 'NO W19 (the last upload follows D536)'} ({time.time() - T0:.0f} s)", flush=True)
if CHECK816:
    ref9, ref8 = json.load(open(O / "t2_d539.json"))["lam1 (descriptive) vs BASE"], json.load(open(O / "t2_d538.json"))["HEADLINE W19F vs BASE (W18 form), draws 816-818"]
    for mine, ref in ((res["H1 vs BASE"], ref9), (res["F vs BASE"], ref8)):
        for k in ("ten", "draws", "six", "min_user", "returning"):
            assert mine[k] == ref[k], (k, mine[k], ref[k])
    print("D540 check816: PASS (D539's lam-1 arm and D538's headline reproduced exactly)", flush=True)
    raise SystemExit(0)
json.dump(res, open(O / "t2_d540.json", "w"), indent=1, default=str)
