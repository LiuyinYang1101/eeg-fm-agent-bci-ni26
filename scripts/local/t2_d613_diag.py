#!/usr/bin/env python
"""T2-D613 STEP 1, the cached gate (registered 92b39e79 before any number; decided on SET A, B–D descriptive after): the reciprocal-neighbour LP
graph. For each fold x draw x stream x refit c, the fb root's teacher post_r<c> on the prefix windows (rows < c) is propagated over the kNN graph of
the members' pooled FM features (= t2_lab.py lpteach / IDEAS._lp_block: centred, L2, cosine, k 10, weights max(cos, 0), alpha 0.5) with
  union      A = maximum(A, A.T)             (D569 as shipped; ASSERTED == the lp root's post_r<c>, <= 1e-6, and == IDEAS._lp_block, <= 1e-12)
  reciprocal A = A where A > 0 and A.T > 0   (an edge only when both windows select each other; an isolated window keeps its posterior)
then the confident half per predicted class (IDEAS._conf, frac 0.5). Counters per group (strangers / returning), micro-summed: the confident
half's label accuracy (truth: a diagnostic only), selected rows, blocks whose selection lacks a class, isolated windows, changed labels (old vs
new accuracy), dropped / added rows.
    mem-guard env OMP_NUM_THREADS=4 python scripts/local/t2_d613_diag.py --set A|B|C|D   ->  outputs/t2_d613_diag_<set>.json
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

T0 = time.time()
S = sys.argv[sys.argv.index("--set") + 1]
D0 = {"A": 825, "B": 828, "C": 831, "D": 834}[S]
for k in [k for k in os.environ if k.startswith("NI26_")]:
    del os.environ[k]
if D0 == 831:
    os.environ["NI26_LAB_D557"] = "1"                    # = t2_d571_local.sh (the teacher / LP roots' env)
O = Path(os.path.expanduser("~/workspace/outputs/neural_interface_challenge"))
TAG = "w20d557" if D0 in (831, 834) else "w20"
FBR, LPR = O / f"t2_d571_fb{TAG}_{D0}/teacher_p1", O / f"t2_d571_lp{TAG}_{D0}/teacher_p1"
sys.argv = ["t2_lab.py", "mix", "--set", S, "--base", "w20"]
src = open("scripts/local/t2_lab.py").read()
CUT = '\nif MODE == "mix":'
assert src.count(CUT) == 1
g = {"__name__": "d613", "__file__": os.path.abspath("scripts/local/t2_lab.py")}
exec(compile(src.split(CUT)[0], "t2_lab_head(d613)", "exec"), g)
IDEAS, Ctx, IX, Y, U, SES, TRU, STREAM, ORD = g["IDEAS"], g["Ctx"], g["IX"], g["Y"], g["U"], g["SES"], g["TRU"], g["STREAM"], g["ORD"]
CKS, DRAWS = tuple(g["CKS"]), g["DRAWS"]
assert CKS == (60, 100, 150)


def lp(Fr_a, T0_, mode, alpha=0.5, kk=10):
    """lpteach's per-stream block, verbatim, with the symmetrisation as the only switch."""
    c = len(T0_)
    X = Fr_a - Fr_a.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Sm = X @ X.T
    np.fill_diagonal(Sm, -np.inf)
    nn = np.argsort(-Sm, 1, kind="stable")[:, :kk]
    A_ = np.zeros((c, c))
    rr = np.repeat(np.arange(c), kk)
    A_[rr, nn.ravel()] = np.maximum(Sm[rr, nn.ravel()], 0)
    if mode == "union":
        A_ = np.maximum(A_, A_.T)
    else:
        A_ = np.where((A_ > 0) & (A_.T > 0), A_, 0.0)
    dg = A_.sum(1) + 1e-9
    Sn = A_ / np.sqrt(dg[:, None] * dg[None, :])
    Fp = np.clip(np.linalg.solve(np.eye(c) - alpha * Sn, T0_), 1e-9, None)
    return Fp / Fp.sum(1, keepdims=True), int((A_.sum(1) == 0).sum())


CT = defaultdict(lambda: defaultdict(int))
CHK = dict(union_vs_root=0.0, union_vs_ideas=0.0, blocks=0)
for f in range(10):
    Fz = IDEAS._knn_feats(Ctx, f, "fm")
    posF = {int(r): i for i, r in enumerate(IX[f])}
    for d in DRAWS:
        zf, zl = np.load(FBR / f"lc9_{f}__s{d}.npz"), np.load(LPR / f"lc9_{f}__s{d}.npz")
        rows = zf["rows"].astype(np.int64)
        assert np.array_equal(rows, zl["rows"])
        Fr = Fz[np.array([posF[int(r)] for r in rows])]
        st_, od_ = STREAM[rows], ORD[rows]
        y = Y[rows]
        grp = np.where(U[rows] == TRU[f], "stranger", np.where(np.isin(SES[rows], ("2", "3")), "returning", "other"))
        for c in CKS:
            PR = zf[f"post_r{c}"].astype(np.float64)
            for s_ in np.unique(st_):
                j = np.flatnonzero(st_ == s_)
                j = j[np.argsort(od_[j], kind="stable")]
                if len(j) <= c:
                    continue
                a = j[:c]
                T0_ = PR[a]
                pu, _ = lp(Fr[a], T0_, "union")
                pr, iso = lp(Fr[a], T0_, "recip")
                CHK["union_vs_root"] = max(CHK["union_vs_root"], float(np.abs(pu - zl[f"post_r{c}"][a].astype(np.float64)).max()))
                CHK["union_vs_ideas"] = max(CHK["union_vs_ideas"], float(np.abs(pu - IDEAS._lp_block(Fr[a], T0_, 0.5, 10)).max()))
                CHK["blocks"] += 1
                yu, ku = IDEAS._conf(pu, 0.5)
                yr, kr = IDEAS._conf(pr, 0.5)
                for gn in ("stranger", "returning"):
                    m = grp[a] == gn
                    if not m.any():
                        continue
                    C_ = CT[gn]
                    ya = y[a]
                    C_["blocks"] += 1
                    C_["isolated"] += int(iso if gn == "stranger" and m.all() else 0)
                    C_["n_union"] += int((ku & m).sum())
                    C_["ok_union"] += int(((yu == ya) & ku & m).sum())
                    C_["n_recip"] += int((kr & m).sum())
                    C_["ok_recip"] += int(((yr == ya) & kr & m).sum())
                    C_["lack_union"] += int(len(np.unique(yu[ku & m])) < 3)
                    C_["lack_recip"] += int(len(np.unique(yr[kr & m])) < 3)
                    ch = ku & kr & m & (yu != yr)
                    C_["n_changed"] += int(ch.sum())
                    C_["fix"] += int(((yr == ya) & (yu != ya) & ch).sum())
                    C_["break"] += int(((yu == ya) & (yr != ya) & ch).sum())
                    dr, ad = ku & ~kr & m, ~ku & kr & m
                    C_["n_dropped"] += int(dr.sum())
                    C_["ok_dropped"] += int(((yu == ya) & dr).sum())
                    C_["n_added"] += int(ad.sum())
                    C_["ok_added"] += int(((yr == ya) & ad).sum())
                    C_["windows"] += int(m.sum())
print(f"[d613] {S}: checks {CHK} ({time.time() - T0:.0f} s)", flush=True)
assert CHK["union_vs_root"] <= 1e-6 and CHK["union_vs_ideas"] <= 1e-12 and CHK["blocks"] > 0, CHK
res = {"set": S, "checks": CHK, "counts": {k: dict(v) for k, v in CT.items()}, "rates": {}}
for k, v in CT.items():
    r_ = lambda a_, b_: round(v[a_] / max(v[b_], 1), 4)
    res["rates"][k] = dict(acc_union=r_("ok_union", "n_union"), acc_recip=r_("ok_recip", "n_recip"), sel_union=v["n_union"], sel_recip=v["n_recip"],
                           lack_class_union=v["lack_union"], lack_class_recip=v["lack_recip"], blocks=v["blocks"], changed=v["n_changed"],
                           fix=v["fix"], brk=v["break"], acc_dropped=r_("ok_dropped", "n_dropped"), acc_added=r_("ok_added", "n_added"),
                           n_dropped=v["n_dropped"], n_added=v["n_added"])
    print(f"[d613] {S} {k}: {res['rates'][k]}", flush=True)
json.dump(res, open(O / f"t2_d613_diag_{S}.json", "w"), indent=1)
print(f"[d613] {S} done ({time.time() - T0:.0f} s)", flush=True)
