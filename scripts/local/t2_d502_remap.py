#!/usr/bin/env python
"""T2-D502 (registered 2026-11-15 before any number; zero GPU): the TRUE-label ceiling of a per-stream output remap on T2-D498's ten
strangers. The stack = T2-D498's shipped shape (M7's shares; FM = 3 plain + 3 CAR ST-L; each family decoded by the causal Sinkhorn
10 / 3 / 0.95 from its cross-fitted start vector, then pooled), 3 draws (CNN seeds 750-752). Per stranger STREAM (each session alone):
  (a) PERM: the best of the 6 class permutations of the decoded outputs, chosen by cell BA on the fit part;
  (b) REMAP: a 3 x 3 log-linear map (multinomial logistic regression on the decoded log-probabilities, C 1) fitted on the fit part;
  fit / score parts: XFIT = interleaved halves (odd / even windows by order, both directions, averaged); C30 / C60 = fit on the first
  30 / 60 windows, score the rest. Gain = remapped BA - the unremapped stack's BA on the SAME scored windows.
Per user = mean over its streams and draws; the four hard strangers (P01 P03 P04 P10) separately.
RULE: the four's mean gain under the best true-label form < +0.05 -> the stranger side closes; else form (2) gets its own row.
    mem-guard python scripts/local/t2_d502_remap.py
"""
import glob
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

warnings.filterwarnings("ignore")
O = Path.home() / "workspace/outputs/neural_interface_challenge"
S, R = O / "sofia_s1", O / "t2_d498"
idx = pd.read_csv(EXP / "index.csv")
U, STREAM, ORD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
CNNF = {"fbc": "route_fbc_lc9/lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz",
        "sh": "route_sh_lc9/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
        "ifn": "route_ifn_lc9/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz"}
DRAWS, HARD = (750, 751, 752), ("P01", "P03", "P04", "P10")


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


def vec(f_out, d, fam):
    per = [MIX[(f, d)][1][fam][U[MIX[(f, d)][0]] == TRU[f]].mean(0) for f in range(10) if f != f_out]
    v = np.mean(per, 0)
    return v / v.sum()


def cell_ba(pred, y, cells):
    out = []
    for c in np.unique(cells):
        m = cells == c
        ks = np.unique(y[m])
        if len(ks) < 2:
            continue
        out.append(np.mean([np.mean(pred[m][y[m] == q] == q) for q in ks]))
    return float(np.mean(out)) if out else np.nan


PERMS = list(itertools.permutations(range(3)))


def fit_score(Q, y, cells, fit, sc):
    """-> (base, perm, remap) BA on the scored windows `sc`, maps fitted on `fit` (true labels)."""
    base = cell_ba(Q[sc].argmax(1), y[sc], cells[sc])
    best = max(PERMS, key=lambda pm: cell_ba(np.array(pm)[Q[fit].argmax(1)], y[fit], cells[fit]))
    perm = cell_ba(np.array(best)[Q[sc].argmax(1)], y[sc], cells[sc])
    if len(np.unique(y[fit])) == 3:
        lr = LogisticRegression(C=1.0, max_iter=500).fit(np.log(Q[fit].clip(1e-9)), y[fit])
        remap = cell_ba(lr.predict(np.log(Q[sc].clip(1e-9))), y[sc], cells[sc])
    else:
        remap = base
    return base, perm, remap


rows = []
for (f, d), (ix, mx) in MIX.items():
    key = (STREAM[ix], ORD[ix])
    Q = sum(W[k] * causal_sinkhorn(mx[k], key, 10, 3, 0.95, start=vec(f, d, k)) for k in W)
    for st_ in np.unique(STREAM[ix][U[ix] == TRU[f]]):
        m = np.flatnonzero(STREAM[ix] == st_)
        m = m[np.argsort(ORD[ix][m])]
        q, y, cells = Q[m], Y[ix][m], CELL[ix][m]
        n = len(m)
        odd, even = np.arange(n) % 2 == 1, np.arange(n) % 2 == 0
        forms = {"XFIT": [(odd, even), (even, odd)]}
        for N in (30, 60):
            if n > N + 30:
                forms[f"C{N}"] = [(np.arange(n) < N, np.arange(n) >= N)]
        for form, splits in forms.items():
            r = np.mean([fit_score(q, y, cells, a, b) for a, b in splits], 0)
            rows.append(dict(user=TRU[f], stream=st_, draw=d, form=form, base=r[0], perm=r[1], remap=r[2]))
T = pd.DataFrame(rows)
out = O / "t2_d502"
out.mkdir(exist_ok=True)
T.to_csv(out / "streams.csv", index=False)
res = {}
for form in ("XFIT", "C30", "C60"):
    t = T[T.form == form]
    pu = t.groupby("user")[["base", "perm", "remap"]].mean()
    g = pu.assign(perm_gain=pu.perm - pu.base, remap_gain=pu.remap - pu.base)
    hard, rest = g.loc[[u for u in HARD if u in g.index]], g.drop([u for u in HARD if u in g.index])
    res[form] = dict(per_user={u: {k: round(float(v), 4) for k, v in g.loc[u].items()} for u in g.index},
                     hard_perm=round(float(hard.perm_gain.mean()), 4), hard_remap=round(float(hard.remap_gain.mean()), 4),
                     rest_perm=round(float(rest.perm_gain.mean()), 4), rest_remap=round(float(rest.remap_gain.mean()), 4))
    print(f"D502 {form}: hard four perm {res[form]['hard_perm']:+.4f} remap {res[form]['hard_remap']:+.4f} | other six perm {res[form]['rest_perm']:+.4f} remap {res[form]['rest_remap']:+.4f}", flush=True)
    print("   per user (base | perm | remap): " + " ".join(f"{u} {r.base:.3f}|{r.perm_gain:+.3f}|{r.remap_gain:+.3f}" for u, r in g.iterrows()), flush=True)
best = max(res[fm_][k] for fm_ in res for k in ("hard_perm", "hard_remap"))
print(f"D502 RULE: the four hard strangers' best true-label gain {best:+.4f} -> "
      + ("< +0.05: the stranger side CLOSES (no label-free form is built)" if best < 0.05 else ">= +0.05: form (2) gets its own row"), flush=True)
res["best_hard_gain"] = best
json.dump(res, open(out / "read.json", "w"), indent=1)
