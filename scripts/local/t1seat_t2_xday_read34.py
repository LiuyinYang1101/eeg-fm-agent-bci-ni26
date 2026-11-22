#!/usr/bin/env python
"""Track1's T2 experiment XDAY, reads 3 and 4 (rules fixed in lead/panel_2026-11-14_outliers/SEAT_T1_on_T2.md before any of these numbers).
Selection split xs (first 80 day-1 windows train, day-1 windows 80+ scored) -> per-user quantities, applied to the xd models on day 2.
  read 3: per-user decoder start b_u (the user's mean blended output on the xs windows, per family) in the causal SK's pseudo-rows, n0 3 / 20,
          at the shipped dose and copy-only; own copy.
  read 4: per-user configuration {per-family | joint decode} x {m 0, 1, 2, copy-only} x {FBCNet share 0.3, 0.5714, 0.8} picked on xs
          (seed-mean BA, ties -> global), scored on day 2 vs the global setting (per-family, m 1, 0.5714) and the day-2 oracle.
    mem-guard python scripts/local/t1seat_t2_xday_read34.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
from t2_cluster_ci import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge/t1seat_t2"
idx = pd.read_csv(O / "export_combo_v1/index.csv", dtype={"session": str})
U = idx.subject.str.split("/").str[-1].to_numpy()
STREAM, ORD, Y = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy()
FAM = {"fbc": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", "sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr"}
SHIP = {"fbc": 1 / 3, "sh": 1 / 2}
DOSE = {"m0": 0.0, "m1": 1.0, "m2": 2.0, "copy": 99.0}
SHARE = (0.3, 0.5714, 0.8)
GLOBAL = ("fam", "m1", 0.5714)
SEEDS = [s for s in range(780, 786) if all((O / f"runs/{p}__{st}tv8cpu__s{s}.npz").exists() for st in FAM.values() for p in ("xd", "xs"))]
rng = np.random.default_rng(0)


def norm(p):
    p = p.astype(np.float64)
    return p / p.sum(-1, keepdims=True)


def blended(proto, s):
    """{dose: {family: (N, 3)}} with the user's own copy, and the rows."""
    Z = {f: np.load(O / f"runs/{proto}__{st}tv8cpu__s{s}.npz") for f, st in FAM.items()}
    ix = Z["fbc"]["test_idx"]
    users = list(Z["fbc"]["users"])
    own, n = np.array([users.index(u) for u in U[ix]]), np.arange(len(ix))
    out = {}
    for d, m in DOSE.items():
        out[d] = {}
        for f in FAM:
            assert (Z[f]["test_idx"] == ix).all() and list(Z[f]["users"]) == users
            r = min(1.0, m * SHIP[f])
            out[d][f] = (1 - r) * norm(Z[f]["prob_last"]) + r * norm(Z[f]["prob_users_test"])[own, n]
    return out, ix


def ba(pred, ix):
    y, u = Y[ix], U[ix]
    return {v: float(np.mean([np.mean(pred[(u == v) & (y == q)] == q) for q in range(3)])) for v in np.unique(u)}


def sk(p, ix, n0=3, start=None):
    if start is None:
        return causal_sinkhorn(p, (STREAM[ix], ORD[ix]), 10, n0, 0.95)
    out = np.empty_like(p)
    for u, b in start.items():                       # one stream per user in both splits
        m = U[ix] == u
        out[m] = causal_sinkhorn(p[m], (STREAM[ix][m], ORD[ix][m]), 10, n0, 0.95, start=b)
    return out


def boot(d):
    d = np.asarray(d)
    b = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
    return float(np.percentile(b, 5)), float(np.percentile(b, 95))


r3, r4 = [], []
for s in SEEDS:
    Ps, ixs = blended("xs", s)
    Pd, ixd = blended("xd", s)
    for d in ("m1", "copy"):                          # read 3
        st = {f: {u: Ps[d][f][U[ixs] == u].mean(0) for u in np.unique(U[ixs])} for f in FAM}
        for arm, n0, use in (("uniform", 3, False), ("start_n3", 3, True), ("start_n20", 20, True), ("uniform_n20", 20, False)):
            mix = sum(w * sk(Pd[d][f], ixd, n0, st[f] if use else None) for f, w in (("fbc", 0.5714), ("sh", 0.4286)))
            r3 += [dict(seed=s, dose=d, arm=arm, user=u, ba=v) for u, v in ba(mix.argmax(1), ixd).items()]
    for split, P, ix in (("xs", Ps, ixs), ("xd", Pd, ixd)):   # read 4
        for d in DOSE:
            dec = {f: sk(P[d][f], ix) for f in FAM}
            for w in SHARE:
                pf = (w * dec["fbc"] + (1 - w) * dec["sh"]).argmax(1)
                pj = sk(w * P[d]["fbc"] + (1 - w) * P[d]["sh"], ix).argmax(1)
                for mode, pr in (("fam", pf), ("joint", pj)):
                    r4 += [dict(seed=s, split=split, mode=mode, dose=d, share=w, user=u, ba=v) for u, v in ba(pr, ix).items()]
R3 = pd.DataFrame(r3)
R3.to_csv(O / "xday_read3_cells.csv", index=False)
res = dict(seeds=SEEDS)
T3 = R3.groupby(["dose", "arm", "user"]).ba.mean().unstack("user")
print(f"XDAY read 3 (per-user decoder start; seeds {SEEDS}):", flush=True)
lead3 = {}
for arm in ("start_n3", "start_n20", "uniform_n20"):
    ok = True
    for d in ("m1", "copy"):
        dd = T3.loc[(d, arm)] - T3.loc[(d, "uniform")]
        lo, hi = boot(dd.to_numpy())
        print(f"   {arm} at {d}: {T3.loc[(d, arm)].mean():.4f} vs uniform {T3.loc[(d, 'uniform')].mean():.4f} = {dd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], users up {int((dd > 0).sum())} / 9", flush=True)
        ok &= bool(dd.mean() >= 0.005 and (dd > 0).sum() >= 6)
        res[f"r3_{arm}_{d}"] = dict(d=float(dd.mean()), ci=[lo, hi], up=int((dd > 0).sum()))
    lead3[arm] = ok
print(f"XDAY read 3 verdict: {'LEAD: ' + ', '.join(a for a, v in lead3.items() if v and a != 'uniform_n20') if any(v for a, v in lead3.items() if a != 'uniform_n20') else 'no lead'}", flush=True)

R4 = pd.DataFrame(r4)
R4.to_csv(O / "xday_read4_cells.csv", index=False)
T4 = R4.groupby(["split", "user", "mode", "dose", "share"]).ba.mean()
users = sorted(R4.user.unique())
cfgs = list(itertools.product(("fam", "joint"), DOSE, SHARE))
sel, orc, glb, picks = {}, {}, {}, {}
for u in users:
    xs_ = {c: T4.loc[("xs", u, *c)] for c in cfgs}
    xd_ = {c: T4.loc[("xd", u, *c)] for c in cfgs}
    best = max(xs_.values())
    pick = GLOBAL if xs_[GLOBAL] >= best - 1e-12 else max(cfgs, key=lambda c: xs_[c])
    picks[u], sel[u], orc[u], glb[u] = pick, xd_[pick], max(xd_.values()), xd_[GLOBAL]
ds = np.array([sel[u] - glb[u] for u in users])
do = np.array([orc[u] - glb[u] for u in users])
lo, hi = boot(ds)
lead4 = bool(ds.mean() >= 0.010 and (ds >= 0).sum() >= 6)
print(f"XDAY read 4 (per-user configuration picked on the day-1 split, scored on day 2): global {np.mean(list(glb.values())):.4f}, selected {np.mean(list(sel.values())):.4f} "
      f"= {ds.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], users not below global {int((ds >= 0).sum())} / 9; day-2 oracle - global {do.mean():+.4f} -> {'LEAD' if lead4 else 'no lead'}", flush=True)
for u in users:
    print(f"   {u}: pick {picks[u]} -> day 2 {sel[u]:.3f} (global {glb[u]:.3f}, oracle {orc[u]:.3f})", flush=True)
res.update(r4=dict(d=float(ds.mean()), ci=[lo, hi], not_below=int((ds >= 0).sum()), oracle=float(do.mean()), lead=lead4, picks={u: list(picks[u]) for u in users}))
# is 'who gains from a factor' stable from the day-1 split to day 2? one-factor changes from the global setting
for name, alt in (("joint decode", ("joint", "m1", 0.5714)), ("dose m2", ("fam", "m2", 0.5714)), ("copy-only", ("fam", "copy", 0.5714)),
                  ("no copy", ("fam", "m0", 0.5714)), ("FBCNet share 0.8", ("fam", "m1", 0.8)), ("FBCNet share 0.3", ("fam", "m1", 0.3))):
    a = np.array([T4.loc[("xs", u, *alt)] - T4.loc[("xs", u, *GLOBAL)] for u in users])
    b = np.array([T4.loc[("xd", u, *alt)] - T4.loc[("xd", u, *GLOBAL)] for u in users])
    rho = float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan")
    print(f"   factor {name}: day-1 split effect {a.mean():+.4f}, day-2 effect {b.mean():+.4f} ({int((b > 0).sum())} / 9 up), rank correlation over users {rho:+.2f}", flush=True)
    res[f"r4_factor_{name}"] = dict(xs=float(a.mean()), xd=float(b.mean()), up=int((b > 0).sum()), rho=rho)
json.dump(res, open(O / "xday_read34.json", "w"), indent=1)
