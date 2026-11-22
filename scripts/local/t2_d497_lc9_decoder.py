#!/usr/bin/env python
"""T2-D497 (seat G14 item B, decoder part; registered 2026-11-14 before any number; zero GPU, diagnostic): do the decoder levers that ship
in W13 / W14 (per-family causal decode + per-family start vectors; T2-D459 / D476b) hold when ALL TEN users serve as strangers?
Every decoder row was read on kit + seen1, i.e. four strangers (P04 P05 P08 P09). Files: T2-D485's K = 9 panel (lc9_<f>, f = 0..9: training =
nine users' session 1, held out = the tenth user's every session + the nine users' sessions 2-3), pooled FBCNet (r1_pulxx_vp2_e180_vr_rc75 /
none) and ShallowNet (r1_pulx_vp2_e180_vr / ea), seeds 400-404 (no copies, no views, no FM: the CNN mix at M7's ratio 0.5714 / 0.4286).
Arms, each decoded per stream (10 it, n0 3, lambda 0.95):
  J      the families pooled, ONE decode, uniform start            (M7's decoder)
  Js     the same with a start vector for the pooled mix         (B3's form)
  F      each family decoded on its own, uniform start, then pooled
  Fs     each family decoded with its own start vector, then pooled (W13's form)
Start vectors cross-fitted by T2-D459's rule over the OTHER nine folds' strangers (mean of per-user mean raw mix rows), per seed.
Scores: strangers (the held-out user's sessions), per user = mean over its cells and the five seeds; typical = non-P04; returning
(the nine users' sessions 2-3) beside. Descriptive: the shuffle stress (D476b's rng 476 order) on Fs vs J.
FLAG (fixed now, informs W14's reading bands; nothing ships from this row): Fs - J on typical strangers <= -0.003, or any stranger below -0.01.
    mem-guard python scripts/local/t2_d497_lc9_decoder.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
RD = O / "sofia_s1/ni26-t2p-d485-k9-5090/runs"
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", 0.3143 / (0.3143 + 0.2357)),
       "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vr", 0.2357 / (0.3143 + 0.2357))}
SEEDS = range(400, 405)
idx = pd.read_csv(EXP / "index.csv")
U, SES = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy()
STREAM, ORD, Y, CELL = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
SHUF = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)


def load(f, s):
    out = {}
    for fam, (stem, _) in FAM.items():
        z = np.load(RD / f"lc9_{f}__{stem}__s{s}.npz")
        out[fam] = (z["test_idx"], z["prob_last"].astype(np.float64))
    assert (out["fbc"][0] == out["sh"][0]).all()
    return out


D = {(f, s): load(f, s) for f in range(10) for s in SEEDS}
TRU = {}
for f in range(10):
    ix = D[(f, 400)]["fbc"][0]
    us = set(U[ix])
    held = [u for u in us if (U[ix] == u).sum() == (U == u).sum()]          # the user whose EVERY window is held out
    assert len(held) == 1, (f, held)
    TRU[f] = held[0]
assert sorted(TRU.values()) == sorted(set(U)), TRU


def mix(f, s, fam=None):
    ix = D[(f, s)]["fbc"][0]
    if fam:
        p = D[(f, s)][fam][1]
        return ix, p / p.sum(1, keepdims=True)
    p = sum(w * D[(f, s)][k][1] for k, (_, w) in FAM.items())
    return ix, p / p.sum(1, keepdims=True)


def vec(f_out, s, fam=None):
    """T2-D459's rule over the OTHER folds' strangers: mean of per-user mean raw mix rows."""
    per = []
    for f in range(10):
        if f == f_out:
            continue
        ix, p = mix(f, s, fam)
        m = U[ix] == TRU[f]
        per.append(p[m].mean(0))
    v = np.mean(per, 0)
    return v / v.sum()


rows = []
for s in SEEDS:
    for f in range(10):
        ix, pj = mix(f, s)
        fam_p = {k: mix(f, s, k)[1] for k in FAM}
        vj = vec(f, s)
        vf = {k: vec(f, s, k) for k in FAM}
        for shuf in (False, True):
            key = (STREAM[ix], (SHUF if shuf else ORD)[ix])
            P = {"J": causal_sinkhorn(pj, key, 10, 3, 0.95), "Js": causal_sinkhorn(pj, key, 10, 3, 0.95, start=vj),
                 "F": sum(w * causal_sinkhorn(fam_p[k], key, 10, 3, 0.95) for k, (_, w) in FAM.items()),
                 "Fs": sum(w * causal_sinkhorn(fam_p[k], key, 10, 3, 0.95, start=vf[k]) for k, (_, w) in FAM.items())}
            for a, Q in P.items():
                pr = Q.argmax(1)
                for c in np.unique(CELL[ix]):
                    mk = CELL[ix] == c
                    u = U[ix][mk][0]
                    rows.append(dict(seed=s, fold=f, shuf=shuf, arm=a, cell=c, user=u, kind="stranger" if u == TRU[f] else "returning",
                                     ba=float(np.mean([np.mean(pr[mk][Y[ix][mk] == q] == q) for q in np.unique(Y[ix][mk])]))))
    print(f"[d497] seed {s} done", flush=True)
R = pd.DataFrame(rows)
out = O / "t2_d497"
out.mkdir(exist_ok=True)
R.to_csv(out / "cells.csv", index=False)
res = {}
for kind in ("stranger", "returning"):
    for shuf in (False, True):
        T = R[(R.kind == kind) & (R.shuf == shuf)]
        pu = T.groupby(["user", "arm"]).ba.mean().unstack()
        typ = pu.drop("P04", errors="ignore").mean()
        tag = f"{kind}{' SHUFFLED' if shuf else ''}"
        res[tag] = dict(typical={a: round(float(typ[a]), 4) for a in pu.columns}, per_user={u: {a: round(float(pu.loc[u, a]), 4) for a in pu.columns} for u in pu.index})
        print(f"D497 {tag}: typical J {typ['J']:.4f} | Js {typ['Js'] - typ['J']:+.4f} F {typ['F'] - typ['J']:+.4f} Fs {typ['Fs'] - typ['J']:+.4f} (vs J)", flush=True)
        if not shuf:
            print("   per user Fs - J: " + " ".join(f"{u} {pu.loc[u, 'Fs'] - pu.loc[u, 'J']:+.4f}" for u in pu.index)
                  + " | Js - J: " + " ".join(f"{u} {pu.loc[u, 'Js'] - pu.loc[u, 'J']:+.4f}" for u in pu.index), flush=True)
st = res["stranger"]
d = st["typical"]["Fs"] - st["typical"]["J"]
worst = min(v["Fs"] - v["J"] for u, v in st["per_user"].items())
print(f"D497 FLAG (typical strangers Fs - J {d:+.4f}, worst stranger {worst:+.4f}): "
      + ("RAISED: W13's decoder levers do not generalise to the ten-stranger panel" if (d <= -0.003 or worst < -0.01) else "not raised"), flush=True)
json.dump(res, open(out / "read.json", "w"), indent=1)
