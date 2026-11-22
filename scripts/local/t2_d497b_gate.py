#!/usr/bin/env python
"""T2-D497b (b) (registered 2026-11-14 before any gate number; part (a) found the per-stranger sign STABLE: 5 of 5 large movers keep it in
>= 4 / 5 seeds): a causal per-stream gate between the joint decode J and W13's per-family decode Fs, on D497's files (T2-D485's K = 9 panel,
pooled FBCNet + ShallowNet, seeds 400-404). Statistics on the FIRST N = 20 windows of each stream (both causal, label-free), exactly:
  g1 = mean over t <= N of the total-variation distance between the two families' rows after each family's own causal decode (uniform start)
  g2 = the L1 distance between the two families' mean RAW rows over t <= N (their class marginals disagree)
Gated stream: J's decode for t <= N, then Fs if (dir * g > dir * theta) else J for t > N. theta and dir in {+1, -1} fitted LEAVE-ONE-USER-OUT:
for each stranger u, on the other nine strangers' streams (all 5 seeds), the (theta, dir) that maximises their mean cell BA (theta over the
deciles of their g), then applied to u's streams; returning users get the theta / dir fitted on all ten strangers.
KILL (the manager's): gated typical strangers < +0.003 over the better fixed choice (J or Fs, both whole-stream), or any user < -0.003 vs it.
Both statistics are read; the better is reported with the two-statistic multiplicity named.
    mem-guard python scripts/local/t2_d497b_gate.py
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
SEEDS, NW = range(400, 405), 20
idx = pd.read_csv(EXP / "index.csv")
U, STREAM, ORD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()


def load(f, s):
    out = {}
    for fam, (stem, _) in FAM.items():
        z = np.load(RD / f"lc9_{f}__{stem}__s{s}.npz")
        p = z["prob_last"].astype(np.float64)
        out[fam] = (z["test_idx"], p / p.sum(1, keepdims=True))
    return out


D = {(f, s): load(f, s) for f in range(10) for s in SEEDS}
TRU = {}
for f in range(10):
    ix = D[(f, 400)]["fbc"][0]
    TRU[f] = [u for u in set(U[ix]) if (U[ix] == u).sum() == (U == u).sum()][0]


def vec(f_out, s, fam=None):
    per = []
    for f in range(10):
        if f == f_out:
            continue
        ix = D[(f, s)]["fbc"][0]
        p = D[(f, s)][fam][1] if fam else sum(w * D[(f, s)][k][1] for k, (_, w) in FAM.items())
        per.append(p[U[ix] == TRU[f]].mean(0))
    v = np.mean(per, 0)
    return v / v.sum()


def cell_ba(pr, rows):
    out = []
    for c in np.unique(CELL[rows]):
        mk = CELL[rows] == c
        out.append(np.mean([np.mean(pr[mk][Y[rows][mk] == q] == q) for q in np.unique(Y[rows][mk])]))
    return float(np.mean(out))


S = []      # one record per (seed, fold, stream): g1, g2, BA under J / Fs / the gated switch (J to t <= N, then Fs)
for s in SEEDS:
    for f in range(10):
        ix = D[(f, s)]["fbc"][0]
        vf = {k: vec(f, s, k) for k in FAM}
        for st_ in np.unique(STREAM[ix]):
            m = np.flatnonzero(STREAM[ix] == st_)
            m = m[np.argsort(ORD[ix][m])]
            rows = ix[m]
            key = (STREAM[rows], ORD[rows])
            pf = {k: D[(f, s)][k][1][m] for k in FAM}
            pj = sum(w * pf[k] for k, (_, w) in FAM.items())
            J = causal_sinkhorn(pj, key, 10, 3, 0.95)
            Fs = sum(w * causal_sinkhorn(pf[k], key, 10, 3, 0.95, start=vf[k]) for k, (_, w) in FAM.items())
            dec = {k: causal_sinkhorn(pf[k], key, 10, 3, 0.95) for k in FAM}
            n = min(NW, len(rows))
            g1 = float(0.5 * np.abs(dec["fbc"][:n] - dec["sh"][:n]).sum(1).mean())
            g2 = float(np.abs(pf["fbc"][:n].mean(0) - pf["sh"][:n].mean(0)).sum())
            sw = J.copy()
            sw[n:] = Fs[n:]
            u = U[rows][0]
            S.append(dict(seed=s, fold=f, stream=st_, user=u, kind="stranger" if u == TRU[f] else "returning", g1=g1, g2=g2,
                          J=cell_ba(J.argmax(1), rows), Fs=cell_ba(Fs.argmax(1), rows), SW=cell_ba(sw.argmax(1), rows)))
    print(f"[d497b] seed {s} done", flush=True)
S = pd.DataFrame(S)
out = O / "t2_d497"
S.to_csv(out / "gate_streams.csv", index=False)


def fit(T, g):
    best = (-1.0, None, None)
    for th in np.quantile(T[g], np.linspace(0.1, 0.9, 9)):
        for dr in (1, -1):
            v = np.where(dr * T[g] > dr * th, T.SW, T.J).mean()
            if v > best[0]:
                best = (v, th, dr)
    return best[1], best[2]


Tst, Tre = S[S.kind == "stranger"], S[S.kind == "returning"]
res = {}
for g in ("g1", "g2"):
    gated = []
    for u in sorted(Tst.user.unique()):
        th, dr = fit(Tst[Tst.user != u], g)
        Tu = Tst[Tst.user == u]
        gated.append(Tu.assign(G=np.where(dr * Tu[g] > dr * th, Tu.SW, Tu.J)))
    G = pd.concat(gated)
    pu = G.groupby("user")[["J", "Fs", "G"]].mean()
    typ = pu.drop("P04", errors="ignore").mean()
    best_fixed = "Fs" if typ["Fs"] >= typ["J"] else "J"
    dlt = pu["G"] - pu[best_fixed]
    th_all, dr_all = fit(Tst, g)
    Rr = Tre.assign(G=np.where(dr_all * Tre[g] > dr_all * th_all, Tre.SW, Tre.J)).groupby("user")[["J", "Fs", "G"]].mean().mean()
    kill = (typ["G"] - typ[best_fixed] < 0.003) or (dlt.drop("P04", errors="ignore").min() < -0.003)
    res[g] = dict(typical={k: round(float(v), 4) for k, v in typ.items()}, best_fixed=best_fixed,
                  per_user_vs_best={u: round(float(v), 4) for u, v in dlt.items()}, returning={k: round(float(v), 4) for k, v in Rr.items()},
                  theta_all=float(th_all), dir_all=int(dr_all), killed=bool(kill))
    print(f"D497b {g}: typical strangers J {typ['J']:.4f} Fs {typ['Fs']:.4f} gated(LOUO) {typ['G']:.4f} -> vs best fixed ({best_fixed}) "
          f"{typ['G'] - typ[best_fixed]:+.4f}; per user " + " ".join(f"{u} {v:+.4f}" for u, v in dlt.items())
          + f" | returning J {Rr['J']:.4f} Fs {Rr['Fs']:.4f} gated {Rr['G']:.4f} -> {'KILLED' if kill else 'PASSES'}", flush=True)
print("D497b: two statistics read (multiplicity 2); a pass needs one of them to pass the kill rule", flush=True)
json.dump(res, open(out / "gate_read.json", "w"), indent=1)
