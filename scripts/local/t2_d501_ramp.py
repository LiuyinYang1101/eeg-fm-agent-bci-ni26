#!/usr/bin/env python
"""T2-D501 (registered 2026-11-15 before any number; zero GPU): the stranger cost of higher copy doses, and three evidence-accumulating
returning-user gate statistics, on T2-D498's local CNN files (lc9 folds 0-9: nine users' session 1 trained, the tenth held out = STRANGER;
P04 / P10 sessions 2-3 in folds that trained them = RETURNING). Per fold the shipped router is refit exactly as t2p_fpcalib blend does
(fit_space on the training rows' fingerprints, the running-mean nearest centroid, every window).
(a) CNN mix (M7's CNN weights normalised), each family decoded by the causal Sinkhorn (10 / 3 / 0.95, uniform start) then pooled; doses
    shipped (FBCNet r 1/3, ShallowNet / IFNet 1/2), m2 (2/3, 1, 1), copy (1, 1, 1); per user, stranger cost vs shipped; returning gain beside.
(b) on each stream's prefix t in {10, 20, 40, 80, 120, end}: g_dist = (d_routed - mean_j d_j) / std_j d_j of the running-mean fingerprint
    (cohort z-norm; small = returning), g_margin = d_best / d_second (small = returning), g_agree = running agreement of the decoded routed-
    copy mix with the decoded pooled mix (large = returning). Threshold at 95 % true accept on returning instances -> strangers' false accept.
KILL: every statistic at every t has false accept > 0.2 -> the ramp is closed.
    mem-guard python scripts/local/t2_d501_ramp.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FC  # noqa: E402
from t2_cluster_ci import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cpu", 0.3143), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cpu", 0.2357),
       "ifn": ("ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cpu", 0.10)}
WS = sum(v for _, v in FAM.values())
DOSE = {"shipped": {"fbc": 1 / 3, "sh": 0.5, "ifn": 0.5}, "m2": {"fbc": 2 / 3, "sh": 1.0, "ifn": 1.0}, "copy": {"fbc": 1.0, "sh": 1.0, "ifn": 1.0}}
TS = (10, 20, 40, 80, 120, 10 ** 6)
SEEDS = (750, 751, 752)
_, idx, meta = B.load_export(str(EXP))
Fp = np.load(O / "t2_d124b/fp_proteus.npy")
subj = idx["subject"].to_numpy()
short = np.array([str(s).split("/")[-1] for s in subj])
Y, CELL, STREAM, ORD, SES = idx.y.to_numpy(), idx.cell.to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.session.astype(str).to_numpy()


def ba(pred, rows):
    out = []
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        out.append(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in np.unique(Y[rows][m])]))
    return float(np.mean(out))


recs, stats = [], []
for f in range(10):
    (_, trm, _, tem), = B.folds(idx, f"lc9_{f}")
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    proj, mu, users = FC.fit_space(Fp[tr], subj[tr])
    held = [u for u in set(short[te]) if (short[te] == u).sum() == (short == u).sum()][0]
    pos = {int(r): j for j, r in enumerate(te)}
    PU = {}
    for s in SEEDS:
        for fam, (stem, _) in FAM.items():
            z = np.load(O / f"sofia_s1/ni26-t2p-d498-{fam}-lc9-5090/runs/lc9_{f}__{stem}__s{s}.npz")
            assert (z["test_idx"] == te).all() and list(z["users"]) == users, (f, fam, s)
            PU[(s, fam)] = (z["prob_last"].astype(np.float64), z["prob_users_test"].astype(np.float64))
    for st_ in np.unique(STREAM[te]):
        rows = te[STREAM[te] == st_]
        rows = rows[np.argsort(ORD[rows])]
        u = short[rows][0]
        kind = "stranger" if u == held else ("returning" if SES[rows][0] in ("2", "3") else None)
        if kind is None:
            continue
        Z = proj(Fp[rows])
        k = FC.stream_route_every(Z, mu, 1)
        cs = np.cumsum(Z, 0) / np.arange(1, len(Z) + 1)[:, None]
        D = np.linalg.norm(cs[:, None, :] - mu[None], axis=2)                       # (T, n_users)
        Ds = np.sort(D, 1)
        g_dist = (D[np.arange(len(D)), k] - D.mean(1)) / D.std(1).clip(1e-9)
        g_margin = Ds[:, 0] / Ds[:, 1].clip(1e-9)
        jj = np.array([pos[int(r)] for r in rows])
        key = (STREAM[rows], ORD[rows])
        for s in SEEDS:
            dec = {}
            for dn, dose in DOSE.items():
                acc = 0.0
                for fam, (_, w) in FAM.items():
                    P0, Pu = PU[(s, fam)]
                    p = FC.blend_route(P0[jj], Pu[:, jj], k, dose[fam])
                    acc = acc + (w / WS) * causal_sinkhorn(p / p.sum(1, keepdims=True), key, 10, 3, 0.95)
                dec[dn] = acc
                recs.append(dict(fold=f, seed=s, stream=st_, user=u, kind=kind, dose=dn, ba=ba(acc.argmax(1), rows)))
            pool = sum((w / WS) * causal_sinkhorn(PU[(s, fam)][0][jj] / PU[(s, fam)][0][jj].sum(1, keepdims=True), key, 10, 3, 0.95)
                       for fam, (_, w) in FAM.items())
            agree = np.cumsum(dec["copy"].argmax(1) == pool.argmax(1)) / np.arange(1, len(rows) + 1)
            for t in TS:
                i = min(t, len(rows)) - 1
                stats.append(dict(fold=f, seed=s, stream=st_, user=u, kind=kind, t=t, g_dist=float(g_dist[i]), g_margin=float(g_margin[i]),
                                  g_agree=float(agree[i])))
    print(f"[d501] fold {f} (held {held}) done", flush=True)
import pandas as pd  # noqa: E402
R, G = pd.DataFrame(recs), pd.DataFrame(stats)
out = O / "t2_d501"
out.mkdir(exist_ok=True)
R.to_csv(out / "doses.csv", index=False)
G.to_csv(out / "gate.csv", index=False)
pu = R.groupby(["kind", "user", "dose"]).ba.mean().unstack()
res = {"a": {}, "b": {}}
for kind in ("stranger", "returning"):
    T = pu.loc[kind]
    d2, dc = T["m2"] - T["shipped"], T["copy"] - T["shipped"]
    typ = (lambda s_: s_.drop("P04", errors="ignore").mean()) if kind == "stranger" else (lambda s_: s_.mean())
    res["a"][kind] = dict(m2_minus_shipped={x: round(float(v), 4) for x, v in d2.items()}, copy_minus_shipped={x: round(float(v), 4) for x, v in dc.items()},
                          typical_m2=round(float(typ(d2)), 4), typical_copy=round(float(typ(dc)), 4), shipped_level=round(float(typ(T["shipped"])), 4))
    print(f"D501 (a) {kind}: shipped {typ(T['shipped']):.4f}; m2 - shipped {typ(d2):+.4f} [" + " ".join(f"{x} {v:+.4f}" for x, v in d2.items())
          + f"]; copy - shipped {typ(dc):+.4f} [" + " ".join(f"{x} {v:+.4f}" for x, v in dc.items()) + "]", flush=True)
best = None
for stat, small in (("g_dist", True), ("g_margin", True), ("g_agree", False)):
    line = []
    for t in TS:
        Gt = G[G.t == t]
        if stat != "g_agree":                                                     # seed-independent: one value per stream instance
            Gt = Gt[Gt.seed == SEEDS[0]]
        ret, stv = Gt[Gt.kind == "returning"][stat].to_numpy(), Gt[Gt.kind == "stranger"][stat].to_numpy()
        th = np.quantile(ret, 0.95) if small else np.quantile(ret, 0.05)
        fa = float(np.mean(stv <= th) if small else np.mean(stv >= th))
        res["b"][f"{stat}@{t}"] = dict(fa=round(fa, 3), threshold=float(th), n_ret=len(ret), n_str=len(stv))
        line.append(f"t{'end' if t > 1000 else t} FA {fa:.2f}")
        if best is None or fa < best[0]:
            best = (fa, stat, t)
    print(f"D501 (b) {stat} (TA 0.95 on returning): " + " | ".join(line), flush=True)
kill = best[0] > 0.2
res["verdict"] = dict(best=dict(fa=best[0], stat=best[1], t=best[2]), killed=bool(kill))
print(f"D501 KILL TEST: best {best[1]} @ t {best[2] if best[2] < 1000 else 'end'} FA {best[0]:.2f} (needs <= 0.2; multiplicity 3 x 6) -> "
      + ("KILLED: no gate protects strangers at 95 % returning detection" if kill else "SURVIVES: cost the ramp"), flush=True)
json.dump(res, open(out / "read.json", "w"), indent=1)
