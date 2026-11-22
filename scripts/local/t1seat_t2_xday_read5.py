#!/usr/bin/env python
"""Track1's T2 experiment XDAY, read 5 (rules fixed in lead/panel_2026-11-14_outliers/SEAT_T1_on_T2.md before its number): the stranger side
and an open-set gate. Folds xl0-2: three Scherer users held out (strangers: their day 1 + day 2), six trained (returning: day 2).
  (a) stranger BA at m 0 / shipped / doubled / copy-only under the package's route; (b) causal open-set statistics Z / M / A: AUC and the
  false-accept rate at 0.95 detection at t = 10 / 20 / 50 / 100 / all; (c) the ramp r_t = P(returning | statistic, log t), cross-fitted
  over folds, vs the shipped dose. PASS iff returning >= +0.020 AND strangers >= -0.003; DIES if FAR@0.95 > 0.2 at 50 windows.
    mem-guard python scripts/local/t1seat_t2_xday_read5.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
import t2p_fpcalib as FC  # noqa: E402
from t2_cluster_ci import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge/t1seat_t2"
idx = pd.read_csv(O / "export_combo_v1/index.csv", dtype={"session": str})
U = idx.subject.str.split("/").str[-1].to_numpy()
SES, STREAM, ORD, Y = idx.session.to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy()
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", 0.5714), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vr", 0.4286)}
SHIP = {"fbc": 1 / 3, "sh": 1 / 2}
DOSE = {"m0": 0.0, "m1": 1.0, "m2": 2.0, "copy": 99.0}
HOLD = {"xl0": ("S1", "S4", "S7"), "xl1": ("S2", "S5", "S8"), "xl2": ("S3", "S6", "S9")}
TS = (10, 20, 50, 100, 10 ** 6)
Fp = np.load(O / "fp_combo.npy")
sch = np.char.startswith(U.astype(str), "S")
rng = np.random.default_rng(0)


def norm(p):
    p = p.astype(np.float64)
    return p / p.sum(-1, keepdims=True)


def dec(p, ix):
    return causal_sinkhorn(p, (STREAM[ix], ORD[ix]), 10, 3, 0.95)


def ba_stream(pred, ix):
    out = {}
    for s in np.unique(STREAM[ix]):
        m = STREAM[ix] == s
        out[s] = float(np.mean([np.mean(pred[m][Y[ix][m] == q] == q) for q in range(3)]))
    return out


S, B = [], []          # S: one row per (fold, stream, window) with the statistics; B: BA rows
for fold, held in HOLD.items():
    seeds = [s for s in range(780, 786) if all((O / f"runs/{fold}__{st}tv8cpu__s{s}.npz").exists() for st, _ in FAM.values())]
    assert seeds, fold
    trm = ~sch | (sch & (SES == "1") & ~np.isin(U, held))
    tr = np.flatnonzero(trm)
    proj, mu, users = FC.fit_space(Fp[tr], U[tr])
    Z0 = np.load(O / f"runs/{fold}__{FAM['fbc'][0]}tv8cpu__s{seeds[0]}.npz")
    ix = Z0["test_idx"]
    assert list(Z0["users"]) == list(users)
    n = np.arange(len(ix))
    k, t_of, zs, ms = np.zeros(len(ix), int), np.zeros(len(ix), int), np.zeros(len(ix)), np.zeros(len(ix))
    for s in np.unique(STREAM[ix]):
        j = np.flatnonzero(STREAM[ix] == s)
        j = j[np.argsort(ORD[ix][j], kind="stable")]
        Zp = proj(Fp[ix[j]])
        zb = np.cumsum(Zp, 0) / np.arange(1, len(j) + 1)[:, None]
        d = np.linalg.norm(zb[:, None] - mu[None], axis=-1)
        o = np.sort(d, 1)
        k[j], t_of[j] = d.argmin(1), np.arange(1, len(j) + 1)
        zs[j] = -(o[:, 0] - o[:, 1:].mean(1)) / (o[:, 1:].std(1) + 1e-12)       # larger = more 'returning'
        ms[j] = (o[:, 1] - o[:, 0]) / (o[:, 1] + 1e-12)
    agree = np.zeros(len(ix))
    P = {}
    for s_ in seeds:
        for f, (st, _) in FAM.items():
            z = np.load(O / f"runs/{fold}__{st}tv8cpu__s{s_}.npz")
            assert (z["test_idx"] == ix).all()
            P[(s_, f)] = (norm(z["prob_last"]), norm(z["prob_users_test"])[k, n])
            agree += (dec(P[(s_, f)][0], ix).argmax(1) == dec(P[(s_, f)][1], ix).argmax(1)) / (len(seeds) * len(FAM))
    a_run = np.zeros(len(ix))
    for s in np.unique(STREAM[ix]):
        j = np.flatnonzero(STREAM[ix] == s)
        j = j[np.argsort(ORD[ix][j], kind="stable")]
        a_run[j] = np.cumsum(agree[j]) / np.arange(1, len(j) + 1)
    ret = ~np.isin(U[ix], held)
    right = np.array(users)[k] == U[ix]
    S.append(pd.DataFrame(dict(fold=fold, row=n, stream=STREAM[ix], user=U[ix], day=SES[ix], t=t_of, ret=ret, right=right,
                               routed=np.array(users)[k], Z=zs, M=ms, A=a_run)))
    for s_ in seeds:
        for d, m in DOSE.items():
            mix = sum(w * dec((1 - min(1, m * SHIP[f])) * P[(s_, f)][0] + min(1, m * SHIP[f]) * P[(s_, f)][1], ix) for f, (_, w) in FAM.items())
            B += [dict(fold=fold, seed=s_, arm=d, stream=s, ba=v) for s, v in ba_stream(mix.argmax(1), ix).items()]
    HOLD[fold] = dict(held=held, seeds=seeds, ix=ix, P=P)
S = pd.concat(S, ignore_index=True)
S["fs"] = S.fold + "|" + S.stream
info = S.groupby("fs").agg(fold=("fold", "first"), user=("user", "first"), day=("day", "first"), ret=("ret", "first"), right=("right", "mean"),
                           routed=("routed", lambda v: v.value_counts().index[0]))
B = pd.DataFrame(B)
B["fs"] = B.fold + "|" + B.stream
BA = B.groupby(["fs", "arm"]).ba.mean().unstack()
res = {}
st_ = info.index[~info.ret]
print("XDAY read 5 (a) strangers (18 fold-streams: 9 users x day 1 / day 2), the package's route: "
      + " ".join(f"{a} {BA.loc[st_, a].mean():.4f}" for a in DOSE)
      + f"; shipped - m0 {float((BA.loc[st_, 'm1'] - BA.loc[st_, 'm0']).mean()):+.4f}, doubled - shipped {float((BA.loc[st_, 'm2'] - BA.loc[st_, 'm1']).mean()):+.4f}, "
      f"copy-only - shipped {float((BA.loc[st_, 'copy'] - BA.loc[st_, 'm1']).mean()):+.4f}", flush=True)
print("   routed to: " + " ".join(f"{info.loc[i, 'user']}d{info.loc[i, 'day']}->{info.loc[i, 'routed']}" for i in st_), flush=True)
rt_ = info.index[info.ret]
print(f"   returning fold-streams (18): own-copy route share {info.loc[rt_, 'right'].mean():.3f}; " + " ".join(f"{a} {BA.loc[rt_, a].mean():.4f}" for a in DOSE), flush=True)
res["a"] = dict(stranger={a: float(BA.loc[st_, a].mean()) for a in DOSE}, returning={a: float(BA.loc[rt_, a].mean()) for a in DOSE})


def at(stat, t):
    g = S[S.t <= t].sort_values("t").groupby("fs").last()
    return g[stat].reindex(info.index)


def auc_far(v, keep):
    y, x = info.ret[keep].to_numpy(), v[keep].to_numpy()
    thr = np.quantile(x[y], 0.05)                      # 0.95 of the returning streams at or above
    return float(roc_auc_score(y, x)), float((x[~y] >= thr).mean())


far50 = {}
for label, keep in (("all nine", np.ones(len(info), bool)), ("route right", (~info.ret | (info.right > 0.5)).to_numpy())):
    for stat in ("Z", "M", "A"):
        cells = [auc_far(at(stat, t), keep) for t in TS]
        print(f"XDAY read 5 (b) {stat} ({label}; {int((info.ret & keep).sum())} returning vs {int((~info.ret & keep).sum())} stranger streams): "
              + " ".join(f"t{'all' if t > 999 else t} AUC {a:.2f} FAR {f:.2f}" for t, (a, f) in zip(TS, cells)), flush=True)
        res[f"b_{stat}_{label}"] = dict(auc=[c[0] for c in cells], far=[c[1] for c in cells])
        if label == "all nine":
            far50[stat] = (cells[2][1], -cells[2][0])
best = min(far50, key=far50.get)
dies = far50[best][0] > 0.2
print(f"XDAY read 5: best statistic {best}: FAR at 0.95 detection, 50 windows = {far50[best][0]:.2f} -> {'DIES (> 0.2)' if dies else 'alive'}", flush=True)
# (c) the ramp, cross-fitted over folds (reported even if the statistic dies: descriptive then)
G = []
for fold, H in HOLD.items():
    trn, te = S[S.fold != fold], S[S.fold == fold].sort_values("row")
    lr = LogisticRegression(C=1.0).fit(np.c_[trn[best], np.log(trn.t)], trn.ret)
    p = lr.predict_proba(np.c_[te[best], np.log(te.t)])[:, 1]
    ix = H["ix"]
    for s_ in H["seeds"]:
        mix = sum(w * dec((1 - p)[:, None] * H["P"][(s_, f)][0] + p[:, None] * H["P"][(s_, f)][1], ix) for f, (_, w) in FAM.items())
        G += [dict(fs=fold + "|" + s, seed=s_, ba=v) for s, v in ba_stream(mix.argmax(1), ix).items()]
    S.loc[te.index, "p"] = p
ramp = pd.DataFrame(G).groupby("fs").ba.mean().reindex(info.index)
d = ramp - BA.m1


def boot(x):
    x = np.asarray(x)
    b = np.array([rng.choice(x, len(x)).mean() for _ in range(4000)])
    return float(np.percentile(b, 5)), float(np.percentile(b, 95))


for label, keep in (("all nine", np.ones(len(info), bool)), ("route right", (~info.ret | (info.right > 0.5)).to_numpy())):
    r_, s__ = d[keep & info.ret.to_numpy()], d[keep & ~info.ret.to_numpy()]
    ok = bool(r_.mean() >= 0.020 and s__.mean() >= -0.003) and not dies
    print(f"XDAY read 5 (c) ramp on {best} ({label}): returning {r_.mean():+.4f} {boot(r_)} vs shipped, strangers {s__.mean():+.4f} {boot(s__)} vs shipped; "
          f"mean P(returning): returning streams {S[S.ret].p.mean():.2f}, strangers {S[~S.ret].p.mean():.2f} -> {'PASS' if ok else 'not passed'}", flush=True)
    res[f"c_{label}"] = dict(returning=float(r_.mean()), stranger=float(s__.mean()), passed=ok)
res.update(best=best, far50={k_: v[0] for k_, v in far50.items()}, dies=bool(dies))
S.drop(columns=["row"]).to_csv(O / "xday_read5_stats.csv", index=False)
BA.assign(ramp=ramp).to_csv(O / "xday_read5_ba.csv")
json.dump(res, open(O / "xday_read5.json", "w"), indent=1)
