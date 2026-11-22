#!/usr/bin/env python
"""Track1's T2 experiment XDAY, the reads (rules fixed in lead/panel_2026-11-14_outliers/SEAT_T1_on_T2.md, section XDAY, before any number).
Scherer2015 day 2 of nine users; p = (1 - r) pooled + r copy per family (tv8c files), the shipped causal decode per family (SK 10 / n0 3 /
lambda 0.95), CNN mix 0.5714 FBCNet + 0.4286 ShallowNet, balanced accuracy per user, mean over the seeds present for BOTH families.
  read 1: validity K1 / K2, then the dose curve (own copy = oracle route; the fingerprint route beside it) and the D500 forecast.
  read 2 (--gap, needs the xh runs): the returning gap = in training (xd) minus held out (the xh fold without the user).
    mem-guard python scripts/local/t1seat_t2_xday_read.py [--gap]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
from t2_cluster_ci import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge/t1seat_t2"
idx = pd.read_csv(O / "export_combo_v1/index.csv", dtype={"session": str})
U = idx.subject.str.split("/").str[-1].to_numpy()
STREAM, ORD, Y = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy()
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", 0.5714), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vr", 0.4286)}
SHIP = {"fbc": 1 / 3, "sh": 1 / 2}
ARMS = {f"m{m:g}": {f: min(1.0, m * SHIP[f]) for f in FAM} for m in (0, 0.5, 1, 1.5, 2)}
ARMS["copy"] = {f: 1.0 for f in FAM}
ARMS.update({f"r{n}": {f: r for f in FAM} for n, r in (("1/3", 1 / 3), ("1/2", 1 / 2), ("2/3", 2 / 3))})
SEEDS = [s for s in range(780, 786) if all((O / f"runs/xd__{st}tv8cpu__s{s}.npz").exists() for st, _ in FAM.values())]
HOLD = {"xh0": ("S1", "S4", "S7"), "xh1": ("S2", "S5", "S8"), "xh2": ("S3", "S6", "S9")}
rng = np.random.default_rng(0)


def ba_users(P, ix):
    """{user: BA} of the mixed, decoded probabilities P = {family: (N, 3)} on rows ix."""
    key = (STREAM[ix], ORD[ix])
    pr = sum(FAM[f][1] * causal_sinkhorn(P[f], key, 10, 3, 0.95) for f in FAM).argmax(1)
    y, u = Y[ix], U[ix]
    return {v: float(np.mean([np.mean(pr[(u == v) & (y == q)] == q) for q in range(3)])) for v in np.unique(u)}


def boot(d):
    d = np.asarray(d)
    b = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
    return float(np.percentile(b, 5)), float(np.percentile(b, 95))


def route(ix_tr, ix_te):
    """the package's fingerprint route (t2p_fpcalib.blend): nearest training-user centroid of the stream's running mean."""
    import t2p_fpcalib as FC
    import t2p_route as RT
    fpf = O / "fp_combo.npy"
    if not fpf.exists():
        X = np.load(O / "export_combo_v1/X.npy", mmap_mode="r")
        names = json.loads((O / "export_combo_v1/meta.json").read_text())["ch_names"]
        np.save(fpf, RT.fingerprint(np.asarray(X), 120.0, names).astype(np.float32))
    Fp = np.load(fpf)
    proj, mu, users = FC.fit_space(Fp[ix_tr], U[ix_tr])
    k = np.zeros(len(ix_te), int)
    for s in np.unique(STREAM[ix_te]):
        j = np.flatnonzero(STREAM[ix_te] == s)
        j = j[np.argsort(ORD[ix_te][j], kind="stable")]
        k[j] = FC.stream_route_every(proj(Fp[ix_te[j]]), mu, 1)
    return k, list(users)


def norm(p):
    p = p.astype(np.float64)
    return p / p.sum(-1, keepdims=True)


rows, kroute = [], None
for s in SEEDS:
    Z = {f: np.load(O / f"runs/xd__{st}tv8cpu__s{s}.npz") for f, (st, _) in FAM.items()}
    ix = Z["fbc"]["test_idx"]
    users = list(Z["fbc"]["users"])
    assert all((Z[f]["test_idx"] == ix).all() and list(Z[f]["users"]) == users for f in FAM)
    sh = {f: norm(Z[f]["prob_last"]) for f in FAM}
    pu = {f: norm(Z[f]["prob_users_test"]) for f in FAM}
    n = np.arange(len(ix))
    own = np.array([users.index(u) for u in U[ix]])
    if kroute is None:
        tr = np.flatnonzero((np.char.startswith(U.astype(str), "P")) | (idx.session.to_numpy() == "1"))
        kroute, ru = route(tr, ix)
        assert ru == users, (ru, users)
        kroute = np.where(kroute < 0, own, kroute)
        hit = {u: float((kroute == own)[U[ix] == u].mean()) for u in np.unique(U[ix])}
        print("XDAY route: share of day-2 windows routed to the user's own copy: " + " ".join(f"{u} {v:.2f}" for u, v in hit.items())
              + f"; mean {np.mean(list(hit.values())):.3f}", flush=True)
    for a, r in ARMS.items():
        for who, k in (("own", own), ("routed", kroute)):
            b = ba_users({f: (1 - r[f]) * sh[f] + r[f] * pu[f][k, n] for f in FAM}, ix)
            rows += [dict(seed=s, arm=a, who=who, user=u, ba=v) for u, v in b.items()]
    sch = [u for u in users if u.startswith("S")]
    for a in ("copy", "m1"):                      # K2: every WRONG Scherer user's copy (cyclic shifts 1..8 over the nine users)
        r = ARMS[a]
        for sft in range(1, 9):
            k = np.array([users.index(sch[(sch.index(u) + sft) % 9]) for u in U[ix]])
            b = ba_users({f: (1 - r[f]) * sh[f] + r[f] * pu[f][k, n] for f in FAM}, ix)
            rows += [dict(seed=s, arm=a, who=f"wrong{sft}", user=u, ba=v) for u, v in b.items()]
C = pd.DataFrame(rows)
C.to_csv(O / "xday_cells.csv", index=False)
T = C[C.who == "own"].groupby(["user", "arm"]).ba.mean().unstack()
R = C[C.who == "routed"].groupby(["user", "arm"]).ba.mean().unstack()
Wr = C[C.who.str.startswith("wrong")].groupby(["user", "arm"]).ba.mean().unstack()
order = ["m0", "m0.5", "m1", "m1.5", "m2", "copy", "r1/3", "r1/2", "r2/3"]
print(f"XDAY read 1: seeds {SEEDS}; per user (own copy):", flush=True)
for u in T.index:
    print(f"   {u}: " + " ".join(f"{a} {T.loc[u, a]:.3f}" for a in order) + f" | wrong-user copy: copy-only {Wr.loc[u, 'copy']:.3f}, shipped {Wr.loc[u, 'm1']:.3f}", flush=True)
k1 = bool(T.m0.mean() >= 0.45 and (T.m0 >= 0.40).sum() >= 6)
dk = T["copy"] - Wr["copy"]
k2 = bool(dk.mean() >= 0.010 and (dk > 0).sum() >= 6)
print(f"XDAY K1 (decodable): pooled mean {T.m0.mean():.4f} (>= 0.45), users >= 0.40: {int((T.m0 >= 0.40).sum())} / 9 -> {k1}", flush=True)
print(f"XDAY K2 (identity): own - wrong at copy-only {dk.mean():+.4f} (>= +0.010), users own > wrong {int((dk > 0).sum())} / 9 -> {k2}; "
      f"at the shipped dose own - wrong {float((T.m1 - Wr.m1).mean()):+.4f}", flush=True)
res = dict(seeds=SEEDS, k1=k1, k2=k2, per_user={u: {a: round(float(T.loc[u, a]), 4) for a in order} for u in T.index},
           wrong={u: {a: round(float(Wr.loc[u, a]), 4) for a in ("copy", "m1")} for u in T.index})
if k1 and k2:
    for name, M in (("own", T), ("routed", R)):
        d = M.m2 - M.m1
        lo, hi = boot(d)
        up, dn = int((d > 0).sum()), int((d < 0).sum())
        fc = "supports doubling" if d.mean() >= 0.005 and up >= 6 else ("against" if d.mean() <= -0.005 and dn >= 6 else "no forecast")
        best = M[["m0", "m0.5", "m1", "m1.5", "m2", "copy"]].idxmax(1)
        lose = int((M["copy"] < M.m1).sum())
        print(f"XDAY VALID. dose curve ({name} copy), nine-user mean: " + " ".join(f"{a} {M[a].mean():.4f}" for a in order), flush=True)
        print(f"   {name}: D = m2 - m1 = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] ({up} up / {dn} down) -> {fc}; copy-only vs shipped "
              f"{float((M['copy'] - M.m1).mean()):+.4f}, users losing at copy-only {lose} / 9; best arm " + " ".join(f"{u}:{b}" for u, b in best.items()), flush=True)
        res[name] = dict(curve={a: round(float(M[a].mean()), 4) for a in order}, D=float(d.mean()), ci=[lo, hi], up=up, down=dn, forecast=fc,
                         lose_copy=lose, best=best.to_dict())
else:
    print("XDAY NOT VALID as a returning-side instrument (a registered kill failed): nothing else is read", flush=True)
if "--gap" in sys.argv and k1 and k2:
    g = []
    for s in SEEDS:
        for h, held in HOLD.items():
            fs = {f: O / f"runs/{h}__{st}tv8__s{s}.npz" for f, (st, _) in FAM.items()}
            if not all(p.exists() for p in fs.values()):
                continue
            Zh = {f: np.load(p) for f, p in fs.items()}
            ixh = Zh["fbc"]["test_idx"]
            b = ba_users({f: norm(Zh[f]["prob_last"]) for f in FAM}, ixh)
            g += [dict(seed=s, user=u, out=b[u]) for u in held]
    G = pd.DataFrame(g)
    seeds_g = sorted(set(G.seed))
    out = G.groupby("user").out.mean()
    Cg = C[(C.who == "own") & C.seed.isin(seeds_g)].groupby(["user", "arm"]).ba.mean().unstack()
    for name, col in (("pooled, user in training (r 0)", "m0"), ("pooled + own copy at the shipped r", "m1")):
        d = Cg[col] - out
        lo, hi = boot(d)
        print(f"XDAY read 2 ({len(seeds_g)} seeds): {name} minus held out = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], users up {int((d > 0).sum())} / 9; "
              + " ".join(f"{u} {v:+.3f}" for u, v in d.items()), flush=True)
        res[f"gap_{col}"] = dict(mean=float(d.mean()), ci=[lo, hi], per_user={u: round(float(v), 4) for u, v in d.items()})
    print(f"   held-out level {out.mean():.4f}; in-training level {Cg.m0.mean():.4f}; with copy {Cg.m1.mean():.4f}", flush=True)
json.dump(res, open(O / ("xday_read_gap.json" if "--gap" in sys.argv else "xday_read.json"), "w"), indent=1)
