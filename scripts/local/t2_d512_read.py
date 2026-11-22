#!/usr/bin/env python
"""T2-D512 read, the full stack (written 2026-11-15 before any full-stack number). Per fold f x draw d (810-812):
  CNN families (fbc / sh / ifn) = the ROUTED member of seed d (copies + tv8c, outputs/t2_d512/route_<fam>); P3 (bag) = the mean of seeds d and d + 30;
  P4 (sbal) = ShallowNet's members from route_shsb instead of route_sh; FM = the mean of six ST-L members 820-825 (3 plain + 3 CAR, sofia), P2 (tilt) =
  the same seeds trained with --fm-tilt 0.15; P1 (in-stream) = sh and ifn slots = (1 - r) * the slot's member part + r * the adapted copy (seed d,
  outputs/t2_d512/p1_<fam>, isSpr; r 1/2), its control = the same with the UNADAPTED pooled net (is0).
  Each family decoded by the causal Sinkhorn (10 / 3 / 0.95) from ITS cross-fitted start vector (the mean posterior of the other folds'
  strangers, computed per variant), pooled at M7's shares. Ten strangers (every window), returning P04 / P10 sessions 2-3.
Columns: BASE, FULL, FULL - Pi (P1 -> the extra-net control in its place), each part alone on BASE.
Decision (registered): a part stays if FULL - (FULL minus Pi) > 0 with >= 2 / 3 draws (P2: >= 6 / 10 strangers) and no guard broken; the final
stack = BASE + the kept parts, re-read: ten strangers > 0 with >= 2 / 3 draws, no stranger < -0.020, returning mean >= -0.003.
    mem-guard python scripts/local/t2_d512_read.py
"""
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S, D = O / "sofia_s1", O / "t2_d512"
idx = pd.read_csv(EXP / "index.csv")
U, SES, STREAM, ORD, Y, CELL = (idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy(), idx.cell.to_numpy())
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
RT = {"fbc": "route_fbc/lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz",
      "sh": "route_sh/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
      "shsb": "route_shsb/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vr_sb16tv8cfb2__s{s}.npz",
      "ifn": "route_ifn/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz"}
P1R = {"sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr", "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vr"}
DRAWS, PARTS, R1 = (810, 811, 812), ("P1", "P2", "P3", "P4"), 0.5


def load(path):
    z = np.load(path)
    p = z["prob_last"].astype(np.float64)
    return z["test_idx"], p / p.sum(1, keepdims=True)


def fm_mean(f, tilt):
    ps, ix = [], None
    for s in range(820, 826):
        arm = ("stlcar" if s >= 823 else "stl") + ("tl" if tilt else "")
        fs = glob.glob(f"{S}/ni26-t2p-d512-{arm}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
        assert len(fs) == 1, (arm, s, f, fs)
        ix_, p = load(fs[0])
        assert ix is None or (ix_ == ix).all()
        ix = ix_
        ps.append(p)
    return ix, np.mean(ps, 0)


def realign(ix_from, p, ix_to):
    pos = {int(r): k for k, r in enumerate(ix_from)}
    return p[np.array([pos[int(r)] for r in ix_to])]


# every fold's family outputs per variant setting, on the FM test order
RAW = {}
for f in range(10):
    ix, fm0 = fm_mean(f, False)
    _, fm1 = fm_mean(f, True)
    for d in DRAWS:
        q = {"fm": {False: fm0, True: realign(_, fm1, ix)}}
        for fam in ("fbc", "sh", "shsb", "ifn"):
            q[fam] = {b: np.mean([realign(*load(D / RT[fam].format(f=f, s=s)), ix) for s in ((d, d + 30) if b else (d,))], 0) for b in (False, True)}
        for fam in ("sh", "ifn"):
            q["p1_" + fam] = {v: realign(*load(D / f"p1_{fam}/runs/lc9_{f}__{P1R[fam]}{v}__s{d}.npz"), ix) for v in ("is0", "isSpr")}
        RAW[(f, d)] = (ix, q)
TRU = {f: [u for u in set(U[RAW[(f, DRAWS[0])][0]]) if (U[RAW[(f, DRAWS[0])][0]] == u).sum() == (U == u).sum()][0] for f in range(10)}


def parts_of(q, on):
    """the four families' raw parts for a set of parts `on` (P1 value: 'adapt' | 'extra' | None)."""
    out = {"fm": q["fm"][bool(on.get("P2"))]}
    out["fbc"] = q["fbc"][bool(on.get("P3"))]
    out["sh"] = q["shsb" if on.get("P4") else "sh"][bool(on.get("P3"))]
    out["ifn"] = q["ifn"][bool(on.get("P3"))]
    if on.get("P1"):
        v = "isSpr" if on["P1"] == "adapt" else "is0"
        for fam in ("sh", "ifn"):
            out[fam] = (1 - R1) * out[fam] + R1 * q["p1_" + fam][v]
    return out


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        ks = np.unique(Y[rows][m])
        if len(ks) >= 2:
            out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
    return out


def stack_score(on):
    """-> DataFrame (draw, kind, user, ba) for the variant `on`; start vectors cross-fitted per variant."""
    P = {k: parts_of(q, on) for k, (ix, q) in RAW.items()}
    vec = {}
    for d in DRAWS:
        for fam in W:
            per = {f: P[(f, d)][fam][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
            for f in range(10):
                v = np.mean([per[g] for g in range(10) if g != f], 0)
                vec[(f, d, fam)] = v / v.sum()
    recs = []
    for (f, d), (ix, q) in RAW.items():
        key = (STREAM[ix], ORD[ix])
        mix = sum(W[fam] * causal_sinkhorn(P[(f, d)][fam], key, 10, 3, 0.95, start=vec[(f, d, fam)]) for fam in W).argmax(1)
        kind = np.where(U[ix] == TRU[f], "stranger", np.where(np.isin(SES[ix], ("2", "3")), "returning", "other"))
        for kd in ("stranger", "returning"):
            for u in np.unique(U[ix][kind == kd]):
                m = (kind == kd) & (U[ix] == u)
                for c, v in cells(mix[m], ix[m]).items():
                    recs.append(dict(fold=f, draw=d, kind=kd, user=u, cell=c, ba=v))
    return pd.DataFrame(recs).groupby(["kind", "draw", "user"]).ba.mean()


FULL = {"P1": "adapt", "P2": True, "P3": True, "P4": True}
variants = {"BASE": {}, "FULL": FULL}
for p in PARTS:
    variants[f"FULL-{p}"] = {k: v for k, v in FULL.items() if k != p} if p != "P1" else dict(FULL, P1="extra")
    variants[f"BASE+{p}"] = {p: FULL[p]} if p != "P1" else {"P1": "adapt"}
variants["FULL-P1(no extra net)"] = {k: v for k, v in FULL.items() if k != "P1"}
variants["BASE+P1extra"] = {"P1": "extra"}
SC = {}
for name, on in variants.items():
    SC[name] = stack_score(on)
    print(f"[d512] scored {name}", flush=True)


def delta(a, b):
    d_ = (SC[a] - SC[b])
    st, rt = d_.loc["stranger"], d_.loc["returning"]
    pu = st.groupby("user").mean()
    return dict(ten=round(float(pu.mean()), 4), draws=[round(float(x), 4) for x in st.groupby("draw").mean()], min_user=round(float(pu.min()), 4),
                min_who=str(pu.idxmin()), users_up=int((pu > 0).sum()), returning=round(float(rt.groupby("user").mean().mean()), 4))


res = {"FULL vs BASE": delta("FULL", "BASE")}
kept = []
for p in PARTS:
    e = delta("FULL", f"FULL-{p}")
    res[f"LOO {p} (FULL - FULL-{p})"] = e
    res[f"alone {p} (BASE+{p} - BASE)" if p != "P1" else "alone P1 (BASE+P1 - BASE+P1extra)"] = delta(f"BASE+{p}", "BASE" if p != "P1" else "BASE+P1extra")
    ok = e["ten"] > 0 and (e["users_up"] >= 6 if p == "P2" else sum(x > 0 for x in e["draws"]) >= 2) and e["min_user"] >= -0.020 and e["returning"] >= -0.003
    if ok:
        kept.append(p)
res["descr: FULL-P1(no extra net) vs FULL-P1(extra)"] = delta("FULL-P1(no extra net)", "FULL-P1")
for k, v in res.items():
    print(f"D512 {k}: ten {v['ten']:+.4f} draws {v['draws']} up {v['users_up']}/10 min {v['min_user']:+.4f} ({v['min_who']}) returning {v['returning']:+.4f}", flush=True)
on = {p: FULL[p] for p in kept}
SC["FINAL"] = stack_score(on)
fin = delta("FINAL", "BASE")
ok = fin["ten"] > 0 and sum(x > 0 for x in fin["draws"]) >= 2 and fin["min_user"] >= -0.020 and fin["returning"] >= -0.003
res["kept"], res["FINAL vs BASE"], res["final_passes"] = kept, fin, bool(ok)
print(f"D512 KEPT {kept} -> FINAL vs BASE: ten {fin['ten']:+.4f} draws {fin['draws']} min {fin['min_user']:+.4f} ({fin['min_who']}) returning {fin['returning']:+.4f} -> "
      + ("PASSES (runtime to be measured)" if ok else "does not pass"), flush=True)
json.dump(res, open(D / "read.json", "w"), indent=1)
