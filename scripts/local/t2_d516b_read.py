#!/usr/bin/env python
"""T2-D516b read (written 2026-11-15 before any D516b number): D516's window-mean GBM bag (outputs/t2_d515/pred_s1-2-3_dm.npz, the same in-fold
models) as a fifth family on T2-D512's FINAL stack (P1 in-stream + P2 tilt; fresh seeds 810-812); shares 0.05 .. 0.15; leave-one-part-out; robustness
per interface, per returning session, and without the streams' first 10 windows. Every family decoded from its cross-fitted start (per variant).
    mem-guard python scripts/local/t2_d516b_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/local")
src = open("scripts/local/t2_d512_read.py").read()
head = src.split("FULL = {")[0].replace("HERE = Path(__file__).resolve().parent", "HERE = Path('scripts/local').resolve()")
exec(compile(head, "t2_d512_read_head", "exec"))            # RAW, TRU, W, parts_of, cells, causal_sinkhorn, U, SES, STREAM, ORD, Y, CELL, DRAWS ...
O_ = Path.home() / "workspace/outputs/neural_interface_challenge"
zp = np.load(O_ / "t2_d515/pred_s1-2-3_dm.npz")
DC = {f: (zp[f"te_{f}"], zp[f"bag_{f}"]) for f in range(10)}
IFC = pd.read_csv(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/index.csv").interface.to_numpy()
NC6 = ("P01", "P03", "P05", "P07", "P09", "P10")


def dc_on(ix, f):
    te, p = DC[f]
    pos = {int(r): k for k, r in enumerate(te)}
    q = p[np.array([pos[int(r)] for r in ix])]
    return q / q.sum(1, keepdims=True)


DCVEC = {}
for f in range(10):
    per = [DC[g][1][U[DC[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f]
    v = np.mean(per, 0)
    DCVEC[f] = v / v.sum()


def mixtures(on, w):
    """-> {(f, d): (ix, pooled decoded posterior)} for the parts `on` + the DC family at share w (0 = none)."""
    P = {k: parts_of(q, on) for k, (ix, q) in RAW.items()}
    vec = {}
    for d in DRAWS:
        for fam in W:
            per = {f: P[(f, d)][fam][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
            for f in range(10):
                v = np.mean([per[g] for g in range(10) if g != f], 0)
                vec[(f, d, fam)] = v / v.sum()
    out = {}
    for (f, d), (ix, q) in RAW.items():
        key = (STREAM[ix], ORD[ix])
        mix = sum(W[fam] * causal_sinkhorn(P[(f, d)][fam], key, 10, 3, 0.95, start=vec[(f, d, fam)]) for fam in W)
        if w > 0:
            mix = (1 - w) * mix + w * causal_sinkhorn(dc_on(ix, f), key, 10, 3, 0.95, start=DCVEC[f])
        out[(f, d)] = (ix, mix)
    return out


def score(A, B, sel=None):
    """paired cell-BA deltas A - B per (kind, draw, user); sel(ix) -> extra row mask (e.g. an interface)."""
    recs = []
    for (f, d), (ix, ma) in A.items():
        mb = B[(f, d)][1]
        kind = np.where(U[ix] == TRU[f], "stranger", np.where(np.isin(SES[ix], ("2", "3")), "returning", "other"))
        m0 = np.ones(len(ix), bool) if sel is None else sel(ix)
        for kd in ("stranger", "returning"):
            for u in np.unique(U[ix][(kind == kd) & m0]):
                m = (kind == kd) & m0 & (U[ix] == u)
                a, b = cells(ma.argmax(1)[m], ix[m]), cells(mb.argmax(1)[m], ix[m])
                for c in a:
                    recs.append(dict(draw=d, kind=kd, user=u, cell=c, d=a[c] - b[c]))
    R = pd.DataFrame(recs)
    st, rt = R[R.kind == "stranger"], R[R.kind == "returning"]
    pu = st.groupby("user").d.mean()
    return dict(ten=round(float(pu.mean()), 4), draws=[round(float(x), 4) for x in st.groupby(["draw", "user"]).d.mean().groupby("draw").mean()],
                six=round(float(pu[[u for u in NC6 if u in pu.index]].mean()), 4), min_user=round(float(pu.min()), 4), min_who=str(pu.idxmin()),
                returning=round(float(rt.groupby("user").d.mean().mean()), 4) if len(rt) else None)


def _pos(ix):
    """each row's position in its stream (0-based, acquisition order)."""
    s_ = pd.DataFrame(dict(st=STREAM[ix], o=ORD[ix]))
    return (s_.groupby("st").o.rank(method="first").to_numpy() - 1).astype(int)


def part(e):
    return e["ten"] > 0 and sum(x > 0 for x in e["draws"]) >= 2 and e["min_user"] >= -0.020 and (e["returning"] is None or e["returning"] >= -0.003)


FINAL = {"P1": "adapt", "P2": True}
res = {}
M_final = mixtures(FINAL, 0)
M_base = mixtures({}, 0)
res["FINAL vs BASE"] = score(M_final, M_base)
gains = {}
for w in (0.05, 0.075, 0.10, 0.125, 0.15):
    M = mixtures(FINAL, w)
    e = score(M, M_final)
    e["part"] = part(e)
    res[f"DC w{w} on FINAL"] = e
    gains[w] = (e["ten"], e["part"])
    if w == 0.075:
        M075 = M
best = max(g for g, _ in gains.values())
ws = sorted(gains)
ship = next((w for i, w in enumerate(ws) if gains[w][0] >= 0.8 * best and gains[w][1] and (i + 1 == len(ws) or gains[ws[i + 1]][1])), None)
res["shippable_share"] = ship
# leave-one-part-out at the primary share 0.075
res["LOO P1 (FINAL+DC - P2+DC with the extra net)"] = score(M075, mixtures({"P1": "extra", "P2": True}, 0.075))
res["LOO P2 (FINAL+DC - P1+DC)"] = score(M075, mixtures({"P1": "adapt"}, 0.075))
res["LOO DC (FINAL+DC - FINAL)"] = res["DC w0.075 on FINAL"]
res["FINAL+DC vs BASE"] = score(M075, M_base)
# robustness of the DC family at 0.075 on FINAL
for ifc in ("Graz", "BrainHero"):
    res[f"DC w0.075 on FINAL, {ifc} only"] = score(M075, M_final, lambda ix, ifc=ifc: IFC[ix] == ifc)
for ses in ("2", "3"):
    res[f"DC w0.075 on FINAL, returning session {ses}"] = score(M075, M_final, lambda ix, ses=ses: (SES[ix] == ses) | ~np.isin(SES[ix], ("2", "3")))
res["DC w0.075 on FINAL, windows >= 10"] = score(M075, M_final, lambda ix: _pos(ix) >= 10)
for k, v in res.items():
    print(f"D516b {k}: {v}", flush=True)
json.dump(res, open(O_ / "t2_d516b.json", "w"), indent=1, default=str)
