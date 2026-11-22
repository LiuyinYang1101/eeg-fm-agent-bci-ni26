#!/usr/bin/env python
"""T2-D517 read (written 2026-11-15 before any D517 number): can a CNN family carry the DC information? On T2-D512's fresh K = 9 panel (draws
810-812), every family decoded from its cross-fitted start (per variant), pooled at M7's shares; ten strangers every window, the six, returning.
  arm (b) keep-mean: D517's routed ShallowNet / IFNet members (copies + tv8c, de-meaning off) IN PLACE of D512's base members (same seeds);
  arms dcmean / dc8 / lvwin / lvstream: each slot (sh, ifn) = 1/2 routed member + 1/2 the side-input POOLED net, vs the control 1/2 routed member +
  1/2 the plain pooled net of the same seed (D512's is0); stacks BASE and BASE + P2 (tilt); each also WITH the DC tree family at 0.075 on top
  (does a net that sees the DC make the tree redundant?); and each side net alone vs is0 alone (decoded, strangers).
    mem-guard python scripts/local/t2_d517_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/local")
src = open("scripts/local/t2_d512_read.py").read()
head = src.split("FULL = {")[0].replace("HERE = Path(__file__).resolve().parent", "HERE = Path('scripts/local').resolve()")
exec(compile(head, "t2_d512_read_head", "exec"))            # RAW, TRU, W, parts_of, cells, causal_sinkhorn, load, realign, U, SES, STREAM, ORD, Y, ...
O_ = Path.home() / "workspace/outputs/neural_interface_challenge"
D7, D2 = O_ / "t2_d517", O_ / "t2_d512"
REC = {"sh": ("shallow", "r1_pulx_vp2_e180_vr"), "ifn": ("ifnet", "r1_pulx_vp2_e180_vr")}
NC6 = ("P01", "P03", "P05", "P07", "P09", "P10")
zp = np.load(O_ / "t2_d515/pred_s1-2-3_dm.npz")
DC = {f: (zp[f"te_{f}"], zp[f"bag_{f}"]) for f in range(10)}
DCVEC = {}
for f in range(10):
    v = np.mean([DC[g][1][U[DC[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
    DCVEC[f] = v / v.sum()


def km_member(fam, f, d, ix):
    m, rec = REC[fam]
    return realign(*load(D7 / f"route_{fam}/lc9_{f}__{m}__ea__n00__{rec}tv8cfb2__km__s{d}.npz"), ix)


def pooled(path, ix):
    return realign(*load(path), ix)


def arm_parts(arm, q, f, d, ix, on):
    P = parts_of(q, on)
    if arm == "base":
        return P
    if arm == "km":
        for fam in ("sh", "ifn"):
            P[fam] = km_member(fam, f, d, ix)
        return P
    for fam in ("sh", "ifn"):
        m, rec = REC[fam]
        if arm == "ctl":
            side = pooled(D2 / f"p1_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}is0__s{d}.npz", ix)
        else:
            side = pooled(D7 / f"side_{arm}_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__side{arm}__s{d}.npz", ix)
        P[fam] = 0.5 * P[fam] + 0.5 * side
    return P


def mixtures(arm, on, w_dc):
    P = {k: arm_parts(arm, q, k[0], k[1], ix, on) for k, (ix, q) in RAW.items()}
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
        if w_dc > 0:
            te, p = DC[f]
            pos = {int(r): k for k, r in enumerate(te)}
            q_ = p[np.array([pos[int(r)] for r in ix])]
            mix = (1 - w_dc) * mix + w_dc * causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCVEC[f])
        out[(f, d)] = (ix, mix)
    return out


def score(A, B):
    recs = []
    for (f, d), (ix, ma) in A.items():
        mb = B[(f, d)][1]
        kind = np.where(U[ix] == TRU[f], "stranger", np.where(np.isin(SES[ix], ("2", "3")), "returning", "other"))
        for kd in ("stranger", "returning"):
            for u in np.unique(U[ix][kind == kd]):
                m = (kind == kd) & (U[ix] == u)
                a, b = cells(ma.argmax(1)[m], ix[m]), cells(mb.argmax(1)[m], ix[m])
                for c in a:
                    recs.append(dict(draw=d, kind=kd, user=u, d=a[c] - b[c]))
    R = pd.DataFrame(recs)
    st, rt = R[R.kind == "stranger"], R[R.kind == "returning"]
    pu = st.groupby("user").d.mean()
    e = dict(ten=round(float(pu.mean()), 4), draws=[round(float(x), 4) for x in st.groupby(["draw", "user"]).d.mean().groupby("draw").mean()],
             six=round(float(pu[[u for u in NC6 if u in pu.index]].mean()), 4), min_user=round(float(pu.min()), 4), min_who=str(pu.idxmin()),
             returning=round(float(rt.groupby("user").d.mean().mean()), 4))
    e["part"] = bool(e["ten"] > 0 and sum(x > 0 for x in e["draws"]) >= 2 and e["min_user"] >= -0.020 and e["returning"] >= -0.003)
    return e


def avail(arm):
    if arm == "km":
        return (D7 / "route_sh").exists() and len(list((D7 / "route_sh").glob("*.npz"))) >= 30 and len(list((D7 / "route_ifn").glob("*.npz"))) >= 30
    if arm in ("base", "ctl"):
        return True
    return all(len(list((D7 / f"side_{arm}_{fam}/runs").glob("*.npz"))) >= 30 for fam in ("sh", "ifn") if (D7 / f"side_{arm}_{fam}/runs").exists()) \
        and all((D7 / f"side_{arm}_{fam}/runs").exists() for fam in ("sh", "ifn"))


res = {}
ARMS = [a for a in ("km", "dcmean", "dc8", "lvwin", "lvstream") if avail(a)]
print(f"[d517] arms with complete files: {ARMS}", flush=True)
for on_name, on in (("BASE", {}), ("BASE+P2", {"P2": True})):
    M = {a: mixtures(a, on, 0.0) for a in ["base", "ctl"] + ARMS}
    MD = {a: mixtures(a, on, 0.075) for a in ["base", "ctl"] + ARMS}
    for a in ARMS:
        ref = "base" if a == "km" else "ctl"
        res[f"{on_name}: {a} vs {ref}"] = score(M[a], M[ref])
        res[f"{on_name}+DC: {a} vs {ref}"] = score(MD[a], MD[ref])
        print(f"D517 {on_name:8s} {a:9s} vs {ref:4s}: {res[f'{on_name}: {a} vs {ref}']}", flush=True)
        print(f"D517 {on_name + '+DC':8s} {a:9s} vs {ref:4s}: {res[f'{on_name}+DC: {a} vs {ref}']}", flush=True)
    for a in ["ctl"] + [x for x in ARMS if x != "km"]:            # added 16:07 after the first read: what SHIPS is the arm vs the base stack
        res[f"{on_name}: {a} vs base"] = score(M[a], M["base"])
        res[f"{on_name}+DC: {a} vs base+DC"] = score(MD[a], MD["base"])
        print(f"D517 {on_name:8s} {a:9s} vs base: {res[f'{on_name}: {a} vs base']}", flush=True)
        print(f"D517 {on_name + '+DC':8s} {a:9s} vs base+DC: {res[f'{on_name}+DC: {a} vs base+DC']}", flush=True)
    res[f"{on_name}: DC tree vs none (ref)"] = score(MD["base"], M["base"])
    print(f"D517 {on_name:8s} DC tree alone on top: {res[f'{on_name}: DC tree vs none (ref)']}", flush=True)
json.dump(res, open(O_ / "t2_d517.json", "w"), indent=1, default=str)
