#!/usr/bin/env python
"""T2-D533 read (written 2026-11-16 before the dump job ran): the side nets' z clamp, stage 1. Inside D532's fresh W17 base (draws 813-815, FM =
CTL's six 826-831; t2_d532_read.py's loaders and arithmetic, exec'd), the S5 side nets come from the D533 re-run (outputs/t2_d533/side_{sh,ifn},
NI26_SIDE_DUMP=1), each net's output recomputed from its dump as softmax(logits + W . z' + b):
  UNC  = z' = z (must equal the dump's prob_last <= 1e-5); CLIP = z' = clip(z, lo, hi), lo / hi = the per-feature 1st / 99th percentile of the
  fold's TRAINING rows' causal z (t2p_bench.side_z; the dumped z must equal it <= 1e-6); NONE = no side nets (only to name W*, the side nets'
  worst-served stranger = the min_who of UNC vs NONE); D532 = D532's own fresh side nets (descriptive: the re-run's noise).
Rule (registered in D533): a candidate W18 switch only if (a) ten CLIP - UNC >= -0.0005, (b) W*'s delta CLIP - UNC >= +0.003, (c) returning
CLIP - UNC >= -0.003; P07 named.
    mem-guard python scripts/local/t2_d533_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0]]                        # D532's head parses its own flags: none here (BASE = fresh draws, no part)
src = open("scripts/local/t2_d532_read.py").read().split("\nBASE = mix([])\n")[0]
exec(compile(src, "t2_d532_read_head", "exec"))  # IX, TRU, DC, load, realign, REC, RT, P1R, CNN, R1, W, mix, score, cells, U, SES, ...
assert not A.check and DRAWS == (813, 814, 815)

sys.path.insert(0, str(Path("tracks/t2_bci/scripts").resolve()))
sys.path.insert(0, str(Path("src").resolve()))
import t2p_bench as B_  # noqa: E402

Xe, idxe, _ = B_.load_export(str(EXP))
assert (idxe["stream"].to_numpy() == np.array([s_.split("/")[-1] for s_ in STREAM])).all() and (idxe["order"].to_numpy() == ORD).all()   # load_export shortens "Dreyer2026Proteus/P01|1" to "P01|1" (format only; fixed before any number)
Fr = B_.side_raw(Xe, "dcmean")
del Xe
ZF, BOUNDS = {}, {}
for f in range(10):
    ((_, trm, _, _),) = B_.folds(idxe, f"lc9_{f}")
    tr = np.flatnonzero(trm)
    ZF[f] = B_.side_z(Fr, idxe, tr, "dcmean")
    BOUNDS[f] = (np.percentile(ZF[f][tr], 1, axis=0), np.percentile(ZF[f][tr], 99, axis=0))
D33 = O / "t2_d533"
CHK = dict(unc_vs_prob_last=0.0, z_vs_side_z=0.0, files=0, clipped_frac=[])


def d533_files(fam, f, d):
    m, rec = REC[fam]
    return [D33 / f"side_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__sidedcmean__s{d + k * 1000}.npz" for k in range(5)]


def side_prob(path, f, mode):
    z = np.load(need(path))
    te = z["test_idx"]
    L, Wh, bh = (z[k].astype(np.float64) for k in ("side_logit_last", "side_w", "side_b"))
    S0 = z["side_te"].astype(np.float64)
    CHK["z_vs_side_z"] = max(CHK["z_vs_side_z"], float(np.abs(S0 - ZF[f][te]).max()))
    Sx = S0 if mode == "unc" else np.clip(S0, BOUNDS[f][0], BOUNDS[f][1])
    if mode == "clip":
        CHK["clipped_frac"].append(float((Sx != S0).mean()))
    lg = L + Sx @ Wh.T + bh
    e = np.exp(lg - lg.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    if mode == "unc":
        CHK["unc_vs_prob_last"] = max(CHK["unc_vs_prob_last"], float(np.abs(p - z["prob_last"].astype(np.float64)).max()))
        CHK["files"] += 1
    return te, p


SIDE_MODE = "unc"


def parts_of(f, d, parts):                       # D532's parts_of with the side nets' source switched (BASE only: no part)
    assert not parts
    ix = IX[f]
    P = {"fbc": np.mean([realign(*load(CNN / RT["fbc"].format(f=f, s=d)), ix)], 0)}
    for fam in ("sh", "ifn"):
        r = np.mean([realign(*load(CNN / RT[fam].format(f=f, s=d)), ix)], 0)
        a = realign(*load(CNN / P1R[fam].format(f=f, s=d)), ix)
        part = (1 - R1) * r + R1 * a
        if SIDE_MODE == "none":
            P[fam] = part
            continue
        if SIDE_MODE == "d532":
            S_ = np.mean([realign(*load(p), ix) for p in side_files(fam, f, d)], 0)
        else:
            S_ = np.mean([realign(*side_prob(p, f, SIDE_MODE), ix) for p in d533_files(fam, f, d)], 0)
        P[fam] = (2.0 / 3.0) * part + (1.0 / 3.0) * S_
    ix_, pfm = fm_slot(f, [])
    assert (ix_ == ix).all()
    P["fm"] = pfm
    return {k: P[k] for k in ("fm", "fbc", "sh", "ifn")}


def per_stranger(Aa, Bb):
    recs = []
    for (f, d), (ix, ma) in Aa.items():
        mb = Bb[(f, d)][1]
        m = U[ix] == TRU[f]
        a, b = cells(ma.argmax(1)[m], ix[m]), cells(mb.argmax(1)[m], ix[m])
        recs += [dict(user=TRU[f], d=a[c] - b[c]) for c in a]
    return pd.DataFrame(recs).groupby("user").d.mean().round(4).to_dict()


M = {}
for mode in ("unc", "clip", "none", "d532"):
    SIDE_MODE = mode
    M[mode] = mix([])
print(f"D533 checks: unclipped recomputation vs prob_last max |dp| {CHK['unc_vs_prob_last']:.1e} over {CHK['files']} net-files; dumped z vs side_z "
      f"max |d| {CHK['z_vs_side_z']:.1e}; z entries clipped {np.mean(CHK['clipped_frac']):.4f}", flush=True)
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
res = {"checks": dict(CHK, clipped_frac=float(np.mean(CHK["clipped_frac"])))}
e_w = score(M["unc"], M["none"])
Wstar = e_w["min_who"]
res["UNC vs NONE (the side nets' gain; names W*)"] = dict(e_w, per_stranger=per_stranger(M["unc"], M["none"]))
e = score(M["clip"], M["unc"])
pu = per_stranger(M["clip"], M["unc"])
e.update(per_stranger=pu, W_star=Wstar, W_star_delta=pu[Wstar], P07=pu.get("P07"))
ok = (e["ten"] >= -0.0005, pu[Wstar] >= 0.003, e["returning"] >= -0.003)
e["rule"] = dict(a_ten=ok[0], b_W_star=ok[1], c_returning=ok[2], PASS=all(ok))
res["CLIP vs UNC (the decision)"] = e
res["UNC vs D532's side nets (re-run noise, descriptive)"] = score(M["unc"], M["d532"])
for k, v in res.items():
    print(f"D533 {k}: {v}", flush=True)
print(f"D533 verdict: {'PASS: a candidate W18 switch (stage 2 may follow)' if all(ok) else 'FAIL: closed; W18 keeps unclipped z'} "
      f"(W* = {Wstar}: {pu[Wstar]:+.4f}; P07 {pu.get('P07'):+.4f})", flush=True)
json.dump(res, open(O / "t2_d533.json", "w"), indent=1, default=str)
