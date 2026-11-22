#!/usr/bin/env python
"""T2-D536 read (written 2026-11-16 before any D536 number): the W19 fresh-draw read. D532's reader head (scripts/local/t2_d532_read.py up to
its BASE) is exec'd with ONE constant redirected, asserted to occur once: the fresh draws are CNN 816-818 (outputs/t2_d536: route_*, p1_*,
side_* with NI26_SIDE_DUMP=1) and the DC bag at LightGBM seeds 7-9 (outputs/t2_d536/pred_s7-8-9_dc.npz). The FM side is REUSED from D532
(D531's lines at 826-831, D532's MI member at 826), as registered: the candidate touches no FM output.
Stacks on the same fresh draws:
  W17F = W17's form (FM = CTL six; side nets UNC);
  BASE = W18's form (FM = D524's SD six with the MI member at 826 = D529; side nets UNC);
  W19F = BASE with D533's clamp (side z clipped at each fold's training 1st / 99th percentiles, D533's construction).
Side nets: UNC = each net's output recomputed from its dump (softmax(logits + W . z + b); must equal prob_last <= 1e-5; the dumped z must
equal t2p_bench.side_z <= 1e-6). The only candidate is the clamp (D535 nominated nothing), so leave-one-out = W19F vs BASE.
Rule (D536): W19 exists iff W19F vs BASE passes the part rule (score()['part']: ten > 0, >= 2 / 3 draws, no stranger < -0.020, returning
>= -0.003) AND ten >= +0.001. Headline = W19F vs BASE; W17F beside (BASE vs W17F, W19F vs W17F, descriptive).
    mem-guard python scripts/local/t2_d536_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0]]
src = open("scripts/local/t2_d532_read.py").read().split("\nBASE = mix([])\n")[0]
old = 'DRAWS, CNN, DCF = (813, 814, 815), O / "t2_d532", O / "t2_d532/pred_s4-5-6_dc.npz"'
new = 'DRAWS, CNN, DCF = (816, 817, 818), O / "t2_d536", O / "t2_d536/pred_s7-8-9_dc.npz"'
assert src.count(old) == 1, "the D532 draw constant moved"
exec(compile(src.replace(old, new), "t2_d532_read_head(d536)", "exec"))
assert not A.check and A.d524_form == "six" and DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")

sys.path.insert(0, str(Path("tracks/t2_bci/scripts").resolve()))
sys.path.insert(0, str(Path("src").resolve()))
import t2p_bench as B_  # noqa: E402

Xe, idxe, _ = B_.load_export(str(EXP))
assert (idxe["stream"].to_numpy() == np.array([s_.split("/")[-1] for s_ in STREAM])).all() and (idxe["order"].to_numpy() == ORD).all()
Fr = B_.side_raw(Xe, "dcmean")
del Xe
ZF, BOUNDS = {}, {}
for f in range(10):
    ((_, trm, _, _),) = B_.folds(idxe, f"lc9_{f}")
    tr = np.flatnonzero(trm)
    ZF[f] = B_.side_z(Fr, idxe, tr, "dcmean")
    BOUNDS[f] = (np.percentile(ZF[f][tr], 1, axis=0), np.percentile(ZF[f][tr], 99, axis=0))
CHK = dict(unc_vs_prob_last=0.0, z_vs_side_z=0.0, files=0, clipped_frac=[])


def side_prob(path, f, mode):
    """= t2_d533_read.side_prob: one side net's output recomputed from its dump, z unclipped ('unc') or clipped ('clip')."""
    z = np.load(need(path))
    refuse_oracle(path, z)                               # T2-D550: the manifest (the name: need)
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


def parts_of(f, d, parts):                       # D532's parts_of (FM parts D524 / D529 only), the side nets through side_prob
    assert set(parts) <= {"D524", "D529"}, parts
    ix = IX[f]
    P = {"fbc": np.mean([realign(*load(CNN / RT["fbc"].format(f=f, s=d)), ix)], 0)}
    for fam in ("sh", "ifn"):
        r = np.mean([realign(*load(CNN / RT[fam].format(f=f, s=d)), ix)], 0)
        a = realign(*load(CNN / P1R[fam].format(f=f, s=d)), ix)
        part = (1 - R1) * r + R1 * a
        S_ = np.mean([realign(*side_prob(p, f, SIDE_MODE), ix) for p in side_files(fam, f, d)], 0)
        P[fam] = (2.0 / 3.0) * part + (1.0 / 3.0) * S_
    ix_, pfm = fm_slot(f, parts)
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


W18 = ["D524", "D529"]
SIDE_MODE = "unc"
W17F, BASE = mix([]), mix(W18)
SIDE_MODE = "clip"
W19F = mix(W18)
CHK["clipped_frac"] = float(np.mean(CHK["clipped_frac"]))
print(f"D536 checks: {CHK}", flush=True)
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
res = {"draws": list(DRAWS), "checks": CHK, "level": {"W17F": level(W17F), "BASE (W18 form)": level(BASE), "W19F": level(W19F)}}
head = score(W19F, BASE)
head["per_stranger"] = per_stranger(W19F, BASE)
head["W19_exists"] = bool(head["part"] and head["ten"] >= 0.001)
res["HEADLINE W19F (clamp) vs BASE (W18 form), same fresh draws"] = head
res["BASE (W18 form) vs W17F (descriptive)"] = dict(score(BASE, W17F), per_stranger=per_stranger(BASE, W17F))
res["W19F vs W17F (descriptive)"] = score(W19F, W17F)
for k, v in res.items():
    print(f"D536 {k}: {v}", flush=True)
print(f"D536 verdict: {'W19 EXISTS (the clamp kept; build needs the owner line)' if head['W19_exists'] else 'NO W19 (the clamp not kept on the fresh draws)'}", flush=True)
json.dump(res, open(O / "t2_d536.json", "w"), indent=1, default=str)
