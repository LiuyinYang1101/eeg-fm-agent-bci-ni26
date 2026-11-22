#!/usr/bin/env python
"""T2-D517 FINAL read (written 2026-11-15 after the BASE / BASE+P2 read, before any FINAL number): the side-input nets inside the SHIPPED
context. FINAL = D512's P1 (adapt: 1/2 routed member + 1/2 the in-stream adapted pooled net) + P2 (tilt FM), the W15 build; FINAL+DC = W16.
Forms for an arm's side nets S (the dcmean / dc8 pooled nets of D517, per family sh / ifn):
  after : part = 1/2 FINAL part + 1/2 S             (the side nets blended after the in-stream blend, before the family corrector)
  thirds: part = 1/3 routed + 1/3 in-stream + 1/3 S
  ctl   : 'after' with the plain pooled net is0 in place of S (the same dilution of FINAL, no side input)
Reads, each vs FINAL (and each +DC vs FINAL+DC): ten strangers every window, the draws, the six, the worst stranger, returning; part rule.
    mem-guard python scripts/local/t2_d517_final.py
"""
import json
import sys
from pathlib import Path

import numpy as np

src = open("scripts/local/t2_d517_read.py").read().split("res = {}")[0]
exec(compile(src, "t2_d517_read_head", "exec"))     # D512 head + km_member / pooled / mixtures' helpers / score / avail, DC / DCVEC

FIN = {"P1": "adapt", "P2": True}


def form_parts(form, arm, q, f, d, ix):
    P = parts_of(q, FIN)
    if form == "final":
        return P
    for fam in ("sh", "ifn"):
        m, rec = REC[fam]
        if arm == "is0":
            S = pooled(D2 / f"p1_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}is0__s{d}.npz", ix)
        else:
            S = pooled(D7 / f"side_{arm}_{fam}/runs/lc9_{f}__{m}__ea__n00__{rec}__side{arm}__s{d}.npz", ix)
        if form == "after":
            P[fam] = 0.5 * P[fam] + 0.5 * S
        else:                                        # thirds: P = 1/2 routed + 1/2 adapted  ->  1/3 routed + 1/3 adapted + 1/3 S
            P[fam] = (2.0 / 3.0) * P[fam] + (1.0 / 3.0) * S
    return P


def mix(form, arm, w_dc):
    P = {k: form_parts(form, arm, q, k[0], k[1], ix) for k, (ix, q) in RAW.items()}
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
        mx = sum(W[fam] * causal_sinkhorn(P[(f, d)][fam], key, 10, 3, 0.95, start=vec[(f, d, fam)]) for fam in W)
        if w_dc > 0:
            te, p = DC[f]
            pos = {int(r): k for k, r in enumerate(te)}
            q_ = p[np.array([pos[int(r)] for r in ix])]
            mx = (1 - w_dc) * mx + w_dc * causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCVEC[f])
        out[(f, d)] = (ix, mx)
    return out


res = {}
for w_dc, tag in ((0.0, "FINAL"), (0.075, "FINAL+DC")):
    ref = mix("final", None, w_dc)
    for form, arm in (("after", "is0"), ("after", "dcmean"), ("thirds", "dcmean"), ("after", "dc8")):
        e = score(mix(form, arm, w_dc), ref)
        res[f"{tag}: {form} {arm} vs {tag}"] = e
        print(f"D517F {tag:8s} {form:6s} {arm:6s} vs {tag}: {e}", flush=True)
json.dump(res, open(O_ / "t2_d517_final.json", "w"), indent=1, default=str)
