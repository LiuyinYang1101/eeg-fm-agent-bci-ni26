#!/usr/bin/env python
"""T2-D529 read (written 2026-11-16 before any D529 member number; the reads were registered in the D529 row 04:35:15).
On D512's K = 9 draws 810-812, the FM slot (share 0.35) = the mean of six tilt ST-L members (plain 820-822 + CAR 823-825):
  ORIG = D512's own six (the shipped FM slot);
  mi1  = ORIG with the PLAIN s820 member replaced by the MI-expert member at s820 (stage 1 on D454's nine MI corpora from the
         public root, stage 2 = D512's plain tilt line);
  ctl1 = ORIG with the plain s820 replaced by D524's CTL s820 (the same line retrained without the MI stage);
  mi2  = ORIG with plain s820 AND s821 replaced by the MI members (descriptive).
Stack: W17 = FINAL + DC tree 0.075 + side nets S5 'thirds'. Reads: mi1 vs ctl1 (THE decision, part rule, final-week floor),
mi1 vs ORIG, ctl1 vs ORIG (the retrain noise floor), mi2 vs ORIG (descriptive); P05 named in each.
    mem-guard python scripts/local/t2_d529_read.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head (S, load, realign, parts_of, FIN), side_files, score, part, DC, DCVEC


DRY = os.environ.get("D529_DRY") == "1"      # code check only: the MI members are read as ORIG (no D529 number)


def member(kind, s, f):
    kind = "orig" if DRY and kind == "mi" else kind
    h = "a" if f < 5 else "b"
    run = {"orig": f"ni26-t2p-d512-{'stlcartl' if s >= 823 else 'stltl'}_s{s}_{h}",
           "mi": f"ni26-t2p-d529-mi_s{s}_{h}",
           "ctl": f"ni26-t2p-d524-stlctl_s{s}_{h}"}[kind]
    fs = glob.glob(f"{S}/{run}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
    assert len(fs) == 1, (kind, s, f, fs)
    return load(fs[0])


ARMS = {"orig": {}, "mi1": {820: "mi"}, "ctl1": {820: "ctl"}, "mi2": {820: "mi", 821: "mi"}}
if not DRY and not all(glob.glob(f"{S}/ni26-t2p-d529-mi_s821_{'a' if f < 5 else 'b'}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s821.npz")
                      for f in range(10)):
    ARMS.pop("mi2")                      # the descriptive mi2 waits for the s821 members; the decision (mi1 vs ctl1) does not
    print("D529: mi2 deferred (the MI s821 members are not all in yet)", flush=True)


def fm_arm(arm, f):
    ps, ixs = [], None
    for s in range(820, 826):
        ix_, p = member(ARMS[arm].get(s, "orig"), s, f)
        if ixs is not None:
            p = realign(ix_, p, ixs)
        else:
            ixs = ix_
        ps.append(p)
    return ixs, np.mean(ps, 0)


FMA = {arm: {f: fm_arm(arm, f) for f in range(10)} for arm in ARMS}
# check: this read's ORIG == the head's FIN FM part (D512's tilt six) on every fold and draw
for (f, d), (ix, q) in RAW.items():
    err = np.abs(realign(*FMA["orig"][f], ix) - parts_of(q, FIN)["fm"]).max()
    assert err < 1e-12, (f, d, err)
print("D529 check: ORIG == the W17 head's FM part (max abs err < 1e-12)", flush=True)


def mix(fm, w_dc=0.075, side=True):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        Pp["fm"] = realign(*FMA[fm][f], ix)
        if side:
            for fam in ("sh", "ifn"):
                S_ = np.mean([pooled(p, ix) for p in side_files("S5", fam, f, d)], 0)
                Pp[fam] = (2.0 / 3.0) * Pp[fam] + (1.0 / 3.0) * S_
        P[(f, d)] = Pp
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


def per_stranger(A, B):
    """each stranger's paired cell-BA delta A - B (mean over draws and cells), as score() pools them."""
    recs = []
    for (f, d), (ix, ma) in A.items():
        mb = B[(f, d)][1]
        m = U[ix] == TRU[f]
        a, b = cells(ma.argmax(1)[m], ix[m]), cells(mb.argmax(1)[m], ix[m])
        recs += [dict(draw=d, user=TRU[f], cell=c, d=a[c] - b[c]) for c in a]
    return pd.DataFrame(recs).groupby("user").d.mean().round(4).to_dict()


def part(e):
    """the part rule with the final-week floor (as t2_d516b_read.part)."""
    return e["ten"] > 0 and sum(x > 0 for x in e["draws"]) >= 2 and e["min_user"] >= -0.020 and (e["returning"] is None or e["returning"] >= -0.003)


M = {arm: mix(arm) for arm in ARMS}
res = {}
for a, b in [c for c in (("mi1", "ctl1"), ("mi1", "orig"), ("ctl1", "orig"), ("mi2", "orig"), ("mi2", "mi1")) if c[0] in ARMS]:   # mi2 vs mi1 = the one-vs-two rule (a), registered 09:0x
    e = score(M[a], M[b])
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    e["part"] = bool(part(e))
    pu = per_stranger(M[a], M[b])
    e["P05"] = pu.get([u for u in pu if u.endswith("P05") or u == "P05"][0]) if any(u.endswith("P05") for u in pu) else None
    e["per_stranger"] = pu
    res[f"W17 stack: FM {a} vs FM {b}"] = e
    print(f"D529 W17 stack: FM {a:4s} vs FM {b:4s}: ten {e['ten']:+.4f} draws {e['draws']} six {e['six']:+.4f} "
          f"min {e['min_user']:+.4f} ({e['min_who']}) returning {e['returning']:+.4f} sealed mix {e['sealed_mix']:+.4f} "
          f"P05 {e['P05']} part {e['part']}", flush=True)
    print(f"    per stranger: {pu}", flush=True)
if not DRY:
    json.dump(res, open(O_ / ("t2_d529.json" if "mi2" in ARMS else "t2_d529_decision.json"), "w"), indent=1, default=str)
