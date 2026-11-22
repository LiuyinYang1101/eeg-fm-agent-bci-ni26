#!/usr/bin/env python
"""T2-D524 read (written 2026-11-15 before any D524 number): the window means INTO the FM members. On D512's K = 9 draws 810-812, the FM slot (share
0.35) = the mean of the six tilt ST-L members (3 plain 820-822 + 3 CAR 823-825):
  SD   = the six with the window-mean side head (t2p_bench --side dcmean --fm-side dcmean);
  CTL  = the six RETRAINED without it on the same seeds (the retrain control);
  ORIG = D512's own tilt members (the shipped FM slot's read; CTL vs ORIG = the retrain noise floor).
Stacks: W17 = FINAL + DC tree 0.075 + side nets S5 'thirds' with each FM slot. Reads: SD vs CTL (the decision, part rule); SD vs ORIG, CTL vs ORIG;
the overlap: the DC tree's gain (W17 vs W17 without the tree) and the side nets' gain (W17 vs W17 without the side nets), each with SD and with CTL.
    mem-guard python scripts/local/t2_d524_read.py
"""
import glob
import json

import numpy as np

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head (S, load, realign, parts_of, FIN), side_files, score, DC, DCVEC


def fm_arm(arm, f):
    ps, ixs = [], None
    for s in range(820, 826):
        base = "stlcar" if s >= 823 else "stl"
        a_ = ("sd" if s < 823 else "ctl") if arm == "plain3" else arm      # the registered fallback (2026-11-16): the head on the plain three only
        fs = glob.glob(f"{S}/ni26-t2p-d524-{base}{a_}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
        assert len(fs) == 1, (arm, s, f, fs)
        ix_, p = load(fs[0])
        assert ixs is None or (ix_ == ixs).all()
        ixs = ix_
        ps.append(p)
    return ixs, np.mean(ps, 0)


FMA = {arm: {f: fm_arm(arm, f) for f in range(10)} for arm in ("sd", "ctl", "plain3")}


def mix(fm, w_dc=0.075, side=True):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        if fm != "orig":
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


M = {fm: mix(fm) for fm in ("sd", "ctl", "orig", "plain3")}
res = {}
for a, b in (("sd", "ctl"), ("sd", "orig"), ("ctl", "orig"), ("plain3", "ctl")):      # plain3 vs ctl = the ONE registered fallback
    e = score(M[a], M[b])
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    res[f"W17 stack: FM {a} vs FM {b}"] = e
    print(f"D524 W17 stack: FM {a:4s} vs FM {b:4s}: {e}", flush=True)
for fm in ("sd", "ctl"):
    e_tree = score(M[fm], mix(fm, w_dc=0.0))
    e_side = score(M[fm], mix(fm, side=False))
    res[f"FM {fm}: the DC tree's gain (W17 vs no tree)"], res[f"FM {fm}: the side nets' gain (W17 vs no side nets)"] = e_tree, e_side
    print(f"D524 FM {fm:4s}: the DC tree's gain {e_tree}", flush=True)
    print(f"D524 FM {fm:4s}: the side nets' gain {e_side}", flush=True)
# P07 named (registered 2026-11-16): each stranger's mean |z| of the side vector under its fold's training prior (how far its window means sit)
sys.path.insert(0, str(Path("tracks/t2_bci/scripts").resolve()))
import t2p_bench as B_  # noqa: E402
Xe, idxe, _ = B_.load_export(str(EXP))
Ue = idxe["subject"].astype(str).str[-3:].to_numpy()
Fr_ = B_.side_raw(Xe, "dcmean")
zabs = {}
for f in range(10):
    ((_, trm_, _, _),) = B_.folds(idxe, f"lc9_{f}")
    Z_ = B_.side_z(Fr_, idxe, np.flatnonzero(trm_), "dcmean")
    zabs[TRU[f]] = round(float(np.abs(Z_[Ue == TRU[f]]).mean()), 3)
res["stranger mean |z| of the side vector (fold prior)"] = zabs
print(f"D524 stranger mean |z| (side vector, fold prior): {zabs}", flush=True)
json.dump(res, open(O_ / "t2_d524.json", "w"), indent=1, default=str)
