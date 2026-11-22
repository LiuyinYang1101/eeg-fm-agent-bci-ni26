#!/usr/bin/env python
"""T2-D530 read (written 2026-11-16 before any D530 student number; CPU): ShallowNet on NESTED out-of-fold cross-user soft targets, in W17
on D512's K = 9 draws 810-812 (ten strangers every window, six, returning).
  REF = W17 = FINAL + DC 0.075 + S5 side nets 'thirds' (asserted equal to t2_d518_read.mix_arm('S5')).
  KD  = the same with the sh slot's routed pooled / copy part (outputs/t2_d530/route_shkd) and its P1 in-stream part
        (outputs/t2_d530/p1_shkd, isSpr) taken from the kd students; the S5 side nets unchanged.
Decision: score(KD, REF), part rule; sealed mix 0.57 x strangers + 0.43 x returning. Descriptive: the sh slot alone (routed + P1, before the
side nets), decoded with its own causal Sinkhorn, kd vs control.
    mem-guard python scripts/local/t2_d530_read.py
"""
import json

import numpy as np

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head (RAW, parts_of, FIN, W, ...), side_files, pooled, score, DC, DCVEC, mix_arm, level

D30 = O_ / "t2_d530"
KDR = "route_shkd/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdtv8cfb2__s{s}.npz"
KDP1 = "p1_shkd/runs/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdisSpr__s{s}.npz"
miss = [p for (f, d) in RAW for p in (D30 / KDR.format(f=f, s=d), D30 / KDP1.format(f=f, s=d)) if not p.exists()]
assert not miss, f"{len(miss)} D530 student files missing, e.g. {miss[0]}"


def q_kd(f, d):
    """RAW's q for (f, d) with the sh slot's routed part (P3 off = seed d alone, as FIN) and its P1 'isSpr' part from the kd students."""
    ix, q = RAW[(f, d)]
    q2 = dict(q)
    q2["sh"] = dict(q["sh"])
    q2["sh"][False] = realign(*load(D30 / KDR.format(f=f, s=d)), ix)
    q2["p1_sh"] = dict(q["p1_sh"])
    q2["p1_sh"]["isSpr"] = realign(*load(D30 / KDP1.format(f=f, s=d)), ix)
    return q2


def mix(kd):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q_kd(f, d) if kd else q, FIN)
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
        te, p = DC[f]
        pos = {int(r): k for k, r in enumerate(te)}
        q_ = p[np.array([pos[int(r)] for r in ix])]
        mx = 0.925 * mx + 0.075 * causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCVEC[f])
        out[(f, d)] = (ix, mx)
    return out


def sh_alone(kd):
    """the sh slot alone (routed + P1 blend, before the side nets), decoded, start = the other folds' stranger mean."""
    P = {(f, d): parts_of(q_kd(f, d) if kd else q, FIN)["sh"] for (f, d), (ix, q) in RAW.items()}
    out = {}
    for d in DRAWS:
        per = {f: P[(f, d)][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
        for f in range(10):
            ix = RAW[(f, d)][0]
            v = np.mean([per[g] for g in range(10) if g != f], 0)
            out[(f, d)] = (ix, causal_sinkhorn(P[(f, d)], (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=v / v.sum()))
    return out


REF = mix(False)
chk = mix_arm("S5")
dmax = max(float(np.abs(REF[k][1] - chk[k][1]).max()) for k in REF)
assert dmax < 1e-9, f"REF != W17 (mix_arm S5): {dmax}"
print(f"[check] REF == W17: max |dp| {dmax:.1e}; level {level(REF)}", flush=True)
KD = mix(True)
e = score(KD, REF)
e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
res = {"D530 W17 with the kd ShallowNet slot vs W17": e}
print(f"D530 W17 with the kd ShallowNet slot vs W17: {e}", flush=True)
A0, A1 = sh_alone(False), sh_alone(True)
res["D530 sh slot alone: control level"], res["D530 sh slot alone: kd level"] = level(A0), level(A1)
res["D530 sh slot alone: kd vs control (descriptive)"] = score(A1, A0)
print(f"D530 sh slot alone (descriptive): control {level(A0)}, kd {level(A1)}; kd vs control {res['D530 sh slot alone: kd vs control (descriptive)']}", flush=True)
json.dump(res, open(O_ / "t2_d530.json", "w"), indent=1, default=str)
