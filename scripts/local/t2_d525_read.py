#!/usr/bin/env python
"""T2-D525 / D526 / D527 / D528 reads (written 2026-11-16 before any of their numbers; CPU, zero GPU): closed-with-positive rows re-read
in W17's context on D512's K = 9 draws 810-812 (ten strangers every window, six, returning). REF = W17 = FINAL + DC 0.075 + S5 side
nets 'thirds' (asserted equal to t2_d518_read.mix_arm('S5')).
  D525: + D487's imcoh family (outputs/t2_d525/pred_imcoh.npz) at w 0.03 (primary), 0.02 / 0.05; alone level, error correlation.
  D526: the family shares re-fitted on an 80-point grid, cross-fitted by stranger halves (folds 0-4 <-> 5-9).
  D527: a per-stream gate between the joint decode (J) and W17's per-family decode (Fs), g2 from the first 20 windows, LOSO (theta, dir).
  D528: the FM part tempered (T 2 primary, 1.5 descriptive) before its corrector.
Part rule; sealed mix 0.57 x strangers + 0.43 x returning.
    mem-guard python scripts/local/t2_d525_read.py [--rows D525,D526,D527,D528]
"""
import argparse
import itertools
import json

import numpy as np

src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
exec(compile(src, "t2_d518_read_head", "exec"))     # D512 head, side_files, pooled, score, cells, DC, DCVEC, mix_arm, level, err_corr

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="D525,D526,D527,D528")
ROWS = ap.parse_args().rows.split(",")
FAMS = list(W)
W_DC = 0.075


def base_parts(temp_fm=None):
    P = {}
    for (f, d), (ix, q) in RAW.items():
        Pp = parts_of(q, FIN)
        for fam in ("sh", "ifn"):
            S_ = np.mean([pooled(p, ix) for p in side_files("S5", fam, f, d)], 0)
            Pp[fam] = (2.0 / 3.0) * Pp[fam] + (1.0 / 3.0) * S_
        if temp_fm:
            z = Pp["fm"] ** (1.0 / temp_fm)
            Pp["fm"] = z / z.sum(1, keepdims=True)
        P[(f, d)] = Pp
    return P


def xvec(P, fam):
    """cross-fitted start vectors: for fold f, the other nine folds' stranger mean of P[(g, d)][fam]."""
    vec = {}
    for d in DRAWS:
        per = {f: P[(f, d)][fam][U[RAW[(f, d)][0]] == TRU[f]].mean(0) for f in range(10)}
        for f in range(10):
            v = np.mean([per[g] for g in range(10) if g != f], 0)
            vec[(f, d)] = v / v.sum()
    return vec


def decoded(P):
    V = {fam: xvec(P, fam) for fam in FAMS}
    return {(f, d): {fam: causal_sinkhorn(P[(f, d)][fam], (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=V[fam][(f, d)]) for fam in FAMS}
            for (f, d), (ix, q) in RAW.items()}


def family_decoded(path):
    """an extra family's per-fold probabilities (te_<f>, bag_<f>) -> {(f, d): decoded rows on RAW's order}, start cross-fitted (as DCVEC)."""
    z = np.load(path)
    F_ = {f: (z[f"te_{f}"], z[f"bag_{f}"]) for f in range(10)}
    V = {}
    for f in range(10):
        v = np.mean([F_[g][1][U[F_[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
        V[f] = v / v.sum()
    out = {}
    for (f, d), (ix, q) in RAW.items():
        te, p = F_[f]
        pos = {int(r): k for k, r in enumerate(te)}
        q_ = p[np.array([pos[int(r)] for r in ix])]
        out[(f, d)] = causal_sinkhorn(q_ / q_.sum(1, keepdims=True), (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=V[f])
    return out


DCD = {}
for (f, d), (ix, q) in RAW.items():
    te, p = DC[f]
    pos = {int(r): k for k, r in enumerate(te)}
    q_ = p[np.array([pos[int(r)] for r in ix])]
    DCD[(f, d)] = causal_sinkhorn(q_ / q_.sum(1, keepdims=True), (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=DCVEC[f])


def combine(Dd, Wt, extra=None, folds=None):
    out = {}
    for (f, d), (ix, q) in RAW.items():
        if folds is not None and f not in folds:
            continue
        mx = sum(Wt[fam] * Dd[(f, d)][fam] for fam in FAMS)
        mx = (1 - W_DC) * mx + W_DC * DCD[(f, d)]
        if extra is not None:
            w, E = extra
            mx = (1 - w) * mx + w * E[(f, d)]
        out[(f, d)] = (ix, mx)
    return out


def sm(e):
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    return e


P0 = base_parts()
D0 = decoded(P0)
REF = combine(D0, W)
chk = mix_arm("S5")
dmax = max(float(np.abs(REF[k][1] - chk[k][1]).max()) for k in REF)
assert dmax < 1e-9, f"REF != W17 (mix_arm S5): {dmax}"
print(f"[check] REF == W17 (t2_d518 mix_arm S5): max |dp| {dmax:.1e}; W17 ten-stranger level {level(REF)}", flush=True)
res = {"REF level": level(REF)}

# cell-BA cache for the stranger rows of each (f, d): base = REF
STR = {(f, d): (U[ix] == TRU[f]) for (f, d), (ix, q) in RAW.items()}


def ucell(pred, f, d):
    ix = RAW[(f, d)][0]
    m = STR[(f, d)]
    return cells(pred[m], ix[m])


BASEC = {(f, d): ucell(REF[(f, d)][1].argmax(1), f, d) for (f, d) in RAW}


def stranger_delta(A, folds):
    """mean over the folds' strangers of (mean over draws and cells of A - REF)."""
    per = []
    for f in folds:
        recs = []
        for d in DRAWS:
            a, b = ucell(A[(f, d)][1].argmax(1), f, d), BASEC[(f, d)]
            recs += [a[c] - b[c] for c in a]
        per.append(np.mean(recs))
    return float(np.mean(per))


if "D525" in ROWS:
    IC = family_decoded(O_ / "t2_d525/pred_imcoh.npz")
    AL = {k: (RAW[k][0], v) for k, v in IC.items()}
    res["D525 imcoh alone level"] = level(AL)
    res["D525 imcoh alone err vs W17"] = err_corr(AL, REF)
    print(f"D525 imcoh alone: level {res['D525 imcoh alone level']}; err vs W17 {res['D525 imcoh alone err vs W17']}", flush=True)
    for w in (0.03, 0.02, 0.05):
        e = sm(score(combine(D0, W, extra=(w, IC)), REF))
        res[f"D525 W17 + imcoh @{w} vs W17"] = e
        print(f"D525 W17 + imcoh @{w} vs W17: {e}", flush=True)

if "D526" in ROWS:
    ship = np.array([W["fbc"], W["sh"], W["ifn"], W["fm"]])
    GRID = []
    for fm, ifn, shf in itertools.product((0.25, 0.30, 0.35, 0.40, 0.45), (0.05, 0.10, 0.15, 0.20), (0.35, W["sh"] / (W["fbc"] + W["sh"]), 0.50, 0.57)):
        rest = 1.0 - fm - ifn
        GRID.append({"fbc": rest * (1 - shf), "sh": rest * shf, "ifn": ifn, "fm": fm})
    assert len(GRID) == 80
    HALF = {"A": range(0, 5), "B": range(5, 10)}
    picks, tables = {}, {}
    for h, fl in HALF.items():
        sc = []
        for g in GRID:
            A = combine(D0, g, folds=set(fl))
            dist = float(np.abs(np.array([g["fbc"], g["sh"], g["ifn"], g["fm"]]) - ship).sum())
            sc.append((stranger_delta(A, fl), -dist, g))
        sc.sort(key=lambda t: (t[0], t[1]), reverse=True)
        picks[h] = sc[0][2]
        tables[h] = [dict(delta=round(s, 4), **{k: round(v, 4) for k, v in g.items()}) for s, _, g in sc[:5]]
        print(f"D526 pick on half {h} (folds {list(fl)}): {tables[h][0]}; top 5 {tables[h]}", flush=True)
    A0, A1 = combine(D0, picks["B"], folds=set(HALF["A"])), combine(D0, picks["A"], folds=set(HALF["B"]))
    ACF = {**A0, **A1}
    e = sm(score(ACF, REF))
    res["D526 picks"] = {h: {k: round(v, 4) for k, v in g.items()} for h, g in picks.items()}
    res["D526 top5 per half (in-half, selection)"] = tables
    res["D526 cross-fitted shares vs W17"] = e
    print(f"D526 cross-fitted shares vs W17: {e}", flush=True)

if "D527" in ROWS:
    VJ = {}
    PJ = {k: {"pool": sum(W[fam] * P0[k][fam] for fam in FAMS)} for k in RAW}
    VJ = xvec(PJ, "pool")
    J = {}
    for (f, d), (ix, q) in RAW.items():
        j0 = causal_sinkhorn(PJ[(f, d)]["pool"], (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=VJ[(f, d)])
        J[(f, d)] = (ix, (1 - W_DC) * j0 + W_DC * DCD[(f, d)])
    eJ = sm(score(J, REF))
    res["D527 J (joint decode) vs Fs (W17), descriptive"] = eJ
    print(f"D527 J vs Fs (descriptive): {eJ}", flush=True)
    # per stream: position and g2 from its first 20 windows (the four families' decoded rows)
    POS, G2, SID = {}, {}, {}
    for (f, d), (ix, q) in RAW.items():
        pos = np.zeros(len(ix), int)
        g2 = {}
        for s in np.unique(STREAM[ix]):
            rows = np.flatnonzero(STREAM[ix] == s)
            rows = rows[np.argsort(ORD[ix][rows], kind="stable")]
            pos[rows] = np.arange(len(rows))
            first = rows[:20]
            M = np.stack([D0[(f, d)][fam][first].mean(0) for fam in FAMS])       # (4, 3)
            g2[s] = float((M.max(0) - M.min(0)).max())
        POS[(f, d)], G2[(f, d)] = pos, g2
    def gated(f, d, theta, sgn, rows_mask=None):
        ix, fs = REF[(f, d)]
        st = STREAM[ix]
        use = np.array([sgn * G2[(f, d)][s] > sgn * theta for s in st]) & (POS[(f, d)] >= 20)
        if rows_mask is not None:
            use &= rows_mask
        return np.where(use[:, None], J[(f, d)][1], fs)
    def user_delta(f, theta, sgn):
        recs = []
        for d in DRAWS:
            a, b = ucell(gated(f, d, theta, sgn).argmax(1), f, d), BASEC[(f, d)]
            recs += [a[c] - b[c] for c in a]
        return float(np.mean(recs))
    def grid_for(folds):
        vals = [G2[(f, d)][s] for f in folds for d in DRAWS for s in np.unique(STREAM[RAW[(f, d)][0]][STR[(f, d)]])]
        return [(float(np.quantile(vals, q / 100)), sgn) for q in range(5, 100, 5) for sgn in (+1, -1)]
    UD = {}
    def fit(folds):
        best = None
        for th, sgn in grid_for(folds):
            v = float(np.mean([UD.setdefault((f, th, sgn), user_delta(f, th, sgn)) for f in folds]))
            if best is None or v > best[0]:
                best = (v, th, sgn)
        return best
    allpick = fit(range(10))
    AG, lopo = {}, {}
    for f in range(10):
        v, th, sgn = fit([g for g in range(10) if g != f])
        lopo[TRU[f]] = dict(theta=round(th, 4), dir=sgn, fit_delta=round(v, 4))
        for d in DRAWS:
            ix = RAW[(f, d)][0]
            g_all = gated(f, d, allpick[1], allpick[2])                  # non-stranger rows: the all-ten pick
            g_str = gated(f, d, th, sgn)                                  # stranger rows: the nine's pick
            AG[(f, d)] = (ix, np.where(STR[(f, d)][:, None], g_str, g_all))
    e = sm(score(AG, REF))
    res["D527 gate picks (LOSO per stranger)"] = lopo
    res["D527 all-ten pick (returning rows)"] = dict(theta=round(allpick[1], 4), dir=allpick[2], fit_delta=round(allpick[0], 4))
    res["D527 gated vs Fs (W17)"] = e
    print(f"D527 LOSO picks {lopo}; all-ten {res['D527 all-ten pick (returning rows)']}", flush=True)
    print(f"D527 gated vs Fs (W17): {e}", flush=True)

if "D528" in ROWS:
    for T in (2.0, 1.5):
        e = sm(score(combine(decoded(base_parts(temp_fm=T)), W), REF))
        res[f"D528 FM temperature T {T} vs W17"] = e
        print(f"D528 FM temperature T {T} vs W17: {e}", flush=True)

json.dump(res, open(O_ / f"t2_d525_{'_'.join(ROWS)}.json", "w"), indent=1, default=str)
