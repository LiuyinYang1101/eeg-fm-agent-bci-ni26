#!/usr/bin/env python
"""T2-D532 read: THE W18 FRESH-DRAW STACK READ (written 2026-11-16 before any fresh-draw number; CPU, zero GPU). Per lc9 fold f x CNN draw d:
  BASE = W17 on the draws: FINAL (P1: the sh / ifn slots = 1/2 routed member + 1/2 the in-stream adapted pooled net, isSpr; P2: the tilt FM =
         the mean of six ST-L members) + the S5 side nets 'thirds' (sh / ifn part = 2/3 FINAL part + 1/3 the mean of 5 side nets, seeds
         d + k * 1000, k 0..4) + the DC tree family at 0.075 (the 82 window means' LightGBM bag); each family decoded by the causal Sinkhorn
         (10 / 3 / 0.95) from ITS cross-fitted start vector (the other folds' stranger mean, per variant), pooled at M7's shares.
  fresh draws (default): CNN 813-815 (outputs/t2_d532: route_*, p1_*, side_*), DC bag seeds 4-6 (outputs/t2_d532/pred_s4-5-6_dc.npz),
         FM = D531's CTL six at 826-831 (sofia run dirs ni26-t2p-d531-stl{,car}ctl_s<seed>_{a,b}-<id>, fetched to outputs/sofia_s1).
  --check: the OLD draws 810-812 with W17's own sources (t2_d512 / t2_d517 / t2_d518, DC pred_s1-2-3_dm.npz, FM = D512's tilt 820-825);
         BASE must equal t2_d518_read.mix_arm('S5') exactly (the reproduction check of this file's loaders and arithmetic).
  --parts (only the parts whose nominating read passed): D524 = the FM slot from D531's SD members (--d524-form six | plain3 = SD 826-828 +
         CTL 829-831); D525 = the imcoh family (outputs/t2_d525/pred_imcoh.npz, seed-free) at 0.03 after the DC family; D529 = the MI-expert
         member at 826 (ni26-t2p-d532-mi_s826_{a,b}-<id>) in place of the s826 member; D530 = the kd ShallowNet (t2_d532/route_shkd and
         t2_d532/p1_shkd) in the sh slot's routed and P1 parts.
Reads: FULL (BASE + every listed part) vs BASE; leave-one-out FULL vs FULL minus each part (KEPT iff the part rule: ten-stranger mean > 0,
>= 2 / 3 draws, no stranger < -0.020, returning >= -0.003); W18-final = BASE + the kept parts vs BASE, once. Sealed mix 0.57 x strangers +
0.43 x returning. Any missing file fails loudly.
    mem-guard python scripts/local/t2_d532_read.py --parts D524,D525,D529,D530 [--d524-form six|plain3]
    mem-guard python scripts/local/t2_d532_read.py --check
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402
from t2_oracle_guard import refuse_oracle  # noqa: E402  (T2-D550: true-label oracle outputs are refused)

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="", help="comma list of the parts that entered: D524, D525, D529, D530")
ap.add_argument("--d524-form", default="six", choices=["six", "plain3"])
ap.add_argument("--check", action="store_true", help="old draws 810-812 with W17's sources; BASE must equal t2_d518 mix_arm('S5')")
ap.add_argument("--d529b", action="store_true", help="T2-D529 rule (b), registered 2026-11-16 09:0x: after the main reads, if D529 is kept, the "
                "second MI member's increment = W18-final with MI s827 in place of the s827 member vs W18-final (ni26-t2p-d529b-mi_s827_{a,b}-<id>)")
A = ap.parse_args()
PARTS = [p for p in A.parts.split(",") if p]
assert set(PARTS) <= {"D524", "D525", "D529", "D530"}, PARTS
assert not (A.check and PARTS), "--check reads BASE only"

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
idx = pd.read_csv(EXP / "index.csv")
U, SES, STREAM, ORD, Y, CELL = (idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy(), idx.cell.to_numpy())
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
R1, W_DC, W_IC = 0.5, 0.075, 0.03
NC6 = ("P01", "P03", "P05", "P07", "P09", "P10")
REC = {"sh": ("shallow", "r1_pulx_vp2_e180_vr"), "ifn": ("ifnet", "r1_pulx_vp2_e180_vr")}
RT = {"fbc": "route_fbc/lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz",
      "sh": "route_sh/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
      "ifn": "route_ifn/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
      "shkd": "route_shkd/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdtv8cfb2__s{s}.npz"}
P1R = {"sh": "p1_sh/runs/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrisSpr__s{s}.npz",
       "ifn": "p1_ifn/runs/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrisSpr__s{s}.npz",
       "shkd": "p1_shkd/runs/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdisSpr__s{s}.npz"}
if A.check:
    DRAWS, CNN, DCF = (810, 811, 812), O / "t2_d512", O / "t2_d515/pred_s1-2-3_dm.npz"
else:
    DRAWS, CNN, DCF = (813, 814, 815), O / "t2_d532", O / "t2_d532/pred_s4-5-6_dc.npz"


def need(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"!! T2-D532: missing {path}")
    refuse_oracle(path)                                  # T2-D550: the name
    return path


def one(pattern):
    fs = glob.glob(str(pattern))
    if len(fs) != 1:
        raise SystemExit(f"!! T2-D532: {len(fs)} files for {pattern} (need exactly 1)")
    return Path(fs[0])


def load(path):
    z = np.load(need(path))
    refuse_oracle(path, z)                               # T2-D550: the manifest (res.twin_suffix)
    p = z["prob_last"].astype(np.float64)
    return z["test_idx"], p / p.sum(1, keepdims=True)


def realign(ix_from, p, ix_to):
    pos = {int(r): k for k, r in enumerate(ix_from)}
    return p[np.array([pos[int(r)] for r in ix_to])]


def side_files(fam, f, d):
    m, rec = REC[fam]
    stem = f"lc9_{f}__{m}__ea__n00__{rec}__sidedcmean__s"
    if A.check:                                     # W17's sources: D517's seed-d net + D518's S5 seeds d + 1000 ... d + 4000
        return [O / f"t2_d517/side_dcmean_{fam}/runs/{stem}{d}.npz"] + [O / f"t2_d518/S5_{fam}/runs/{stem}{d + k * 1000}.npz" for k in range(1, 5)]
    return [CNN / f"side_{fam}/runs/{stem}{d + k * 1000}.npz" for k in range(5)]


def fm_member(s, f, form):
    """one FM member's file for fold f: form 'orig' = D512's tilt (W17), 'ctl' / 'sd' = D531's fresh lines, 'mi' = D532's MI member."""
    h = "a" if f < 5 else "b"
    stem = f"runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz"
    if form == "orig":
        return one(f"{S}/ni26-t2p-d512-{'stlcar' if s >= 823 else 'stl'}tl_s{s}_{h}-[0-9]*/{stem}")
    if form == "mi":
        return one(f"{S}/ni26-t2p-{'d529b' if s == 827 else 'd532'}-mi_s{s}_{h}-[0-9]*/{stem}")
    return one(f"{S}/ni26-t2p-d531-{'stlcar' if s >= 829 else 'stl'}{form}_s{s}_{h}-[0-9]*/{stem}")


def fm_forms(parts):
    """the FM slot's six (seed, form) pairs for a set of parts."""
    if A.check:
        return [(s, "orig") for s in range(820, 826)]
    seeds = list(range(826, 832))
    if "D524" in parts:
        forms = {s: ("sd" if (A.d524_form == "six" or s <= 828) else "ctl") for s in seeds}
    else:
        forms = {s: "ctl" for s in seeds}
    if "D529" in parts:
        forms[826] = "mi"
    if "D529b" in parts:                    # T2-D529 rule (b): the second MI member, in place of the plain s827 member
        assert "D529" in parts
        forms[827] = "mi"
    return [(s, forms[s]) for s in seeds]


# the FM slot per (fold, form-set) and the fold's test order (= the FM test order, as T2-D512's RAW)
_FMC = {}


def fm_slot(f, parts):
    key = (f, tuple(fm_forms(parts)))
    if key not in _FMC:
        ps, ix = [], None
        for s, form in key[1]:
            ix_, p = load(fm_member(s, f, form))
            assert ix is None or (ix_ == ix).all(), (f, s, form)
            ix = ix_
            ps.append(p)
        _FMC[key] = (ix, np.mean(ps, 0))
    return _FMC[key]


IX = {f: fm_slot(f, [])[0] for f in range(10)}
TRU = {f: [u for u in set(U[IX[f]]) if (U[IX[f]] == u).sum() == (U == u).sum()][0] for f in range(10)}
zp = np.load(need(DCF))
DC = {f: (zp[f"te_{f}"], zp[f"bag_{f}"]) for f in range(10)}
DCVEC = {}
for f in range(10):
    v = np.mean([DC[g][1][U[DC[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
    DCVEC[f] = v / v.sum()
if "D525" in PARTS:
    zi = np.load(need(O / "t2_d525/pred_imcoh.npz"))
    IC = {f: (zi[f"te_{f}"], zi[f"bag_{f}"]) for f in range(10)}
    ICVEC = {}
    for f in range(10):
        v = np.mean([IC[g][1][U[IC[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
        ICVEC[f] = v / v.sum()


def parts_of(f, d, parts):
    """the four families' raw parts (before their correctors) on IX[f]'s order."""
    ix = IX[f]
    shf = "shkd" if "D530" in parts else "sh"
    P = {"fbc": np.mean([realign(*load(CNN / RT["fbc"].format(f=f, s=d)), ix)], 0)}
    for fam, src in (("sh", shf), ("ifn", "ifn")):
        r = np.mean([realign(*load(CNN / RT[src].format(f=f, s=d)), ix)], 0)
        a = realign(*load(CNN / P1R[src].format(f=f, s=d)), ix)
        part = (1 - R1) * r + R1 * a                                       # P1 (adapt)
        S_ = np.mean([realign(*load(p), ix) for p in side_files(fam, f, d)], 0)
        P[fam] = (2.0 / 3.0) * part + (1.0 / 3.0) * S_                    # the side nets, 'thirds'
    ix_, pfm = fm_slot(f, parts)
    assert (ix_ == ix).all()
    P["fm"] = pfm
    return {k: P[k] for k in ("fm", "fbc", "sh", "ifn")}


def mix(parts):
    P = {(f, d): parts_of(f, d, parts) for f in range(10) for d in DRAWS}
    vec = {}
    for d in DRAWS:
        for fam in W:
            per = {f: P[(f, d)][fam][U[IX[f]] == TRU[f]].mean(0) for f in range(10)}
            for f in range(10):
                v = np.mean([per[g] for g in range(10) if g != f], 0)
                vec[(f, d, fam)] = v / v.sum()
    out = {}
    for f in range(10):
        ix = IX[f]
        key = (STREAM[ix], ORD[ix])
        te, p = DC[f]
        pos = {int(r): k for k, r in enumerate(te)}
        q_ = p[np.array([pos[int(r)] for r in ix])]
        dcd = causal_sinkhorn(q_ / q_.sum(1, keepdims=True), key, 10, 3, 0.95, start=DCVEC[f])
        icd = None
        if "D525" in parts:
            te_i, p_i = IC[f]
            pos_i = {int(r): k for k, r in enumerate(te_i)}
            qi = p_i[np.array([pos_i[int(r)] for r in ix])]
            icd = causal_sinkhorn(qi / qi.sum(1, keepdims=True), key, 10, 3, 0.95, start=ICVEC[f])
        for d in DRAWS:
            mx = sum(W[fam] * causal_sinkhorn(P[(f, d)][fam], key, 10, 3, 0.95, start=vec[(f, d, fam)]) for fam in W)
            mx = (1 - W_DC) * mx + W_DC * dcd
            if icd is not None:
                mx = (1 - W_IC) * mx + W_IC * icd
            out[(f, d)] = (ix, mx)
    return out


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        ks = np.unique(Y[rows][m])
        if len(ks) >= 2:
            out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
    return out


def score(Aa, Bb):
    recs = []
    for (f, d), (ix, ma) in Aa.items():
        mb = Bb[(f, d)][1]
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
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    return e


def level(Aa):
    per = {}
    for (f, d), (ix, mx) in Aa.items():
        m = U[ix] == TRU[f]
        c = cells(mx.argmax(1)[m], ix[m])
        per.setdefault(TRU[f], []).append(np.mean(list(c.values())))
    return round(float(np.mean([np.mean(v) for v in per.values()])), 4)


BASE = mix([])
res = {"draws": list(DRAWS), "BASE level": level(BASE)}
if A.check:
    ns = {}
    src = open("scripts/local/t2_d518_read.py").read().split("\nap = argparse.ArgumentParser()\n")[0]
    exec(compile(src, "t2_d518_read_head", "exec"), ns)
    REF = ns["mix_arm"]("S5")
    assert set(REF) == set(BASE), (sorted(REF)[:3], sorted(BASE)[:3])
    dmax = max(float(np.abs(BASE[k][1] - REF[k][1]).max()) for k in REF)
    same_ix = all((BASE[k][0] == REF[k][0]).all() for k in REF)
    print(f"[check] old draws 810-812, W17's sources: BASE vs t2_d518 mix_arm('S5'): max |dp| {dmax:.1e}, test orders equal {same_ix}; level {res['BASE level']}",
          flush=True)
    assert same_ix and dmax == 0.0, f"T2-D532 reproduction check FAILED: max |dp| {dmax}"
    res["check"] = dict(max_abs_dp=dmax, ok=True)
    print("[check] OK", flush=True)
    json.dump(res, open(O / "t2_d532_check.json", "w"), indent=1, default=str)
    raise SystemExit(0)
print(f"D532 fresh draws {DRAWS}: BASE (W17) ten-stranger level {res['BASE level']}; parts {PARTS} (D524 form {A.d524_form})", flush=True)
if not PARTS:
    raise SystemExit("D532: no part entered -> nothing to read (BASE only)")
FULL = mix(PARTS)
res["FULL vs BASE"] = score(FULL, BASE)
print(f"D532 FULL ({'+'.join(PARTS)}) vs BASE: {res['FULL vs BASE']}", flush=True)
kept = []
for p in PARTS:
    rest = [q for q in PARTS if q != p]
    e = score(FULL, mix(rest) if rest else BASE)
    res[f"LOO {p}: FULL vs FULL minus {p}"] = e
    print(f"D532 leave-one-out {p}: FULL vs FULL - {p}: {e} -> {'KEPT' if e['part'] else 'dropped'}", flush=True)
    if e["part"]:
        kept.append(p)
res["kept"] = kept
if kept:
    FIN = FULL if kept == PARTS else mix(kept)
    res["W18-final vs BASE"] = score(FIN, BASE)
    res["W18-final level"] = level(FIN)
    print(f"D532 W18-final = BASE + {'+'.join(kept)} vs BASE: {res['W18-final vs BASE']}; level {res['W18-final level']}", flush=True)
else:
    print("D532: no part kept -> W18-final = BASE (W17)", flush=True)
if A.d529b:                                  # T2-D529 rule (b): the second MI member's increment on the fresh draws
    if "D529" in kept:
        e = score(mix(kept + ["D529b"]), FIN)
        e["rule_b"] = bool(e["ten"] > 0 and sum(x > 0 for x in e["draws"]) >= 2 and e["min_user"] >= -0.020 and e["returning"] >= -0.003)
        res["D529 rule (b): W18-final + MI s827 vs W18-final"] = e
        print(f"D532 / D529 rule (b): W18-final + MI s827 vs W18-final: {e} -> {'(b) holds' if e['rule_b'] else '(b) fails: one MI member'}", flush=True)
    else:
        print("D532 / D529 rule (b): D529 not kept -> no second MI member", flush=True)
json.dump(res, open(O / f"t2_d532_{'_'.join(PARTS)}{'_d529b' if A.d529b else ''}{'_' + A.d524_form if 'D524' in PARTS else ''}.json", "w"), indent=1, default=str)
