#!/usr/bin/env python
"""T2 W19 NOMINATION reader (T2-D540a; committed before the items' inputs exist): every stack item vs W18's form on D536's draws 816-818.
Nomination bar (D540a, registered 2026-11-16 20:03:46): ten > 0, >= 2 / 3 draws > 0, no stranger < -0.020, returning >= -0.003 (= score's 'part').
D536's reader head is exec'd unchanged (W18's form = FM SD 827-831 + MI s826, side nets UNC, DC bag LightGBM 7-9); each item changes one thing:
  D538  FBCNet side nets at r 1/3 (outputs/t2_d538/side_fbc; = t2_d538_read.py's arm)
  D541  slot 0: MI s826 -> MI-SD s826 (sofia ni26-t2p-d541-misd_s826_p*)
  D542  the DC bag: LightGBM 7-9 -> XGBoost 7-9 (outputs/t2_d535/pred_s7-8-9_dc_xgb.npz)
  D543  side nets S5 -> S10 (k 5..9 from outputs/t2_d543/side_{sh,ifn})
  D544  slot 1: SD s827 -> word-α s827 (sofia ni26-t2p-d544-wa_s827_p*)
  D545  shares: IFNet 0.10 -> 0.15, FBCNet : ShallowNet ratio kept, FM 0.35
  D546a / D546b  slots 2 / 4 = the D546 jobs' own bases (BASE') -> their 'ca' / 'cb' continuations (vs BASE', paired)
  D547  ShallowNet side nets: 180 -> 65 epochs (outputs/t2_d547/side_sh65)
An item whose inputs are not all on disk is reported 'pending' (never partially read).
    mem-guard env OMP_NUM_THREADS=1 python scripts/local/t2_w19_nom_read.py --items D541,D542,...
"""
import glob
import json
import sys
import time

import numpy as np

T0 = time.time()
ITEMS = [x for x in (sys.argv[sys.argv.index("--items") + 1] if "--items" in sys.argv else "").split(",") if x]
SRC36 = open("scripts/local/t2_d536_read.py").read()
CUT = '\nW18 = ["D524", "D529"]\n'
assert SRC36.count(CUT) == 1, "the D536 reader's tail moved"
exec(compile(SRC36.split(CUT)[0], "t2_d536_read_head(w19nom)", "exec"))
assert DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")
D536 = json.load(open(O / "t2_d536.json"))
W18 = ["D524", "D529"]
SIDE_MODE = "unc"

# ---- the overridable pieces (resolved by the head's functions at call time) ----
FM_SWAP = {}
_fm_forms_d532 = fm_forms
_fm_member_d532 = fm_member


def fm_forms(parts):
    return [(s, FM_SWAP.get(s, form)) for s, form in _fm_forms_d532(parts)]


def fm_member(s, f, form):
    stem = f"runs/lc9_{f}__fm-st_large-mean__sz__n00__*"
    p = f"_p{f // 2}"
    if form == "misd":
        return one(f"{S}/ni26-t2p-d541-misd_s{s}{p}-[0-9]*/{stem}sddcmean__sidedcmean__s{s}.npz")
    if form == "wa":
        return one(f"{S}/ni26-t2p-d544-wa_s{s}{p}-[0-9]*/{stem}sddcmean__sidedcmean__s{s}.npz")
    if form in ("cbase", "ca", "cb"):
        kind = "plain" if s in (828, 835) else "car"
        suf = "" if form == "cbase" else form
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d546-{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{suf}__{car}sidedcmean__s{s}.npz")
    return _fm_member_d532(s, f, form)


SIDE_EXTRA = {}                                   # fam -> extra side-net dirs (D543); SIDE_REPL = fam -> replacement (D547)
SIDE_REPL = {}
_side_files_d532 = side_files


def side_files(fam, f, d):
    if fam in SIDE_REPL:
        dr, rec = SIDE_REPL[fam]
        m_ = REC[fam][0]
        return [dr / f"lc9_{f}__{m_}__ea__n00__{rec}__sidedcmean__s{d + k * 1000}.npz" for k in range(5)]
    fs = list(_side_files_d532(fam, f, d))
    if fam in SIDE_EXTRA:
        m_, rec = REC[fam]
        fs += [SIDE_EXTRA[fam] / f"lc9_{f}__{m_}__ea__n00__{rec}__sidedcmean__s{d + k * 1000}.npz" for k in range(5, 10)]
    return fs


WITH_FBC = False
FBC_SIDE = O / "t2_d538/side_fbc/runs"
FBC_STEM = "lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr__sidedcmean__s{s}.npz"
_parts_of_d536 = parts_of


def parts_of(f, d, parts):
    P = _parts_of_d536(f, d, parts)
    if WITH_FBC:
        ix = IX[f]
        S_ = np.mean([realign(*side_prob(FBC_SIDE / FBC_STEM.format(f=f, s=d + k * 1000), f, "unc"), ix) for k in range(5)], 0)
        P = dict(P)
        P["fbc"] = (2.0 / 3.0) * P["fbc"] + (1.0 / 3.0) * S_
    return P


W_W18 = dict(W)
DC_W18, DCVEC_W18 = dict(DC), dict(DCVEC)


def set_dc(path):
    global DC, DCVEC
    zp_ = np.load(need(path))
    DC = {f: (zp_[f"te_{f}"], zp_[f"bag_{f}"]) for f in range(10)}
    DCVEC = {}
    for f in range(10):
        v = np.mean([DC[g][1][U[DC[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
        DCVEC[f] = v / v.sum()


def reset():
    global FM_SWAP, SIDE_EXTRA, SIDE_REPL, WITH_FBC, W, DC, DCVEC
    FM_SWAP, SIDE_EXTRA, SIDE_REPL, WITH_FBC = {}, {}, {}, False
    W, DC, DCVEC = dict(W_W18), dict(DC_W18), dict(DCVEC_W18)


def have(pattern_list):
    return all(glob.glob(str(p)) for p in pattern_list)


def inputs(item):
    """the files an item needs (all must exist, else 'pending')."""
    if item == "D538":
        return [FBC_SIDE / FBC_STEM.format(f=f, s=d + k * 1000) for f in range(10) for d in DRAWS for k in range(5)]
    if item == "D541":
        return [f"{S}/ni26-t2p-d541-misd_s826_p{f // 2}-[0-9]*/runs/lc9_{f}__*__s826.npz" for f in range(10)]
    if item == "D542":
        return [O / "t2_d535/pred_s7-8-9_dc_xgb.npz"]
    if item == "D543":
        return [O / f"t2_d543/side_{fam}/runs/lc9_{f}__{REC[fam][0]}__ea__n00__{REC[fam][1]}__sidedcmean__s{d + k * 1000}.npz"
                for fam in ("sh", "ifn") for f in range(10) for d in DRAWS for k in range(5, 10)]
    if item == "D544":
        return [f"{S}/ni26-t2p-d544-wa_s827_p{f // 2}-[0-9]*/runs/lc9_{f}__*__s827.npz" for f in range(10)]
    if item == "D545":
        return []
    if item in ("D546a", "D546b"):
        a = item[-1]
        return [f"{S}/ni26-t2p-d546-{k_}_s{s}_p{f // 2}-[0-9]*/runs/lc9_{f}__*sddcmean{x}__{'car__' if k_ == 'car' else ''}sidedcmean__s{s}.npz"
                for k_, s in (("plain", 828), ("car", 830)) for f in range(10) for x in ("", "c" + a)]
    if item == "D547":
        return [O / f"t2_d547/side_sh65/runs/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e65_vr__sidedcmean__s{d + k * 1000}.npz"
                for f in range(10) for d in DRAWS for k in range(5)]
    raise SystemExit(f"unknown item {item}")


def arm(item):
    """-> (ITEM, BASE_for_item) mixes."""
    reset()
    base = None
    if item == "D538":
        globals()["WITH_FBC"] = True
    elif item == "D541":
        FM_SWAP[826] = "misd"
    elif item == "D542":
        set_dc(O / "t2_d535/pred_s7-8-9_dc_xgb.npz")
    elif item == "D543":
        SIDE_EXTRA.update(sh=O / "t2_d543/side_sh/runs", ifn=O / "t2_d543/side_ifn/runs")
    elif item == "D544":
        FM_SWAP[827] = "wa"
    elif item == "D545":
        s_ = (0.65 - 0.15) / (W_W18["fbc"] + W_W18["sh"])
        globals()["W"] = {"fbc": W_W18["fbc"] * s_, "sh": W_W18["sh"] * s_, "ifn": 0.15, "fm": 0.35}
        assert abs(sum(W.values()) - 1.0) < 1e-9, W
    elif item in ("D546a", "D546b"):
        FM_SWAP.update({828: "cbase", 830: "cbase"})
        base = mix(W18)                                  # BASE' = the D546 jobs' own bases in slots 2 / 4
        FM_SWAP.update({828: "c" + item[-1], 830: "c" + item[-1]})
    elif item == "D547":
        SIDE_REPL["sh"] = (O / "t2_d547/side_sh65/runs", "r1_pulx_vp2_e65_vr")
    it = mix(W18)
    reset()
    return it, (base if base is not None else BASE)


reset()
BASE = mix(W18)
lv = level(BASE)
assert abs(lv - D536["level"]["BASE (W18 form)"]) < 5e-5, (lv, D536["level"])
print(f"W19 nom: BASE (W18 form) level {lv} = D536's", flush=True)
res = {"draws": list(DRAWS), "base_level": lv}
for item in ITEMS:
    miss = [p for p in inputs(item) if not glob.glob(str(p))]
    if miss:
        res[item] = dict(state="pending", missing=len(miss), example=str(miss[0]))
        print(f"W19 nom {item}: PENDING ({len(miss)} inputs missing, e.g. {miss[0]})", flush=True)
        continue
    n0 = CHK["files"]
    it, b = arm(item)
    sc = score(it, b)
    sc = dict(per_stranger=per_stranger(it, b), **sc)
    sc["nominated"] = bool(sc["part"])
    sc["level_item"], sc["level_base"] = level(it), level(b)
    sc["side_files_read"] = CHK["files"] - n0
    res[item] = sc
    print(f"W19 nom {item}: {'NOMINATED' if sc['nominated'] else 'not nominated'} {sc}", flush=True)
    assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
res["checks"] = dict(CHK)
fn = O / f"t2_w19_nom_{'_'.join(ITEMS) or 'none'}.json"
json.dump(res, open(fn, "w"), indent=1, default=str)
print(f"W19 nom: wrote {fn} ({time.time() - T0:.0f} s)", flush=True)
