#!/usr/bin/env python
"""T2 DIAGNOSIS (the manager ≈ 23:1x, under the owner's error-anatomy rule; zero GPU, descriptive only: no item, no weight re-pick):
three components gained ALONE and lost in the mix on D536's draws 816-818. Why?
  D544  FM slot member s827: W18's SD -> word-α (one of six FM members; FM share 0.35)
  D547  ShallowNet side nets (5 per draw): 180 -> 65 epochs (the side part = 1/3 of the ShallowNet family; family share 0.2357)
  D538  FBCNet side nets (5 per draw) JOINED at r 1/3 (FBCNet family share 0.3143)
For each: (1) alone: stranger cell BA of the old and the new component, and per-class recall (0 MI, 1 subtraction, 2 word);
(2) error anatomy on stranger windows vs the STACK (W18's form, the BASE of the nomination read): fixes (old wrong, new right) and breaks
(old right, new wrong), the share of each where the stack is already right (a fix the stack does not need / a break the stack absorbs);
P(stack wrong | component wrong) for old and new, per stranger and per class;
(3) weight: the new component's weight swept (D544: the member's share of the FM slot 1/6, 2/7, 3/8, 1/2; D547 / D538: the side share r of
its family 1/3, 1/2, 2/3) for ITEM(w) vs BASE and, where a like-for-like exists, BASE(w) (the OLD component at the same weight) vs BASE;
cross-fitted on stranger halves: the w that maximises ten on strangers of folds 0-4 is read on folds 5-9 and vice versa.
The nomination reader's head (scripts/local/t2_w19_nom_read.py up to its BASE) is exec'd unchanged; the weights enter through two
overrides of functions it resolves at call time (fm_slot, parts_of).
    mem-guard env OMP_NUM_THREADS=1 python scripts/local/t2_gain_carry_diag.py
"""
import json
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
sys.argv = [sys.argv[0], "--items", ""]
SRC = open("scripts/local/t2_w19_nom_read.py").read()
CUT = "\nreset()\nBASE = mix(W18)\n"
assert SRC.count(CUT) == 1, "the nomination reader's tail moved"
exec(compile(SRC.split(CUT)[0], "t2_w19_nom_read_head(gaindiag)", "exec"))
CLS = {0: "MI", 1: "sub", 2: "word"}

# ---- overrides: the FM member weights and the side share of a family ----
FMW = {}                                      # seed -> weight (others 1); {} = the plain mean (the reader's own fm_slot)
_fm_slot_plain = fm_slot


def fm_slot(f, parts):
    if not FMW:
        return _fm_slot_plain(f, parts)
    ps, ws, ix = [], [], None
    for s, form in fm_forms(parts):
        ix_, p = load(fm_member(s, f, form))
        assert ix is None or (ix_ == ix).all()
        ix = ix_
        ps.append(p)
        ws.append(FMW.get(s, 1.0))
    ws = np.asarray(ws) / np.sum(ws)
    return ix, np.tensordot(ws, np.asarray(ps), 1)


RSIDE = {}                                    # fam -> side share r (the reader's own = 1/3)
_parts_of_nom = parts_of


def side_mean(fam, f, d):
    ix = IX[f]
    if fam == "fbc":
        return np.mean([realign(*side_prob(FBC_SIDE / FBC_STEM.format(f=f, s=d + k * 1000), f, "unc"), ix) for k in range(5)], 0)
    return np.mean([realign(*side_prob(p, f, SIDE_MODE), ix) for p in side_files(fam, f, d)], 0)


def parts_of(f, d, parts):
    P = _parts_of_nom(f, d, parts)
    if not RSIDE:
        return P
    P = dict(P)
    for fam, r in RSIDE.items():
        if fam == "fbc" and not WITH_FBC:
            continue
        S_ = side_mean(fam, f, d)
        core = (P[fam] - S_ / 3.0) * 1.5      # P = (2/3) core + (1/3) S_  (the reader's 'thirds')
        P[fam] = (1.0 - r) * core + r * S_
    return P


# ---- components alone, on IX[f] ----
FORM827 = dict(fm_forms(W18))[827]


def comp(item, which, f, d):
    ix = IX[f]
    if item == "D544":
        return realign(*load(fm_member(827, f, FORM827 if which == "old" else "wa")), ix)
    if item == "D547":
        if which == "old":
            return np.mean([realign(*side_prob(p, f, SIDE_MODE), ix) for p in _side_files_d532("sh", f, d)], 0)
        return np.mean([realign(*side_prob(O / f"t2_d547/side_sh65/runs/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e65_vr__sidedcmean__s{d + k * 1000}.npz",
                                           f, SIDE_MODE), ix) for k in range(5)], 0)
    if item == "D538":
        if which == "old":
            return realign(*load(CNN / RT["fbc"].format(f=f, s=d)), ix)
        return np.mean([realign(*side_prob(FBC_SIDE / FBC_STEM.format(f=f, s=d + k * 1000), f, "unc"), ix) for k in range(5)], 0)
    raise SystemExit(item)


def stranger_rows(f):
    return U[IX[f]] == TRU[f]


def alone_and_anatomy(item, base):
    recs, anat = [], []
    for f in range(10):
        ix, m = IX[f], stranger_rows(f)
        y = Y[ix][m]
        for d in DRAWS:
            po, pn = comp(item, "old", f, d)[m].argmax(1), comp(item, "new", f, d)[m].argmax(1)
            pb = base[(f, d)][1][m].argmax(1)
            co, cn = cells(po, ix[m]), cells(pn, ix[m])
            recs.append(dict(user=TRU[f], draw=d, old=np.mean(list(co.values())), new=np.mean(list(cn.values())),
                             **{f"rec_old_{CLS[k]}": float(np.mean(po[y == k] == k)) for k in CLS},
                             **{f"rec_new_{CLS[k]}": float(np.mean(pn[y == k] == k)) for k in CLS}))
            wo, wn, wb = po != y, pn != y, pb != y
            for k in list(CLS) + [None]:
                s_ = (y == k) if k is not None else np.ones(len(y), bool)
                fix, brk = wo & ~wn & s_, ~wo & wn & s_
                anat.append(dict(user=TRU[f], draw=d, cls=CLS.get(k, "all"), n=int(s_.sum()), fix=int(fix.sum()), brk=int(brk.sum()),
                                 fix_stack_right=int((fix & ~wb).sum()), brk_stack_right=int((brk & ~wb).sum()),
                                 old_wrong=int((wo & s_).sum()), new_wrong=int((wn & s_).sum()),
                                 old_wrong_stack_wrong=int((wo & wb & s_).sum()), new_wrong_stack_wrong=int((wn & wb & s_).sum()),
                                 stack_wrong=int((wb & s_).sum())))
    R, A_ = pd.DataFrame(recs), pd.DataFrame(anat)
    pu = R.groupby("user")[["old", "new"]].mean()
    out = dict(alone_old=round(float(pu.old.mean()), 4), alone_new=round(float(pu.new.mean()), 4),
               alone_delta=round(float((pu.new - pu.old).mean()), 4),
               alone_delta_per_stranger=(pu.new - pu.old).round(4).to_dict(),
               recall_old={CLS[k]: round(float(R[f"rec_old_{CLS[k]}"].mean()), 4) for k in CLS},
               recall_new={CLS[k]: round(float(R[f"rec_new_{CLS[k]}"].mean()), 4) for k in CLS})
    an = {}
    for c, g in A_.groupby("cls"):
        t = g.sum(numeric_only=True)
        an[c] = dict(n=int(t.n), fix=int(t.fix), brk=int(t.brk), net=int(t.fix - t.brk),
                     fix_stack_already_right=round(float(t.fix_stack_right / max(t.fix, 1)), 3),
                     brk_stack_still_right=round(float(t.brk_stack_right / max(t.brk, 1)), 3),
                     p_stack_wrong_given_old_wrong=round(float(t.old_wrong_stack_wrong / max(t.old_wrong, 1)), 3),
                     p_stack_wrong_given_new_wrong=round(float(t.new_wrong_stack_wrong / max(t.new_wrong, 1)), 3),
                     stack_err=round(float(t.stack_wrong / max(t.n, 1)), 3))
    out["anatomy"] = an
    pu_net = A_[A_.cls == "all"].groupby("user")[["fix", "brk", "fix_stack_right", "brk_stack_right"]].sum()
    pu_net["useful"] = (pu_net.fix - pu_net.fix_stack_right) - (pu_net.brk - pu_net.brk_stack_right)
    out["per_stranger_fix_brk_useful"] = {u: [int(r.fix), int(r.brk), int(r.useful)] for u, r in pu_net.iterrows()}
    return out


def half_ten(ps, half):
    us = [TRU[f] for f in (range(5) if half == 0 else range(5, 10))]
    return float(np.mean([ps[u] for u in us if u in ps]))


def sweep(item, grid, base):
    rows = []
    for w in grid:
        for like in ((False, True) if item != "D538" else (False,)):
            reset()
            if item == "D544":
                if not like:
                    FM_SWAP[827] = "wa"
                FMW.clear()
                FMW[827] = w
            else:
                fam = "sh" if item == "D547" else "fbc"
                if item == "D547" and not like:
                    SIDE_REPL["sh"] = (O / "t2_d547/side_sh65/runs", "r1_pulx_vp2_e65_vr")
                if item == "D538":
                    globals()["WITH_FBC"] = True
                RSIDE.clear()
                RSIDE[fam] = w
            mx = mix(W18)
            sc = score(mx, base)
            ps = per_stranger(mx, base)
            rows.append(dict(w=round(float(w), 4), arm="BASE(w): old at w" if like else "ITEM(w)", ten=sc["ten"], draws=sc["draws"],
                             min_user=sc["min_user"], min_who=sc["min_who"], returning=sc["returning"], h0=round(half_ten(ps, 0), 4), h1=round(half_ten(ps, 1), 4)))
            print(f"  {item} w {w:.3f} {rows[-1]['arm']}: ten {sc['ten']:+.4f} draws {sc['draws']} returning {sc['returning']:+.4f} "
                  f"halves {rows[-1]['h0']:+.4f} / {rows[-1]['h1']:+.4f}", flush=True)
            FMW.clear()
            RSIDE.clear()
            reset()
    cf = {}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm]
        w0 = max(rr, key=lambda r: r["h0"])         # picked on folds 0-4, read on 5-9
        w1 = max(rr, key=lambda r: r["h1"])
        cf[arm] = dict(pick_on_h0=w0["w"], read_h1=w0["h1"], pick_on_h1=w1["w"], read_h0=w1["h0"],
                       crossfit_ten=round((w0["h1"] + w1["h0"]) / 2, 4))
    return rows, cf


reset()
BASE = mix(W18)
assert abs(level(BASE) - D536["level"]["BASE (W18 form)"]) < 5e-5
print(f"gain-carry diag: BASE level {level(BASE)} (= D536's); FM slot form of s827 = {FORM827}", flush=True)
GRID = {"D544": [1.0, 2.0, 3.0, 5.0],          # member weight vs 1 for each of the other five: share 1/6, 2/7, 3/8, 1/2
        "D547": [1 / 3, 1 / 2, 2 / 3],
        "D538": [1 / 3, 1 / 2, 2 / 3]}
res = {}
for item in ("D544", "D547", "D538"):
    print(f"== {item}", flush=True)
    a = alone_and_anatomy(item, BASE)
    print(f"  alone: old {a['alone_old']} new {a['alone_new']} (delta {a['alone_delta']:+.4f}); recall old {a['recall_old']} new {a['recall_new']}", flush=True)
    for c, v in a["anatomy"].items():
        print(f"  anatomy {c}: {v}", flush=True)
    rows, cf = sweep(item, GRID[item], BASE)
    nom = [v for fn in sorted(O.glob("t2_w19_nom_*.json")) for k, v in json.load(open(fn)).items()
           if k == item and isinstance(v, dict) and "ten" in v]          # self-check: ITEM at the default weight = the nomination read
    nom = nom[-1] if nom else json.load(open(O / "t2_d538.json"))["HEADLINE W19F vs BASE (W18 form), draws 816-818"]   # D538: its own reader
    r0 = [r for r in rows if r["arm"] == "ITEM(w)"][0]
    assert r0["ten"] == nom["ten"] and r0["draws"] == nom["draws"] and r0["returning"] == nom["returning"], (item, r0, nom)
    print(f"  self-check: ITEM at the default weight = the nomination read ({nom['ten']:+.4f})", flush=True)
    a["sweep"], a["crossfit"] = rows, cf
    print(f"  cross-fit: {cf}", flush=True)
    res[item] = a
    reset()
res["checks"] = dict(CHK)
json.dump(res, open(O / "t2_gain_carry_diag.json", "w"), indent=1, default=str)
print(f"gain-carry diag: wrote {O / 't2_gain_carry_diag.json'} ({time.time() - T0:.0f} s)", flush=True)
