#!/usr/bin/env python
"""T2-D540a STACK read (registered 2026-11-16 20:03:46; committed before any 819-821 output exists).
  --draws 819 (the decision): W18's form RE-TRAINED at 819-821 (outputs/t2_d540 chain; DC bag LightGBM 10-12) with the FM slots that any
     launched FM item touches at FRESH seeds: slot 0 = MI s832 (head-less; D541's confirm base), slot 1 = SD s833 (D544's confirm base) when
     D544 is an item, slots 2 / 4 = the D546 confirm jobs' own bases (s835 / s836) when D546a / D546b is an item. FULL = BASE + every item;
     leave-one-out (ONE pass): an item is dropped if FULL vs FULL - item has ten <= 0 or fewer than 2 / 3 draws > 0; W19-final = BASE + the
     kept items, read ONCE vs BASE. STACK BAR (the manager's decision 1): ten >= +0.001, >= 2 / 3 draws > 0, no stranger < -0.020,
     returning >= -0.003, sealed mix > 0.
  --draws 816 (the reader's own CHECK, never a decision): the same code on D536's draws with the nomination files; each single item vs its
     base must reproduce t2_w19_nom_read.py's numbers (asserted for the items given that the nomination JSON holds).
Items: D538 (FBCNet side nets), D539b (H at lam 1 on every side net present), D541 (slot 0 -> MI-SD), D543 (side S10), D544 (slot 1 ->
word-α), D545 (IFNet 0.15), D546a / D546b (slots 2 / 4 continued), D547 (ShallowNet side nets at 65 epochs).
    mem-guard env OMP_NUM_THREADS=1 python scripts/local/t2_d540a_stack_read.py --draws 819 --items D538,D545,...
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
from pathlib import Path
from scipy.optimize import minimize

T0 = time.time()
ARGV = list(sys.argv)
DSET = int(ARGV[ARGV.index("--draws") + 1]) if "--draws" in ARGV else 819
ITEMS = [x for x in (ARGV[ARGV.index("--items") + 1] if "--items" in ARGV else "").split(",") if x]
assert DSET in (816, 819, 822, 825, 828, 831, 834, 860) and ITEMS, (DSET, ITEMS)   # T2-D615: 860 = set E
KNOWN = {"D538", "D539b", "D541", "D543", "D544", "D545", "D546a", "D546b", "D547", "D526", "MIX", "D553", "D555", "D557", "D567", "D568", "D568B", "D582", "D582B", "D583", "D583B", "D586", "D586B", "D588", "D588B", "D596", "D596B", "D611", "D611B", "D618", "D618B"}
assert set(ITEMS) <= KNOWN and not {"D546a", "D546b"} <= set(ITEMS), ITEMS
D567_ON = "D567" in ITEMS                                 # T2-D567 (registered 2026-11-18 06:40): slot 0 -> the augmented MI line (B or C only)
assert not D567_ON or (DSET in (828, 831) and "D557" in ITEMS), (DSET, ITEMS)
assert "D568" not in ITEMS or (DSET == 831 and "D557" in ITEMS), (DSET, ITEMS)   # T2-D568 (registered 2026-11-18 11:34:11): C with D557 only
assert "D582" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS), (DSET, ITEMS)   # T2-D582 (registered 2026-11-18): C with D557, or D
assert "D583" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS and not {"D568", "D582"} & set(ITEMS)), (DSET, ITEMS)   # T2-D583: C with D557, or D (its own roots)
assert "D596" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS and not {"D568", "D582", "D583", "D586", "D588"} & set(ITEMS)), (DSET, ITEMS)   # T2-D596: C with D557, or D
assert "D596B" not in ITEMS or (DSET == 828 and "D557" in ITEMS and not {"D582B", "D568B", "D583B", "D586B", "D588B"} & set(ITEMS)), (DSET, ITEMS)   # T2-D596 B arm (V forms)
assert "D588" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS and not {"D568", "D582", "D583", "D586"} & set(ITEMS)), (DSET, ITEMS)   # T2-D588: C with D557, or D
assert "D588B" not in ITEMS or (DSET == 828 and "D557" in ITEMS and not {"D582B", "D568B", "D583B", "D586B"} & set(ITEMS)), (DSET, ITEMS)   # T2-D588 B arm (f forms)
assert "D583B" not in ITEMS or (DSET == 828 and "D557" in ITEMS and not {"D582B", "D568B", "D586B"} & set(ITEMS)), (DSET, ITEMS)   # T2-D583 B arm (w forms)
assert "D586B" not in ITEMS or (DSET == 828 and "D582B" in ITEMS and os.environ.get("NI26_LAB_D582B") == "ref"), (DSET, ITEMS)   # T2-D586 B arm: D582's REF + 827
assert "D586" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS and not {"D568", "D582", "D583"} & set(ITEMS)), (DSET, ITEMS)   # T2-D586: C with D557, or D
assert "D611" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS and not {"D568", "D582", "D583", "D586", "D588", "D596"} & set(ITEMS)), (DSET, ITEMS)   # T2-D611: C with D557, or D
assert "D611B" not in ITEMS or (DSET == 828 and "D582B" in ITEMS and os.environ.get("NI26_LAB_D582B") == "ref"), (DSET, ITEMS)   # T2-D611 B arm: D582's REF + 831
assert "D618" not in ITEMS or (DSET in (831, 834) and "D557" in ITEMS and not {"D568", "D582", "D583", "D586", "D588", "D596", "D611"} & set(ITEMS)), (DSET, ITEMS)   # T2-D618: C with D557, or D
assert "D618B" not in ITEMS or (DSET == 828 and "D582B" in ITEMS and "D611B" not in ITEMS and os.environ.get("NI26_LAB_D582B") == "ref"), (DSET, ITEMS)   # T2-D618 B arm: D582's REF + 831
D584 = os.environ.get("NI26_LAB_D584") == "1"             # T2-D584: D539b's CNN side-head refits weighted on every prefix window (the lab arm w20_fb_wref)
assert not D584 or "D539b" in ITEMS, "NI26_LAB_D584 acts on D539b's side-head refits: D539b must be on"
D582B = os.environ.get("NI26_LAB_D582B", "")              # T2-D582 B: ref = D557's dose re-run with dumps (r forms), arm = + vp 2 (s forms)
assert D582B in ("", "ref", "arm") and (("D582B" in ITEMS) == bool(D582B)) and ("D582B" not in ITEMS or (DSET == 828 and "D557" in ITEMS)), (D582B, DSET, ITEMS)
D568B = os.environ.get("NI26_LAB_D568B", "")              # T2-D568's B stage (amended 11:39): ref = D557's dose re-run with dumps (n forms), arm = the stronger dose (o forms)
assert D568B in ("", "ref", "arm") and (("D568B" in ITEMS) == bool(D568B)) and ("D568B" not in ITEMS or (DSET == 828 and "D557" in ITEMS)), (D568B, DSET, ITEMS)
ITEMS_ALL, ITEMS = ITEMS, [x for x in ITEMS if x not in ("D567", "D568", "D568B", "D582", "D582B", "D583", "D583B", "D586", "D586B", "D588", "D588B", "D596", "D596B", "D611", "D611B", "D618", "D618B")]   # the older item checks below see the set without D567 / D568
LMI567 = os.environ.get("NI26_D567_LOCAL", "")           # T2-D567's local backup line(s) on the 5090 ('842', '832' or '832,842'): form ami -> t2_d567/mi_s<s>
D551 = bool({"D526", "MIX"} & set(ITEMS))                # T2-D551 (registered 2026-11-17 09:4x): the owner's 'little help' stack read
assert not D551 or (DSET == 819 and not {"D526", "D545"} <= set(ITEMS)), (DSET, ITEMS)
W19C = ["D539b", "D543", "D546b"]                         # T2-D553 (registered 2026-11-17): REF = W19c at 822-824, ARM = REF + D553
assert DSET not in (822, 825, 828) or sorted(ITEMS) == sorted(W19C + ["D553"] + {822: [], 825: ["D555"], 828: ["D557"]}[DSET]), (DSET, ITEMS)
assert DSET != 831 or sorted(ITEMS) in (sorted(W19C + ["D553", "D555"]), sorted(W19C + ["D553", "D557"])), (DSET, ITEMS)   # T2-D558: ONE item
assert DSET not in (834, 860) or sorted(ITEMS) == sorted(W19C + ["D553", "D557"]), (DSET, ITEMS)   # T2-D578: set D = W20+D557's members only
assert ("D553" in ITEMS) == (DSET in (822, 825, 828, 831, 834, 860)) and ("D555" in ITEMS) <= (DSET in (825, 831)) and ("D557" in ITEMS) <= (DSET in (828, 831, 834, 860)), (DSET, ITEMS)

src = open("scripts/local/t2_d536_read.py").read()
CUT = '\nW18 = ["D524", "D529"]\n'
assert src.count(CUT) == 1, "the D536 reader's tail moved"
src = src.split(CUT)[0]
NEWD = {819: ("(819, 820, 821)", "t2_d540", "pred_s10-11-12_dc.npz"), 822: ("(822, 823, 824)", "t2_d553", "pred_s13-14-15_dc.npz"),
        825: ("(825, 826, 827)", "t2_d555", "pred_s16-17-18_dc.npz"), 828: ("(828, 829, 830)", "t2_d557", "pred_s19-20-21_dc.npz"),
        831: ("(831, 832, 833)", "t2_d558", "pred_s22-23-24_dc.npz"), 834: ("(834, 835, 836)", "t2_d578", "pred_s25-26-27_dc.npz"),
        860: ("(860, 861, 862)", "t2_d615", "pred_s28-29-30_dc.npz")}   # T2-D615: set E
D559 = (os.environ.get("NI26_D559_ROOT", ""), os.environ.get("NI26_D559_CKS", ""))   # T2-D559: one schedule's root + its refit points
assert bool(D559[0]) == bool(D559[1]) and (not D559[0] or (DSET in (825, 828, 831, 834, 860) and D559[0].startswith(("t2_d559_", "t2_d569_", "t2_d571_", "t2_d576_", "t2_d580_", "t2_d589_", "t2_d594b_", "t2_d638_", "t2_d642_")) and D559[0].endswith(f"_{DSET}"))), D559
assert not D559[0].startswith(("t2_d569_", "t2_d571_", "t2_d576_", "t2_d580_", "t2_d589_", "t2_d594b_", "t2_d638_", "t2_d642_")) or D559[1] == "60,100,150", D559   # T2-D569: a P1 root at the shipped refits (its own teacher_p1 + students)
if DSET in NEWD:
    dr_, dir_, dc_ = NEWD[DSET]
    if D559[0]:                                          # the root holds symlinks to the chain's CNN members and DC bag
        dir_ = D559[0]
    for a, b in (("new = 'DRAWS, CNN, DCF = (816, 817, 818), O / \"t2_d536\", O / \"t2_d536/pred_s7-8-9_dc.npz\"'",
                  f"new = 'DRAWS, CNN, DCF = {dr_}, O / \"{dir_}\", O / \"{dir_}/{dc_}\"'"),
                 ('assert not A.check and A.d524_form == "six" and DRAWS == (816, 817, 818) and str(CNN).endswith("t2_d536")',
                  f'assert not A.check and A.d524_form == "six" and DRAWS == {dr_} and str(CNN).endswith("{dir_}")')):
        assert src.count(a) == 1, (a[:60], src.count(a))
        src = src.replace(a, b)
# Track3's lab-vs-template note (lead/diag_word_2026-11-18/RESULT_D595P.md, the manager 2026-11-18): before the first installation the lab's
# sh / ifn part is (1 - R1) R + R1 U' (every P1 file holds the bench's pooled twin U' on windows < 60), the template's is R alone (is_net None,
# t2p_submission_w21lp.py :1339 / :1566). NI26_LAB_PRE_R=1 = template-faithful: R alone where the window order < the first refit (the template
# decodes window index c with the student after the refit at is_n == c). Default off = the lab as every earlier read (reproducible).
PRE_R = os.environ.get("NI26_LAB_PRE_R", "") == "1"
assert os.environ.get("NI26_LAB_PRE_R", "") in ("", "0", "1"), os.environ.get("NI26_LAB_PRE_R")
PRE_R_FIRST = min(int(c) for c in D559[1].split(",")) if D559[1] else 60
assert not PRE_R or PRE_R_FIRST == 60, PRE_R_FIRST
if PRE_R:
    a_ = "        part = (1 - R1) * r + R1 * a\n"
    assert src.count(a_) == 1, ("the D536 part line moved", src.count(a_))
    src = src.replace(a_, a_ + "        part = np.where((ORD[ix] < PRE_R_FIRST)[:, None], r, part)   # NI26_LAB_PRE_R (template-faithful pre-refit)\n"
                                "        CHK_PRE_R.append(float((ORD[ix] < PRE_R_FIRST).mean()))\n"
                                "        if len(CHK_PRE_R) == 1: print(f'[pre_r] NI26_LAB_PRE_R on: R alone on windows < {PRE_R_FIRST} "
                                "({CHK_PRE_R[0]:.4f} of fold {f} rows; changed {float(np.abs(part - ((1 - R1) * r + R1 * a)).max()):.3g})', flush=True)\n")
CHK_PRE_R = []
# T2-D616 (the P1 student-rep ensemble; the manager's GO 2026-11-19): every root holds three student reps per refit (isSpr / isSprr1 / isSprr2 =
# --instream-reps 3, fine-tune seed + 1000 k); NI26_LAB_P1REPS=k averages the first k reps' probabilities as the P1 part's student (1 = rep 0
# only = every earlier read, the shipped template).
P1REPS = int(os.environ.get("NI26_LAB_P1REPS", "") or 1)
assert P1REPS in (1, 2, 3), P1REPS
if P1REPS > 1:
    a_ = "        a = realign(*load(CNN / P1R[fam].format(f=f, s=d)), ix)\n"
    assert src.count(a_) == 1, ("the D536 P1 load line moved", src.count(a_))
    src = src.replace(a_, "        a = np.mean([realign(*load(CNN / P1R[fam].format(f=f, s=d).replace('isSpr__', sfx_ + '__')), ix) "
                          "for sfx_ in ('isSpr', 'isSprr1', 'isSprr2')[:P1REPS]], 0)   # NI26_LAB_P1REPS (T2-D616)\n")
sys.argv = [ARGV[0]]
exec(compile(src, f"t2_d536_read_head(d540a,{DSET})", "exec"))
assert DRAWS == {816: (816, 817, 818), 819: (819, 820, 821), 822: (822, 823, 824), 825: (825, 826, 827), 828: (828, 829, 830), 831: (831, 832, 833), 834: (834, 835, 836), 860: (860, 861, 862)}[DSET]
W18 = ["D524", "D529"]
SIDE_MODE = "unc"

# ---------------- FM slots: seed s of W18's six -> (form, actual seed) ----------------
FM_SWAP = {}
_fm_forms_d532 = fm_forms
_fm_member_d532 = fm_member


def fm_forms(parts):
    return [(s, FM_SWAP.get(s, form)) for s, form in _fm_forms_d532(parts)]


def fm_member(s, f, form):
    if "@" not in form and form not in ("misd", "wa", "cbase", "ca", "cb", "dsd", "dbase", "dcb", "esd", "ebase", "ecb", "emi", "gsd", "gbase", "gcb", "hsd", "hbase", "hcb", "hmi", "ksd", "kbase", "kcb", "lmi", "lsd", "ami", "msd", "mbase", "mcb", "nsd", "nbase", "ncb", "osd", "obase", "ocb", "psd", "pbase", "pcb", "pmi", "qsd", "qbase", "qcb", "rsd", "rbase", "rcb", "ssd", "sbase", "scb", "tsd", "tbase", "tcb", "usd", "ubase", "ucb", "vsd", "wsd", "wbase", "wcb", "xsd", "xbase", "xcb", "ysd", "zsd", "jsd", "jbase", "jcb", "fsd", "fbase", "fcb", "isd", "ibase", "icb", "Usd", "Ubase", "Ucb", "Vsd", "Vbase", "Vcb", "Wsd", "Wbase", "Wcb"):
        return _fm_member_d532(s, f, form)
    form, s = (form.split("@")[0], int(form.split("@")[1])) if "@" in form else (form, s)
    stem = f"runs/lc9_{f}__fm-st_large-mean__sz__n00__*"
    p = f"_p{f // 2}"
    if form == "mi":                                                    # D541's confirm base (head-less MI line)
        return one(f"{S}/ni26-t2p-d541-mi_s{s}{p}-[0-9]*/{stem}__s{s}.npz")
    if form == "misd":
        return one(f"{S}/ni26-t2p-d541-misd_s{s}{p}-[0-9]*/{stem}sddcmean__sidedcmean__s{s}.npz")
    if form == "sd":                                                    # D544's confirm base
        return one(f"{S}/ni26-t2p-d544-sd_s{s}{p}-[0-9]*/{stem}sddcmean__sidedcmean__s{s}.npz")
    if form == "wa":
        return one(f"{S}/ni26-t2p-d544-wa_s{s}{p}-[0-9]*/{stem}sddcmean__sidedcmean__s{s}.npz")
    if form in ("cbase", "ca", "cb"):
        kind = "plain" if s in (828, 835) else "car"
        suf = "" if form == "cbase" else form
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d546-{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{suf}__{car}sidedcmean__s{s}.npz")
    if form == "dsd":                                                   # T2-D553: D531's SD line re-run with the head dump (fold halves)
        return one(f"{S}/ni26-t2p-d553-{'stlcarsd' if s >= 829 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("dbase", "dcb"):                                        # T2-D553: D546's confirm job re-run with the dump, arm b only
        kind = "plain" if s == 835 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d553-{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form == 'dcb' else ''}__{car}sidedcmean__s{s}.npz")
    if form == "esd":                                                   # T2-D555: the same lines re-run with the head + feature dumps
        return one(f"{S}/ni26-t2p-d555-{'stlcarsd' if s >= 829 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("ebase", "ecb"):
        kind = "plain" if s == 835 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d555-{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form == 'ecb' else ''}__{car}sidedcmean__s{s}.npz")
    if form == "emi":                                                   # T2-D555: the head-less MI member re-run with the feature dump
        return one(f"{S}/ni26-t2p-d555-mi_s{s}{p}-[0-9]*/{stem}__s{s}.npz")
    if form == "gsd":                                                   # T2-D557: the SD lines re-trained with the combined augmentation
        return one(f"{S}/ni26-t2p-d557-{'stlcarsd' if s >= 829 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("gbase", "gcb"):
        kind = "plain" if s == 835 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d557-{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form == 'gcb' else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("hsd", "ksd"):                                          # T2-D558: the new-seed lines (h = plain, k = D557-augmented)
        grp = "plain" if form == "hsd" else "aug"
        return one(f"{S}/ni26-t2p-d558-{grp}_{'stlcarsd' if s >= 838 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("hbase", "hcb", "kbase", "kcb"):
        grp = "plain" if form[0] == "h" else "aug"
        kind = "plain" if s == 840 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d558-{grp}_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form.endswith('cb') else ''}__{car}sidedcmean__s{s}.npz")
    if form == "hmi":
        return one(f"{S}/ni26-t2p-d558-plain_mi_s{s}{p}-[0-9]*/{stem}__s{s}.npz")
    if form == "msd":                                                   # T2-D568: D558's k lines at the stronger augmentation dose (same seeds)
        return one(f"{S}/ni26-t2p-d568-aug_{'stlcarsd' if s >= 838 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("mbase", "mcb"):
        kind = "plain" if s == 840 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d568-aug_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form == 'mcb' else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("qsd", "rsd", "ssd", "tsd"):                            # T2-D582: + vp 2 (q = C, s = B arm, t = D) / B REF re-run (r)
        grp = {"q": "c", "r": "bref", "s": "barm", "t": "d"}[form[0]]
        cut = {"q": 838, "r": 829, "s": 829, "t": 844}[form[0]]
        return one(f"{S}/ni26-t2p-d582-{grp}_{'stlcarsd' if s >= cut else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("qbase", "qcb", "rbase", "rcb", "sbase", "scb", "tbase", "tcb"):
        grp = {"q": "c", "r": "bref", "s": "barm", "t": "d"}[form[0]]
        kind = "plain" if s in (840, 835, 846) else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d582-{grp}_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form.endswith('cb') else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("wsd", "xsd"):                                          # T2-D583: B ARM (w, seeds 827 / 829 / 831) / D (x, 843 / 844 / 845)
        grp, cut = ("barm", 829) if form[0] == "w" else ("d", 844)
        return one(f"{S}/ni26-t2p-d583-{grp}_{'stlcarsd' if s >= cut else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("wbase", "wcb", "xbase", "xcb"):                        # T2-D583: B ARM (835 plain / 836 CAR) / D (846 plain / 847 CAR)
        grp = "barm" if form[0] == "w" else "d"
        kind = "plain" if s in (835, 846) else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d583-{grp}_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form.endswith('cb') else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("Usd", "Vsd", "Wsd"):                                   # T2-D596 (the WORD margin): C (U, 837-839) / B ARM (V, 827 / 829 / 831) / D (W, 843-845)
        grp, cut = {"U": ("c", 838), "V": ("barm", 829), "W": ("d", 844)}[form[0]]
        return one(f"{S}/ni26-t2p-d596-{grp}_{'stlcarsd' if s >= cut else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("Ubase", "Ucb", "Vbase", "Vcb", "Wbase", "Wcb"):        # T2-D596: the base + cb lines (C 840 / 841, B 835 / 836, D 846 / 847)
        grp = {"U": "c", "V": "barm", "W": "d"}[form[0]]
        kind = "plain" if s in (840, 835, 846) else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d596-{grp}_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form.endswith('cb') else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("jsd", "fsd", "isd"):                                   # T2-D588 (consistency): C (j, 837-839) / B ARM (f, 827 / 829 / 831) / D (i, 843-845)
        grp, cut = {"j": ("c", 838), "f": ("barm", 829), "i": ("d", 844)}[form[0]]
        return one(f"{S}/ni26-t2p-d588-{grp}_{'stlcarsd' if s >= cut else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("jbase", "jcb", "fbase", "fcb", "ibase", "icb"):        # T2-D588: the base + cb lines (C 840 / 841, B 835 / 836, D 846 / 847)
        grp = {"j": "c", "f": "barm", "i": "d"}[form[0]]
        kind = "plain" if s in (840, 835, 846) else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d588-{grp}_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form.endswith('cb') else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("bpcsd", "bpbsd", "bpdsd"):                             # T2-D611: the 8-30 Hz specialist of the CAR SD slot 831: C 839 / B ARM 831 / D 845
        grp = {"bpcsd": "c", "bpbsd": "barm", "bpdsd": "d"}[form]
        assert s == {"bpcsd": 839, "bpbsd": 831, "bpdsd": 845}[form], (form, s)
        return one(f"{S}/ni26-t2p-d611-{grp}_stlcarsd_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("mrcsd", "mrbsd", "mrdsd"):                             # T2-D618: the left-right mirror twin of the CAR SD slot 831: C 839 / B ARM 831 / D 845
        grp = {"mrcsd": "c", "mrbsd": "barm", "mrdsd": "d"}[form]
        assert s == {"mrcsd": 839, "mrbsd": 831, "mrdsd": 845}[form], (form, s)
        return one(f"{S}/ni26-t2p-d618-{grp}_stlcarsd_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}mr0.5*__s{s}.npz")
    if form in ("ysd", "zsd"):                                          # T2-D586: B ARM (y, the plain SD 827) / D (z, the plain SD 843)
        grp = "barm" if form[0] == "y" else "d"
        assert s == (827 if form[0] == "y" else 843), (form, s)
        return one(f"{S}/ni26-t2p-d586-{grp}_stlsd_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form == "vsd":                                                   # T2-D586 C: the plain SD line 837 + the shared-phase surrogate
        assert s == 837, s
        return one(f"{S}/ni26-t2p-d586-c_stlsd_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form == "usd":                                                   # T2-D583 C: the SD lines + segment recombination (837 plain / 838, 839 CAR)
        return one(f"{S}/ni26-t2p-d583-c_{'stlcarsd' if s >= 838 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("ubase", "ucb"):                                        # T2-D583 C: the base + cb lines + segment recombination (840 plain / 841 CAR)
        kind = "plain" if s == 840 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d583-c_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form == 'ucb' else ''}__{car}sidedcmean__s{s}.npz")
    if form == "psd":                                                   # T2-D578: set D's augmented SD lines (843 plain / 844, 845 CAR)
        return one(f"{S}/ni26-t2p-d578-aug_{'stlcarsd' if s >= 844 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("pbase", "pcb"):                                        # T2-D578: set D's augmented base + cb lines (846 plain / 847 CAR)
        kind = "plain" if s == 846 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d578-aug_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form == 'pcb' else ''}__{car}sidedcmean__s{s}.npz")
    if form == "pmi":                                                   # T2-D578: set D's head-less MI line (seed 848)
        return one(f"{S}/ni26-t2p-d578-plain_mi_s{s}{p}-[0-9]*/{stem}__s{s}.npz")
    if form in ("nsd", "osd"):                                          # T2-D568 B: D557's B lines re-run with dumps (n = REF dose, o = stronger dose)
        grp = "bref" if form[0] == "n" else "barm"
        return one(f"{S}/ni26-t2p-d568-{grp}_{'stlcarsd' if s >= 829 else 'stlsd'}_s{s}_{'a' if f < 5 else 'b'}-[0-9]*/{stem}__s{s}.npz")
    if form in ("nbase", "ncb", "obase", "ocb"):
        grp = "bref" if form[0] == "n" else "barm"
        kind = "plain" if s == 835 else "car"
        car = "car__" if kind == "car" else ""
        return one(f"{S}/ni26-t2p-d568-{grp}_{kind}_s{s}{p}-[0-9]*/{stem}sddcmean{'cb' if form.endswith('cb') else ''}__{car}sidedcmean__s{s}.npz")
    if form in ("lmi", "lsd"):                                          # T2-D565 C': a D558 line fine-tuned on the local 5090 (all 10 folds, one dir)
        return one(f"{O}/t2_d565/{'mi' if form == 'lmi' else 'sd'}_s{s}/{stem}__s{s}.npz")
    if form == "ami":                                                   # T2-D567: the head-less MI line + D557's augmentation (seed 832 B / 842 C)
        if str(s) in LMI567.split(","):                                 # the local 5090 backup line (all 10 folds in one dir, as D565's lmi)
            return one(f"{O}/t2_d567/mi_s{s}/{stem}__s{s}.npz")
        return one(f"{S}/ni26-t2p-d567-aug_mi_s{s}{p}-[0-9]*/{stem}__s{s}.npz")
    raise SystemExit(f"unknown FM form {form}")


# ---------------- side nets: S10 (D543), ShallowNet at 65 epochs (D547), FBCNet (D538), H at lam 1 (D539b) ----------------
_FMSPL = os.environ.get("NI26_LAB_FMSPLICE", "")         # T2-D642: a dir of spliced FM member files under the originals' basenames; absent = unchanged
if _FMSPL:
    assert os.path.isdir(_FMSPL) and os.path.basename(_FMSPL.rstrip("/")).startswith("fmspl_"), _FMSPL
    _fm_member_d642 = fm_member

    def fm_member(s, f, form):                          # noqa: F811 (T2-D642: the spliced copy of exactly the file the read would load)
        p_ = _fm_member_d642(s, f, form)
        o_ = os.path.join(_FMSPL, os.path.basename(p_))
        return o_ if os.path.exists(o_) else p_
S10, SH65, WITH_FBC, H_ON = False, False, False, False
SIDE_DIR = {816: dict(fbc=O / "t2_d538/side_fbc/runs", s10=O / "t2_d543", sh65=O / "t2_d547/side_sh65/runs"),
            819: dict(fbc=O / "t2_d540/side_fbc/runs", s10=O / "t2_d543", sh65=O / "t2_d547/side_sh65/runs"),
            825: dict(fbc=None, s10=O / "t2_d555/s10", sh65=None),
            828: dict(fbc=None, s10=O / "t2_d557/s10", sh65=None),
            831: dict(fbc=None, s10=O / "t2_d558/s10", sh65=None),
            822: dict(fbc=None, s10=O / "t2_d553/s10", sh65=None),
            834: dict(fbc=None, s10=O / "t2_d578/s10", sh65=None),
            860: dict(fbc=None, s10=O / "t2_d615/s10", sh65=None)}[DSET]   # T2-D615: set E's S10 half   # T2-D578 (Track3 F1 on 1b3a149f): set D's S10 half (t2_d578_chain_local.sh)
if D559[0].startswith("t2_d580_"):                         # T2-D580: the root's own S10 half (the mixed side nets; S5 is root-relative already)
    SIDE_DIR = dict(SIDE_DIR, s10=O / D559[0] / "s10")
FBC_STEM = "lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr__sidedcmean__s{s}.npz"
_side_files_d532 = side_files


def side_files(fam, f, d):
    m_, rec = REC[fam]
    if SH65 and fam == "sh":
        fs = [SIDE_DIR["sh65"] / f"lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e65_vr__sidedcmean__s{d + k * 1000}.npz" for k in range(5)]
    else:
        fs = list(_side_files_d532(fam, f, d))
    if S10:
        fs += [SIDE_DIR["s10"] / f"side_{fam}/runs/lc9_{f}__{m_}__ea__n00__{rec}__sidedcmean__s{d + k * 1000}.npz" for k in range(5, 10)]
    return fs


# D539's in-stream head re-fit, verbatim (scripts/local/t2_d539_read.py at 3fd109bf), lam 1 (D539b), teacher = CNN / teacher_p1
TEACH = Path(os.environ["NI26_TEACH_DIR"]) if os.environ.get("NI26_TEACH_DIR") else CNN / "teacher_p1"   # T2-D564: a heads-only teacher variant
assert not os.environ.get("NI26_TEACH_DIR") or ("/t2_d564_" in os.environ["NI26_TEACH_DIR"] and os.environ["NI26_TEACH_DIR"].endswith(f"_{DSET}/teacher_p1")), os.environ.get("NI26_TEACH_DIR")
CKS = tuple(int(x) for x in D559[1].split(",")) if D559[1] else (60, 100, 150)
CPRIME = set(x for x in os.environ.get("NI26_CPRIME", "").split(",") if x)   # T2-D565: the reduced local C's fresh lines (831 only)
assert CPRIME <= {"mi842", "sd837", "sd838"} and (not CPRIME or DSET == 831), CPRIME   # T2-D559 overrides the refit points with its root
_TCACHE = {}


def teacher(f, d):
    if (f, d) not in _TCACHE:
        z = np.load(need(TEACH / f"lc9_{f}__s{d}.npz"))
        rows = z["rows"].astype(np.int64)
        _TCACHE[(f, d)] = ({int(r): k for k, r in enumerate(rows)}, {c: z[f"post_r{c}"].astype(np.float64) for c in CKS})
    return _TCACHE[(f, d)]


def conf_half(pp):
    ya, cf = pp.argmax(1), pp.max(1)
    keep = np.zeros(len(ya), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return ya, keep


def d584_weights(pp):
    """T2-D584 (registered 2026-11-18 16:26:28; the cap 3 folded in at 16:42:11, before any number): EVERY prefix window, label = argmax of the
    refit's teacher posterior, weight = max_q p(q) x min(c / (3 n_yhat), 3) (c = the prefix length), normalised to mean 1."""
    ya = pp.argmax(1)
    n_ = np.bincount(ya, minlength=3).astype(np.float64)
    w = pp.max(1) * np.minimum(len(pp) / (3.0 * n_[ya]), 3.0)
    return ya, w / w.mean()


def fit_head(Ls, Ss, ys, W0, b0, lam, sw=None):
    if sw is not None:                                    # T2-D584: the per-sample weighted loss (mean-1 weights); None = the code below
        return _fit_head_w(Ls, Ss, ys, W0, b0, lam, np.asarray(sw, np.float64))
    K, D = W0.shape
    n = len(ys)
    th0 = np.concatenate([W0.ravel(), b0])
    Y1 = np.eye(K)[ys]

    def fg(th):
        W_, b_ = th[:K * D].reshape(K, D), th[K * D:]
        z = Ls + Ss @ W_.T + b_
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        G = (np.exp(z - lse[:, None]) - Y1) / n
        dth = th - th0
        f_ = float((lse - z[np.arange(n), ys]).mean() + 0.5 * lam * (dth @ dth))
        return f_, np.concatenate([(G.T @ Ss).ravel(), G.sum(0)]) + lam * dth

    r = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(gtol=1e-8, maxiter=500))
    return r.x[:K * D].reshape(K, D), r.x[K * D:], bool(r.success)


def _fit_head_w(Ls, Ss, ys, W0, b0, lam, sw):
    """T2-D584: fit_head with the CE terms weighted by sw (the same pull lam toward the trained head; sw all-ones == fit_head)."""
    K, D = W0.shape
    n = len(ys)
    assert sw.shape == (n,) and np.isfinite(sw).all() and (sw >= 0).all(), sw.shape
    th0 = np.concatenate([W0.ravel(), b0])
    Y1 = np.eye(K)[ys]

    def fg(th):
        W_, b_ = th[:K * D].reshape(K, D), th[K * D:]
        z = Ls + Ss @ W_.T + b_
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        G = sw[:, None] * (np.exp(z - lse[:, None]) - Y1) / n
        dth = th - th0
        f_ = float((sw * (lse - z[np.arange(n), ys])).mean() + 0.5 * lam * (dth @ dth))
        return f_, np.concatenate([(G.T @ Ss).ravel(), G.sum(0)]) + lam * dth

    r = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(gtol=1e-8, maxiter=500))
    return r.x[:K * D].reshape(K, D), r.x[K * D:], bool(r.success)


def softmax(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def side_prob_h(path, f, lam=1.0):
    z = np.load(need(path))
    refuse_oracle(path, z)                               # T2-D550: the manifest (the name: need)
    te = z["test_idx"].astype(np.int64)
    L, Wh, bh = (z[k].astype(np.float64) for k in ("side_logit_last", "side_w", "side_b"))
    S0 = z["side_te"].astype(np.float64)
    d = int(re.search(r"__s(\d+)\.npz$", str(path)).group(1)) % 1000
    assert d in DRAWS, (path, d)
    p = softmax(L + S0 @ Wh.T + bh)
    pos, post = teacher(f, d)
    st_te, od_te = STREAM[te], ORD[te]
    for s in np.unique(st_te):
        j = np.flatnonzero(st_te == s)
        j = j[np.argsort(od_te[j], kind="stable")]
        if len(j) <= CKS[0]:
            continue
        for k, c in enumerate(CKS):
            if len(j) <= c:
                break
            e = CKS[k + 1] if k + 1 < len(CKS) else len(j)
            a = j[:c]
            pp = np.stack([post[c][pos[int(r)]] for r in te[a]])
            fin = np.isfinite(pp).all(1)
            if D584:                                     # T2-D584: every prefix window, weighted (P1's own selection is the root's, unchanged)
                ya, wv = d584_weights(pp[fin])
                sel, ys = a[fin], ya.astype(np.int64)
                CHK["d584_side_fits"] = CHK.get("d584_side_fits", 0) + 1
            else:
                ya, keep = conf_half(pp[fin])
                sel, ys, wv = a[fin][keep], ya[keep].astype(np.int64), None
            W_, b_, _ = fit_head(L[sel], S0[sel], ys, Wh, bh, lam, sw=wv)
            q = j[c:e]
            p[q] = softmax(L[q] + S0[q] @ W_.T + b_)
    return te, p


_side_prob_unc = side_prob


def side_prob(path, f, mode):
    if H_ON:
        _side_prob_unc(path, f, "unc")                   # keeps the z / recomputation checks on every file
        return side_prob_h(path, f)
    return _side_prob_unc(path, f, mode)


# ---------------- T2-D553: the re-run FM members' probabilities from their head dumps; the in-stream refit of their side heads ----------------
FM_REFIT, FM_MODE = False, "side"
_load_d532 = load
_FMD = {}


def fm_refit_probs(L, S0, Wh, bh, te, f, d, lam=1.0):
    """D539b's per-stream loop (side_prob_h's, verbatim in substance) on an FM member's pre-head logits: re-fits at CKS against draw d's teacher."""
    p = softmax(L + S0 @ Wh.T + bh)
    pos, post = teacher(f, d)
    st_te, od_te = STREAM[te], ORD[te]
    for s in np.unique(st_te):
        j = np.flatnonzero(st_te == s)
        j = j[np.argsort(od_te[j], kind="stable")]
        if len(j) <= CKS[0]:
            continue
        for k, c in enumerate(CKS):
            if len(j) <= c:
                break
            e = CKS[k + 1] if k + 1 < len(CKS) else len(j)
            a = j[:c]
            pp = np.stack([post[c][pos[int(r)]] for r in te[a]])
            fin = np.isfinite(pp).all(1)
            ya, keep = conf_half(pp[fin])
            sel, ys = a[fin][keep], ya[keep].astype(np.int64)
            W_, b_, ok = fit_head(L[sel], S0[sel], ys, Wh, bh, lam)
            CHK["fm_refit_fits"] = CHK.get("fm_refit_fits", 0) + 1
            CHK["fm_refit_failed"] = CHK.get("fm_refit_failed", 0) + int(not ok)
            q = j[c:e]
            p[q] = softmax(L[q] + S0[q] @ W_.T + b_)
    return p


LAM_MAIN = 1.0                                            # T2-D555: the joint refit's pull (1 primary, 10 descriptive)


def fm_dump_prob(path, fd=None, mode="side"):
    """a D553 / D555 re-run member on IX[f]'s order, from its dumps (both arms). D553 files: softmax(L + z W^T + b); D555 files: the base
    logits are the main head on the dumped features (feat W_m^T + b_m), + the side head where the member has one. fd = (f, d) -> re-fitted
    in-stream: mode 'side' = D553 (the side head only; a head-less member is not re-fitted), 'joint' = D555 (main + side jointly, lam LAM_MAIN)."""
    key = (str(path), fd, mode, LAM_MAIN if mode == "joint" else None)
    if key not in _FMD:
        z = np.load(need(path))
        refuse_oracle(path, z)
        f = int(re.search(r"/lc9_(\d)__", str(path)).group(1))
        te = z["test_idx"].astype(np.int64)
        has_side = "side_w" in z.files
        if "fm_feat_te" in z.files:
            Fm, Wm, bm = (z[k].astype(np.float64) for k in ("fm_feat_te", "fm_out_w", "fm_out_b"))
            L = Fm @ Wm.T + bm
        else:
            Fm = None
            L = z["side_logit_last"].astype(np.float64)
        if has_side:
            Wh, bh = z["side_w"].astype(np.float64), z["side_b"].astype(np.float64)
            S0 = z["side_te"].astype(np.float64)
            p0 = softmax(L + S0 @ Wh.T + bh)
        else:
            p0 = softmax(L)
        pl = z["prob_last"].astype(np.float64)
        CHK["fm_dump_vs_prob_last"] = max(CHK.get("fm_dump_vs_prob_last", 0.0), float(np.abs(p0 - pl).max()))
        CHK["fm_dump_argmax_agree_min"] = min(CHK.get("fm_dump_argmax_agree_min", 1.0), float(np.mean(p0.argmax(1) == pl.argmax(1))))
        CHK["fm_dump_files"] = CHK.get("fm_dump_files", 0) + int(fd is None)
        if fd is None or (mode == "side" and not has_side):
            p = p0
        elif mode == "side":
            p = fm_refit_probs(L, S0, Wh, bh, te, *fd)
        else:
            assert mode == "joint" and Fm is not None, (path, mode)
            X = np.concatenate([Fm, S0], 1) if has_side else Fm
            W0 = np.concatenate([Wm, Wh], 1) if has_side else Wm
            b0 = bm + bh if has_side else bm
            p = fm_refit_probs(np.zeros_like(L), X, W0, b0, te, *fd, lam=LAM_MAIN)
            CHK["fm_joint_refits"] = CHK.get("fm_joint_refits", 0) + 1
        assert set(te.tolist()) == set(IX[f].tolist()), (path, "test rows")
        _FMD[key] = (IX[f], realign(te, p, IX[f]))
    return _FMD[key]


def load(path):
    if any(f"/ni26-t2p-{t_}-" in str(path) for t_ in ("d553", "d555", "d557", "d558")) or "/t2_d565/" in str(path):
        return fm_dump_prob(path)
    return _load_d532(path)


_parts_of_d536 = parts_of


def parts_of(f, d, parts):
    P = _parts_of_d536(f, d, parts)
    if WITH_FBC:
        ix = IX[f]
        S_ = np.mean([realign(*side_prob(SIDE_DIR["fbc"] / FBC_STEM.format(f=f, s=d + k * 1000), f, "unc"), ix) for k in range(5)], 0)
        P = dict(P)
        P["fbc"] = (2.0 / 3.0) * P["fbc"] + (1.0 / 3.0) * S_
    if FM_REFIT:                                          # T2-D553: the FM slot with its headed members re-fitted against draw d's teacher
        ps = []
        for s_, form in fm_forms(parts):
            path = fm_member(s_, f, form)
            dumped = any(f"/ni26-t2p-{t_}-" in str(path) for t_ in ("d553", "d555", "d557", "d558")) or "/t2_d565/" in str(path)
            ix_, p_ = fm_dump_prob(path, (f, d), FM_MODE) if dumped else load(path)
            assert (ix_ == IX[f]).all(), (f, s_, form)
            ps.append(p_)
        P = dict(P)
        P["fm"] = np.mean(ps, 0)
    return P


W_W18 = dict(W)

# ---------------- T2-D551 items: D526 (the all-ten share pick on 810-812, fixed before this read) and MIX (LightGBM + XGBoost DC bags) ----------
DC_LGB, DCVEC_LGB = DC, DCVEC
if D551:
    W_D526 = json.load(open(O / "t2_d551_d526_pick.json"))["D526 all-ten pick (810-812, W17 context; the registered single form)"]
    assert set(W_D526) == set(W_W18) and abs(sum(W_D526.values()) - 1.0) < 1e-9, W_D526

    def dc_combine(paths):
        """D535's combine(): the equal-weight mean of several DC bags (same test rows), with its own cross-fitted start vectors."""
        zs = [np.load(need(p)) for p in paths]
        D_ = {}
        for f in range(10):
            te = zs[0][f"te_{f}"]
            assert all((z[f"te_{f}"] == te).all() for z in zs)
            D_[f] = (te, np.mean([z[f"bag_{f}"] for z in zs], 0))
        V = {}
        for f in range(10):
            v = np.mean([D_[g][1][U[D_[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
            V[f] = v / v.sum()
        return D_, V

    _d0, _v0 = dc_combine([DCF])
    assert all((_d0[f][0] == DC_LGB[f][0]).all() and np.array_equal(_d0[f][1], DC_LGB[f][1]) and np.array_equal(_v0[f], DCVEC_LGB[f])
               for f in range(10)), "dc_combine([LightGBM]) != the reader's DC"
    DC_MIX, DCVEC_MIX = dc_combine([DCF, O / "t2_d535/pred_s10-11-12_dc_xgb.npz"])
    print(f"D551: D526 shares {W_D526} (W18 {W_W18}); MIX = LightGBM 10-12 + XGBoost 10-12 bags", flush=True)


def base_swaps():
    """the FM slots BASE uses (fresh seeds at 819 for every launched FM item)."""
    sw = {}
    if DSET == 831 and CPRIME:                            # T2-D565 C': fresh local lines where present, D555's fits (A / B's) elsewhere
        return {826: "lmi@842" if "mi842" in CPRIME else "emi@832", 827: "lsd@837" if "sd837" in CPRIME else "esd",
                829: "lsd@838" if "sd838" in CPRIME else "esd", 831: "esd", 828: "ebase@835", 830: "ebase@836"}
    if DSET in (834, 860):                                # T2-D578: set D = W20+D557's members at seeds 843-848 (no plain lines exist on D); T2-D615: set E = the same FM members
        return {826: "pmi@848", 827: "psd@843", 829: "psd@844", 831: "psd@845", 828: "pbase@846", 830: "pbase@847"}
    if DSET == 831:                                       # T2-D558: the second FM-seed set (837-842), every member with the dumps
        return {826: "hmi@842", 827: "hsd@837", 829: "hsd@838", 831: "hsd@839", 828: "hbase@840", 830: "hbase@841"}
    if DSET in (825, 828):                                # T2-D555 / D557: all six members = D555's re-runs with the head + feature dumps
        return {826: "emi@832", 827: "esd", 829: "esd", 831: "esd", 828: "ebase@835", 830: "ebase@836"}
    if DSET == 822:                                       # T2-D553: every headed member = its re-run with the dump; slot 0 = MI s832
        return {826: "mi@832", 827: "dsd", 829: "dsd", 831: "dsd", 828: "dbase@835", 830: "dbase@836"}
    if DSET == 819:
        sw[826] = "mi@832"
        if "D544" in ITEMS:
            sw[827] = "sd@833"
        if "D546a" in ITEMS or "D546b" in ITEMS:
            sw[828], sw[830] = "cbase@835", "cbase@836"
    else:
        if "D546a" in ITEMS or "D546b" in ITEMS:
            sw[828], sw[830] = "cbase", "cbase"
    return sw


PLAIN = False                                             # check mode only: the nomination reader's BASE (W18's form, no fresh-slot swaps)


def configure(on):
    """set the globals for BASE + the items in `on`."""
    global FM_SWAP, S10, SH65, WITH_FBC, H_ON, W, DC, DCVEC, FM_REFIT, FM_MODE
    FM_SWAP = {} if PLAIN else base_swaps()
    S10, SH65, WITH_FBC, H_ON = "D543" in on, "D547" in on, "D538" in on, "D539b" in on
    W = dict(W_W18)
    DC, DCVEC = (DC_MIX, DCVEC_MIX) if "MIX" in on else (DC_LGB, DCVEC_LGB)
    if "D526" in on:
        W = dict(W_D526)
    if "D545" in on:
        s_ = (0.65 - 0.15) / (W_W18["fbc"] + W_W18["sh"])
        W = {"fbc": W_W18["fbc"] * s_, "sh": W_W18["sh"] * s_, "ifn": 0.15, "fm": 0.35}
    sfx = "@832" if DSET == 819 else ""
    if "D541" in on:
        FM_SWAP[826] = "misd" + sfx
    if "D544" in on:
        FM_SWAP[827] = "wa" + ("@833" if DSET == 819 else "")
    for a in ("a", "b"):
        if f"D546{a}" in on:
            FM_SWAP[828], FM_SWAP[830] = (f"c{a}@835", f"c{a}@836") if DSET == 819 else (f"c{a}", f"c{a}")
    if DSET == 822 and "D546b" in on:
        FM_SWAP[828], FM_SWAP[830] = "dcb@835", "dcb@836"
    if DSET in (825, 828) and "D546b" in on:
        FM_SWAP[828], FM_SWAP[830] = "ecb@835", "ecb@836"
    if DSET in (834, 860) and "D546b" in on:                      # T2-D578: D546b's continuation on set D's base lines
        FM_SWAP[828], FM_SWAP[830] = "pcb@846", "pcb@847"
    if DSET == 831 and "D546b" in on:
        FM_SWAP[828], FM_SWAP[830] = ("ecb@835", "ecb@836") if CPRIME else ("hcb@840", "hcb@841")
    if DSET == 831 and "D557" in on:                      # T2-D558: the augmented new-seed members
        FM_SWAP.update({827: "ksd@837", 829: "ksd@838", 831: "ksd@839"})
        FM_SWAP[828], FM_SWAP[830] = ("kcb@840", "kcb@841") if "D546b" in on else ("kbase@840", "kbase@841")
    if DSET == 828 and "D557" in on:                      # T2-D557: the five headed members -> their augmented re-runs (same seeds)
        FM_SWAP.update({827: "gsd", 829: "gsd", 831: "gsd"})
        FM_SWAP[828], FM_SWAP[830] = ("gcb@835", "gcb@836") if "D546b" in on else ("gbase@835", "gbase@836")
    if "D567" in on:                                      # T2-D567: slot 0 (the head-less MI member) -> its augmented re-run, same seed
        assert DSET in (828, 831), DSET
        FM_SWAP[826] = {828: "ami@832", 831: "ami@842"}[DSET]
    if "D568" in on:                                      # T2-D568: the five k members -> their stronger-dose twins (m forms, same seeds)
        assert DSET == 831 and "D557" in on, (DSET, on)
        FM_SWAP.update({827: "msd@837", 829: "msd@838", 831: "msd@839"})
        FM_SWAP[828], FM_SWAP[830] = ("mcb@840", "mcb@841") if "D546b" in on else ("mbase@840", "mbase@841")
    if "D582" in on:                                      # T2-D582: the five headed members -> their vp 2 twins (C: q forms; D: t forms), same seeds
        assert DSET in (831, 834) and ("D557" in on or DSET == 834), (DSET, on)
        pre, sd_, bs_ = ("q", (837, 838, 839), (840, 841)) if DSET == 831 else ("t", (843, 844, 845), (846, 847))
        FM_SWAP.update({827: f"{pre}sd@{sd_[0]}", 829: f"{pre}sd@{sd_[1]}", 831: f"{pre}sd@{sd_[2]}"})
        FM_SWAP[828], FM_SWAP[830] = (f"{pre}cb@{bs_[0]}", f"{pre}cb@{bs_[1]}") if "D546b" in on else (f"{pre}base@{bs_[0]}", f"{pre}base@{bs_[1]}")
    if "D583" in on:                                      # T2-D583: the five headed members -> their recombination twins (C: u forms; D: x forms), same seeds
        assert DSET in (831, 834) and ("D557" in on or DSET == 834) and not {"D568", "D582"} & set(on), (DSET, on)
        pre, sd_, bs_ = ("u", (837, 838, 839), (840, 841)) if DSET == 831 else ("x", (843, 844, 845), (846, 847))
        FM_SWAP.update({827: f"{pre}sd@{sd_[0]}", 829: f"{pre}sd@{sd_[1]}", 831: f"{pre}sd@{sd_[2]}"})
        FM_SWAP[828], FM_SWAP[830] = (f"{pre}cb@{bs_[0]}", f"{pre}cb@{bs_[1]}") if "D546b" in on else (f"{pre}base@{bs_[0]}", f"{pre}base@{bs_[1]}")
    if "D586" in on:                                      # T2-D586: ONE member, the plain SD slot 827 (C: 837 -> v; D: 843 -> z)
        assert DSET in (831, 834) and ("D557" in on or DSET == 834) and not {"D568", "D582", "D583"} & set(on), (DSET, on)
        assert FM_SWAP.get(827) == {831: "ksd@837", 834: "psd@843"}[DSET], (DSET, FM_SWAP.get(827))
        FM_SWAP[827] = {831: "vsd@837", 834: "zsd@843"}[DSET]
    if "D611" in on:                                      # T2-D611: ONE member, the headed CAR SD slot 831 (C: 839 -> bpcsd; D: 845 -> bpdsd)
        assert DSET in (831, 834) and ("D557" in on or DSET == 834) and not {"D568", "D582", "D583", "D586", "D588", "D596"} & set(on), (DSET, on)
        assert FM_SWAP.get(831) == {831: "ksd@839", 834: "psd@845"}[DSET], (DSET, FM_SWAP.get(831))
        FM_SWAP[831] = {831: "bpcsd@839", 834: "bpdsd@845"}[DSET]
    if "D618" in on:                                      # T2-D618: ONE member, the headed CAR SD slot 831 (C: 839 -> mrcsd; D: 845 -> mrdsd)
        assert DSET in (831, 834) and ("D557" in on or DSET == 834) and not {"D568", "D582", "D583", "D586", "D588", "D596", "D611"} & set(on), (DSET, on)
        assert FM_SWAP.get(831) == {831: "ksd@839", 834: "psd@845"}[DSET], (DSET, FM_SWAP.get(831))
        FM_SWAP[831] = {831: "mrcsd@839", 834: "mrdsd@845"}[DSET]
    if "D582B" in on:                                     # T2-D582 B: the five headed B members -> the re-run REF (r) or the vp arm (s), same seeds
        assert DSET == 828 and D582B in ("ref", "arm"), (DSET, on, D582B)   # the lab's B 'on' carries no D557 (all five headed slots replaced here)
        pre = "r" if D582B == "ref" else "s"
        FM_SWAP.update({827: f"{pre}sd", 829: f"{pre}sd", 831: f"{pre}sd"})
        FM_SWAP[828], FM_SWAP[830] = (f"{pre}cb@835", f"{pre}cb@836") if "D546b" in on else (f"{pre}base@835", f"{pre}base@836")
    if "D588" in on:                                      # T2-D588: the five headed members -> their consistency twins (C: j forms; D: i forms), same seeds
        assert DSET in (831, 834) and ("D557" in on or DSET == 834) and not {"D568", "D582", "D583", "D586"} & set(on), (DSET, on)
        pre, sd_, bs_ = ("j", (837, 838, 839), (840, 841)) if DSET == 831 else ("i", (843, 844, 845), (846, 847))
        FM_SWAP.update({827: f"{pre}sd@{sd_[0]}", 829: f"{pre}sd@{sd_[1]}", 831: f"{pre}sd@{sd_[2]}"})
        FM_SWAP[828], FM_SWAP[830] = (f"{pre}cb@{bs_[0]}", f"{pre}cb@{bs_[1]}") if "D546b" in on else (f"{pre}base@{bs_[0]}", f"{pre}base@{bs_[1]}")
    if "D596" in on:                                      # T2-D596: the five headed members -> their WORD-margin twins (C: U forms; D: W forms), same seeds
        assert DSET in (831, 834) and ("D557" in on or DSET == 834) and not {"D568", "D582", "D583", "D586", "D588"} & set(on), (DSET, on)
        pre, sd_, bs_ = ("U", (837, 838, 839), (840, 841)) if DSET == 831 else ("W", (843, 844, 845), (846, 847))
        FM_SWAP.update({827: f"{pre}sd@{sd_[0]}", 829: f"{pre}sd@{sd_[1]}", 831: f"{pre}sd@{sd_[2]}"})
        FM_SWAP[828], FM_SWAP[830] = (f"{pre}cb@{bs_[0]}", f"{pre}cb@{bs_[1]}") if "D546b" in on else (f"{pre}base@{bs_[0]}", f"{pre}base@{bs_[1]}")
    if "D596B" in on:                                     # T2-D596 B ARM: the five headed B members -> their WORD-margin twins (V forms, same seeds)
        assert DSET == 828 and not {"D582B", "D568B", "D583B", "D586B", "D588B"} & set(on), (DSET, on)
        FM_SWAP.update({827: "Vsd", 829: "Vsd", 831: "Vsd"})
        FM_SWAP[828], FM_SWAP[830] = ("Vcb@835", "Vcb@836") if "D546b" in on else ("Vbase@835", "Vbase@836")
    if "D588B" in on:                                     # T2-D588 B ARM: the five headed B members -> their consistency twins (f forms, same seeds)
        assert DSET == 828 and not {"D582B", "D568B", "D583B", "D586B"} & set(on), (DSET, on)
        FM_SWAP.update({827: "fsd", 829: "fsd", 831: "fsd"})
        FM_SWAP[828], FM_SWAP[830] = ("fcb@835", "fcb@836") if "D546b" in on else ("fbase@835", "fbase@836")
    if "D583B" in on:                                     # T2-D583 B ARM: the five headed B members -> their recombination twins (w forms, same seeds)
        assert DSET == 828 and not {"D582B", "D568B", "D586B"} & set(on), (DSET, on)
        FM_SWAP.update({827: "wsd", 829: "wsd", 831: "wsd"})
        FM_SWAP[828], FM_SWAP[830] = ("wcb@835", "wcb@836") if "D546b" in on else ("wbase@835", "wbase@836")
    if "D586B" in on:                                     # T2-D586 B ARM: D582's REF (r forms) with ONLY the plain SD slot 827 -> its surrogate twin
        assert DSET == 828 and "D582B" in on and D582B == "ref" and FM_SWAP.get(827) == "rsd", (DSET, on, D582B, FM_SWAP.get(827))
        FM_SWAP[827] = "ysd@827"
    if "D611B" in on:                                     # T2-D611 B ARM: D582's REF (r forms) with ONLY the CAR SD slot 831 -> its band twin
        assert DSET == 828 and "D582B" in on and D582B == "ref" and FM_SWAP.get(831) == "rsd", (DSET, on, D582B, FM_SWAP.get(831))
        FM_SWAP[831] = "bpbsd@831"
    if "D618B" in on:                                     # T2-D618 B ARM: D582's REF (r forms) with ONLY the CAR SD slot 831 -> its mirror twin
        assert DSET == 828 and "D582B" in on and D582B == "ref" and FM_SWAP.get(831) == "rsd", (DSET, on, D582B, FM_SWAP.get(831))
        FM_SWAP[831] = "mrbsd@831"
    if "D568B" in on:                                     # T2-D568 B: the five headed B members -> the re-run REF (n) or the stronger dose (o), same seeds
        assert DSET == 828 and D568B in ("ref", "arm"), (DSET, on, D568B)
        pre = "n" if D568B == "ref" else "o"
        FM_SWAP.update({827: f"{pre}sd", 829: f"{pre}sd", 831: f"{pre}sd"})
        FM_SWAP[828], FM_SWAP[830] = (f"{pre}cb@835", f"{pre}cb@836") if "D546b" in on else (f"{pre}base@835", f"{pre}base@836")
    FM_REFIT = "D553" in on or "D555" in on
    FM_MODE = "joint" if "D555" in on else "side"


def read(on):
    configure(on)
    out = mix(W18)
    configure([])
    return out


def judge(Aa, Bb):
    sc = score(Aa, Bb)
    return dict(per_stranger=per_stranger(Aa, Bb), **sc)


BASE = read([])
res = {"draws": list(DRAWS), "items": ITEMS, "base_fm_swaps": base_swaps(), "level": {"BASE": level(BASE)}}
if DSET == 831:                                           # T2-D558: the SECOND read of D555 or D557 (one item), new FM seeds, draws 831-833
    it = "D555" if "D555" in ITEMS else "D557"
    REF = read(W19C + ["D553"])
    ARM = read(W19C + (["D555"] if it == "D555" else ["D553", "D557"]))
    res["level"].update({"W19": level(REF), f"W19 + {it}": level(ARM)})
    res["W19 vs BASE (descriptive)"] = judge(REF, BASE)
    sc = judge(ARM, REF)
    sc["second_read_pass"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
    res[f"{it}-2: W19 + {it} vs W19 (the second read)"] = sc
    res["checks"] = dict(CHK)
    assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6 and CHK["fm_dump_vs_prob_last"] <= 0.05, CHK
    for k, v in res.items():
        print(f"D558 {k}: {v}", flush=True)
    print(f"D558 {it}-2 verdict: {'PASS' if sc['second_read_pass'] else 'FAIL'} ({time.time() - T0:.0f} s)", flush=True)
    json.dump(res, open(O / f"t2_d558_{it.lower()}_read.json", "w"), indent=1, default=str)
    raise SystemExit(0)
if DSET == 828:                                           # T2-D557: ARM = W19 with the five headed members augmented vs REF = W19, 828-830
    REF = read(W19C + ["D553"])
    ARM = read(W19C + ["D553", "D557"])
    res["level"].update({"W19": level(REF), "W19 + D557": level(ARM)})
    res["W19 vs BASE (descriptive)"] = judge(REF, BASE)
    sc = judge(ARM, REF)
    sc["stack_bar"] = bool(sc["ten"] >= 0.001 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020
                           and sc["returning"] >= -0.003 and sc["sealed_mix"] > 0)
    res["D557: W19 + D557 vs W19 (the decision, D540a's stack bar)"] = sc
    res["checks"] = dict(CHK)
    assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6 and CHK["fm_dump_vs_prob_last"] <= 0.05, CHK
    for k, v in res.items():
        print(f"D557 {k}: {v}", flush=True)
    print(f"D557 verdict: {'PASS' if sc['stack_bar'] else 'FAIL'} ({time.time() - T0:.0f} s)", flush=True)
    json.dump(res, open(O / "t2_d557_read.json", "w"), indent=1, default=str)
    raise SystemExit(0)
if DSET == 825:                                           # T2-D555: ARM = W19 with the joint main + side refit vs REF = W19 (D553), 825-827
    REF = read(W19C + ["D553"])
    ARM = read(W19C + ["D555"])
    LAM_MAIN = 10.0
    ARM10 = read(W19C + ["D555"])
    LAM_MAIN = 1.0
    res["level"].update({"W19": level(REF), "W19 + D555 (lam 1)": level(ARM), "W19 + D555 (lam 10)": level(ARM10)})
    res["W19 vs BASE (descriptive)"] = judge(REF, BASE)
    sc = judge(ARM, REF)
    sc["pass"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
    res["D555: W19 + D555 (lam 1) vs W19 (the decision)"] = sc
    res["D555 lam 10 vs W19 (descriptive)"] = judge(ARM10, REF)
    res["checks"] = dict(CHK)
    assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6 and CHK["fm_dump_vs_prob_last"] <= 0.05, CHK
    assert CHK.get("fm_joint_refits", 0) > 0 and CHK["fm_refit_fits"] > 0, CHK
    for k, v in res.items():
        print(f"D555 {k}: {v}", flush=True)
    print(f"D555 verdict: {'PASS' if sc['pass'] else 'FAIL'} ({time.time() - T0:.0f} s)", flush=True)
    json.dump(res, open(O / "t2_d555_read.json", "w"), indent=1, default=str)
    raise SystemExit(0)
if DSET == 822:                                           # T2-D553: ARM = W19c + the FM side-head refit vs REF = W19c, fresh draws 822-824
    REF = read(W19C)
    ARM = read(W19C + ["D553"])
    res["level"].update({"W19c": level(REF), "W19c + D553": level(ARM)})
    res["W19c vs BASE (descriptive)"] = judge(REF, BASE)
    sc = judge(ARM, REF)
    sc["pass"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
    res["D553: W19c + D553 vs W19c (the decision)"] = sc
    res["checks"] = dict(CHK)
    assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6 and CHK["fm_dump_vs_prob_last"] <= 0.05, CHK
    assert CHK["fm_dump_files"] == 70 and CHK["fm_refit_fits"] > 0, CHK   # BASE: SD 827 / 829 / 831 + bases 835 / 836; REF adds cb 835 / 836 (x 10 folds)
    for k, v in res.items():
        print(f"D553 {k}: {v}", flush=True)
    print(f"D553 verdict: {'PASS' if sc['pass'] else 'FAIL'} ({time.time() - T0:.0f} s)", flush=True)
    json.dump(res, open(O / "t2_d553_read.json", "w"), indent=1, default=str)
    raise SystemExit(0)
if DSET == 816:                                           # the reader's own check vs the nomination reader (single items)
    nom = {}
    for fn in glob.glob(str(O / "t2_w19_nom_*.json")):
        for k, v in json.load(open(fn)).items():
            if isinstance(v, dict) and "ten" in v:
                nom[k] = v
    for it in ITEMS:
        if it == "D539b":
            continue
        PLAIN = not it.startswith("D546")                 # D546 vs its own jobs' bases (BASE'); every other item vs W18's form
        sc = judge(read([it]), read([]))
        PLAIN = False
        res[f"{it} vs BASE (check)"] = sc
        if it in nom:
            for k in ("ten", "draws", "min_user", "returning"):
                assert sc[k] == nom[it][k], (it, k, sc[k], nom[it][k])
            print(f"D540a check816 {it}: reproduces the nomination reader exactly", flush=True)
    print(f"D540a check816: PASS ({time.time() - T0:.0f} s)", flush=True)
    json.dump(res, open(O / "t2_d540a_check816.json", "w"), indent=1, default=str)
    raise SystemExit(0)

if D551:                                                  # regression check: D540a's W19-final headline, exactly, before any new number
    ref = json.load(open(O / "t2_d540a_stack.json"))["HEADLINE W19-final vs BASE (819-821)"]
    chk = judge(read(["D539b", "D543", "D546b"]), BASE)
    for k in ("ten", "draws", "min_user", "returning", "sealed_mix", "per_stranger"):
        assert chk[k] == ref[k], (k, chk[k], ref[k])
    print(f"D551 check: read(D539b, D543, D546b) vs BASE reproduces D540a's headline exactly ({time.time() - T0:.0f} s)", flush=True)
FULL = read(ITEMS)
res["level"]["FULL"] = level(FULL)
res["FULL vs BASE"] = judge(FULL, BASE)
keep = []
for it in ITEMS:
    rest = [x for x in ITEMS if x != it]
    sc = judge(FULL, read(rest))
    sc["kept"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2)
    res[f"LOO {it}: FULL vs FULL - {it}"] = sc
    if sc["kept"]:
        keep.append(it)
    print(f"D540a LOO {it}: {'KEPT' if sc['kept'] else 'DROPPED'} {sc}", flush=True)
FINAL = read(keep) if keep else BASE
res["kept"] = keep
res["level"]["W19-final"] = level(FINAL)
head = judge(FINAL, BASE) if keep else None
if head is not None:
    head["stack_bar"] = bool(head["ten"] >= 0.001 and sum(x > 0 for x in head["draws"]) >= 2 and head["min_user"] >= -0.020
                             and head["returning"] >= -0.003 and head["sealed_mix"] > 0)
res["HEADLINE W19-final vs BASE (819-821)"] = head
res["checks"] = dict(CHK)
assert CHK["unc_vs_prob_last"] <= 1e-5 and CHK["z_vs_side_z"] <= 1e-6, CHK
for k, v in res.items():
    print(f"D540a {k}: {v}", flush=True)
w19 = bool(head and head["stack_bar"])
print(f"D540a verdict: {'W19 = BASE + ' + ', '.join(keep) + ' (draft the owner line; build on W18 b2)' if w19 else 'NO W19 (D536 last-upload rule)'}"
      f" ({time.time() - T0:.0f} s)", flush=True)
json.dump(res, open(O / ("t2_d551_stack.json" if D551 else "t2_d540a_stack.json"), "w"), indent=1, default=str)
