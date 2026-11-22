#!/usr/bin/env python
"""T2-D552 (DESCRIPTIVE, no decision; written before its numbers; the manager's ask after D551): the PODIUM GAP ANALYSIS on draws 819-821.
Where would +0.004 (the manager's case: W19 ~ W18 + 0.004) or +0.008 (D551: no W19; W18's expected 0.839 -> the podium line 0.847) on the
sealed mix (0.57 x ten strangers + 0.43 x returning, the local proxy) have to come from? The stack reader's head
(scripts/local/t2_d540a_stack_read.py --draws 819) is exec'd unchanged; BASE = W18's form re-trained at 819-821 (the reference);
W19c = BASE + D540a's three parts (D539b, D543, D546b), DESCRIPTIVE only (D551: no W19 by the rule). Per group, for BASE and W19c:
  levels: every stranger (P01-P10), the ten, the returning users (later sessions of the nine training users per fold), per user;
  per class (MI / sub / word): recall averaged over cells, then users (strangers; returning); the strangers' confusion matrix;
  the share of windows wrong; the windows per class per cell (to turn a level gain into windows).
The required gain for each gap G: all from strangers (ten + G / 0.57), all from returning (+ G / 0.43), all from the four weakest strangers
named by the manager (P03 / P06 / P07 / P10), all from one class's recall (via each cell's class count), each also in windows per cell.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d552_gap.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
O_ = Path.home() / "workspace/outputs/neural_interface_challenge"
KEPT = ["D539b", "D543", "D546b"]                        # D540a's composition, descriptive (D551: no W19 by the rule)
ITEMS = KEPT                                              # BASE's fresh-slot swaps = D540a's (D546b's confirm bases s835 / s836)
sys.argv = [sys.argv[0], "--draws", "819", "--items", ",".join(ITEMS)]
SRC = open("scripts/local/t2_d540a_stack_read.py").read()
CUT = "\nBASE = read([])\n"
assert SRC.count(CUT) == 1, "the stack reader's tail moved"
exec(compile(SRC.split(CUT)[0], "t2_d540a_stack_read_head(d552)", "exec"))
assert DRAWS == (819, 820, 821)
CLS = {0: "MI", 1: "sub", 2: "word"}
WEAK = ["P03", "P06", "P07", "P10"]
GAPS = (0.004, 0.008)


def anatomy(A):
    recs, conf = [], np.zeros((3, 3))
    for (f, d), (ix, mx) in A.items():
        pred = mx.argmax(1)
        kind = np.where(U[ix] == TRU[f], "stranger", np.where(np.isin(SES[ix], ("2", "3")), "returning", "other"))
        for kd in ("stranger", "returning"):
            for u in np.unique(U[ix][kind == kd]):
                m = (kind == kd) & (U[ix] == u)
                rows, pr = ix[m], pred[m]
                for c in np.unique(CELL[rows]):
                    mc = CELL[rows] == c
                    yc, pc = Y[rows][mc], pr[mc]
                    ks = np.unique(yc)
                    if len(ks) < 2:
                        continue
                    rec = {int(q): float(np.mean(pc[yc == q] == q)) for q in ks}
                    recs.append(dict(draw=d, kind=kd, user=str(u), cell=str(c), ba=float(np.mean(list(rec.values()))), n=int(mc.sum()),
                                     nk=len(ks), wrong=int((pc != yc).sum()),
                                     **{f"rec_{CLS[q]}": rec.get(q, np.nan) for q in CLS}, **{f"n_{CLS[q]}": int((yc == q).sum()) for q in CLS}))
                    if kd == "stranger":
                        for t, p_ in zip(yc, pc):
                            conf[int(t), int(p_)] += 1
    return pd.DataFrame(recs), conf


out = {"items_W19c (descriptive)": KEPT}
TAB = {}
for name, on in (("BASE", []), ("W19c", KEPT)):
    R, conf = anatomy(read(on))
    TAB[name] = R
    st, rt = R[R.kind == "stranger"], R[R.kind == "returning"]
    pu = st.groupby("user").ba.mean()
    e = dict(ten=round(float(pu.mean()), 4), per_stranger=pu.round(4).to_dict(),
             returning=round(float(rt.groupby("user").ba.mean().mean()), 4), returning_per_user=rt.groupby("user").ba.mean().round(4).to_dict(),
             n_returning_users=int(rt.user.nunique()))
    e["sealed_mix_level"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    for kd, T in (("stranger", st), ("returning", rt)):
        e[f"{kd}_recall"] = {CLS[q]: round(float(T.groupby("user")[f"rec_{CLS[q]}"].mean().mean()), 4) for q in CLS}
        e[f"{kd}_wrong_share"] = round(float(T.wrong.sum() / T.n.sum()), 4)
        e[f"{kd}_windows_per_class_per_cell"] = {CLS[q]: round(float(T[f"n_{CLS[q]}"][T[f"n_{CLS[q]}"] > 0].mean()), 1) for q in CLS}
    e["stranger_confusion_rows_true"] = {CLS[t]: {CLS[p]: round(float(conf[t, p] / conf[t].sum()), 4) for p in CLS} for t in CLS}
    out[name] = e
    print(f"D552 {name}: {json.dumps(e)}", flush=True)

# ---- the required gains for each gap on the sealed mix (local proxy), on BASE's cell structure (W18's form)
st = TAB["BASE"][TAB["BASE"].kind == "stranger"]
for GAP in GAPS:
    need = {"all_from_strangers_ten": round(GAP / 0.57, 4), "all_from_returning": round(GAP / 0.43, 4)}
    need["all_from_P03_P06_P07_P10_each"] = round(GAP / 0.57 * 10 / len(WEAK), 4)
    for q in CLS:
        k = CLS[q]
        # one class's recall up by x in every stranger cell that holds it -> that cell's BA up by x / (classes in the cell)
        fac = float(st.groupby("user").apply(lambda T: np.mean(np.where(T[f"n_{k}"] > 0, 1.0 / T.nk, 0.0))).mean())
        need[f"strangers_{k}_recall"] = round(GAP / 0.57 / fac, 4)
        need[f"strangers_{k}_extra_right_windows_per_cell"] = round(GAP / 0.57 / fac * float(st[f"n_{k}"][st[f"n_{k}"] > 0].mean()), 2)
    out[f"required_for_+{GAP}_sealed"] = need
    print(f"D552 required for +{GAP} sealed: {need}", flush=True)
out["W19c_vs_BASE_per_stranger"] = {u: round(out["W19c"]["per_stranger"][u] - out["BASE"]["per_stranger"][u], 4) for u in out["BASE"]["per_stranger"]}
print(f"D552 W19c vs BASE per stranger: {out['W19c_vs_BASE_per_stranger']}", flush=True)
json.dump(out, open(O_ / "t2_d552_gap.json", "w"), indent=1, default=str)
print(f"D552 wrote {O_ / 't2_d552_gap.json'} ({time.time() - T0:.0f} s)", flush=True)
