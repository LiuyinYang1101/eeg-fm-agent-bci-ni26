#!/usr/bin/env python
"""T2-D594 (a set-valued loss for ambiguous P1 rows) — the CACHED DIAGNOSTIC, descriptive, before any D594 job:
on the P1 students' own selected rows (W21's LP teacher, refits 60 / 100 / 150, 'confR': the prefix re-decoded posterior; the confident half
of each predicted class = t2p_bench labels_), a row is AMBIGUOUS iff the pre-LP teacher (the fb root's teacher_p1) and the LP teacher (the lp
root's) disagree specifically MI vs WORD (argmaxes {MI, WORD}, different) AND both rank SUB last. Reported per set (A / B / C / D), strangers
(the fold's held-out user) and returning (P04 / P10's later sessions) apart: coverage (ambiguous / selected), the true-class mix of the
ambiguous rows, how often the forced winner (the LP label) is wrong there, how often the set {MI, WORD} holds the truth, and for comparison
the forced-label error on all selected rows. Labels MI 0 / SUB 1 / WORD 2 (index.csv trig_description).
    mem-guard env OMP_NUM_THREADS=2 python scripts/local/t2_d594_diag.py  ->  outputs/t2_d594_diag.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

O = Path.home() / "workspace/outputs/neural_interface_challenge"
IDX = pd.read_csv(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/index.csv",
                  usecols=["y", "subject", "stream", "order"])
Y, ST, OD, SUBJ = IDX["y"].to_numpy(), IDX["stream"].to_numpy(), IDX["order"].to_numpy(), IDX["subject"].to_numpy()
MI, SUB, WORD = 0, 1, 2
SETS = {"A": ("t2_d571_fbw20_825", "t2_d571_lpw20_825", (825, 826, 827)), "B": ("t2_d571_fbw20_828", "t2_d571_lpw20_828", (828, 829, 830)),
        "C": ("t2_d571_fbw20d557_831", "t2_d571_lpw20d557_831", (831, 832, 833)),
        "D": ("t2_d571_fbw20d557_834", "t2_d571_lpw20d557_834", (834, 835, 836))}
CKS = (60, 100, 150)


def conf_half(pp):
    ya, cf = pp.argmax(1), pp.max(1)
    keep = np.zeros(len(ya), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return ya, keep


def main():
    out = {}
    for S, (fb, lp, draws) in SETS.items():
        acc = {g: dict(sel=0, sel_wrong=0, amb=0, amb_true=[0, 0, 0], amb_lp_wrong=0, amb_pre_wrong=0, amb_set_true=0,
                       lp_word_fixes=0, lp_mi_breaks=0) for g in ("stranger", "returning")}
        for f in range(10):
            for d in draws:
                zl = np.load(O / lp / "teacher_p1" / f"lc9_{f}__s{d}.npz")
                zf = np.load(O / fb / "teacher_p1" / f"lc9_{f}__s{d}.npz")
                rows = zl["rows"].astype(np.int64)
                assert np.array_equal(rows, zf["rows"].astype(np.int64)), (S, f, d)
                held = zl["held"].astype(bool)
                pos = {int(r): k for k, r in enumerate(rows)}
                for s in np.unique(ST[rows]):
                    j = rows[ST[rows] == s]
                    j = j[np.argsort(OD[j], kind="stable")]
                    for c in CKS:
                        if len(j) <= c:
                            break
                        a = j[:c]
                        k_ = np.array([pos[int(r)] for r in a])
                        pl, pf = zl[f"post_r{c}"][k_].astype(np.float64), zf[f"post_r{c}"][k_].astype(np.float64)
                        fin = np.isfinite(pl).all(1) & np.isfinite(pf).all(1)
                        a, k_, pl, pf = a[fin], k_[fin], pl[fin], pf[fin]
                        ya, keep = conf_half(pl)
                        a, k_, pl, pf, ya = a[keep], k_[keep], pl[keep], pf[keep], ya[keep]
                        yp = pf.argmax(1)
                        amb = (np.isin(ya, (MI, WORD)) & np.isin(yp, (MI, WORD)) & (ya != yp) & (pl.argmin(1) == SUB) & (pf.argmin(1) == SUB))
                        for g, gm in (("stranger", held[k_]), ("returning", ~held[k_])):
                            t = Y[a[gm]]
                            A = acc[g]
                            A["sel"] += int(gm.sum())
                            A["sel_wrong"] += int((ya[gm] != t).sum())
                            m = amb[gm]
                            ta, la, pa = t[m], ya[gm][m], yp[gm][m]
                            A["amb"] += int(m.sum())
                            for q in (MI, SUB, WORD):
                                A["amb_true"][q] += int((ta == q).sum())
                            A["amb_lp_wrong"] += int((la != ta).sum())
                            A["amb_pre_wrong"] += int((pa != ta).sum())
                            A["amb_set_true"] += int(np.isin(ta, (MI, WORD)).sum())
                            A["lp_word_fixes"] += int(((ta == WORD) & (la == WORD)).sum())   # the LP label says WORD and is right (a set loss gives this up)
                            A["lp_mi_breaks"] += int(((ta == MI) & (la == WORD)).sum())      # the LP label says WORD and the truth is MI (a set loss avoids this)
        res = {}
        for g, A in acc.items():
            n = max(A["amb"], 1)
            res[g] = dict(selected=A["sel"], forced_error_all_selected=round(A["sel_wrong"] / max(A["sel"], 1), 4), ambiguous=A["amb"],
                          coverage=round(A["amb"] / max(A["sel"], 1), 4), ambiguous_truth_MI_SUB_WORD=A["amb_true"],
                          lp_label_wrong_on_ambiguous=round(A["amb_lp_wrong"] / n, 4), pre_label_wrong_on_ambiguous=round(A["amb_pre_wrong"] / n, 4),
                          set_MI_WORD_holds_truth=round(A["amb_set_true"] / n, 4), lp_says_WORD_correct=A["lp_word_fixes"],
                          lp_says_WORD_truth_MI=A["lp_mi_breaks"])
        out[S] = res
        print(f"[d594] set {S}: " + " | ".join(f"{g}: sel {r['selected']}, forced err {r['forced_error_all_selected']}, amb {r['ambiguous']} "
                                              f"(cov {r['coverage']}), truth MI/SUB/WORD {r['ambiguous_truth_MI_SUB_WORD']}, LP wrong {r['lp_label_wrong_on_ambiguous']}, "
                                              f"pre wrong {r['pre_label_wrong_on_ambiguous']}, set holds truth {r['set_MI_WORD_holds_truth']}"
                                              for g, r in res.items()), flush=True)
    (O / "t2_d594_diag.json").write_text(json.dumps(out, indent=1))
    print(f"[d594] saved {O / 't2_d594_diag.json'}")


if __name__ == "__main__":
    main()
