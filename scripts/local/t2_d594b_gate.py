#!/usr/bin/env python
"""T2-D594b — the registered ZERO-GPU GATE (before any root): on the P1 students' selected rows (W21's LP teacher, refits 60 / 100 / 150,
'confR', the confident half of each predicted class = t2p_bench labels_), a row is IN the rule iff the LP teacher's top-2 classes are {MI, WORD}
and p_top1 - p_top2 < 0.20 (pinned). On cached sets A and B: coverage among the STRANGER selected rows and the share whose truth lies inside
{MI, WORD}; the gate passes iff coverage >= 3 % AND truth-in-set >= 90 % (both sets pooled). Returning reported beside it, with WORD-fixes
(the LP label WORD and right: supervision the set loss WEAKENS, the risk) vs the LP label WORD with the truth MI (errors it REPAIRS). Labels MI 0 / SUB 1 / WORD 2.
    mem-guard env OMP_NUM_THREADS=2 python scripts/local/t2_d594b_gate.py  ->  outputs/t2_d594b_gate.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

O = Path.home() / "workspace/outputs/neural_interface_challenge"
IDX = pd.read_csv(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/index.csv",
                  usecols=["y", "stream", "order"])
Y, ST, OD = IDX["y"].to_numpy(), IDX["stream"].to_numpy(), IDX["order"].to_numpy()
MI, SUB, WORD, THR = 0, 1, 2, 0.20
SETS = {"A": ("t2_d571_lpw20_825", (825, 826, 827)), "B": ("t2_d571_lpw20_828", (828, 829, 830))}


def conf_half(pp):
    ya, cf = pp.argmax(1), pp.max(1)
    keep = np.zeros(len(ya), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return ya, keep


def main():
    tot = {g: dict(sel=0, rule=0, in_set=0, lp_wrong=0, word_right_weakened=0, word_on_mi_repaired=0, truth=[0, 0, 0]) for g in ("stranger", "returning")}
    per = {}
    for S, (lp, draws) in SETS.items():
        acc = {g: dict(sel=0, rule=0, in_set=0, lp_wrong=0, word_right_weakened=0, word_on_mi_repaired=0, truth=[0, 0, 0]) for g in ("stranger", "returning")}
        for f in range(10):
            for d in draws:
                z = np.load(O / lp / "teacher_p1" / f"lc9_{f}__s{d}.npz")
                rows, held = z["rows"].astype(np.int64), z["held"].astype(bool)
                pos = {int(r): k for k, r in enumerate(rows)}
                for s in np.unique(ST[rows]):
                    j = rows[ST[rows] == s]
                    j = j[np.argsort(OD[j], kind="stable")]
                    for c in (60, 100, 150):
                        if len(j) <= c:
                            break
                        a = j[:c]
                        k_ = np.array([pos[int(r)] for r in a])
                        pl = z[f"post_r{c}"][k_].astype(np.float64)
                        fin = np.isfinite(pl).all(1)
                        a, k_, pl = a[fin], k_[fin], pl[fin]
                        ya, keep = conf_half(pl)
                        a, k_, pl, ya = a[keep], k_[keep], pl[keep], ya[keep]
                        o = np.sort(pl, 1)
                        top2 = np.argsort(-pl, 1)[:, :2]
                        rule = (np.sort(top2, 1) == np.array([MI, WORD])).all(1) & ((o[:, 2] - o[:, 1]) < THR)
                        for g, gm in (("stranger", held[k_]), ("returning", ~held[k_])):
                            t, r_, l_ = Y[a[gm]], rule[gm], ya[gm]
                            for A in (acc[g], tot[g]):
                                A["sel"] += int(gm.sum())
                                A["rule"] += int(r_.sum())
                                A["in_set"] += int(np.isin(t[r_], (MI, WORD)).sum())
                                A["lp_wrong"] += int((l_[r_] != t[r_]).sum())
                                A["word_right_weakened"] += int(((l_[r_] == WORD) & (t[r_] == WORD)).sum())
                                A["word_on_mi_repaired"] += int(((l_[r_] == WORD) & (t[r_] == MI)).sum())
                                for q in (MI, SUB, WORD):
                                    A["truth"][q] += int((t[r_] == q).sum())
        per[S] = acc

    def summ(A):
        n = max(A["rule"], 1)
        return dict(selected=A["sel"], in_rule=A["rule"], coverage=round(A["rule"] / max(A["sel"], 1), 4), truth_in_set=round(A["in_set"] / n, 4),
                    lp_label_wrong=round(A["lp_wrong"] / n, 4), truth_MI_SUB_WORD=A["truth"], lp_WORD_right=A["word_right_weakened"], lp_WORD_truth_MI=A["word_on_mi_repaired"])
    out = {S: {g: summ(A) for g, A in acc.items()} for S, acc in per.items()}
    out["pooled_AB"] = {g: summ(A) for g, A in tot.items()}
    st = out["pooled_AB"]["stranger"]
    out["gate"] = dict(threshold=THR, coverage_bar=0.03, truth_bar=0.90, coverage=st["coverage"], truth_in_set=st["truth_in_set"],
                       passes=bool(st["coverage"] >= 0.03 and st["truth_in_set"] >= 0.90))
    for k, v in out.items():
        print(f"[d594b] {k}: {v}", flush=True)
    (O / "t2_d594b_gate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
