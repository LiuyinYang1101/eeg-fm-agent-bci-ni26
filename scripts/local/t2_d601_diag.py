#!/usr/bin/env python
"""T2-D601 STEP 1, the cached gate (registered dcb953e2 before any number): family-excluded P1 teaching. W21's P1 teacher q at refit c =
Σ_k TW_k prefix_decode(rows_k <= c) (CNN rows = the raw routed parts R_k of the teacher file, FM rows = the arm's own current FM part), then the
FM-graph LP (α 0.5, k 10) and the confident half per predicted class. For recipient f (sh / ifn): q_f = (q - TW_f pd_f) / (1 - TW_f), the same LP
and confident half. Checks (asserted): the lab's own fbteach call (IDEAS.fm_feedback(proto 0.5, fm_rows own, record); env as t2_d571_local.sh's
fbteach) reproduces the fb root's post_r<c> (float32); the implied FM component (q - Σ_CNN TW_k pd_k) / TW_fm is a distribution; LP of the fb
file == the lp root's post_r<c>. Counters per recipient x group (strangers / returning), micro-summed over folds x draws x streams x refits:
the confident half's label accuracy common vs excluded (truth: a diagnostic only), changed labels (old vs new accuracy), dropped / added rows.
    mem-guard env OMP_NUM_THREADS=4 python scripts/local/t2_d601_diag.py --set A|B|C|D   ->  outputs/t2_d601_diag_<set>.json
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

T0 = time.time()
S = sys.argv[sys.argv.index("--set") + 1]
D0 = {"A": 825, "B": 828, "C": 831, "D": 834}[S]
for k in [k for k in os.environ if k.startswith("NI26_")]:
    del os.environ[k]
if D0 == 831:
    os.environ["NI26_LAB_D557"] = "1"                    # = t2_d571_local.sh (C: W20+D557's members; on 834 they ARE the base members)
O = Path(os.path.expanduser("~/workspace/outputs/neural_interface_challenge"))
TAG = "w20d557" if D0 in (831, 834) else "w20"
FBR, LPR = O / f"t2_d571_fb{TAG}_{D0}/teacher_p1", O / f"t2_d571_lp{TAG}_{D0}/teacher_p1"
sys.argv = ["t2_lab.py", "mix", "--set", S, "--base", "w20"]
src = open("scripts/local/t2_lab.py").read()
CUT = '\nif MODE == "mix":'
assert src.count(CUT) == 1, "the lab's mix branch moved"
g = {"__name__": "d601", "__file__": os.path.abspath("scripts/local/t2_lab.py")}
exec(compile(src.split(CUT)[0], "t2_lab_head(d601)", "exec"), g)
IDEAS, Ctx, IX, Y, U, SES, TRU = g["IDEAS"], g["Ctx"], g["IX"], g["Y"], g["U"], g["SES"], g["TRU"]
assert not g["PRE_R"] and tuple(g["CKS"]) == (60, 100, 150)
TW = dict(IDEAS.TW)
assert abs(sum(TW.values()) - 1.0) < 1e-12, TW
rec = {}
IDEAS.fm_feedback(Ctx, proto=0.5, fm_rows="own", record=rec)            # = t2_lab.py fbteach --fm-rows own --proto 0.5 (W21's teacher)
print(f"[d601] {S}: teacher re-built ({sum(len(v) for v in rec.values())} stream x refit blocks; {time.time() - T0:.0f} s)", flush=True)
CNN = ("fbc", "sh", "ifn")
CT = defaultdict(lambda: defaultdict(int))
CHK = dict(rec_vs_fb=0.0, fm_min=1.0, fm_sum_err=0.0, lp_vs_root=0.0, blocks=0)
for (f, d), blocks in sorted(rec.items()):
    R, V, _ = IDEAS._tfile(Ctx, f, d)
    zf, zl = np.load(FBR / f"lc9_{f}__s{d}.npz"), np.load(LPR / f"lc9_{f}__s{d}.npz")
    assert np.array_equal(zf["rows"], zl["rows"])
    pos = {int(r): k for k, r in enumerate(zf["rows"])}
    Fz = IDEAS._knn_feats(Ctx, f, "fm")
    for c, a, q in blocks:
        rows = IX[f][a]
        fi = np.array([pos[int(r)] for r in rows])
        T32 = zf[f"post_r{c}"][fi].astype(np.float64)
        CHK["rec_vs_fb"] = max(CHK["rec_vs_fb"], float(np.abs(q.astype(np.float32).astype(np.float64) - T32).max()))
        pd = {k: IDEAS._prefix_decode(R[k][a], V[k]) for k in CNN}
        fmi = (q - sum(TW[k] * pd[k] for k in CNN)) / TW["fm"]
        CHK["fm_min"] = min(CHK["fm_min"], float(fmi.min()))
        CHK["fm_sum_err"] = max(CHK["fm_sum_err"], float(np.abs(fmi.sum(1) - 1.0).max()))
        Fg = Fz[a]
        lp0 = IDEAS._lp_block(Fg, T32, 0.5, 10)
        CHK["lp_vs_root"] = max(CHK["lp_vs_root"], float(np.abs(lp0 - zl[f"post_r{c}"][fi].astype(np.float64)).max()))
        CHK["blocks"] += 1
        y0, k0 = IDEAS._conf(lp0, 0.5)
        y = Y[rows]
        grp = np.where(U[rows] == TRU[f], "stranger", np.where(np.isin(SES[rows], ("2", "3")), "returning", "other"))
        for fam in ("sh", "ifn"):
            qx = (q - TW[fam] * pd[fam]) / (1.0 - TW[fam])
            lpx = IDEAS._lp_block(Fg, qx.astype(np.float32).astype(np.float64), 0.5, 10)
            y1, k1 = IDEAS._conf(lpx, 0.5)
            for gn in ("stranger", "returning"):
                m = grp == gn
                C_ = CT[f"{fam}|{gn}"]
                C_["n_common"] += int((k0 & m).sum())
                C_["ok_common"] += int(((y0 == y) & k0 & m).sum())
                C_["n_excl"] += int((k1 & m).sum())
                C_["ok_excl"] += int(((y1 == y) & k1 & m).sum())
                ch = k0 & k1 & m & (y0 != y1)
                C_["n_changed"] += int(ch.sum())
                C_["ok_old_changed"] += int((y0 == y)[ch].sum())
                C_["ok_new_changed"] += int((y1 == y)[ch].sum())
                dr, ad = k0 & ~k1 & m, ~k0 & k1 & m
                C_["n_dropped"] += int(dr.sum())
                C_["ok_dropped"] += int(((y0 == y) & dr).sum())
                C_["n_added"] += int(ad.sum())
                C_["ok_added"] += int(((y1 == y) & ad).sum())
print(f"[d601] {S}: checks {CHK}", flush=True)
assert CHK["rec_vs_fb"] <= 1e-6, ("the lab's teacher != the fb root's post_r<c>", CHK)
assert CHK["fm_min"] >= -1e-9 and CHK["fm_sum_err"] <= 1e-9, ("the implied FM component is not a distribution", CHK)
assert CHK["lp_vs_root"] <= 1e-6, ("LP(fb) != the lp root's post_r<c>", CHK)
res = {"set": S, "checks": CHK, "counts": {k: dict(v) for k, v in CT.items()}}
for k, v in CT.items():
    r_ = lambda a_, b_: round(v[a_] / max(v[b_], 1), 4)
    res.setdefault("rates", {})[k] = dict(acc_common=r_("ok_common", "n_common"), acc_excl=r_("ok_excl", "n_excl"),
                                          changed_share=r_("n_changed", "n_common"), acc_old_changed=r_("ok_old_changed", "n_changed"),
                                          acc_new_changed=r_("ok_new_changed", "n_changed"), acc_dropped=r_("ok_dropped", "n_dropped"),
                                          acc_added=r_("ok_added", "n_added"), n_common=v["n_common"], n_changed=v["n_changed"])
    print(f"[d601] {S} {k}: {res['rates'][k]}", flush=True)
json.dump(res, open(O / f"t2_d601_diag_{S}.json", "w"), indent=1)
print(f"[d601] {S} done ({time.time() - T0:.0f} s)", flush=True)
