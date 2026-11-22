#!/usr/bin/env python
"""T2-D628 (DIAGNOSTIC; registered in docs/DECISIONS_T2.md before any code or number; GPT-6-astra R9 diagnostic 2): is the in-stream
teacher's error asymmetric NOISE or feature-dependent MISINFORMATION?
At each refit c in (60, 100, 150) of every scored stream: the prefix rows (order < c) and W21's ACTUAL P1 pseudo-labels (argmax of the lp root's
teacher_p1 post_r<c>) are held fixed; identical ridge heads on the FM features (IDEAS._knn_feats(Ctx, f, 'fm'); the ridge of Track3's
lead/diag_word_2026-11-18/causal_oracle.py: centred, row-normalised, alpha 1) are fit on (i) the actual labels and (ii) TEN within-TRUE-class
permutations of them (seeds 6280-6289: the same confusion counts, the wrong labels moved to other windows of the same true class), and scored on
the next segment [c, next c) (the last to the stream end) by cell BA. PRIMARY rows = the SELECTED prefix rows (conf_half: the confident half per
pseudo-class); descriptive: all prefix rows. Strangers pooled (user mean), P04 / P10 separate.
Runs inside the lab (runpy of t2_lab.py mix with the w20_fb idea wrapped, as causal_oracle.py), cwd = the repo:
  A: mem-guard env OMP_NUM_THREADS=4 NI26_D559_ROOT=t2_d571_lpw20_825 NI26_D559_CKS=60,100,150 NI26_LAB_PRE_R=1 python scripts/local/t2_d628_teacher_err.py A
  C: ... NI26_LAB_D557=1 NI26_D559_ROOT=t2_d571_lpw20d557_831 ... python scripts/local/t2_d628_teacher_err.py C
-> outputs/t2_d628/d628_<set>.csv + summary_<set>.json (scripts/local/t2_d628_teacher_err.py report writes RESULT_D628.md from both).
"""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O = Path.home() / "workspace/outputs/neural_interface_challenge"
OUT = O / "t2_d628"
PRO = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
LPR = {"A": "t2_d571_lpw20_825", "C": "t2_d571_lpw20d557_831"}
CKS, ALPHA, NPERM, SEED0 = (60, 100, 150), 1.0, 10, 6280


def ridge(Ftr, ytr, Fte):                              # = causal_oracle.py's ridge (Track3)
    mu = Ftr.mean(0)
    Xtr, Xte = Ftr - mu, Fte - mu
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
    Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    T = np.eye(3)[ytr]
    A = np.linalg.solve(Xtr @ Xtr.T + ALPHA * np.eye(len(ytr)), T - T.mean(0))
    return (Xte @ Xtr.T @ A + T.mean(0)).argmax(1)


def conf_half(P):                                      # = causal_oracle.py's conf_half (the confident half per pseudo-class)
    ya, cf = P.argmax(1), P.max(1)
    keep = np.zeros(len(P), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return ya, keep


def perm_within(ya, yt, rng):
    """the pseudo-labels permuted among the windows of each TRUE class: per true class the multiset of pseudo-labels (the confusion row) is kept."""
    out = ya.copy()
    for q in np.unique(yt):
        jj = np.flatnonzero(yt == q)
        out[jj] = ya[jj][rng.permutation(len(jj))]
    return out


def wrapper(SET):
    def w(Ctx, **kw):
        sys.path.insert(0, "scripts/local")
        import t2_lab_ideas as IDEAS
        G = Ctx.lab_mix.__globals__
        I = pd.read_csv(PRO / "index.csv", low_memory=False)
        CELL = I["cell"].to_numpy()
        Y, U, ST, OD = np.asarray(Ctx.Y).astype(int), np.asarray(Ctx.U), np.asarray(G["STREAM"]), np.asarray(G["ORD"])
        SES = np.asarray(G["SES"])
        assert np.array_equal(ST, I["stream"].to_numpy())

        def cellba(pred, rows):
            v = []
            for c in np.unique(CELL[rows]):
                m = CELL[rows] == c
                ks = np.unique(Y[rows][m])
                if len(ks) >= 2:
                    v.append(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
            return float(np.mean(v)) if v else np.nan

        recs, done_ret = [], set()
        for f in Ctx.folds:
            Fz = IDEAS._knn_feats(Ctx, f, "fm").astype(np.float64)
            ix = np.asarray(Ctx.IX[f])
            posF = {int(r): i for i, r in enumerate(ix)}
            tru = Ctx.TRU[f]
            grp = np.where(U[ix] == tru, "stranger", np.where(np.isin(SES[ix], ("2", "3")), "ret", "other"))
            for s_ in np.unique(ST[ix]):
                j = np.flatnonzero(ST[ix] == s_)
                g_ = grp[j[0]]
                if g_ == "other":
                    continue
                user = U[ix][j[0]]
                if g_ == "ret":                           # a returning stream is in every fold's IX: read it once (fold of first sight)
                    if s_ in done_ret:
                        continue
                    done_ret.add(s_)
                rows = ix[j][np.argsort(OD[ix[j]], kind="stable")]
                F = Fz[[posF[int(r)] for r in rows]]
                L = len(rows)
                for k, c in enumerate(CKS):
                    if L <= c:
                        continue
                    e = min(CKS[k + 1] if k + 1 < len(CKS) else L, L)
                    tr_, te_ = np.arange(c), np.arange(c, e)
                    rte, yt = rows[te_], Y[rows[tr_]]
                    for d in Ctx.draws:
                        zt = np.load(O / LPR[SET] / "teacher_p1" / f"lc9_{f}__s{d}.npz")
                        pos_t = {int(x): i for i, x in enumerate(zt["rows"])}
                        P_ = zt[f"post_r{c}"][[pos_t[int(x)] for x in rows[tr_]]].astype(np.float64)
                        assert np.isfinite(P_).all(), (f, d, s_, c)
                        ya, keep = conf_half(P_)
                        r = dict(set=SET, fold=f, draw=d, user=user, grp="stranger" if g_ == "stranger" else user, stream=s_, seg=c,
                                 n_test=len(te_), n_sel=int(keep.sum()), err_sel=float(np.mean(ya[keep] != yt[keep])), err_all=float(np.mean(ya != yt)))
                        for nm, sel in (("sel", keep), ("all", np.ones(c, bool))):
                            Ftr = F[tr_][sel]
                            r[f"i_{nm}"] = cellba(ridge(Ftr, ya[sel], F[te_]), rte)
                            pp = []
                            for kk in range(NPERM):
                                rng = np.random.default_rng(SEED0 + kk)
                                yp = perm_within(ya[sel], yt[sel], rng)
                                assert np.array_equal(np.bincount(yt[sel] * 3 + yp, minlength=9), np.bincount(yt[sel] * 3 + ya[sel], minlength=9))
                                pp.append(cellba(ridge(Ftr, yp, F[te_]), rte))
                            r[f"ii_{nm}"], r[f"ii_{nm}_sd"] = float(np.nanmean(pp)), float(np.nanstd(pp))
                            r[f"wins_{nm}"] = int(sum(x > r[f"i_{nm}"] for x in pp))
                        recs.append(r)
        R = pd.DataFrame(recs)
        OUT.mkdir(exist_ok=True)
        R.to_csv(OUT / f"d628_{SET}.csv", index=False)
        summ = {}
        for g_ in ("stranger", "P04", "P10"):
            sub = R[R.grp == g_]
            if sub.empty:
                continue
            pu = sub.groupby("user")[["i_sel", "ii_sel", "i_all", "ii_all", "err_sel", "err_all"]].mean()
            summ[g_] = dict(n_users=int(len(pu)), **{k: round(float(pu[k].mean()), 4) for k in pu.columns},
                            d_sel=round(float((pu.ii_sel - pu.i_sel).mean()), 4), d_all=round(float((pu.ii_all - pu.i_all).mean()), 4),
                            users_pos_sel=int(((pu.ii_sel - pu.i_sel) > 0).sum()),
                            per_user_d_sel={u: round(float(v), 4) for u, v in (pu.ii_sel - pu.i_sel).items()},
                            per_seg_d_sel={int(s): round(float(v), 4) for s, v in
                                           sub.assign(dd=sub.ii_sel - sub.i_sel).groupby(["seg", "user"]).dd.mean().groupby("seg").mean().items()},
                            wins_share_sel=round(float(sub.wins_sel.sum() / (NPERM * len(sub))), 3))
            print(f"[d628] {SET} {g_}: {json.dumps(summ[g_])}", flush=True)
        (OUT / f"summary_{SET}.json").write_text(json.dumps(summ, indent=1))
        raise SystemExit(0)
    return w


def report():
    S = {s: json.loads((OUT / f"summary_{s}.json").read_text()) for s in ("A", "C")}
    dA, dC = S["A"]["stranger"]["d_sel"], S["C"]["stranger"]["d_sel"]
    verdict = ("MISINFORMATION" if dA >= 0.02 and dC >= 0.02 else "NOISE-LIKE" if abs(dA) < 0.01 and abs(dC) < 0.01 else "INTERMEDIATE")
    L = ["# T2-D628 RESULT — is the in-stream teacher's error noise or feature-dependent misinformation? (DIAGNOSTIC; truth used; no ship path)", "",
         "Registered in docs/DECISIONS_T2.md before any code or number (GPT-6-astra R9 diagnostic 2). Ridge heads on the FM features fit at refits "
         "60 / 100 / 150 on the prefix rows with (i) W21's actual P1 pseudo-labels vs (ii) ten within-TRUE-class permutations (identical confusion counts); "
         "scored on the next segment by cell BA; strangers = user mean; PRE_R lab, all 3 draws.", "",
         f"**Verdict (the registered reading, strangers, selected rows): {verdict}** — Δ = mean(ii) − (i): A {dA:+.4f}, C {dC:+.4f} "
         "(MISINFORMATION needs ≥ +0.02 on both; NOISE-LIKE needs |Δ| < 0.01 on both).", "",
         "| set | group | (i) actual, sel | (ii) permuted, sel | Δ sel | users Δ > 0 | perms beating (i) | (i) all | (ii) all | Δ all | teacher err sel / all |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("A", "C"):
        for g_, v in S[s].items():
            L.append(f"| {s} | {g_} | {v['i_sel']:.4f} | {v['ii_sel']:.4f} | {v['d_sel']:+.4f} | {v['users_pos_sel']} / {v['n_users']} | {v['wins_share_sel']:.0%} | "
                     f"{v['i_all']:.4f} | {v['ii_all']:.4f} | {v['d_all']:+.4f} | {v['err_sel']:.3f} / {v['err_all']:.3f} |")
    L += ["", "Per stranger Δ (selected rows): " + "; ".join(f"{s}: " + ", ".join(f"{u} {d:+.3f}" for u, d in S[s]["stranger"]["per_user_d_sel"].items())
                                                         for s in ("A", "C")),
          "Per segment Δ (strangers, selected rows): " + "; ".join(f"{s}: " + ", ".join(f"seg {k} {d:+.4f}" for k, d in S[s]["stranger"]["per_seg_d_sel"].items())
                                                                  for s in ("A", "C")), ""]
    (OUT / "RESULT_D628.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    SET = sys.argv[1]
    if SET == "report":
        report()
        raise SystemExit(0)
    assert SET in LPR and os.environ.get("NI26_D559_ROOT") == LPR[SET] and os.environ.get("NI26_LAB_PRE_R") == "1", "set the registered env (see the docstring)"
    assert (os.environ.get("NI26_LAB_D557") == "1") == (SET == "C"), "NI26_LAB_D557=1 on set C only"
    sys.path.insert(0, "scripts/local")
    import t2_lab_ideas as IDEAS
    reg0 = IDEAS.registry
    IDEAS.registry = lambda: [(n, wrapper(SET) if n == "w20_fb" else fn, p) for n, fn, p in reg0()]
    sys.argv = ["scripts/local/t2_lab.py", "mix", "--set", SET, "--base", "w20", "--idea", "w20_fb", "--out", "/dev/null"]
    try:
        runpy.run_path("scripts/local/t2_lab.py", run_name="__main__")
    except SystemExit as e:
        assert e.code in (0, None), e.code
