#!/usr/bin/env python
"""T2-D619 (DIAGNOSTIC, registered in docs/DECISIONS_T2.md before any number): mirror canonicalisation of the TEST stream, FBCNet member-alone.
Reads outputs/t2_d619/cnn_fbc/runs (t2p_bench --test-mirror: plain / mr, tv8 / tv8mr; lc9; draws 860-862) and writes
outputs/t2_d619/RESULT_D619.md + result.json.
  per user (STRANGER fold: lc9_i holds out user i): cell-balanced accuracy (cells = the export's 'cell' = subject | session | interface) per draw,
  mean over the 3 draws, mirrored - original (plain and tv8); P04 / P10 returning (sessions 2-3 in the 9 folds that train them) descriptive.
  (a) users with delta >= +0.03.  (b) PRIMARY rule: per user, the mean prediction entropy over the PREFIX (order < 60 of each of the user's test
  streams), averaged over draws; pick the LOWER; 'better' = the higher draw-mean cell BA on the whole stranger stream (an exact tie counts as
  right); success = >= 8 / 10 users (plain). Descriptive: 'better' on order >= 60 only, per (user, draw) agreement / 30, the max-prob and the
  class-balance (KL(mean prediction || uniform), lower) rules, the oracle gain mean(max(0, delta)) and the rule's realised gain; the plain output
  vs set E's t2_d615 plain files at the same seeds.
  python scripts/local/t2_d619_mirror_read.py        (OMP / MKL / OPENBLAS threads capped by the caller)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402
import t2p_bench as B  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
RUNS, E_RUNS = O / "t2_d619/cnn_fbc/runs", O / "t2_d615/cnn_fbc/runs"
REC, DRAWS, PREFIX = "r1_pulxx_vp2_e180_vr_rc75", (860, 861, 862), 60
NAMED = ("P03", "P05", "P06", "P07")


def fname(fold, sfx, s, root=RUNS):
    return root / f"lc9_{fold}__fbcnet_bboth__none__n00__{REC}{sfx}__s{s}.npz"


def load(fold, sfx, s, root=RUNS):
    z = np.load(fname(fold, sfx, s, root))
    return z["test_idx"], z["prob_last"].astype(np.float64), {k: z[k] for k in z.files if k.startswith("mirror_")}


def cba(p, rows, y, cell):
    return cell_balanced_accuracy(y[rows], p.argmax(1), cell[rows])[0]


def entropy(p):
    q = np.clip(p, 1e-12, 1)
    return float(-(q * np.log(q)).sum(1).mean())


def main():
    _, idx, _ = B.load_export(str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    sub = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
    ses, y, cell, order = idx["session"].astype(str).to_numpy(), idx["y"].to_numpy(), idx["cell"].astype(str).to_numpy(), idx["order"].to_numpy()
    users = sorted(np.unique(sub))
    assert len(users) == 10
    forms = {"plain": ("", "mr"), "tv8": ("tv8", "tv8mr")}
    st, rt, checks = {}, {}, dict(id_maxdiff=[], pairs=set(), vs_E=[])
    for i, u in enumerate(users):
        for s in DRAWS:
            for fm, (so, sm) in forms.items():
                te, po, _ = load(i, so, s)
                te2, pm, ex = load(i, sm, s)
                assert np.array_equal(te, te2) and po.shape == pm.shape
                if "mirror_id_maxdiff" in ex:
                    checks["id_maxdiff"].append(float(ex["mirror_id_maxdiff"]))
                checks["pairs"].add(int(ex["mirror_pairs"]))
                sm_ = sub[te] == u
                assert sm_.any() and set(sub[te][~sm_]) <= {"P04", "P10"}, (i, u)
                rows, ps, pms = te[sm_], po[sm_], pm[sm_]
                pre, post = order[rows] < PREFIX, order[rows] >= PREFIX
                st[(u, s, fm)] = dict(o=cba(ps, rows, y, cell), m=cba(pms, rows, y, cell),
                                      o_post=cba(ps[post], rows[post], y, cell), m_post=cba(pms[post], rows[post], y, cell),
                                      ent_o=entropy(ps[pre]), ent_m=entropy(pms[pre]),
                                      mx_o=float(ps[pre].max(1).mean()), mx_m=float(pms[pre].max(1).mean()),
                                      kl_o=float((ps[pre].mean(0) * np.log(3 * ps[pre].mean(0))).sum()),
                                      kl_m=float((pms[pre].mean(0) * np.log(3 * pms[pre].mean(0))).sum()))
                for r_ in ("P04", "P10"):
                    mr_ = (sub[te] == r_) & np.isin(ses[te], ("2", "3"))
                    if mr_.any() and r_ != u:
                        rr = te[mr_]
                        rt.setdefault((r_, fm), []).append(cba(pm[mr_], rr, y, cell) - cba(po[mr_], rr, y, cell))
            if E_RUNS.exists() and fname(i, "", s, E_RUNS).exists():
                teE, pE, _ = load(i, "", s, E_RUNS)
                te0, p0, _ = load(i, "", s)
                checks["vs_E"].append(float(np.abs(pE - p0).max()) if np.array_equal(teE, te0) else float("nan"))
    out = dict(checks=dict(id_maxdiff=max(checks["id_maxdiff"]) if checks["id_maxdiff"] else None, pairs=sorted(checks["pairs"]),
                           vs_E_max=(float(np.nanmax(checks["vs_E"])) if checks["vs_E"] else None),
                           vs_E_n=len(checks["vs_E"]), vs_E_exact=int(sum(v == 0.0 for v in checks["vs_E"]))))
    lines = ["# T2-D619 RESULT — mirror canonicalisation of the TEST stream, FBCNet member-alone (DIAGNOSTIC; no ship path)", "",
             "Registered in docs/DECISIONS_T2.md before any number. lc9, draws 860 / 861 / 862, set E's FBCNet line without copies; twins from "
             "t2p_bench --test-mirror (the trained nets unchanged; the TEST windows' homolog channels swapped at the net input).", "",
             f"Checks: unmirrored re-predict == prob_last, max diff {out['checks']['id_maxdiff']}; mirror pairs {out['checks']['pairs']}; "
             f"the plain output vs set E's t2_d615 plain files (same seeds): {out['checks']['vs_E_exact']} / {out['checks']['vs_E_n']} exact, "
             f"max |dp| {out['checks']['vs_E_max']}.", ""]
    for fm in forms:
        D = {u: np.mean([st[(u, s, fm)]["m"] - st[(u, s, fm)]["o"] for s in DRAWS]) for u in users}
        Dp = {u: np.mean([st[(u, s, fm)]["m_post"] - st[(u, s, fm)]["o_post"] for s in DRAWS]) for u in users}
        lev = {u: np.mean([st[(u, s, fm)]["o"] for s in DRAWS]) for u in users}
        rule = {}
        for nm, ko, km, lower in (("entropy", "ent_o", "ent_m", True), ("maxprob", "mx_o", "mx_m", False), ("balance", "kl_o", "kl_m", True)):
            a_ = {u: np.mean([st[(u, s, fm)][ko] for s in DRAWS]) for u in users}
            b_ = {u: np.mean([st[(u, s, fm)][km] for s in DRAWS]) for u in users}
            pick = {u: bool(b_[u] < a_[u]) if lower else bool(b_[u] > a_[u]) for u in users}   # True = mirrored
            right = sum((pick[u] and D[u] >= 0) or (not pick[u] and D[u] <= 0) for u in users)
            right_post = sum((pick[u] and Dp[u] >= 0) or (not pick[u] and Dp[u] <= 0) for u in users)
            pd_ = 0
            for u in users:
                for s in DRAWS:
                    e = st[(u, s, fm)]
                    pk = (e[km] < e[ko]) if lower else (e[km] > e[ko])
                    d_ = e["m"] - e["o"]
                    pd_ += int((pk and d_ >= 0) or (not pk and d_ <= 0))
            rule[nm] = dict(picks_mirror=[u for u in users if pick[u]], right=int(right), right_post=int(right_post), right_user_draw=int(pd_),
                            realised=round(float(np.mean([D[u] if pick[u] else 0.0 for u in users])), 4))
        ret = {r_: round(float(np.mean(rt[(r_, fm)])), 4) for r_ in ("P04", "P10") if (r_, fm) in rt}
        res = dict(level_ten=round(float(np.mean(list(lev.values()))), 4), delta_ten=round(float(np.mean(list(D.values()))), 4),
                   delta=dict((u, round(float(D[u]), 4)) for u in users), delta_post=dict((u, round(float(Dp[u]), 4)) for u in users),
                   gain_ge_003=[u for u in users if D[u] >= 0.03], oracle=round(float(np.mean([max(0.0, D[u]) for u in users])), 4),
                   returning=ret, rule=rule, b_pass=bool(rule["entropy"]["right"] >= 8))
        out[fm] = res
        lines += [f"## {fm} ({'the pooled net, plain predict' if fm == 'plain' else 'the pooled net, VP-8 TTA'})", "",
                  f"Stranger level (original) ten {res['level_ten']}; mirrored − original ten **{res['delta_ten']:+.4f}**; oracle per-user pick "
                  f"mean(max(0, Δ)) {res['oracle']:+.4f}; returning (mirrored − original) {ret}.", "",
                  "| user | level (orig) | Δ whole stream | Δ windows ≥ 60 | prefix entropy orig / mirr | picked (entropy) |", "|---|---|---|---|---|---|"]
        for u in users:
            eo = np.mean([st[(u, s, fm)]["ent_o"] for s in DRAWS]); em = np.mean([st[(u, s, fm)]["ent_m"] for s in DRAWS])
            lines.append(f"| {u}{' *' if u in NAMED else ''} | {lev[u]:.4f} | {D[u]:+.4f} | {Dp[u]:+.4f} | {eo:.4f} / {em:.4f} | "
                         f"{'mirrored' if u in rule['entropy']['picks_mirror'] else 'original'} |")
        lines += ["", f"**(a)** users with Δ ≥ +0.03: {res['gain_ge_003'] or 'none'} (named in advance: {', '.join(NAMED)}).",
                  f"**(b) primary (entropy):** right for {rule['entropy']['right']} / 10 users (bar 8) → **{'PASS' if res['b_pass'] else 'FAIL'}**; "
                  f"realised ten gain {rule['entropy']['realised']:+.4f}; right on windows ≥ 60 {rule['entropy']['right_post']} / 10; per (user, draw) "
                  f"{rule['entropy']['right_user_draw']} / 30.",
                  f"Descriptive rules: maxprob right {rule['maxprob']['right']} / 10 (realised {rule['maxprob']['realised']:+.4f}); balance right "
                  f"{rule['balance']['right']} / 10 (realised {rule['balance']['realised']:+.4f}).", ""]
    spect = bool(out["plain"]["b_pass"] and out["plain"]["rule"]["entropy"]["realised"] >= 0.01)
    out["spectacular"] = spect
    lines += [f"**Spectacular (registered: (b) passes and the rule's realised ten gain ≥ +0.01, plain):** {spect}. "
              "Diagnostic only: nothing here enters a ship line without a new registration, T2-D590's bar and the owner's line.", ""]
    (O / "t2_d619/RESULT_D619.md").write_text("\n".join(lines))
    (O / "t2_d619/result.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
