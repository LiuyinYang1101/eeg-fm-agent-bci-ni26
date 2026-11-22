#!/usr/bin/env python
"""T2-D630 reader (registered + amended in docs/DECISIONS_T2.md before any code or number): the pooled-training composition, B (unweighted, the
ship composition) vs C (--user-balance: 1 / the user's training sessions), PAIRED (seeds 880-882), from outputs/t2_d630/{B,C}.
  PRIMARY strangers (ss8_<u>, the 8 single-session users): per family the seed-mean POOLED tv8 probability -> per-family causal Sinkhorn per stream
  (t2p_report.causal_sinkhorn, iters 10, n0 3, lam 0.95, uniform start) -> the CNN mix at W21's family weights renormalised (fbc .3143 / sh .2357
  / ifn .10) -> argmax -> cell BA (the export's 'cell') on u's windows; dS = mean_u (C - B).
  RETURNING (copies ON): the ROUTED form (t2p_fpcalib blend, <rec>tv8c fb<tag>) through the same decode / mix, on X's scored sessions:
  d(i) = mean_X (C - B) on rt1_X (X trained on session 1, scored 2-3), d(ii) on rt2_X (trained 1-2, scored 3), X in {P04, P10}.
  DECISION: PASS = dS >= +0.004 with >= 6 / 8 users >= 0 and none < -0.02 AND 0.57 dS + 0.38 d(i) + 0.06 d(ii) >= +0.002; dS < +0.002 = CLOSED;
  dS in [+0.002, +0.004) = UNDETERMINED (no rebuild); dS >= +0.004 failing a guard or the sealed arithmetic = NOT PASS.
  Descriptive: the routed strangers, the plain (no TTA) pooled strangers, each family alone, the pooled-only returning, A = set E's lc9 cache
  (t2_d615 cnn_*, draws 860-862, the fold holding u out, pooled tv8) levels.
  python scripts/local/t2_d630_read.py    (OMP / MKL / OPENBLAS threads capped by the caller) -> outputs/t2_d630/RESULT_D630.md + result.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402
import t2p_bench as B  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d630"
SEEDS, SEEDS_A = (880, 881, 882), (860, 861, 862)
FAM = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75", "3"), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr", "2"),
       "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr", "2")}
WF = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10}
STRANGERS = ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P09")


def path(arm, fam, fold, form, s):
    model, al, rec, tag = FAM[fam]
    xt = "__ubw" if arm == "C" else ""
    if form == "routed":
        return R / arm / f"route_{fam}" / f"{fold}__{model}__{al}__n00__{rec}tv8cfb{tag}{xt}__s{s}.npz"
    sfx = {"tv8": "tv8", "plain": ""}[form]
    if arm == "A":
        return O / "t2_d615" / f"cnn_{fam}" / "runs" / f"{fold}__{model}__{al}__n00__{rec}{sfx}__s{s}.npz"
    return R / arm / f"cnn_{fam}" / "runs" / f"{fold}__{model}__{al}__n00__{rec}{sfx}{xt}__s{s}.npz"


def main():
    _, idx, _ = B.load_export(str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    sub = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
    ses, y = idx["session"].astype(str).to_numpy(), idx["y"].to_numpy().astype(int)
    cell, stream, order = idx["cell"].astype(str).to_numpy(), idx["stream"].astype(str).to_numpy(), idx["order"].to_numpy()
    users = sorted(np.unique(sub))

    def mix(arm, fold, form, rows_keep, seeds=SEEDS, fams=tuple(FAM)):
        """the decoded CNN mix on the fold's test rows, restricted to rows_keep (a boolean over the export) -> (rows, mixed prob)."""
        te0, acc = None, 0.0
        for fam in fams:
            ps = []
            for s in seeds:
                z = np.load(path(arm, fam, fold, form, s))
                te = z["test_idx"]
                if te0 is None:
                    te0 = te
                assert np.array_equal(te, te0), (arm, fam, fold, form, s)
                ps.append(z["prob_last"].astype(np.float64))
            P = np.mean(ps, 0)
            D = causal_sinkhorn(P, (stream[te0], order[te0]), iters=10, n0=3, lam=0.95)
            acc = acc + WF[fam] * D
        acc = acc / sum(WF[f] for f in fams)
        k = rows_keep[te0]
        return te0[k], acc[k]

    def cba(rows, p):
        return cell_balanced_accuracy(y[rows], p.argmax(1), cell[rows])[0]

    out, RAW = {}, {}                                   # Track3 LOW (783c6e45): the verdict reads the UNROUNDED values; out is for display
    # strangers: primary pooled tv8; descriptive routed / plain / per family / A
    for form in ("tv8", "routed", "plain"):
        lv = {}
        for arm in ("B", "C"):
            lv[arm] = {u: cba(*mix(arm, f"ss8_{u}", form, sub == u)) for u in STRANGERS}
        d = {u: lv["C"][u] - lv["B"][u] for u in STRANGERS}
        RAW[f"stranger_{form}"] = dict(dS=float(np.mean(list(d.values()))), min=float(min(d.values())), n_ge0=int(sum(v >= 0 for v in d.values())))
        out[f"stranger_{form}"] = dict(B=round(float(np.mean(list(lv["B"].values()))), 4), C=round(float(np.mean(list(lv["C"].values()))), 4),
                                       dS=round(float(np.mean(list(d.values()))), 4), n_ge0=int(sum(v >= 0 for v in d.values())),
                                       min=round(float(min(d.values())), 4), min_who=min(d, key=d.get), per_user={u: round(float(v), 4) for u, v in d.items()})
    for fam in FAM:
        d = {u: cba(*mix("C", f"ss8_{u}", "tv8", sub == u, fams=(fam,))) - cba(*mix("B", f"ss8_{u}", "tv8", sub == u, fams=(fam,))) for u in STRANGERS}
        out[f"stranger_tv8_{fam}_alone"] = dict(dS=round(float(np.mean(list(d.values()))), 4), n_ge0=int(sum(v >= 0 for v in d.values())))
    try:
        la = {u: cba(*mix("A", f"lc9_{users.index(u)}", "tv8", sub == u, seeds=SEEDS_A)) for u in STRANGERS}
        out["stranger_A_lc9_level"] = round(float(np.mean(list(la.values()))), 4)
    except FileNotFoundError as e:
        out["stranger_A_lc9_level"] = f"missing: {e.filename}"
    # returning: primary routed (copies ON); descriptive pooled tv8
    for form in ("routed", "tv8"):
        for k, scored in (("i", ("2", "3")), ("ii", ("3",))):
            dd = {}
            for x in ("P04", "P10"):
                fold = f"rt{1 if k == 'i' else 2}_{x}"
                keep = (sub == x) & np.isin(ses, scored)
                dd[x] = (cba(*mix("C", fold, form, keep)), cba(*mix("B", fold, form, keep)))
            RAW[f"returning_{k}_{form}"] = float(np.mean([c - b for c, b in dd.values()]))
            out[f"returning_{k}_{form}"] = dict(delta=round(float(np.mean([c - b for c, b in dd.values()])), 4),
                                                per_X={x: dict(B=round(b, 4), C=round(c, 4), d=round(c - b, 4)) for x, (c, b) in dd.items()})
    S, SR = out["stranger_tv8"], RAW["stranger_tv8"]
    di, dii = RAW["returning_i_routed"], RAW["returning_ii_routed"]
    sealed = 0.57 * SR["dS"] + 0.38 * di + 0.06 * dii
    out["sealed_arith"] = round(float(sealed), 4)
    out["unrounded"] = dict(dS=SR["dS"], min=SR["min"], d_i=di, d_ii=dii, sealed=sealed)
    if SR["dS"] < 0.002:
        verdict = "CLOSED (dS < +0.002)"
    elif SR["dS"] < 0.004:
        verdict = "UNDETERMINED (dS in [+0.002, +0.004); no rebuild)"
    elif SR["n_ge0"] >= 6 and SR["min"] >= -0.02 and sealed >= 0.002:
        verdict = "PASS (nominates ONE headed FM member B vs C; the promotion boundary applies)"
    else:
        verdict = "NOT PASS (dS >= +0.004 but a guard or the sealed arithmetic fails)"
    out["verdict"] = verdict
    L = ["# T2-D630 RESULT — the pooled-training composition: user-balanced (C) vs the shipped all-sessions composition (B), CNN families", "",
         "Registered + amended before any code or number (docs/DECISIONS_T2.md T2-D630). Paired seeds 880–882; ss8 = the 8 single-session strangers, "
         "rt1 / rt2 = P04 / P10 returning arms (copies ON, routed). Decode: per-family causal Sinkhorn (10 / 3 / 0.95), the CNN mix at W21's weights.", "",
         f"**Verdict: {verdict}**", "",
         f"- **Strangers, pooled tv8 (PRIMARY):** B {S['B']:.4f} → C {S['C']:.4f}, **ΔS {S['dS']:+.4f}**, users ≥ 0: {S['n_ge0']} / 8, min {S['min']:+.4f} ({S['min_who']}).",
         "  Per user: " + ", ".join(f"{u} {v:+.4f}" for u, v in S["per_user"].items()),
         f"- **Returning (routed, copies ON):** Δ(i) {di:+.4f} {out['returning_i_routed']['per_X']}; Δ(ii) {dii:+.4f} {out['returning_ii_routed']['per_X']}.",
         f"- **Sealed arithmetic** 0.57·ΔS + 0.38·Δ(i) + 0.06·Δ(ii) = **{sealed:+.4f}** (bar +0.002).",
         f"- The verdict reads the UNROUNDED values: ΔS {SR['dS']:+.6f}, min {SR['min']:+.6f}, Δ(i) {di:+.6f}, Δ(ii) {dii:+.6f}, sealed {sealed:+.6f}.", "",
         "Descriptive:",
         f"- strangers routed: ΔS {out['stranger_routed']['dS']:+.4f} ({out['stranger_routed']['n_ge0']} / 8 ≥ 0); plain (no TTA): ΔS {out['stranger_plain']['dS']:+.4f} "
         f"({out['stranger_plain']['n_ge0']} / 8); per family alone (tv8): " + ", ".join(f"{f} {out[f'stranger_tv8_{f}_alone']['dS']:+.4f}" for f in FAM),
         f"- returning pooled-only (tv8): Δ(i) {out['returning_i_tv8']['delta']:+.4f}, Δ(ii) {out['returning_ii_tv8']['delta']:+.4f}",
         f"- A (set E's lc9 cache, 9 users × session 1, draws 860–862) stranger level: {out['stranger_A_lc9_level']}", ""]
    R.joinpath("RESULT_D630.md").write_text("\n".join(L) + "\n")
    R.joinpath("result.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
