#!/usr/bin/env python
"""T2-D638 reader (registered + amended ×2 in docs/DECISIONS_T2.md before any number): the full-stack 2 × 2 in ship composition.
REFUSES unless the identity root PASSED (outputs/t2_d638/identity.log) AND the GPU chain finished EVERY cell ('### done d638 cell <c>' for
bw21 bw23 cw21 cw23 in outputs/t2_d638/chain.log; no 3-of-4 read).
  AFTER adaptation (the full stack): per cell, `t2_lab.py mix --set E --base w20 --idea w20_fb` on the lp root t2_d638_lp<cell>_860 with
  NI26_LAB_PRE_R=1, CKS 60,100,150; chronological + decode-only reorders (NI26_LAB_ORDER runs_perm / in_run, seeds 6310-6312; PARTIAL).
  Scored on the 8 spliced lc9 folds: strangers = TRU[f]'s rows (cell BA per user, mean over draws); P04 / P10 = their sessions 2-3 rows (cell BA
  per fold x draw, mean); deltas vs BW21 (and the candidates vs BW23); the per-draw values first (the student seed-to-seed sd).
  BEFORE adaptation (the CNN mix at the blend stage, D632's decode on D630's caches, the same cells' routed files outputs/t2_d638/route_<cell>):
  strangers (ss8), rt1 (sessions 2-3) and rt2 (session 3; no full-stack analogue in lc9) per X.
  DECISIONS (pinned): (a) W23 VETO: BW23 vs BW21 strangers dS <= 0 with <= 4 / 8 >= 0 chronologically AND < 0 under both reorders -> W23 OUT.
  (b) W24': CW21 / CW23 vs BW21: dS >= +0.004, >= 6 / 8 >= 0, dS >= 0 under both reorders, the CNN-stage d(ii) (C vs B, that router) >= -0.010,
  the full-stack d(i) >= -0.003, 0.57 dS + 0.38 d(i) + 0.06 d(ii, CNN stage) >= +0.002; both pass -> the higher chronological dS; it replaces W23
  only if its strangers beat BW23's by >= +0.002 (paired, chronological).
  mem-guard env OMP_NUM_THREADS=4 ... python scripts/local/t2_d638_read.py [--jobs 2]   -> outputs/t2_d638/RESULT_D638.md + result.json
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d638"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
PY = sys.executable
CELLS = ("bw21", "bw23", "cw21", "cw23")
DRAWS, SEEDS = (860, 861, 862), (880, 881, 882)
USERS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
SPLICED = [f for f in range(10) if USERS[f] not in ("P04", "P10")]
STRANGERS = [USERS[f] for f in SPLICED]
ORDERS = ["chrono"] + [f"{k}:{s}" for k in ("runs_perm", "in_run") for s in (6310, 6311, 6312)]
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3", 0.3143), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2", 0.2357),
       "ifn": ("ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2", 0.10)}


def refuse_unless_complete():
    idl = (R / "identity.log").read_text() if (R / "identity.log").exists() else ""
    assert "[d638 identity] PASS" in idl, "REFUSED: the identity root has not PASSED"
    ch = (R / "chain.log").read_text() if (R / "chain.log").exists() else ""
    assert "[d638 identity (ii)]" in ch and "STOP" not in ch, "REFUSED: identity (ii) missing or a STOP in chain.log"
    missing = [c for c in CELLS + ("nz",) if f"### done d638 cell {c} " not in ch]   # the noise cell too (no partial read)
    assert not missing, f"REFUSED: partial cells (no 3-of-4 read): {missing} not done"


def mix_path(cell, order):
    return R / f"mix_{cell}_{order.replace(':', '_')}.npz"


def run_mix(cell, order):
    out = mix_path(cell, order)
    if out.exists():
        return out
    assert cell != "ref", "the reference mix comes from the identity run"
    env = {k: v for k, v in os.environ.items() if not k.startswith("NI26_")}
    env.update(NI26_LAB_PRE_R="1", NI26_D559_ROOT=f"t2_d638_lp{cell}_860", NI26_D559_CKS="60,100,150")
    if order != "chrono":
        env["NI26_LAB_ORDER"] = order
    log = open(R / f"mix_{cell}.log", "a")
    subprocess.run([PY, "-u", "scripts/local/t2_lab.py", "mix", "--set", "E", "--base", "w20", "--idea", "w20_fb", "--out", str(out)],
                   env=env, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    z = np.load(out)
    assert str(z["idea"]) == "w20_fb" and str(z["base"]) == "w20" and str(z["order"]) == ("" if order == "chrono" else order), (cell, order)
    return out


def main():
    refuse_unless_complete()
    jobs = int(dict(zip(sys.argv[1::2], sys.argv[2::2])).get("--jobs", "2"))
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda co: run_mix(*co), [(c, o) for o in ORDERS for c in CELLS] + [("nz", "chrono")]))

    def cba(rows, p):
        return cell_balanced_accuracy(Y[rows], p.argmax(1), CELL[rows])[0]

    # ---- AFTER adaptation (the full stack) ----
    def score(cell, order):
        z = np.load(mix_path(cell, order))
        st = {u: [] for u in STRANGERS}
        rt = {x: [] for x in ("P04", "P10")}
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                m = U[ix] == USERS[f]
                st[USERS[f]].append(cba(ix[m], mx[m]))
                for x in rt:
                    m = (U[ix] == x) & np.isin(SES[ix], ("2", "3"))
                    rt[x].append(cba(ix[m], mx[m]))
        return st, rt                                    # st[u] = per draw (3); rt[x] = per (fold, draw) (8 x 3, fold-major)

    S = {(c, o): score(c, o) for c in CELLS for o in ORDERS}
    S[("nz", "chrono")], S[("ref", "chrono")] = score("nz", "chrono"), score("ref", "chrono")   # the noise floor: nz vs the reference root

    def delta(a, b, order):
        (sa, ra), (sb, rb) = S[(a, order)], S[(b, order)]
        du = {u: float(np.mean(np.subtract(sa[u], sb[u]))) for u in STRANGERS}
        dd = [float(np.mean([sa[u][j] - sb[u][j] for u in STRANGERS])) for j in range(len(DRAWS))]
        dx = {x: float(np.mean(np.subtract(ra[x], rb[x]))) for x in ra}
        return dict(dS=float(np.mean(list(du.values()))), n_ge0=int(sum(v >= 0 for v in du.values())), min=float(min(du.values())),
                    per_user=du, per_draw=dd, sd_draws=float(np.std(dd, ddof=1)), d_i=float(np.mean(list(dx.values()))), per_X=dx)

    def reorder(a, b):
        out = {}
        for k in ("runs_perm", "in_run"):
            ds = [delta(a, b, f"{k}:{s}") for s in (6310, 6311, 6312)]
            out[k] = dict(dS=float(np.mean([x["dS"] for x in ds])), n_ge0=[x["n_ge0"] for x in ds], d_i=float(np.mean([x["d_i"] for x in ds])))
        return out

    lev = {c: dict(S=float(np.mean([np.mean(v) for v in S[(c, "chrono")][0].values()])),
                   per_draw=[float(np.mean([S[(c, "chrono")][0][u][j] for u in STRANGERS])) for j in range(len(DRAWS))],
                   P04=float(np.mean(S[(c, "chrono")][1]["P04"])), P10=float(np.mean(S[(c, "chrono")][1]["P10"]))) for c in CELLS}
    pairs = {f"{a}-{b}": dict(chrono=delta(a, b, "chrono"), orders=reorder(a, b))
             for a, b in (("bw23", "bw21"), ("cw21", "bw21"), ("cw23", "bw21"), ("cw21", "bw23"), ("cw23", "bw23"))}

    # ---- BEFORE adaptation (the CNN mix at the blend stage; D632's decode on the cells' routed files) ----
    def cnn_mix(cell, fold, keep):
        te0, acc = None, 0.0
        for fam, (stem, w) in FAM.items():
            ps = []
            for s in SEEDS:
                zz = np.load(R / f"route_{cell}" / f"route_{fam}" / f"{fold}__{stem}__s{s}.npz")
                te0 = zz["test_idx"] if te0 is None else te0
                assert np.array_equal(zz["test_idx"], te0)
                ps.append(zz["prob_last"].astype(np.float64))
            acc = acc + w * causal_sinkhorn(np.mean(ps, 0), (STREAM[te0], ORD[te0]), iters=10, n0=3, lam=0.95)
        acc = acc / sum(w for _, w in FAM.values())
        k = keep[te0]
        return cba(te0[k], acc[k])

    pre = {}
    for c in CELLS:
        pre[c] = dict(S={u: cnn_mix(c, f"ss8_{u}", U == u) for u in STRANGERS},
                      i={x: cnn_mix(c, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                      ii={x: cnn_mix(c, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")})

    def pre_delta(a, b):
        du = {u: pre[a]["S"][u] - pre[b]["S"][u] for u in STRANGERS}
        return dict(dS=float(np.mean(list(du.values()))), n_ge0=int(sum(v >= 0 for v in du.values())),
                    d_i=float(np.mean([pre[a]["i"][x] - pre[b]["i"][x] for x in ("P04", "P10")])),
                    d_ii=float(np.mean([pre[a]["ii"][x] - pre[b]["ii"][x] for x in ("P04", "P10")])),
                    per_X_i={x: pre[a]["i"][x] - pre[b]["i"][x] for x in ("P04", "P10")}, per_X_ii={x: pre[a]["ii"][x] - pre[b]["ii"][x] for x in ("P04", "P10")})
    pre_pairs = {k: pre_delta(*k.split("-")) for k in pairs}

    # ---- the noise floor (the manager, with identity (ii)'s amendment): nz (both student generations regenerated, IFNet's GPU drift only) vs ref ----
    NZ = delta("nz", "ref", "chrono")
    FL_S, FL_I = abs(NZ["dS"]), abs(NZ["d_i"])

    def z(x, fl):                                        # a delta smaller than the floor is read as zero (raw values reported beside)
        return 0.0 if abs(x) < fl else x

    # ---- the pinned rules (on floor-zeroed deltas) ----
    A = pairs["bw23-bw21"]
    veto = bool(z(A["chrono"]["dS"], FL_S) <= 0 and A["chrono"]["n_ge0"] <= 4 and z(A["orders"]["runs_perm"]["dS"], FL_S) < 0
                and z(A["orders"]["in_run"]["dS"], FL_S) < 0)
    cand = {}
    for c, rb in (("cw21", "bw21"), ("cw23", "bw23")):     # the CNN-stage d(ii): C vs B under THAT router
        P_, ch = pairs[f"{c}-bw21"], pairs[f"{c}-bw21"]["chrono"]
        dii = pre_pairs[f"{c}-bw21"]["d_ii"] if rb == "bw21" else pre_delta(c, "bw23")["d_ii"]
        dS_, di_ = z(ch["dS"], FL_S), z(ch["d_i"], FL_I)
        sealed = 0.57 * dS_ + 0.38 * di_ + 0.06 * dii
        checks = dict(dS=dS_ >= 0.004, n6=ch["n_ge0"] >= 6, orders=all(z(P_["orders"][k]["dS"], FL_S) >= 0 for k in P_["orders"]),
                      cnn_dii=dii >= -0.010, d_i=di_ >= -0.003, sealed=sealed >= 0.002)
        cand[c] = dict(passes=bool(all(checks.values())), checks=checks, cnn_d_ii=dii, sealed=sealed,
                       vs_bw23=z(pairs[f"{c}-bw23"]["chrono"]["dS"], FL_S))
    winners = [c for c in cand if cand[c]["passes"]]
    best = max(winners, key=lambda c: pairs[f"{c}-bw21"]["chrono"]["dS"]) if winners else None
    replaces = bool(best and cand[best]["vs_bw23"] >= 0.002)
    verdict = dict(a_W23_veto=veto, b_candidates={c: cand[c]["passes"] for c in cand}, b_best=best, b_replaces_W23=replaces)

    out = dict(levels=lev, pairs=pairs, pre=pre, pre_pairs=pre_pairs, noise=dict(nz_vs_ref=NZ, floor_S=FL_S, floor_i=FL_I), rules=dict(cand=cand, verdict=verdict))
    R.joinpath("result.json").write_text(json.dumps(out, indent=1, default=float))

    def fmt(dct):
        return ", ".join(f"{k} {v:+.4f}" for k, v in dct.items())
    L = ["# T2-D638 RESULT — the full-stack 2 × 2 in ship composition (nets B / C × router W21 / session-1)", "",
         "Registered + amended ×2 before any number. Identity root PASSED (bitwise base / fb / lp teachers and mix); identity (ii) in chain.log.",
         "Hybrid caveat: CNN members ship composition (D630), FM / DC / side heads lc9 (set E). Order evidence decode-only (PARTIAL). "
         "rt2 has no full-stack analogue in lc9 (CNN stage only).", "",
         f"## The noise floor (nz = both student generations regenerated on the no-op routes vs the reference root; IFNet GPU drift only): "
         f"ΔS {NZ['dS']:+.4f} (per draw {', '.join(f'{v:+.4f}' for v in NZ['per_draw'])}; per user {fmt(NZ['per_user'])}), Δ(i) {NZ['d_i']:+.4f} "
         f"({fmt(NZ['per_X'])}) -> floors |ΔS| {FL_S:.4f}, |Δ(i)| {FL_I:.4f}: smaller deltas are read as zero in the rules (raw values reported)", "",
         "## Seed-to-seed (per draw 860 / 861 / 862) FIRST",
         *[f"- {c}: strangers per draw {', '.join(f'{v:.4f}' for v in lev[c]['per_draw'])}" for c in CELLS],
         *[f"- {k}: ΔS per draw {', '.join(f'{v:+.4f}' for v in p['chrono']['per_draw'])} (sd {p['chrono']['sd_draws']:.4f})" for k, p in pairs.items()], "",
         "## Levels (full stack, chronological)",
         *[f"- {c}: strangers {lev[c]['S']:.4f}; P04 s2–3 {lev[c]['P04']:.4f}; P10 s2–3 {lev[c]['P10']:.4f}" for c in CELLS], "",
         "## Pairs: AFTER adaptation (full stack) | BEFORE (CNN blend stage)"]
    for k, p in pairs.items():
        ch, pp = p["chrono"], pre_pairs[k]
        L += [f"- **{k}**: ΔS {ch['dS']:+.4f} ({ch['n_ge0']} / 8, worst {ch['min']:+.4f}); P04 {ch['per_X']['P04']:+.4f}, P10 {ch['per_X']['P10']:+.4f} "
              f"(Δ(i) {ch['d_i']:+.4f}); orders runs_perm {p['orders']['runs_perm']['dS']:+.4f} / in_run {p['orders']['in_run']['dS']:+.4f} | "
              f"CNN stage ΔS {pp['dS']:+.4f} ({pp['n_ge0']} / 8), Δ(i) {pp['d_i']:+.4f} ({fmt(pp['per_X_i'])}), Δ(ii) {pp['d_ii']:+.4f} ({fmt(pp['per_X_ii'])})",
              f"  per stranger (full stack): {fmt(ch['per_user'])}"]
    L += ["", "## Rules (pinned)",
          f"- **(a) W23 veto:** {'W23 OUT' if veto else 'not triggered (W23 stays)'} (BW23 − BW21: ΔS {A['chrono']['dS']:+.6f}, {A['chrono']['n_ge0']} / 8; "
          f"runs_perm {A['orders']['runs_perm']['dS']:+.6f}, in_run {A['orders']['in_run']['dS']:+.6f}).",
          *[f"- **(b) {c}:** {'PASS' if cand[c]['passes'] else 'fails'} — {cand[c]['checks']}; CNN-stage Δ(ii) {cand[c]['cnn_d_ii']:+.4f}; "
            f"sealed {cand[c]['sealed']:+.4f}; vs BW23 {cand[c]['vs_bw23']:+.4f}" for c in cand],
          f"- **Verdict:** {verdict}", ""]
    R.joinpath("RESULT_D638.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
