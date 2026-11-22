#!/usr/bin/env python
"""T2-D631 reader (registered in docs/DECISIONS_T2.md, b10aabc0, before any number of the alternative): the copy share r x 0.5 under the ship
composition, on D630's B-arm caches (outputs/t2_d630/B, seeds 880-882; no training).
  S = the shipped r (fbc 1/3, sh 0.5, ifn 0.5), recomputed from the pu files (t2p_fpcalib fit_space / stream_route_every / blend_route); it
      must reproduce D630's B route files (max |dp| <= 1e-6) or the reader STOPS. H = r x 0.5 (the same route k). r = 0 = the pooled tv8 (reference).
  Decode = D630's routed form: per family the seed-mean -> causal Sinkhorn per stream (10 / 3 / 0.95) -> W21's CNN weights renormalised -> argmax
  -> cell BA. dS = mean over the 8 ss8 strangers of H - S on u's windows; d(i) = mean over X in {P04, P10} of H - S on rt1_X (sessions 2-3),
  d(ii) on rt2_X (session 3).
  DECISION (unrounded; AMENDED 12:55:19 before any number, the dilution-scaled bar): PASS = dS >= +0.004 AND 0.57 dS + 0.38 d(i) + 0.06 d(ii)
  >= +0.004 AND >= 6 / 8 users >= 0 AND the worst user >= -0.020, and the ORDER GUARD (D448's orders, as D624): under runs_perm and under in_run
  (3 shuffles each, seeds 6310-6312; the route k and the decode re-run in the permuted order; the members' own aligners stay in recording order)
  the sealed arithmetic of the shuffle-mean deltas stays >= 0; else CLOSED (NOT PASS when only the guard fails). A PASS nominates only.
  Descriptive: route shares per copy, per copy routed - pooled decoded window accuracy, per stranger the forced-route cost of each copy.
  python scripts/local/t2_d631_read.py    (OMP / MKL / OPENBLAS threads capped by the caller) -> outputs/t2_d631/RESULT_D631.md + result.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FC  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
RB, OUT = O / "t2_d630" / "B", O / "t2_d631"
SEEDS = (880, 881, 882)
FAM = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75", "3", 1 / 3), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr", "2", 0.5),
       "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr", "2", 0.5)}
WF = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10}
STRANGERS = ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P09")
RETURN = (("i", "rt1", ("2", "3")), ("ii", "rt2", ("3",)))
SCALES = {"S": 1.0, "H": 0.5, "pooled": 0.0}
ORDER_SEEDS = (6310, 6311, 6312)


def main():
    X, idx, _ = B.load_export(str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    del X
    subj = idx["subject"].to_numpy()                       # as t2p_fpcalib blend (the pu files' user names)
    sub = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
    ses, y = idx["session"].astype(str).to_numpy(), idx["y"].to_numpy().astype(int)
    cell, stream, order = idx["cell"].astype(str).to_numpy(), idx["stream"].astype(str).to_numpy(), idx["order"].to_numpy()
    run = idx["run"].to_numpy().astype(int)
    Fp = np.load(O / "t2_d124b" / "fp_proteus.npy")
    assert len(Fp) == len(idx)

    def okeys(kind, rng):
        """t2_d448_orderfit.keys over the whole export: chrono | runs_perm | in_run."""
        if kind == "chrono":
            return order.astype(float)
        k = np.empty(len(idx))
        for s in np.unique(stream):
            m = np.flatnonzero(stream == s)
            runs = np.unique(run[m])
            pos = dict(zip(runs, rng.permutation(len(runs)) if kind == "runs_perm" else np.arange(len(runs))))
            for r_ in runs:
                mr = m[run[m] == r_]
                inner = rng.permutation(len(mr)) if kind == "in_run" else np.argsort(np.argsort(order[mr]))
                k[mr] = pos[r_] * 1000 + inner
        return k

    ORD = {("chrono", 0): okeys("chrono", None)}
    for kind in ("runs_perm", "in_run"):
        for sd in ORDER_SEEDS:
            ORD[(kind, sd)] = okeys(kind, np.random.default_rng(sd))

    folds = [f"ss8_{u}" for u in STRANGERS] + [f"{p}_{x}" for _, p, _ in RETURN for x in ("P04", "P10")]
    F = {}
    check = 0.0
    for fold in folds:
        (_, trm, _, tem), = B.folds(idx, fold)
        tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
        proj, mu, users = FC.fit_space(Fp[tr], subj[tr])
        Z = proj(Fp[te])
        mem = {}
        for fam, (model, al, rec, tag, r) in FAM.items():
            lst = []
            for s in SEEDS:
                z = np.load(RB / f"cnn_{fam}" / "runs" / f"{fold}__{model}__{al}__n00__{rec}tv8cpu__s{s}.npz")
                assert np.array_equal(z["test_idx"], te) and list(z["users"]) == users, (fold, fam, s)
                lst.append((z["prob_last"].astype(np.float64), z["prob_users_test"].astype(np.float64)))
            mem[fam] = lst
        F[fold] = dict(te=te, Z=Z, users=[str(u).split("/")[-1] for u in users], mem=mem, mu=mu)

    def route(fold, key):
        f = F[fold]
        te, k = f["te"], np.full(len(f["te"]), -1)
        for s in np.unique(stream[te]):
            m = np.flatnonzero(stream[te] == s)
            m = m[np.argsort(key[te][m], kind="stable")]
            k[m] = FC.stream_route_every(f["Z"][m], f["mu"], 1)
        return k

    def decode(fold, k, scale, key, fams=tuple(FAM)):
        f = F[fold]
        te, acc = f["te"], 0.0
        for fam in fams:
            P = np.mean([FC.blend_route(sh, Pu, k, FAM[fam][4] * scale) for sh, Pu in f["mem"][fam]], 0)
            acc = acc + WF[fam] * causal_sinkhorn(P, (stream[te], key[te]), iters=10, n0=3, lam=0.95)
        return acc / sum(WF[f_] for f_ in fams)

    def cba(fold, p, keep):
        rows = F[fold]["te"][keep]
        return cell_balanced_accuracy(y[rows], p[keep].argmax(1), cell[rows])[0]

    def keep_of(fold):
        te = F[fold]["te"]
        if fold.startswith("ss8_"):
            return sub[te] == fold[4:]
        p, x = fold.split("_")
        return (sub[te] == x) & np.isin(ses[te], dict((q, sc) for _, q, sc in RETURN)[p])

    # 1. S reproduces D630's B route files (chrono)
    K0 = {fold: route(fold, ORD[("chrono", 0)]) for fold in folds}
    for fold in folds:
        for fam, (model, al, rec, tag, r) in FAM.items():
            for s, (sh, Pu) in zip(SEEDS, F[fold]["mem"][fam]):
                z = np.load(RB / f"route_{fam}" / f"{fold}__{model}__{al}__n00__{rec}tv8cfb{tag}__s{s}.npz")
                assert np.array_equal(z["test_idx"], F[fold]["te"])
                check = max(check, float(np.abs(FC.blend_route(sh, Pu, K0[fold], r) - z["prob_last"].astype(np.float64)).max()))
    if check > 1e-6:
        raise SystemExit(f"STOP: S does not reproduce D630's B route files (max |dp| {check:.2e})")
    print(f"S reproduces D630's B route files: max |dp| {check:.2e}", flush=True)

    # 2. levels per (order, fold, arm)
    LV = {}
    for (kind, sd), key in ORD.items():
        for fold in folds:
            k = K0[fold] if kind == "chrono" else route(fold, key)
            kp = keep_of(fold)
            for arm, sc in SCALES.items():
                if kind != "chrono" and arm == "pooled":
                    continue
                LV[(kind, sd, fold, arm)] = cba(fold, decode(fold, k, sc, key), kp)
        print(f"levels done: {kind} {sd}", flush=True)

    def deltas(kind, sds):
        dS_u = {u: float(np.mean([LV[(kind, sd, f"ss8_{u}", "H")] - LV[(kind, sd, f"ss8_{u}", "S")] for sd in sds])) for u in STRANGERS}
        dr = {}
        for k_, p, _ in RETURN:
            dr[k_] = {x: float(np.mean([LV[(kind, sd, f"{p}_{x}", "H")] - LV[(kind, sd, f"{p}_{x}", "S")] for sd in sds])) for x in ("P04", "P10")}
        dS = float(np.mean(list(dS_u.values())))
        di, dii = float(np.mean(list(dr["i"].values()))), float(np.mean(list(dr["ii"].values())))
        return dict(dS=dS, n_ge0=int(sum(v >= 0 for v in dS_u.values())), min=float(min(dS_u.values())), min_who=min(dS_u, key=dS_u.get),
                    per_user=dS_u, d_i=di, d_ii=dii, per_X_i=dr["i"], per_X_ii=dr["ii"], sealed=0.57 * dS + 0.38 * di + 0.06 * dii)

    C = deltas("chrono", (0,))
    G = {kind: deltas(kind, ORDER_SEEDS) for kind in ("runs_perm", "in_run")}
    guard = all(G[k_]["sealed"] >= 0 for k_ in G)
    fails = [w for w, bad in (("dS < +0.004", C["dS"] < 0.004), ("sealed < +0.004", C["sealed"] < 0.004), ("< 6 / 8 users >= 0", C["n_ge0"] < 6),
                              ("worst user < -0.020", C["min"] < -0.020)) if bad]
    if not fails:
        verdict = ("PASS (NOMINATES r x 0.5 only: the owner's line + every W21 gate; the full-stack effect stays unmeasured under the ship "
                   "composition; build decision by ~ Nov 20 06:00)") if guard else "NOT PASS (the chronological read passes but the D448 order guard fails)"
    else:
        verdict = "CLOSED (" + ", ".join(fails) + ")"

    # 3. descriptive anatomy (chrono)
    key0 = ORD[("chrono", 0)]
    lev = {arm: {fold: LV[("chrono", 0, fold, arm)] for fold in folds} for arm in SCALES}
    share, percopy = {}, {}
    for fold in folds:
        kp = keep_of(fold)
        us = F[fold]["users"]
        share[fold] = {us[j] if j >= 0 else "none": int(n) for j, n in sorted(Counter(K0[fold][kp].tolist()).items())}
        if fold.startswith("ss8_"):
            pS, p0 = decode(fold, K0[fold], 1.0, key0), decode(fold, K0[fold], 0.0, key0)
            rows = F[fold]["te"]
            for j in np.unique(K0[fold][kp]):
                m = kp & (K0[fold] == j)
                nm = us[j] if j >= 0 else "none"
                d = percopy.setdefault(nm, dict(n=0, right_S=0, right_pooled=0))
                d["n"] += int(m.sum()); d["right_S"] += int((pS[m].argmax(1) == y[rows[m]]).sum()); d["right_pooled"] += int((p0[m].argmax(1) == y[rows[m]]).sum())
    for nm, d in percopy.items():
        d["acc_S_minus_pooled"] = round((d["right_S"] - d["right_pooled"]) / max(d["n"], 1), 4)
    forced = {}
    for u in STRANGERS:
        fold = f"ss8_{u}"
        kp, te = keep_of(fold), F[fold]["te"]
        base = lev["pooled"][fold]
        forced[u] = {nm: round(cba(fold, decode(fold, np.full(len(te), j), 1.0, key0), kp) - base, 4) for j, nm in enumerate(F[fold]["users"])}
    print("anatomy done", flush=True)

    out = dict(check_max_dp=check, verdict=verdict, chrono=C, order_guard=G, guard_ok=guard,
               levels={arm: {fold: round(v, 4) for fold, v in d.items()} for arm, d in lev.items()},
               route_share=share, per_copy_strangers=percopy, forced_route_cost=forced)
    m8 = lambda arm: float(np.mean([lev[arm][f"ss8_{u}"] for u in STRANGERS]))  # noqa: E731
    L = ["# T2-D631 RESULT — the copy share r × 0.5 under the ship composition (D630's B-arm caches, CNN mix at the blend stage)", "",
         f"Registered before any number (docs/DECISIONS_T2.md T2-D631, b10aabc0). S reproduces D630's B route files: max |dp| {check:.1e}.", "",
         f"**Verdict: {verdict}**", "",
         f"- **Strangers (8, ss8):** pooled {m8('pooled'):.4f} / S {m8('S'):.4f} / H {m8('H'):.4f}; **ΔS (H − S) {C['dS']:+.4f}**, users ≥ 0: {C['n_ge0']} / 8, "
         f"min {C['min']:+.4f} ({C['min_who']}). Per user: " + ", ".join(f"{u} {v:+.4f}" for u, v in C["per_user"].items()),
         f"- **Returning:** Δ(i) rt1 {C['d_i']:+.4f} ({', '.join(f'{x} {v:+.4f}' for x, v in C['per_X_i'].items())}); Δ(ii) rt2 {C['d_ii']:+.4f} "
         f"({', '.join(f'{x} {v:+.4f}' for x, v in C['per_X_ii'].items())}).",
         f"- **Sealed arithmetic** 0.57·ΔS + 0.38·Δ(i) + 0.06·Δ(ii) = **{C['sealed']:+.4f}** (bar +0.004, dilution-scaled; ΔS bar +0.004, worst user ≥ −0.020).",
         f"- Unrounded: ΔS {C['dS']:+.6f}, min {C['min']:+.6f}, Δ(i) {C['d_i']:+.6f}, Δ(ii) {C['d_ii']:+.6f}, sealed {C['sealed']:+.6f}.",
         "- **Order guard (D448; shuffle-mean deltas, seeds 6310–6312):** " + "; ".join(
             f"{k_}: ΔS {g['dS']:+.4f} ({g['n_ge0']} / 8), Δ(i) {g['d_i']:+.4f}, Δ(ii) {g['d_ii']:+.4f}, sealed {g['sealed']:+.4f}" for k_, g in G.items())
         + f" → {'holds' if guard else 'FAILS'}.", "",
         "Descriptive (chronological):",
         "- levels pooled / S / H: " + "; ".join(f"{fold} {lev['pooled'][fold]:.4f} / {lev['S'][fold]:.4f} / {lev['H'][fold]:.4f}" for fold in folds),
         "- route share (scored windows → copy): " + "; ".join(f"{fold} {share[fold]}" for fold in folds),
         "- per copy over the 8 strangers (windows routed; S − pooled decoded window accuracy): "
         + "; ".join(f"{nm} n {d['n']}, {d['acc_S_minus_pooled']:+.4f}" for nm, d in sorted(percopy.items())),
         "- forced-route cost per stranger (every window → copy k at the shipped r, cell BA − pooled):"]
    L += [f"  - {u}: " + ", ".join(f"{nm} {v:+.4f}" for nm, v in forced[u].items()) for u in STRANGERS]
    L += ["", "Caveat (registered): rt1 / rt2 are only P04 / P10, so the 0.38 regime rests on two people; the read is the CNN mix at the blend stage "
          "(W21's in-stream copies, side refits, P1 copies, FM, DC and the family decode are not in it).", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.joinpath("RESULT_D631.md").write_text("\n".join(L) + "\n")
    OUT.joinpath("result.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
