#!/usr/bin/env python
"""T2-D632 reader (registered 6aa58d4c + amended 3ac8b5a6 in docs/DECISIONS_T2.md before any number): THE ROUTE HUB, on D630's B-arm caches
(outputs/t2_d630/B, seeds 880-882; no training), the frame of scripts/local/t2_d631_read.py.
  S  = the shipped route (t2p_fpcalib fit_space on the fold's training fingerprints, 10 user centroids, nearest, running mean, every window);
       must reproduce D630's B route files (max |dp| <= 1e-6) or the reader STOPS.
  P1 = PRIMARY: the route space (scaler, LDA W, the user centroids) fitted on the fold's SESSION-1 training rows only (every training user must
       have session-1 rows, else STOP); the same route rule, copies and r (fbc 1/3, sh 0.5, ifn 0.5).
  Q  = DESCRIPTIVE: S's space, one centroid per TRAINING SESSION, a stream routes to the user owning the nearest session centroid.
  Decode = D630's routed form. dS = mean over the 8 ss8 strangers of P1 - S; d(i) rt1_X, d(ii) rt2_X, X in {P04, P10}.
  DECISION (unrounded): PASS = dS >= +0.004 AND 0.57 dS + 0.38 d(i) + 0.06 d(ii) >= +0.004 AND >= 6 / 8 strangers >= 0 AND worst >= -0.020 AND the
  D448 order guard (runs_perm, in_run; seeds 6310-6312; route + decode re-run in the permuted order; the sealed arithmetic of the shuffle-mean
  deltas >= 0 under both); else CLOSED (NOT PASS when only the guard fails). A PASS nominates only.
  Descriptive: route shares (S / P1 / Q), the self-route plurality on rt1 / rt2, each form's centroid nearest the origin, Q's deltas.
  python scripts/local/t2_d632_read.py    (OMP / MKL / OPENBLAS threads capped by the caller) -> outputs/t2_d632/RESULT_D632.md + result.json
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
RB, OUT = O / "t2_d630" / "B", O / "t2_d632"
SEEDS = (880, 881, 882)
FAM = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75", "3", 1 / 3), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr", "2", 0.5),
       "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr", "2", 0.5)}
WF = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10}
STRANGERS = ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P09")
RETURN = (("i", "rt1", ("2", "3")), ("ii", "rt2", ("3",)))
ORDER_SEEDS = (6310, 6311, 6312)
FORMS = ("S", "P1", "Q")


def main():
    X, idx, _ = B.load_export(str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    del X
    subj = idx["subject"].to_numpy()
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
    for fold in folds:
        (_, trm, _, tem), = B.folds(idx, fold)
        tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
        proj, mu, users = FC.fit_space(Fp[tr], subj[tr])
        tr1 = tr[ses[tr] == "1"]
        proj1, mu1, users1 = FC.fit_space(Fp[tr1], subj[tr1])
        if users1 != users:
            raise SystemExit(f"STOP: {fold}: a training user has no session-1 rows ({users} vs {users1})")
        Ztr = proj(Fp[tr])
        sk = sorted(set(zip(subj[tr].tolist(), ses[tr].tolist())))
        muQ = np.stack([Ztr[(subj[tr] == u) & (ses[tr] == s)].mean(0) for u, s in sk])
        mapQ = np.array([users.index(u) for u, _ in sk])
        mem = {}
        for fam, (model, al, rec, tag, r) in FAM.items():
            lst = []
            for s in SEEDS:
                z = np.load(RB / f"cnn_{fam}" / "runs" / f"{fold}__{model}__{al}__n00__{rec}tv8cpu__s{s}.npz")
                assert np.array_equal(z["test_idx"], te) and list(z["users"]) == users, (fold, fam, s)
                lst.append((z["prob_last"].astype(np.float64), z["prob_users_test"].astype(np.float64)))
            mem[fam] = lst
        F[fold] = dict(te=te, users=[str(u).split("/")[-1] for u in users], mem=mem,
                       space={"S": (proj(Fp[te]), mu, np.arange(len(mu))), "P1": (proj1(Fp[te]), mu1, np.arange(len(mu1))),
                              "Q": (proj(Fp[te]), muQ, mapQ)},
                       origin={"S": [str(users[j]).split("/")[-1] for j in np.argsort(np.linalg.norm(mu, axis=1))[:2]],
                               "P1": [str(users[j]).split("/")[-1] for j in np.argsort(np.linalg.norm(mu1, axis=1))[:2]]},
                       origin_norm={"S": np.round(np.sort(np.linalg.norm(mu, axis=1))[:2], 2).tolist(),
                                    "P1": np.round(np.sort(np.linalg.norm(mu1, axis=1))[:2], 2).tolist()})

    def route(fold, key, form):
        f = F[fold]
        te, k = f["te"], np.full(len(f["te"]), -1)
        Z, mu_, map_ = f["space"][form]
        for s in np.unique(stream[te]):
            m = np.flatnonzero(stream[te] == s)
            m = m[np.argsort(key[te][m], kind="stable")]
            j = FC.stream_route_every(Z[m], mu_, 1)
            k[m] = np.where(j >= 0, map_[np.maximum(j, 0)], -1)
        return k

    def decode(fold, k, scale, key):
        f = F[fold]
        te, acc = f["te"], 0.0
        for fam in FAM:
            P = np.mean([FC.blend_route(sh, Pu, k, FAM[fam][4] * scale) for sh, Pu in f["mem"][fam]], 0)
            acc = acc + WF[fam] * causal_sinkhorn(P, (stream[te], key[te]), iters=10, n0=3, lam=0.95)
        return acc / sum(WF.values())

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
    key0 = ORD[("chrono", 0)]
    K = {(fold, form): route(fold, key0, form) for fold in folds for form in FORMS}
    check = 0.0
    for fold in folds:
        for fam, (model, al, rec, tag, r) in FAM.items():
            for s, (sh, Pu) in zip(SEEDS, F[fold]["mem"][fam]):
                z = np.load(RB / f"route_{fam}" / f"{fold}__{model}__{al}__n00__{rec}tv8cfb{tag}__s{s}.npz")
                assert np.array_equal(z["test_idx"], F[fold]["te"])
                check = max(check, float(np.abs(FC.blend_route(sh, Pu, K[(fold, "S")], r) - z["prob_last"].astype(np.float64)).max()))
    if check > 1e-6:
        raise SystemExit(f"STOP: S does not reproduce D630's B route files (max |dp| {check:.2e})")
    print(f"S reproduces D630's B route files: max |dp| {check:.2e}", flush=True)

    # 2. levels: chrono for S / P1 / Q / pooled; the order guard for S / P1
    LV = {}
    for (kind, sd), key in ORD.items():
        for fold in folds:
            kp = keep_of(fold)
            for form in (FORMS if kind == "chrono" else ("S", "P1")):
                k = K[(fold, form)] if kind == "chrono" else route(fold, key, form)
                LV[(kind, sd, fold, form)] = cba(fold, decode(fold, k, 1.0, key), kp)
            if kind == "chrono":
                LV[(kind, sd, fold, "pooled")] = cba(fold, decode(fold, K[(fold, "S")], 0.0, key), kp)
        print(f"levels done: {kind} {sd}", flush=True)

    def deltas(kind, sds, arm):
        dS_u = {u: float(np.mean([LV[(kind, sd, f"ss8_{u}", arm)] - LV[(kind, sd, f"ss8_{u}", "S")] for sd in sds])) for u in STRANGERS}
        dr = {k_: {x: float(np.mean([LV[(kind, sd, f"{p}_{x}", arm)] - LV[(kind, sd, f"{p}_{x}", "S")] for sd in sds])) for x in ("P04", "P10")}
              for k_, p, _ in RETURN}
        dS = float(np.mean(list(dS_u.values())))
        di, dii = float(np.mean(list(dr["i"].values()))), float(np.mean(list(dr["ii"].values())))
        return dict(dS=dS, n_ge0=int(sum(v >= 0 for v in dS_u.values())), min=float(min(dS_u.values())), min_who=min(dS_u, key=dS_u.get),
                    per_user=dS_u, d_i=di, d_ii=dii, per_X_i=dr["i"], per_X_ii=dr["ii"], sealed=0.57 * dS + 0.38 * di + 0.06 * dii)

    C = deltas("chrono", (0,), "P1")
    CQ = deltas("chrono", (0,), "Q")
    G = {kind: deltas(kind, ORDER_SEEDS, "P1") for kind in ("runs_perm", "in_run")}
    guard = all(G[k_]["sealed"] >= 0 for k_ in G)
    fails = [w for w, bad in (("dS < +0.004", C["dS"] < 0.004), ("sealed < +0.004", C["sealed"] < 0.004), ("< 6 / 8 strangers >= 0", C["n_ge0"] < 6),
                              ("worst < -0.020", C["min"] < -0.020)) if bad]
    if not fails:
        verdict = ("PASS (NOMINATES the session-1 route space only: the owner's line + every W21 gate; the full-stack effect stays unmeasured "
                   "under the ship composition; build decision by ~ Nov 20 06:00)") if guard else "NOT PASS (the chronological read passes but the D448 order guard fails)"
    else:
        verdict = "CLOSED (" + ", ".join(fails) + ")"

    # 3. descriptive
    share = {form: {} for form in FORMS}
    selfp = {}
    for fold in folds:
        kp = keep_of(fold)
        us = F[fold]["users"]
        for form in FORMS:
            c_ = Counter(K[(fold, form)][kp].tolist())
            share[form][fold] = {us[j] if j >= 0 else "none": int(n) for j, n in sorted(c_.items())}
            if not fold.startswith("ss8_"):
                x = fold.split("_")[1]
                top = max(share[form][fold].items(), key=lambda t: t[1])[0]
                selfp[(form, fold)] = dict(own=share[form][fold].get(x, 0), n=int(kp.sum()), plurality=top == x)
    lev = {form: {fold: LV[("chrono", 0, fold, form)] for fold in folds} for form in FORMS + ("pooled",)}
    m8 = lambda form: float(np.mean([lev[form][f"ss8_{u}"] for u in STRANGERS]))  # noqa: E731
    out = dict(check_max_dp=check, verdict=verdict, chrono_P1=C, chrono_Q=CQ, order_guard_P1=G, guard_ok=guard,
               levels={f_: {fold: round(v, 4) for fold, v in d.items()} for f_, d in lev.items()}, route_share=share,
               self_route={f"{a}|{b}": v for (a, b), v in selfp.items()}, origin={fold: F[fold]["origin"] for fold in folds},
               origin_norm={fold: F[fold]["origin_norm"] for fold in folds})

    def line(D):
        return (f"ΔS {D['dS']:+.4f} ({D['n_ge0']} / 8 ≥ 0, worst {D['min']:+.4f} {D['min_who']}); Δ(i) {D['d_i']:+.4f} "
                f"({', '.join(f'{x} {v:+.4f}' for x, v in D['per_X_i'].items())}); Δ(ii) {D['d_ii']:+.4f} "
                f"({', '.join(f'{x} {v:+.4f}' for x, v in D['per_X_ii'].items())}); sealed {D['sealed']:+.4f}")
    L = ["# T2-D632 RESULT — the route hub: the session-1 route space (PRIMARY) and per-session centroids (descriptive), D630's B-arm caches", "",
         f"Registered 6aa58d4c + amended 3ac8b5a6 before any number. S reproduces D630's B route files: max |dp| {check:.1e}. CNN mix at the blend stage.", "",
         f"**Verdict: {verdict}**", "",
         f"- **Strangers (8, ss8) levels:** pooled {m8('pooled'):.4f} / S {m8('S'):.4f} / **P1 {m8('P1'):.4f}** / Q {m8('Q'):.4f}.",
         f"- **P1 − S (PRIMARY, chronological):** {line(C)} (bars: ΔS ≥ +0.004, sealed ≥ +0.004, ≥ 6 / 8, worst ≥ −0.020).",
         "  Per stranger: " + ", ".join(f"{u} {v:+.4f}" for u, v in C["per_user"].items()),
         f"- Unrounded: ΔS {C['dS']:+.6f}, worst {C['min']:+.6f}, Δ(i) {C['d_i']:+.6f}, Δ(ii) {C['d_ii']:+.6f}, sealed {C['sealed']:+.6f}.",
         "- **Order guard (P1 − S, shuffle means):** " + "; ".join(f"{k_}: {line(g)}" for k_, g in G.items()) + f" → {'holds' if guard else 'FAILS'}.", "",
         "Descriptive (chronological):",
         f"- **Q − S (per-session centroids):** {line(CQ)}.",
         "- levels pooled / S / P1 / Q: " + "; ".join(f"{fold} {lev['pooled'][fold]:.4f} / {lev['S'][fold]:.4f} / {lev['P1'][fold]:.4f} / {lev['Q'][fold]:.4f}" for fold in folds)]
    for form in FORMS:
        L.append(f"- route share {form}: " + "; ".join(f"{fold} {share[form][fold]}" for fold in folds))
    L.append("- self-route (X's scored windows to X's own copy; plurality): " + "; ".join(
        f"{form} {fold} {v['own']} / {v['n']}{' ✓' if v['plurality'] else ' ✗'}" for (form, fold), v in selfp.items()))
    L.append("- the two centroids nearest the origin (S | P1): " + "; ".join(
        f"{fold} {F[fold]['origin']['S']} {F[fold]['origin_norm']['S']} | {F[fold]['origin']['P1']} {F[fold]['origin_norm']['P1']}" for fold in folds))
    L += ["", "Caveat (registered): rt1 / rt2 are only P04 / P10 (the 0.38 regime rests on two people); the read is the CNN mix at the blend stage "
          "(W21's in-stream copies, side refits, P1 copies, FM, DC and the family decode are not in it).", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.joinpath("RESULT_D632.md").write_text("\n".join(L) + "\n")
    OUT.joinpath("result.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
