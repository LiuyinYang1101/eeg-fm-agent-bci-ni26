#!/usr/bin/env python
"""T2-D656 (registered 544853f8 before any number): THE ORDER HEDGE READ. Decode-only (PARTIAL) mixes of W25-ALL6's bw21 C' (the all6rep TE tags n:
roots s / p04 / p10, draws 860-862) through t2_lab.py with NI26_LAB_LAM (the lambda of every decode of the final saved mix) and NI26_LAB_ORDER
(runs_perm / in_run / in_sess at seeds 6310-6312). Per (lambda, order): strangers dS (8 users, root s; = t2_d642_set_read.score_s), returning
(P04 / P10 sessions 2-3 on roots p04 / p10; = score_x), OLD sealed = 0.57 dS + 0.38 dret (d(ii) = 0: no member changes), each vs lambda 0.95 under the
SAME order and seed. PICK RULE (the manager's, verbatim): 'the hedge file = the config with the best OLD sealed under the session shuffle, provided it
is >= +0.002 over the shipped config there; else no hedge file.' AMENDED before any number (b48639f4 / 78b668c5): the TIED set = configs whose in_sess OLD sealed is within 0.001 of the best AND itself >= +0.002; among them the higher CHRONOLOGICAL OLD sealed is the pick (chrono cost reported, not gating). PHASES (grid order only): the decision cells (chrono + in_sess) first, the decision written; then runs_perm / in_run (reported) appended. Identity first: lambda 0.95 chrono on root s with the knob set == the set read's
mix_ns_chrono.npz bitwise (else STOP).
    flock <setread/bw21/.cpu.lock> mem-guard python scripts/local/t2_d656_order_read.py [--jobs 4]
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
sys.path.insert(0, str(ROOT / "src"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
SR = O / "t2_d642" / "setread" / "bw21"
OUT = SR / "d656"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
PY = sys.executable
IDEA = "w20_fb"
TAG = "n"                                                # all6rep's TE = W25-ALL6's C' (the shipped members)
LAMS = (0.95, 0.98, 1.0)
KINDS = ("runs_perm", "in_run", "in_sess")
SEEDS = (6310, 6311, 6312)
ORDERS = ["chrono"] + [f"{k}:{s}" for k in KINDS for s in SEEDS]
DRAWS = (860, 861, 862)
USERS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
SPLICED = [f for f in range(10) if USERS[f] not in ("P04", "P10")]
STRANGERS = [USERS[f] for f in SPLICED]
JOBS = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4


def mix_path(tag, order, lam):
    return OUT / f"mix_{tag}_{order.replace(':', '_')}_l{lam:.2f}.npz"


def run_mix(tag, order, lam):
    out = mix_path(tag, order, lam)
    if not out.exists():
        env = {k: v for k, v in os.environ.items() if not k.startswith("NI26_")}
        env.update(NI26_LAB_PRE_R="1", NI26_D559_ROOT=f"t2_d642_lp{tag}_bw21_860", NI26_D559_CKS="60,100,150", NI26_LAB_FMSPLICE=str(SR / f"fmspl_{tag}"),
                   NI26_LAB_LAM=repr(float(lam)), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if tag[1] == "p":
            env["NI26_LAB_FMSTART"] = f"freeze:{SR / 'fmstart.npz'}"
        if order != "chrono":
            env["NI26_LAB_ORDER"] = order
        with open(OUT / f"mix_{tag}.log", "a") as log:
            subprocess.run([PY, "-u", "scripts/local/t2_lab.py", "mix", "--set", "E", "--base", "w20", "--idea", IDEA, "--out", str(out)],
                           env=env, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    z = np.load(out)
    assert str(z["idea"]) == IDEA and str(z["order"]) == ("" if order == "chrono" else order) and abs(float(z["lam"]) - lam) < 1e-12, (tag, order, lam)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # identity: the knob at the shipped lambda == the set read's own chronological mix, bitwise
    run_mix(f"{TAG}s", "chrono", 0.95)
    a, b = np.load(mix_path(f"{TAG}s", "chrono", 0.95)), np.load(SR / f"mix_{TAG}s_chrono.npz")
    ks = [k for k in b.files if k.startswith(("ix_", "mx_"))]
    assert ks and all(np.array_equal(a[k], b[k]) for k in ks), "T2-D656 IDENTITY FAILED: lambda 0.95 with the knob != the set read's mix_ns_chrono.npz"
    print(f"[d656] identity PASS: lambda 0.95 via NI26_LAB_LAM == mix_{TAG}s_chrono.npz bitwise ({len(ks)} arrays)", flush=True)
    DEC = [o for o in ORDERS if o == "chrono" or o.startswith("in_sess")]
    runs1 = [(f"{TAG}{r}", o, lam) for r in ("s", "p04", "p10") for lam in LAMS for o in DEC]
    runs2 = [(f"{TAG}{r}", o, lam) for r in ("s", "p04", "p10") for lam in LAMS for o in ORDERS if o not in DEC]
    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(lambda t: run_mix(*t), runs1))
    report(DEC, final=False)
    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(lambda t: run_mix(*t), runs2))
    report(ORDERS, final=True)


def report(orders_done, final):

    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()

    def cba(rows, p):
        return cell_balanced_accuracy(Y[rows], p.argmax(1), CELL[rows])[0]

    def score_s(path):
        z = np.load(path)
        st = {u: [] for u in STRANGERS}
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = U[ix] == USERS[f]
                st[USERS[f]].append(cba(ix[k], mx[k]))
        return st

    def score_x(path, x):
        z = np.load(path)
        v = []
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = (U[ix] == x) & np.isin(SES[ix], ("2", "3"))
                v.append(cba(ix[k], mx[k]))
        return v

    S = {(lam, o): score_s(mix_path(f"{TAG}s", o, lam)) for lam in LAMS for o in orders_done}
    X = {(lam, o): {x: score_x(mix_path(f"{TAG}p{x[1:]}", o, lam), x) for x in ("P04", "P10")} for lam in LAMS for o in orders_done}
    kinds = [k for k in ("chrono",) + KINDS if any(o == k or o.startswith(k + ":") for o in orders_done)]
    lev = lambda st: float(np.mean([np.mean(v) for v in st.values()]))                       # noqa: E731
    levx = lambda xx: float(np.mean([np.mean(v) for v in xx.values()]))                      # noqa: E731

    def cell(lam, o):                                    # lam vs 0.95 under the same order / seed
        du = {u: float(np.mean(np.subtract(S[(lam, o)][u], S[(0.95, o)][u]))) for u in STRANGERS}
        dS = float(np.mean(list(du.values())))
        dret = float(np.mean([np.mean(np.subtract(X[(lam, o)][x], X[(0.95, o)][x])) for x in ("P04", "P10")]))
        return dict(dS=dS, n_ge0=int(sum(v >= 0 for v in du.values())), min=float(min(du.values())), per_user=du, dret=dret, sealed=0.57 * dS + 0.38 * dret)

    res = {"levels": {}, "cells": {}}
    for lam in LAMS:
        for kind in kinds:
            os_ = ["chrono"] if kind == "chrono" else [f"{kind}:{s}" for s in SEEDS]
            res["levels"][f"{lam}|{kind}"] = dict(strangers=float(np.mean([lev(S[(lam, o)]) for o in os_])), returning=float(np.mean([levx(X[(lam, o)]) for o in os_])))
            if lam != 0.95:
                cs = [cell(lam, o) for o in os_]
                res["cells"][f"{lam}|{kind}"] = dict(dS=float(np.mean([c["dS"] for c in cs])), dret=float(np.mean([c["dret"] for c in cs])),
                                                    sealed=float(np.mean([c["sealed"] for c in cs])), n_ge0=[c["n_ge0"] for c in cs],
                                                    per_seed_sealed=[c["sealed"] for c in cs],
                                                    per_user=({u: float(np.mean([c["per_user"][u] for c in cs])) for u in STRANGERS}))
    sh = {lam: res["cells"][f"{lam}|in_sess"]["sealed"] for lam in (0.98, 1.0)}
    best = max(sh, key=sh.get)
    chs = {lam: res["cells"][f"{lam}|chrono"]["sealed"] for lam in (0.98, 1.0)}
    tied = [lam for lam in (0.98, 1.0) if sh[lam] >= 0.002 and sh[lam] >= sh[best] - 0.001]   # AMENDED (b48639f4 / 78b668c5)
    pick = max(tied, key=lambda lam: chs[lam]) if tied else None
    res.update(pick=pick, best_under_session_shuffle=best, sealed_under_session_shuffle=sh, tied=tied, chrono_sealed=chs, final=final,
               orders_done=list(orders_done),
               rule=("the manager's: gate = the best in_sess OLD sealed >= +0.002 over lambda 0.95; AMENDED: tied = in_sess OLD sealed within 0.001 "
                     "of the best AND >= +0.002; among the tied, the higher chrono OLD sealed is the pick (chrono cost reported, not gating)"),
               caveat=("DECODE-ONLY (PARTIAL): members / students / in-stream refits in recording order; AND the shipped tta.lam also moves the "
                       "in-stream refit labels (confR, is_tcfg), which no lab read covers (unread)"))
    (OUT / "d656.json").write_text(json.dumps(res, indent=1))
    L = ["# T2-D656 — the order hedge read (W25-ALL6's bw21 C' = tags n; lambda vs the shipped 0.95 under the same order)", "",
         "**CAVEATS:** " + res["caveat"] + ".", "",
         f"**PICK: {('lambda ' + str(pick)) if pick else 'NO hedge file'}** (in_sess OLD sealed: lambda 0.98 {sh[0.98]:+.4f}, lambda 1.0 {sh[1.0]:+.4f}; bar +0.002; "
         f"tied {tied}; chrono OLD sealed (tie-break, reported cost): 0.98 {chs[0.98]:+.4f}, 1.0 {chs[1.0]:+.4f})"
         + ("" if final else " — runs_perm / in_run (reported only) still running"), "",
         "| lambda vs 0.95 | order | strangers dS (n >= 0 per seed) | returning dret | OLD sealed (per seed) |", "|---|---|---|---|---|"]
    for lam in (0.98, 1.0):
        for kind in kinds:
            c = res["cells"][f"{lam}|{kind}"]
            L.append(f"| {lam} | {kind} | {c['dS']:+.4f} ({', '.join(map(str, c['n_ge0']))} / 8) | {c['dret']:+.4f} | {c['sealed']:+.4f} ({', '.join(f'{v:+.4f}' for v in c['per_seed_sealed'])}) |")
    L += ["", "Levels (strangers / returning mean cell-BA):"]
    for lam in LAMS:
        L.append("- lambda " + str(lam) + ": " + "; ".join(f"{k} {res['levels'][f'{lam}|{k}']['strangers']:.4f} / {res['levels'][f'{lam}|{k}']['returning']:.4f}" for k in kinds))
    L += ["", "Per user under the session shuffle (mean of 3 seeds): " + "; ".join(
        f"lambda {lam}: " + ", ".join(f"{u} {v:+.4f}" for u, v in res["cells"][f"{lam}|in_sess"]["per_user"].items()) for lam in (0.98, 1.0)),
        ]
    (OUT / "RESULT_D656.md").write_text("\n".join(L) + "\n")
    print(f"{'### D656 reported cells added;' if final else '=== D656'} PICK {('lambda ' + str(pick)) if pick else 'NONE'}: in_sess OLD sealed 0.98 {sh[0.98]:+.4f} / 1.0 {sh[1.0]:+.4f}; "
          f"tied {tied}; chrono 0.98 {chs[0.98]:+.4f} / 1.0 {chs[1.0]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
