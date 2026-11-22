#!/usr/bin/env python
"""T2-D573 (P3, the honest forecast for the owner; registered 2026-11-18 before any number; DESCRIPTIVE, selects nothing).

Saved lab mixes (t2_lab.py mix: ix_<f>_<d> / mx_<f>_<d>) of several forms on ONE draw set are scored per cell (subject | session | interface,
the official cell) with the reader's cells(): BA per cell, argmax decisions.
  * EQUAL-USER (the lab's): strangers = mean over the 10 held-out users of (mean over draws of the user's mean cell BA); returning = the same over
    P04 / P10's sessions 2-3; sealed mix = 0.57 strangers + 0.43 returning.
  * CELL-WEIGHTED (the official target averages cells): strangers = mean over ALL stranger cells (pooled over users) per draw, then over draws;
    returning the same; mixed 0.57 / 0.43 (the sealed cell shares of the two groups, as in the reader).
  * Paired deltas vs the REF form, with a PARTICIPANT bootstrap (B reps): stranger users resampled with replacement, and within each drawn user
    its sessions resampled with replacement (cells = that session's interfaces); the 2 returning users likewise (their sessions 2-3). The same
    resample is applied to every form (paired). 5 / 50 / 95 % percentiles.
  * Forecast: W18 b1's sealed 0.835 + k x (form - W18) for k in (0.5, 1.0), per weighting, with the bootstrap percentiles.
    mem-guard env OMP_NUM_THREADS=2 python scripts/local/t2_p3_forecast.py --set C --ref W18 --form W18=<npz> --form W20D557=<npz> ...
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=("A", "B", "C", "D"))
ap.add_argument("--ref", required=True)
ap.add_argument("--form", action="append", required=True, help="name=path of a saved lab mix")
ap.add_argument("--reps", type=int, default=2000)
ap.add_argument("--seed", type=int, default=573)
ap.add_argument("--out", required=True)
a = ap.parse_args()
FORMS = dict(kv.split("=", 1) for kv in a.form)
assert a.ref in FORMS, (a.ref, list(FORMS))
D0 = {"A": 825, "B": 828, "C": 831, "D": 834}[a.set]
ITEMS = {825: "D539b,D543,D546b,D553,D555", 828: "D539b,D543,D546b,D553,D557", 831: "D539b,D543,D546b,D553,D555", 834: "D539b,D543,D546b,D553,D557"}[D0]
for k in [k for k in os.environ if k.startswith("NI26_")]:
    del os.environ[k]
sys.argv = [sys.argv[0], "--draws", str(D0), "--items", ITEMS]
src = open("scripts/local/t2_d540a_stack_read.py").read()
g = {"__name__": "p3_head", "__file__": os.path.abspath("scripts/local/t2_d540a_stack_read.py")}
exec(compile(src.split("\nBASE = read([])\n")[0], "t2_d540a_stack_read_head(p3)", "exec"), g)
U, SES, CELL, TRU, cells, DRAWS = g["U"], g["SES"], g["CELL"], g["TRU"], g["cells"], g["DRAWS"]
# F2 (Track3): read([]) = BASE must be W18 b1's uploaded form in its numbers (family weights, decode, DC weight); the FM lines are printed
import zipfile  # noqa: E402
Z18 = os.path.expanduser("~/workspace/outputs/neural_interface_challenge/lead/UPLOAD_TODAY_2026-11-17/1_T2__t2p_fin1_w18.zip")
with zipfile.ZipFile(Z18) as zf:
    C18 = json.loads(zf.read("config.json"))
g["configure"]([])
W_B = dict(g["W"])
w18 = {"fbc": 1.0 - C18["fm_weight"] - C18["cnn2"]["weight"] - C18["cnn3"]["weight"], "sh": C18["cnn2"]["weight"], "ifn": C18["cnn3"]["weight"], "fm": C18["fm_weight"]}
assert set(W_B) == set(w18) and all(abs(W_B[k] - w18[k]) < 5e-5 for k in w18), (W_B, w18)   # the reader rounds the shares to 4 decimals
print(f"[p3] F2: family weights reader {W_B} vs W18 b1 {dict((k, round(v, 7)) for k, v in w18.items())} (max |diff| {max(abs(W_B[k] - w18[k]) for k in w18):.1e})", flush=True)
assert abs(g["W_DC"] - C18["dcfam"]["weight"]) < 1e-12 and (C18["tta"]["iters"], C18["tta"]["n0"], C18["tta"]["lam"]) == (10, 3, 0.95), (g["W_DC"], C18["tta"])
assert all(k in C18 for k in ("instream", "side", "fm_side")) and not any(k in C18 for k in ("fm_proto", "fm_side_refit")), sorted(C18)
print(f"[p3] F2: BASE = W18 b1's numbers (weights {W_B}, decode 10 / 3 / 0.95, DC {g['W_DC']}; W18 b1 carries instream + side + fm_side, no fm_proto / "
      f"fm_side_refit); BASE FM slots {g['base_swaps']()} vs W18 b1's members "
      f"{[(m.get('seed'), m.get('mixup'), m.get('tilt'), m.get('align')) for m in C18['fm_members']]}", flush=True)


def load(p):
    z = np.load(p)
    return {(int(k.split("_")[1]), int(k.split("_")[2])): (z[k], z["mx_" + k[3:]]) for k in z.files if k.startswith("ix_")}


M = {n: load(p) for n, p in FORMS.items()}
keys = sorted(M[a.ref])
assert all(sorted(M[n]) == keys and all(np.array_equal(M[n][k][0], M[a.ref][k][0]) for k in keys) for n in M), "the forms' rows differ"
# per form: rec[(group, user, session, cell, draw)] = BA
REC = {}
for n, mm in M.items():
    r = {}
    for (f, d), (ix, mx) in mm.items():
        kind = np.where(U[ix] == TRU[f], "s", np.where(np.isin(SES[ix], ("2", "3")), "r", "o"))
        for kd in ("s", "r"):
            m = kind == kd
            for c, v in cells(mx.argmax(1)[m], ix[m]).items():
                u, ses, _ = str(c).split("|")
                r.setdefault((kd, u.split("/")[-1], ses, str(c), d), []).append(v)   # F1 (Track3): returning cells recur in 9 folds
    REC[n] = {k: float(np.mean(v)) for k, v in r.items()}                            # averaged over folds within (user, session, cell, draw)
K0 = sorted(REC[a.ref])
assert all(sorted(REC[n]) == K0 for n in REC), "cell sets differ between forms"
users = {kd: sorted({k[1] for k in K0 if k[0] == kd}) for kd in ("s", "r")}
sess = {(kd, u): sorted({k[2] for k in K0 if k[0] == kd and k[1] == u}) for kd in ("s", "r") for u in users[kd]}
print(f"[p3] set {a.set}: strangers {users['s']} ({sum(1 for k in K0 if k[0] == 's') // len(DRAWS)} cells / draw), returning {users['r']} "
      f"({sum(1 for k in K0 if k[0] == 'r') // len(DRAWS)} cells / draw); sessions per user {[len(sess[('s', u)]) for u in users['s']]}", flush=True)
# arrays: for each form, value[(kd, u, ses)] = list over draws of the cells' BAs in that session
V = {n: {} for n in REC}
for n, r in REC.items():
    for (kd, u, ses, c, d), v in r.items():
        V[n].setdefault((kd, u, ses), {}).setdefault(d, []).append(v)


def agg(n, pick):
    """pick = {kd: [(u, [sessions...]), ...]} (a resample) -> (equal-user strangers, returning, cell-weighted strangers, returning)."""
    out = {}
    for kd in ("s", "r"):
        eu, cw = [], []
        for d in DRAWS:
            per_user, allc = [], []
            for u, ss in pick[kd]:
                cs = [v for s_ in ss for v in V[n][(kd, u, s_)][d]]
                per_user.append(np.mean(cs))
                allc += cs
            eu.append(np.mean(per_user))
            cw.append(np.mean(allc))
        out[kd] = (float(np.mean(eu)), float(np.mean(cw)))
    return dict(eu_s=out["s"][0], eu_r=out["r"][0], cw_s=out["s"][1], cw_r=out["r"][1],
                eu_mix=0.57 * out["s"][0] + 0.43 * out["r"][0], cw_mix=0.57 * out["s"][1] + 0.43 * out["r"][1])


full = {kd: [(u, sess[(kd, u)]) for u in users[kd]] for kd in ("s", "r")}
LEV = {n: agg(n, full) for n in V}
rng = np.random.default_rng(a.seed)
BS = {n: [] for n in V}
for _ in range(a.reps):
    pick = {}
    for kd in ("s", "r"):
        us = rng.choice(users[kd], size=len(users[kd]), replace=True)
        pick[kd] = [(u, list(rng.choice(sess[(kd, u)], size=len(sess[(kd, u)]), replace=True))) for u in us]
    for n in V:
        BS[n].append(agg(n, pick))
# F1 self-check (Track3): the equal-user point deltas == the reader's judge() ten / returning (unrounded), every form vs REF
g["round"] = lambda x, *k: x
for n in V:
    if n != a.ref:
        jd = g["score"](M[n], M[a.ref])
        e1, e2 = LEV[n]["eu_s"] - LEV[a.ref]["eu_s"], LEV[n]["eu_r"] - LEV[a.ref]["eu_r"]
        assert abs(e1 - jd["ten"]) < 1e-9 and abs(e2 - jd["returning"]) < 1e-9, (n, e1, jd["ten"], e2, jd["returning"])
print("[p3] F1 self-check: equal-user deltas == the reader's score() ten / returning to 1e-9 for every form", flush=True)
res = {"set": a.set, "ref": a.ref, "forms": FORMS, "reps": a.reps, "seed": a.seed, "levels": LEV, "deltas": {}, "forecast_on_w18_0835": {}}
pct = lambda x: [round(float(np.percentile(x, q)), 4) for q in (5, 50, 95)]
for n in V:
    if n == a.ref:
        continue
    dd = {}
    for key in ("eu_s", "eu_r", "eu_mix", "cw_s", "cw_r", "cw_mix"):
        dlt = np.array([b[key] for b in BS[n]]) - np.array([b[key] for b in BS[a.ref]])
        dd[key] = dict(point=round(LEV[n][key] - LEV[a.ref][key], 4), p5_p50_p95=pct(dlt))
    res["deltas"][n] = dd
    if a.ref == "W18":
        fc = {}
        for k_ in (0.5, 1.0):
            for key in ("eu_mix", "cw_mix"):
                dlt = np.array([b[key] for b in BS[n]]) - np.array([b[key] for b in BS[a.ref]])
                fc[f"k{k_}_{key}"] = dict(point=round(0.835 + k_ * dd[key]["point"], 4), p5_p50_p95=[round(0.835 + k_ * x, 4) for x in pct(dlt)])
        res["forecast_on_w18_0835"][n] = fc
for n, l in LEV.items():
    print(f"[p3] {n:>14}: equal-user strangers {l['eu_s']:.4f} returning {l['eu_r']:.4f} mix {l['eu_mix']:.4f} | cell-weighted strangers "
          f"{l['cw_s']:.4f} returning {l['cw_r']:.4f} mix {l['cw_mix']:.4f}", flush=True)
for n, dd in res["deltas"].items():
    print(f"[p3] {n} - {a.ref}: " + "; ".join(f"{k} {v['point']:+.4f} {v['p5_p50_p95']}" for k, v in dd.items()), flush=True)
for n, fc in res["forecast_on_w18_0835"].items():
    print(f"[p3] forecast {n} (0.835 + k x delta): " + "; ".join(f"{k} {v['point']:.4f} {v['p5_p50_p95']}" for k, v in fc.items()), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print(f"[p3] saved {a.out}", flush=True)
