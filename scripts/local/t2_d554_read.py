#!/usr/bin/env python
"""T2-D554 kill-read + stack read (registered 2026-11-17 before any specnet number; written before the outputs land).
The stack reader's head (scripts/local/t2_d540a_stack_read.py --draws 822 --items D539b,D543,D546b,D553) is exec'd unchanged: its
draws 822-824, CNN files (outputs/t2_d553), FM dump members, decoder (causal_sinkhorn 10 / 3 / 0.95) and scoring.
  (1) KILL-READ: specnet ALONE = each fold's pooled specnet probabilities (sofia ni26-t2p-d554-spec_s<d>-*/runs) through its own causal
      Sinkhorn with a cross-fitted start (as each family inside mix()), the ten-stranger typical level averaged over draws 822-824;
      references in the same decode: FBCNet / ShallowNet / IFNet (their routed files at the same draws). KILLED if < 0.70.
  (2) STACK (only if alone >= 0.70): W19 = read(D539b, D543, D546b, D553) vs ARM = W19 + specnet as a fourth family at w 0.10 (the
      other four families x 0.9; primary) and w 0.05 (descriptive). PASS iff ten >= +0.003, >= 2 / 3 draws > 0, no stranger < -0.020,
      returning >= -0.003. Descriptive: phi of specnet's errors with W19's, specnet's accuracy on W19's wrong stranger windows.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d554_read.py
T2-D554b (registered 2026-11-17 before its number): the same read for `specnet_sf` from the local 5090 runs (outputs/t2_d554b/runs) with the
kill line 0.626 (this decode's weakest shipped family, ShallowNet), output t2_d554b_read.json:
    ... python scripts/local/t2_d554_read.py --model specnet_sf --runs outputs/t2_d554b/runs --kill 0.626 --out t2_d554b_read.json
"""
import argparse
import glob
import json
import sys
import time

import numpy as np

T0 = time.time()
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="specnet")
_ap.add_argument("--runs", default="", help="a local runs dir (D554b); default = D554's sofia dirs")
_ap.add_argument("--kill", type=float, default=0.70)
_ap.add_argument("--out", default="t2_d554_read.json")
A554 = _ap.parse_args()
sys.argv = [sys.argv[0], "--draws", "822", "--items", "D539b,D543,D546b,D553"]
SRC = open("scripts/local/t2_d540a_stack_read.py").read()
CUT = "\nBASE = read([])\n"
assert SRC.count(CUT) == 1, "the stack reader's tail moved"
exec(compile(SRC.split(CUT)[0], "t2_d540a_stack_read_head(d554)", "exec"))
assert DRAWS == (822, 823, 824)
SPEC = {}
for d in DRAWS:
    if A554.runs:
        rd = A554.runs if A554.runs.startswith("/") else str(O / A554.runs.split("outputs/neural_interface_challenge/")[-1].removeprefix("outputs/"))
    else:
        dd = glob.glob(str(S / f"ni26-t2p-d554-spec_s{d}-[0-9]*"))
        assert len(dd) == 1, (d, dd)
        rd = f"{dd[0]}/runs"
    for f in range(10):
        SPEC[(f, d)] = one(f"{rd}/lc9_{f}__{A554.model}__ea__n00__r1_spec_e180_mx4__s{d}.npz")


def alone(get):
    """{(f, d): (IX[f], decoded)} for one family's raw parts on IX[f], decoded with a cross-fitted start per draw."""
    P = {(f, d): get(f, d) for f in range(10) for d in DRAWS}
    out = {}
    for d in DRAWS:
        per = {f: P[(f, d)][U[IX[f]] == TRU[f]].mean(0) for f in range(10)}
        for f in range(10):
            v = np.mean([per[g] for g in range(10) if g != f], 0)
            ix = IX[f]
            out[(f, d)] = (ix, causal_sinkhorn(P[(f, d)], (STREAM[ix], ORD[ix]), 10, 3, 0.95, start=v / v.sum()))
    return out


def spec_p(f, d):
    return realign(*load(SPEC[(f, d)]), IX[f])


res = {"draws": list(DRAWS)}
A_spec = alone(spec_p)
res["alone_level"] = {"specnet": level(A_spec)}
res["model"], res["kill_line"] = A554.model, A554.kill
for fam in ("fbc", "sh", "ifn"):
    res["alone_level"][fam] = level(alone(lambda f, d, fam=fam: realign(*load(CNN / RT[fam].format(f=f, s=d)), IX[f])))
print(f"D554 alone (ten-stranger typical level, decoded): {res['alone_level']} ({time.time() - T0:.0f} s)", flush=True)
res["killed"] = bool(res["alone_level"]["specnet"] < A554.kill)
if not res["killed"]:
    REF = read(["D539b", "D543", "D546b", "D553"])
    SPEC_W = 0.0
    _configure0, _parts0 = configure, parts_of

    def configure(on):
        global W
        _configure0(on)
        if SPEC_W > 0:
            W = {k: v * (1.0 - SPEC_W) for k, v in W.items()}
            W["spec"] = SPEC_W

    def parts_of(f, d, parts):
        P = _parts0(f, d, parts)
        if SPEC_W > 0:
            P = dict(P)
            P["spec"] = spec_p(f, d)
        return P
    for w in (0.10, 0.05):
        SPEC_W = w
        ARM = read(["D539b", "D543", "D546b", "D553"])
        SPEC_W = 0.0
        sc = judge(ARM, REF)
        sc["pass"] = bool(sc["ten"] >= 0.003 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
        res[f"W19 + specnet @{w} vs W19" + (" (THE DECISION)" if w == 0.10 else " (descriptive)")] = sc
        print(f"D554 stack w {w}: {sc}", flush=True)
    wr = {"n_stack_wrong": 0, "spec_right_on_stack_wrong": 0, "both_wrong": 0, "stack_wrong": 0, "spec_wrong": 0, "n": 0}
    for (f, d), (ix, mx) in REF.items():
        m = U[ix] == TRU[f]
        y = Y[ix][m]
        sw = mx.argmax(1)[m] != y
        pw = A_spec[(f, d)][1].argmax(1)[m] != y
        wr["n"] += int(m.sum()); wr["stack_wrong"] += int(sw.sum()); wr["spec_wrong"] += int(pw.sum()); wr["both_wrong"] += int((sw & pw).sum())
        wr["spec_right_on_stack_wrong"] += int((sw & ~pw).sum())
    p1, p2, p12 = wr["stack_wrong"] / wr["n"], wr["spec_wrong"] / wr["n"], wr["both_wrong"] / wr["n"]
    res["error_overlap"] = dict(wr, phi=round((p12 - p1 * p2) / np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)), 4),
                                acc_on_stack_wrong=round(wr["spec_right_on_stack_wrong"] / max(wr["stack_wrong"], 1), 4))
    print(f"D554 error overlap with W19 (strangers): {res['error_overlap']}", flush=True)
res["checks"] = {k: v for k, v in CHK.items() if not isinstance(v, list)}
verdict = f"KILLED (alone < {A554.kill})" if res["killed"] else ("PASS" if res["W19 + specnet @0.1 vs W19 (THE DECISION)"]["pass"] else "FAIL (stack)")
res["verdict"] = verdict
print(f"D554 verdict: {verdict} ({time.time() - T0:.0f} s)", flush=True)
json.dump(res, open(O / A554.out, "w"), indent=1, default=str)
