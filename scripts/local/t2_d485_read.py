#!/usr/bin/env python
"""T2-D485 read (written 2026-11-14 before any number): the five nearest CNN misses whose mechanism should grow with users, at K = 9 and K = 5.
Files: outputs/sofia_s1/ni26-t2p-d485-k9-5090/runs (scripts/local/t2_d485_k9_local.sh; t2p_bench lc5 / lc9 x 10 cyclic folds, pooled, last,
seeds 400-404). Per (K, fold, seed) the T2-D410 mix: FBCNet 0.571 + ShallowNet 0.429, causal SK lambda (10 iters, n0 3, 0.95) per stream over
the fold's test windows (no start vector: D410's form), cell-BA per cell; NEW = cells of users the fold did not train (D410's grouping).
  ctl    : FBCNet rc75 + ShallowNet vr                 sw2    : ShallowNet x2 width (D325) in the ShallowNet slot
  rc75m1 : FBCNet Bures-midpoint re-colouring (D237)   us5    : user re-styling in both slots (D365)
  fms5   : FBCNet same-class MixStyle (D297)           eegnex : 0.9 x ctl's mix + 0.1 EEGNeX (D346's eegnex10 form)
Per seed: typical NEW(K) = the mean over the nine non-P04 users of each user's mean cell-BA over the folds where it is untrained; gain = lever - ctl
(paired by fold and seed). BAR (each lever at K = 9): typical NEW gain >= +0.003 with >= 4 / 5 seeds > 0, no typical user below -0.010 (one
untrained fold per user at K = 9). Reported beside: the gain at K = 5, growth = gain(9) - gain(5), RETURNING (P04 / P10 sessions 2-3 when trained),
each family alone. Five levers are read: the multiplicity goes with any pass.
Code check: D485_FAKE=1 -> every lever's files are the controls' (EEGNeX = ctl's mix) -> every gain = 0 exactly.
    OMP_NUM_THREADS=8 mem-guard python scripts/local/t2_d485_read.py   ->  outputs/t2_d485/{cells.csv, read.txt}
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
RUNS = O / "sofia_s1/ni26-t2p-d485-k9-5090/runs"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
FAKE = bool(os.environ.get("D485_FAKE"))
OUT = O / "t2_d485"
OUT.mkdir(exist_ok=True)
SEEDS, KS = range(400, 405), (5, 9)
F = {"fbc": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", "sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr",
     "sw2": "shallow_w2__ea__n00__r1_pulx_vp2_e180_vr", "rc75m1": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75m1",
     "fus5": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_us5", "sus5": "shallow__ea__n00__r1_pulx_vp2_e180_vr_us5",
     "fms5": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_ms5", "eegnex": "eegnex__ea__n00__r1_pulx_vp2_e180_vr"}
FAKEMAP = {"sw2": "sh", "rc75m1": "fbc", "fus5": "fbc", "sus5": "sh", "fms5": "fbc"}   # FAKE eegnex = ctl's mix (below)
WF, WS = 0.314 / 0.55, 0.236 / 0.55
ARMS = {"ctl": [("fbc", WF), ("sh", WS)], "sw2": [("fbc", WF), ("sw2", WS)], "rc75m1": [("rc75m1", WF), ("sh", WS)],
        "us5": [("fus5", WF), ("sus5", WS)], "fms5": [("fms5", WF), ("sh", WS)],
        "eegnex": [("fbc", 0.9 * WF), ("sh", 0.9 * WS), ("eegnex", 0.1)],
        # each family alone (descriptive)
        "a_fbc": [("fbc", 1.0)], "a_sh": [("sh", 1.0)], "a_sw2": [("sw2", 1.0)], "a_rc75m1": [("rc75m1", 1.0)], "a_fus5": [("fus5", 1.0)],
        "a_sus5": [("sus5", 1.0)], "a_fms5": [("fms5", 1.0)], "a_eegnex": [("eegnex", 1.0)]}
LEVERS = {"sw2": "D325 ShallowNet x2 width", "rc75m1": "D237 Bures-midpoint re-colouring", "us5": "D365 user re-styling (both)",
          "fms5": "D297 FBCNet MixStyle", "eegnex": "D346 EEGNeX third family (0.1)"}
ALONE = {"a_sw2": "a_sh", "a_rc75m1": "a_fbc", "a_fus5": "a_fbc", "a_sus5": "a_sh", "a_fms5": "a_fbc", "a_eegnex": "a_sh"}
d = pd.read_csv(EXP / "index.csv", dtype={"subject": str, "session": str})
strm, ordr, y, cell = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy(), d.cell.to_numpy()
sub, ses = d.subject.str[-3:].to_numpy(), d.session.astype(str).to_numpy()


def bal(yy, p):
    return float(np.mean([np.mean(p[yy == c] == c) for c in np.unique(yy)]))


def load(fold, key, s):
    if FAKE and key == "eegnex":                                # 0.9 x ctl + 0.1 x ctl = ctl
        (ix, pf), (_, ps) = load(fold, "fbc", s), load(fold, "sh", s)
        return ix, WF * pf + WS * ps
    k = FAKEMAP.get(key, key) if FAKE else key
    fs = glob.glob(f"{RUNS}/{fold}__{F[k]}__s{s}.npz")
    assert len(fs) == 1, (fold, key, s, fs)
    z = np.load(fs[0])
    return z["test_idx"], z["prob_last"].astype(np.float64)


rows = []
for K in KS:
    for i in range(10):
        fold = f"lc{K}_{i}"
        for s in SEEDS:
            P = {k: load(fold, k, s) for k in F}
            ix = P["fbc"][0]
            assert all(np.array_equal(ix, v[0]) for v in P.values())
            kk = (strm[ix], ordr[ix])
            for arm, mem in ARMS.items():
                assert abs(sum(w for _, w in mem) - 1) < 1e-9
                q = causal_sinkhorn(sum(w * P[k][1] for k, w in mem), kk, 10, 3, 0.95).argmax(1)
                for c in np.unique(cell[ix]):
                    m = cell[ix] == c
                    u, se = sub[ix][m][0], ses[ix][m][0]
                    trained_u = (u in ("P04", "P10")) and se in ("2", "3") and not (sub[ix] == u).sum() == (sub == u).sum()
                    rows.append((K, i, s, arm, u, se, c, "returning" if trained_u else "new", bal(y[ix][m], q[m])))
    print(f"K{K} done", flush=True)
R = pd.DataFrame(rows, columns=["K", "fold", "seed", "arm", "user", "session", "cell", "group", "ba"])
R.to_csv(OUT / ("cells_fake.csv" if FAKE else "cells.csv"), index=False)


def per_user(arm, K, group="new"):
    """(seed x user) table: each user's mean cell-BA over the folds where it is in `group`."""
    t = R[(R.arm == arm) & (R.K == K) & (R.group == group)]
    return t.groupby(["seed", "user", "fold"]).ba.mean().groupby(["seed", "user"]).mean().unstack("user")


def gain(a, b, K, group="new"):
    A, B = per_user(a, K, group), per_user(b, K, group)
    D = A - B
    typ = D.drop(columns="P04", errors="ignore").mean(axis=1) if group == "new" else D.mean(axis=1)
    return typ, D.mean(axis=0), A.drop(columns="P04", errors="ignore").mean(axis=1).mean() if group == "new" else A.mean(axis=1).mean()


lines = []


def say(t):
    print(t, flush=True)
    lines.append(t)


if FAKE:
    mx = max(float(np.abs(gain(a, "ctl", K)[0]).max()) for a in LEVERS for K in KS)
    mxa = max(float(np.abs(gain(a, b, 9)[0]).max()) for a, b in ALONE.items() if a != "a_eegnex")
    say(f"FAKE code check: max |gain| over levers and K {mx:.2e}, alone {mxa:.2e} -> {'OK' if mx == 0 and mxa == 0 else 'MISMATCH'}")
    sys.exit(0)
c9, c5 = gain("ctl", "ctl", 9)[2], gain("ctl", "ctl", 5)[2]
say(f"D485 ctl typical NEW level: K9 {c9:.4f}  K5 {c5:.4f}  (T2-D410, seed 400 on the H200: K9 0.690, K5 0.647)")
for a, name in LEVERS.items():
    t9, u9, _ = gain(a, "ctl", 9)
    t5, _, _ = gain(a, "ctl", 5)
    tr, _, _ = gain(a, "ctl", 9, "returning")
    ut = u9.drop("P04", errors="ignore")
    ok = t9.mean() >= 0.003 and (t9 > 0).sum() >= 4 and ut.min() >= -0.010
    say(f"D485 BAR {a} ({name}) at K9: typical NEW {t9.mean():+.4f} ({int((t9 > 0).sum())}/5 seeds) per user "
        + " ".join(f"{u} {v:+.4f}" for u, v in u9.items()) + f" -> {'PASS' if ok else 'no'}")
    say(f"     K5 {t5.mean():+.4f} ({int((t5 > 0).sum())}/5); growth K9 - K5 {t9.mean() - t5.mean():+.4f} "
        f"(per seed {' '.join(f'{v:+.4f}' for v in (t9 - t5).to_numpy())}); RETURNING at K9 {tr.mean():+.4f}")
for a, b in ALONE.items():
    t9, _, _ = gain(a, b, 9)
    t5, _, _ = gain(a, b, 5)
    say(f"D485 descriptive alone {a[2:]} vs {b[2:]}: K9 {t9.mean():+.4f} ({int((t9 > 0).sum())}/5)  K5 {t5.mean():+.4f}  growth {t9.mean() - t5.mean():+.4f}")
(OUT / "read.txt").write_text("\n".join(lines) + "\n")
