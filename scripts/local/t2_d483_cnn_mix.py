#!/usr/bin/env python
"""T2-D483 (the manager's advance 1, 2026-11-14; registered before any number; zero GPU): CAR as a second INPUT VIEW inside a CNN family,
read as a MIX (D475d's lesson for the ST-L: the CAR swap lost -0.0010, the CAR mix passed +0.0064 / +0.0055). D475 arm B read FBCNet-CAR
only as a swap (-0.0025, closed); it was never read mixed. Stored predictions only, under W13's decode, on T2-D475g's CNN15 shape:
  each CNN family = the mean of 15 routed nets from its 24-seed pool, plain seed j of draw i = pool[(2i + j) mod 24] (FBCNet D419 324-335 +
  D430 348-359; ShallowNet D429 + D424; IFNet D429 + M7's stored); FM = five plain ST-L (D455, seed index i + k); M7's family weights;
  per-family causal SK lambda (10 iters, n0 3, 0.95), each family's start vector cross-fitted from the OTHER split (the reader's rule); 10 draws,
  kit + seen1, test + val; typical = non-P04.
CAR pool for FBCNet = D475 arm B's 12 routed CAR nets (D419's recipe + __car, seeds 324-335). In draw i the CAR nets are taken cyclically from
324 + (i mod 12), those whose plain twin (same seed) is NOT among the draw's plain nets first (a build trains distinct seeds).
  ref : FBCNet 15 plain                     mK  : FBCNet K plain (the first K of ref's, nested) + (15 - K) CAR, K = 12, 10, 8
  descriptive: p12 = FBCNet 12 plain only (does dropping 3 plain nets cost?); the same arms stacked on W14's FM shape (five plain + three CAR
  ST-L, seed index i + 5 .. i + 7) = refW / mKW.
BAR (each mK vs ref): typical >= +0.003 with >= 8 / 10, no typical user below -0.005 (zero-runtime form). GATE on a pass: under D476b's
within-stream shuffle order the form loses at most 0.003 more than ref loses. Pick = the largest typical gain among passing forms (three forms
read; the multiplicity goes with it). A pass -> CAR nets for ShallowNet and IFNet on their controls' seeds (5090, weights only), then all three
families mixed; a template change (per-net CAR inside a vmapped CNN family) = the user's line before any package.
Code check: D483_FAKE=1 -> the "CAR" nets are the plain pool's next nets, so every mK must equal ref exactly.
    OMP_NUM_THREADS=8 mem-guard python scripts/local/t2_d483_cnn_mix.py   ->  outputs/t2_d483/{cells.csv, read.txt}
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
FAKE = bool(os.environ.get("D483_FAKE"))
OUT = O / "t2_d483"
OUT.mkdir(exist_ok=True)
idx = pd.read_csv(EXP / "index.csv")
C = json.load(open(O / "t2_d442/spec.json"))["arms"]["M7f"]
FAMW = {"fbc": C["fbc_tv8c"], "sh": C["sh_tv8c"], "ifn": C["ifn"], "fm": sum(v for k, v in C.items() if k.startswith("n_"))}
assert abs(sum(FAMW.values()) - 1) < 1e-9
N, NNET = 10, 15
POOL = {}
for fam, srcs in (("fbc", [("t2_d419/spec.json", "fbc_e180"), ("t2_d430/spec_member_fbc.json", "fbc_ctl")]),
                  ("sh", [("t2_d429/spec.json", "sh_ctl"), ("t2_d424/spec_member.json", "sh_ctl")]),
                  ("ifn", [("t2_d429/spec.json", "ifn_ctl"), ("t2_d408/spec_m7.json", "ifn")])):
    POOL[fam] = []
    for f, k in srcs:
        mm = json.load(open(O / f))["members"][k]
        POOL[fam] += [(mm, mm["offset"] + q) for q in range(12)]
CARPOOL = {"fbc": [({p: str(O / f"t2_d475/route_fbc_car/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__car__s{{s}}.npz")
                     for p in ("kit", "seen1")}, 324 + q) for q in range(12)]}


def stl(pat):
    out = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{S}/{pat.format(pr=pr)}-[0-9]*")
        assert len(ds) == 1, (pat, pr, ds)
        f = glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s472*.npz")
        assert len(f) == 1, f
        out[pr] = f[0].replace("__s472.npz", "__s{s}.npz")
    return out


PP, PC = stl("ni26-t2p-d455-ctl-{pr}"), stl("ni26-t2p-d475-stlcar-{pr}")


def plain(fam, i, proto, n):
    return [mm[proto].format(s=sd) for mm, sd in (POOL[fam][(2 * i + j) % 24] for j in range(n))]


def car(fam, i, proto, k):
    """15 - k CAR nets for draw i, twins of the draw's k plain nets last."""
    if FAKE:
        return [mm[proto].format(s=sd) for mm, sd in (POOL[fam][(2 * i + k + m) % 24] for m in range(NNET - k))]
    used = {sd for _, sd in (POOL[fam][(2 * i + j) % 24] for j in range(k))}
    cyc = [CARPOOL[fam][(i + m) % 12] for m in range(12)]
    cyc = [c for c in cyc if c[1] not in used] + [c for c in cyc if c[1] in used]
    return [mm[proto].format(s=sd) for mm, sd in cyc[:NNET - k]]


def arm_files(proto, i, fbc_k, w14):
    fams = {f: plain(f, i, proto, NNET) for f in ("sh", "ifn")}
    fams["fbc"] = plain("fbc", i, proto, NNET) if fbc_k is None else plain("fbc", i, proto, fbc_k) + car("fbc", i, proto, fbc_k)
    fams["fm"] = [PP[proto].format(s=472 + (i + k) % 14) for k in range(5)] + ([PC[proto].format(s=472 + (i + k) % 14) for k in (5, 6, 7)] if w14 else [])
    return fams


ARMS = {}
for w14, sfx in ((False, ""), (True, "W")):
    ARMS["ref" + sfx] = (None, w14)
    for k in (12, 10, 8):
        ARMS[f"m{k}{sfx}"] = (k, w14)
ARMS["p12"] = ("p12", False)


def files_for(a, proto, i):
    k, w14 = ARMS[a]
    if k == "p12":
        f = arm_files(proto, i, None, False)
        f["fbc"] = plain("fbc", i, proto, 12)
        return f
    return arm_files(proto, i, k, w14)


MIX, cache = {}, {}
for proto in ("kit", "seen1"):
    for i in range(N):
        for a in ARMS:
            fl = files_for(a, proto, i)
            for split in ("test", "val"):
                ix, mx = None, {}
                for fam, fs in fl.items():
                    acc = 0.0
                    for f in fs:
                        if f not in cache:
                            z = np.load(f)
                            cache[f] = {"test": (z["test_idx"], z["prob_last"].astype(np.float64)), "val": (z["val_idx"], z["prob_val_last"].astype(np.float64))}
                        ix_, p = cache[f][split]
                        assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                        ix, acc = ix_, acc + p / len(fs)
                    mx[fam] = acc / acc.sum(1, keepdims=True)
                MIX[(a, proto, i, split)] = (ix, mx)
        cache.clear()
    print(f"{proto} loaded", flush=True)
VEC = {}
for a in ARMS:
    for sp in ("test", "val"):
        other = [v for k, v in MIX.items() if k[0] == a and k[3] != sp]
        VEC[(a, sp)] = {f: np.concatenate([o[1][f] for o in other]).mean(0) for f in FAMW}
ORD_SHUF = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order (= D481's)
Y, CELL, SUBJ, STREAM = idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy()


def cells(arms, shuf=False):
    order = ORD_SHUF if shuf else idx.order.to_numpy()
    out = []
    for (a, proto, i, split), (ix, mx) in MIX.items():
        if a not in arms:
            continue
        kk = (STREAM[ix], order[ix])
        P = sum(FAMW[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
        y, cl, sb = Y[ix], CELL[ix], SUBJ[ix]
        for c in np.unique(cl):
            mk = cl == c
            out.append((a, proto, split, i, c, sb[mk][0], np.mean([np.mean(P[mk][y[mk] == q] == q) for q in np.unique(y[mk])])))
    return pd.DataFrame(out, columns=["arm", "proto", "split", "i", "cell", "subj", "ba"])


KEY = ["proto", "split", "i", "cell"]


def cmp(R, a, b, bar=0.003):
    m = R[R.arm == a].merge(R[R.arm == b][KEY + ["ba"]], on=KEY, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    ok = d.mean() >= bar and (d > 0).sum() >= 8 and u.drop("P04", errors="ignore").min() >= -0.005
    return ok, float(d.mean()), (f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{N}) level {t.ba.mean():.4f} vs {t.ba_b.mean():.4f}  "
                                 + " ".join(f"{s} {v:+.4f}" for s, v in u.items()))


R = cells(list(ARMS))
R.to_csv(OUT / ("cells_fake.csv" if FAKE else "cells.csv"), index=False)
lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


if FAKE:
    for k in (12, 10, 8):
        for sfx in ("", "W"):
            d = np.abs(R[R.arm == f"m{k}{sfx}"].ba.to_numpy() - R[R.arm == f"ref{sfx}"].ba.to_numpy()).max()
            say(f"FAKE code check m{k}{sfx} vs ref{sfx}: max |dBA| {d:.2e} -> {'OK' if d < 1e-12 else 'MISMATCH'}")
    sys.exit(0)
passing = {}
for k in (12, 10, 8):
    ok, g, s = cmp(R, f"m{k}", "ref")
    say(f"D483 BAR m{k} (FBCNet {k} plain + {NNET - k} CAR) vs ref (15 plain): {s} -> {'PASS' if ok else 'no'}")
    if ok:
        passing[f"m{k}"] = g
say("D483 descriptive p12 (12 plain only) vs ref: " + cmp(R, "p12", "ref")[2])
say("D483 descriptive m12 vs p12 (three CAR nets vs none, 12 plain each): " + cmp(R, "m12", "p12")[2])
for k in (10, 8):
    say(f"D483 descriptive m{k} vs m12: " + cmp(R, f"m{k}", "m12")[2])
for k in (12, 10, 8):
    say(f"D483 descriptive on W14's FM (5 + 3 CAR ST-L) m{k}W vs refW: " + cmp(R, f"m{k}W", "refW")[2])
say("D483 descriptive refW vs ref (W14's ST-L CAR on CNN15, = D475g a): " + cmp(R, "refW", "ref")[2])
gate_arms = ["ref"] + [f"m{k}" for k in (12, 10, 8)]
Rs = cells(gate_arms, shuf=True)
lose = {a: R[(R.arm == a) & (R.subj != "P04")].ba.mean() - Rs[(Rs.arm == a) & (Rs.subj != "P04")].ba.mean() for a in gate_arms}
for a in gate_arms[1:]:
    gok = lose[a] - lose["ref"] <= 0.003
    say(f"D483 shuffle gate {a}: loses {lose[a]:+.4f} under the within-stream shuffle, ref {lose['ref']:+.4f} -> {'ok' if gok else 'FAILS'}"
        + ("" if a in passing else " (reported; binding only on a pass)"))
    if a in passing and not gok:
        passing.pop(a)
pick = max(passing, key=passing.get) if passing else None
say(f"D483 PICK: {pick or 'none (no form passes the bar and the gate)'}  [three forms read]")
(OUT / "read.txt").write_text("\n".join(lines) + "\n")
