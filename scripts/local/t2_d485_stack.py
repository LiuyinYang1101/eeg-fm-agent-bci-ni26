#!/usr/bin/env python
"""T2-D485 stack read (registered 2026-11-14 10:50 before its numbers; zero GPU): EEGNeX, the one lever that passed at K = 9, as a FOURTH
CNN family inside C under W13's decode, on T2-D475g's CNN15 shape (as T2-D483):
  each CNN family = the mean of 15 routed nets from its 24-seed pool, seed j of draw i = pool[(2i + j) mod 24] (FBCNet D419 + D430,
  ShallowNet D429 + D424, IFNet D429 + M7's stored, EEGNeX D346 156-167 + D346r 168-179, tv8c copies routed); FM = five plain ST-L (D455,
  seed index i + k) or W14's 5 + 3 (CAR at i + 5 .. i + 7); per-family causal SK lambda (10 / 3 / 0.95), start vectors cross-fitted from the
  OTHER split; 10 draws, kit + seen1, test + val; typical = non-P04.
  ref / refW : M7's weights (FBCNet 0.3143, ShallowNet 0.2357, IFNet 0.10, FM 0.35)
  nex / nexW : EEGNeX at 0.10 taken from the CNN share (the three CNN families x 0.55 / 0.65; D346's eegnex10 form)
  descriptive: nexW at 0.05 and 0.15
HARM CHECK (binding, nexW vs refW): typical >= -0.002, no typical user below -0.005, shuffle gate (loses <= 0.003 more than refW).
Code check: D485S_FAKE=1 -> EEGNeX's weight is 0 inside nex / nexW (its family still loaded and decoded), which must then equal ref / refW exactly.
    OMP_NUM_THREADS=8 mem-guard python scripts/local/t2_d485_stack.py   ->  outputs/t2_d485/{stack_cells.csv, stack.txt}
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
FAKE = bool(os.environ.get("D485S_FAKE"))
OUT = O / "t2_d485"
OUT.mkdir(exist_ok=True)
idx = pd.read_csv(EXP / "index.csv")
C = json.load(open(O / "t2_d442/spec.json"))["arms"]["M7f"]
M7W = {"fbc": C["fbc_tv8c"], "sh": C["sh_tv8c"], "ifn": C["ifn"], "fm": sum(v for k, v in C.items() if k.startswith("n_"))}
assert abs(sum(M7W.values()) - 1) < 1e-9
N, NNET = 10, 15
POOL = {}
for fam, srcs in (("fbc", [("t2_d419/spec.json", "fbc_e180"), ("t2_d430/spec_member_fbc.json", "fbc_ctl")]),
                  ("sh", [("t2_d429/spec.json", "sh_ctl"), ("t2_d424/spec_member.json", "sh_ctl")]),
                  ("ifn", [("t2_d429/spec.json", "ifn_ctl"), ("t2_d408/spec_m7.json", "ifn")]),
                  ("nex", [("t2_d346/spec.json", "eegnex"), ("t2_d346r/spec.json", "eegnex")])):
    POOL[fam] = []
    for f, k in srcs:
        mm = json.load(open(O / f))["members"][k]
        POOL[fam] += [(mm, mm["offset"] + q) for q in range(12)]


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


def weights(w_nex, has_nex):
    if FAKE:
        w_nex = 0.0
    cnn = 1 - M7W["fm"]
    sc = (cnn - w_nex) / cnn
    return {"fbc": M7W["fbc"] * sc, "sh": M7W["sh"] * sc, "ifn": M7W["ifn"] * sc, "fm": M7W["fm"], **({"nex": w_nex} if has_nex else {})}


ARMS = {"ref": (0.0, False), "nex": (0.10, False), "refW": (0.0, True), "nexW": (0.10, True), "nexW05": (0.05, True), "nexW15": (0.15, True)}


def files(proto, i, w14, with_nex):
    fams = {f: [mm[proto].format(s=sd) for mm, sd in (POOL[f][(2 * i + j) % 24] for j in range(NNET))] for f in ("fbc", "sh", "ifn")}
    if with_nex:
        fams["nex"] = [mm[proto].format(s=sd) for mm, sd in (POOL["nex"][(2 * i + j) % 24] for j in range(NNET))]
    fams["fm"] = [PP[proto].format(s=472 + (i + k) % 14) for k in range(5)] + ([PC[proto].format(s=472 + (i + k) % 14) for k in (5, 6, 7)] if w14 else [])
    return fams


MIX, cache = {}, {}
for proto in ("kit", "seen1"):
    for i in range(N):
        for a, (wn, w14) in ARMS.items():
            fl = files(proto, i, w14, wn > 0)
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
        VEC[(a, sp)] = {f: np.concatenate([o[1][f] for o in other]).mean(0) for f in other[0][1]}
ORD_SHUF = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order (= D481's, D483's)
Y, CELL, SUBJ, STREAM = idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy()


def cells(arms, shuf=False):
    order = ORD_SHUF if shuf else idx.order.to_numpy()
    out = []
    for (a, proto, i, split), (ix, mx) in MIX.items():
        if a not in arms:
            continue
        W = weights(ARMS[a][0], "nex" in mx)
        assert abs(sum(W.values()) - 1) < 1e-9 and set(W) == set(mx), (a, W.keys(), mx.keys())
        kk = (STREAM[ix], order[ix])
        P = sum(W[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
        y, cl, sb = Y[ix], CELL[ix], SUBJ[ix]
        for c in np.unique(cl):
            mk = cl == c
            out.append((a, proto, split, i, c, sb[mk][0], np.mean([np.mean(P[mk][y[mk] == q] == q) for q in np.unique(y[mk])])))
    return pd.DataFrame(out, columns=["arm", "proto", "split", "i", "cell", "subj", "ba"])


KEY = ["proto", "split", "i", "cell"]


def cmp(R, a, b):
    m = R[R.arm == a].merge(R[R.arm == b][KEY + ["ba"]], on=KEY, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    return float(d.mean()), u, (f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{N}) level {t.ba.mean():.4f} vs {t.ba_b.mean():.4f}  "
                               + " ".join(f"{s_} {v:+.4f}" for s_, v in u.items()))


R = cells(list(ARMS))
R.to_csv(OUT / ("stack_cells_fake.csv" if FAKE else "stack_cells.csv"), index=False)
lines = []


def say(t):
    print(t, flush=True)
    lines.append(t)


if FAKE:
    for a, b in (("nex", "ref"), ("nexW", "refW")):
        dmax = np.abs(R[R.arm == a].ba.to_numpy() - R[R.arm == b].ba.to_numpy()).max()
        say(f"FAKE code check {a} (EEGNeX weight 0) vs {b}: max |dBA| {dmax:.2e} -> {'OK' if dmax < 1e-12 else 'MISMATCH'}")
    sys.exit(0)
g, u, txt = cmp(R, "nexW", "refW")
ut = u.drop("P04", errors="ignore")
Rs = cells(["refW", "nexW"], shuf=True)
lose = {a: R[(R.arm == a) & (R.subj != "P04")].ba.mean() - Rs[(Rs.arm == a) & (Rs.subj != "P04")].ba.mean() for a in ("refW", "nexW")}
gate = lose["nexW"] - lose["refW"] <= 0.003
ok = g >= -0.002 and ut.min() >= -0.005 and gate
say(f"D485 STACK HARM CHECK nexW (EEGNeX 0.10 on W14's 5 + 3) vs refW: {txt}; shuffle: nexW loses {lose['nexW']:+.4f}, refW {lose['refW']:+.4f} -> "
    + ("PASSES the harm check" if ok else "FAILS the harm check"))
say("D485 stack descriptive nex vs ref (five plain FM): " + cmp(R, "nex", "ref")[2])
for a in ("nexW05", "nexW15"):
    say(f"D485 stack descriptive {a} vs refW: " + cmp(R, a, "refW")[2])
say("D485 stack descriptive refW vs ref (= D475g a): " + cmp(R, "refW", "ref")[2])
(OUT / "stack.txt").write_text("\n".join(lines) + "\n")
