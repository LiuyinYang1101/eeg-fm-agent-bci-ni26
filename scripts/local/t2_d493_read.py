#!/usr/bin/env python
"""T2-D493 read (written before any number): does a uniform weight SOUP of five ST-L fine-tunes (same CPT init, same recipe =
T2-D455's control line, seeds s + 1000 j, j = 0..4) hold the score of the ENSEMBLE of the same five nets? One soup forward replaces
five, so a hold frees ≈ 17 ms / window for different-in-kind members (the manager's 'buy the budget back').
Draw i = soup task seed 730 + i (kit + seen1, test + val): SINGLE = its seed-s member (the main file), ENS5 = mean of the five members'
probabilities (main + soup5m1..m4), SOUP5 = the soup twin. Every arm is decoded with W13's per-family causal Sinkhorn (10 it, n0 3,
λ 0.95) and a per-family start vector cross-fitted over the other split (t2_d475_read.py's A13 rule). CNN slots = the A13 base (M7's
weights; FBCNet D419 / ShallowNet + IFNet D429 at offset + i). Typical = non-P04.
  stack, W14 shape : CNNs + fm family {five plain at w/8 each (ENS5) | SOUP5 at 5w/8 | SINGLE at 5w/8} + 3 CAR nets (T2-D475's, seed
                     index i + 5 .. i + 7 as A13) at w/8 each          -> wE / wS / w1
  stack, W13 shape : CNNs + fm family {ENS5 at w/5 each | SOUP5 at w | SINGLE at w}                                   -> pE / pS / p1
  alone            : the fm family only (ENS5 | SOUP5 | SINGLE), weight 1                                               -> aE / aS / a1
HOLD (the bar, fixed before numbers): wS − wE AND pS − pE each typical ≥ −0.002 with no typical user below −0.005 (the W13 leg added
after the FAKE run: one plain net alone is −0.0019 vs five in the W14 shape). Otherwise the 'buy the budget back' premise fails and is
closed. Descriptive: aS − aE, ENS5 − SINGLE (what five nets are worth), SOUP5 − SINGLE, P05.
D493_FAKE=1 (code check before the nets land): members = T2-D455's plain nets at seed index i + k, soup = single = index i, 10 draws ->
wE = A13's add3 and pE = A13's pl exactly (levels 0.8396 / 0.8349), wS = w1, pS = p1, aS = a1.
    mem-guard python scripts/local/t2_d493_read.py
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
FAKE = bool(os.environ.get("D493_FAKE"))
SEEDS = list(range(730, 736))
N = 10 if FAKE else len(SEEDS)
K = 5
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    Sp = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = Sp[src]
C = J["arms"]["M7f"]
FM5 = [k for k in C if k.startswith("n_")]
w_fm = sum(C[m] for m in FM5)
CNN = [("fbc", "fbc_tv8c"), ("sh", "sh_tv8c"), ("ifn", "ifn")]
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]


def stl(pat):
    out = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{S}/{pat.format(pr=pr)}-[0-9]*")
        assert len(ds) == 1, (pat, pr, ds)
        f = glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s472*.npz")
        assert len(f) == 1, f
        out[pr] = f[0].replace("__s472.npz", "__s{s}.npz")
    return out


pp, pc = stl("ni26-t2p-d455-ctl-{pr}"), stl("ni26-t2p-d475-stlcar-{pr}")


def st(k):
    return lambda i: 472 + (i + k) % 14


SOUPF = {}       # (proto, draw) -> {"single": f, "soup": f, "members": [f0..f4]}
for pr in ("kit", "seen1"):
    for i in range(N):
        if FAKE:
            mem = [pp[pr].format(s=st(k)(i)) for k in range(K)]
            SOUPF[(pr, i)] = {"single": mem[0], "soup": mem[0], "members": mem}
            continue
        sd = SEEDS[i]
        f = glob.glob(f"{S}/ni26-t2p-d493-{pr}-s{sd}-[0-9]*/runs/{pr}__fm-*__s{sd}.npz")
        f = [x for x in f if "soup" not in Path(x).name]
        assert len(f) == 1, (pr, sd, f)
        f = f[0]
        tw = [f.replace(f"__s{sd}.npz", f"soup{K}m{j}__s{sd}.npz") for j in range(1, K)]
        sp = f.replace(f"__s{sd}.npz", f"soup{K}__s{sd}.npz")
        assert all(Path(x).exists() for x in tw + [sp]), (pr, sd)
        SOUPF[(pr, i)] = {"single": f, "soup": sp, "members": [f] + tw}

w8 = w_fm / 8


def arm(shape, form):
    base = [(f, C[m], ("M", m)) for f, m in CNN] if shape in ("w", "p") else []
    wf = 1.0 if shape == "a" else w_fm
    share = 5 / 8 if shape == "w" else 1.0
    if form == "E":
        fm = [("fm", wf * share / K, ("mem", j)) for j in range(K)]
    else:
        fm = [("fm", wf * share, ("soup" if form == "S" else "single", None))]
    car = [("fm", w8, ("car", st(5 + k))) for k in range(3)] if shape == "w" else []
    return base + fm + car


ARMS = {f"{sh}{fo}": arm(sh, fo) for sh in ("w", "p", "a") for fo in ("E", "S", "1")}
keys_ = idx.stream.to_numpy(), idx.order.to_numpy()
MIX = {}
for proto in ("kit", "seen1"):
    for i in range(N):
        cache = {}

        def get(src):
            kind, x = src
            if kind == "M":
                f = M[x][proto].format(s=M[x].get("offset", 0) + i)
            elif kind == "car":
                f = pc[proto].format(s=x(i))
            elif kind == "mem":
                f = SOUPF[(proto, i)]["members"][x]
            else:
                f = SOUPF[(proto, i)][kind]
            if f not in cache:
                z = np.load(f)
                cache[f] = {"test": (z["test_idx"], z["prob_last"].astype(np.float64)), "val": (z["val_idx"], z["prob_val_last"].astype(np.float64))}
            return cache[f]
        for a, mem in ARMS.items():
            for split in ("test", "val"):
                ix, fam, fw = None, {}, {}
                for f, w, src in mem:
                    ix_, p = get(src)[split]
                    assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                    ix, fam[f], fw[f] = ix_, fam.get(f, 0.0) + w * p, fw.get(f, 0.0) + w
                MIX[(a, proto, i, split)] = (ix, {f: v / v.sum(1, keepdims=True) for f, v in fam.items()}, fw)
    print(f"D493 {proto} loaded", flush=True)
assert all(abs(sum(v[2].values()) - 1.0) < 1e-9 for v in MIX.values())
VEC = {}
for a in ARMS:
    for sp in ("test", "val"):
        other = [v for k, v in MIX.items() if k[0] == a and k[3] != sp]
        VEC[(a, sp)] = {f: np.concatenate([o[1][f] for o in other]).mean(0) for f in other[0][1]}
rows = []
for (a, proto, i, split), (ix, mx, fw) in MIX.items():
    kk = (keys_[0][ix], keys_[1][ix])
    P = sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
    y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
    for c in np.unique(cells):
        mk = cells == c
        rows.append(dict(arm=a, proto=proto, split=split, i=i, cell=c, subj=subj[mk][0],
                         ba=np.mean([np.mean(P[mk][y[mk] == k] == k) for k in np.unique(y[mk])])))
R = pd.DataFrame(rows)
(O / "t2_d493").mkdir(exist_ok=True)
R.to_csv(O / ("t2_d493/cells" + ("_fake" if FAKE else "") + ".csv"), index=False)


def cmp(a, b, hold=None):
    m = R[R.arm == a].merge(R[R.arm == b][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    ut = u.drop("P04", errors="ignore")
    s = (f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{N} up, min draw {d.min():+.4f}) level {t.ba.mean():.4f} vs {t.ba_b.mean():.4f}  "
         + " ".join(f"{s_} {v:+.4f}" for s_, v in u.items()))
    if hold is not None:
        ok = d.mean() >= hold and ut.min() >= -0.005
        s += f"  -> {'HOLDS' if ok else 'FAILS'}"
    return s


# The bar has TWO legs (fixed before numbers, after the FAKE run showed that ONE plain net already sits at −0.0019 vs five in the W14
# shape, so the W14 leg alone cannot tell a soup from a single): W14 shape AND W13 shape (the five-plain family, where one net costs
# −0.0055), each SOUP5 − ENS5 typical ≥ −0.002 with no typical user below −0.005.
b14, b13 = cmp("wS", "wE", -0.002), cmp("pS", "pE", -0.002)
print("D493 BAR leg W14 shape SOUP5 vs ENS5:", b14, flush=True)
print("D493 BAR leg W13 shape SOUP5 vs ENS5:", b13, flush=True)
print("D493 VERDICT:", "HOLDS" if b14.endswith("HOLDS") and b13.endswith("HOLDS") else "FAILS (the soup does not hold the ensemble)", flush=True)
for a_, b_, lab in (("aS", "aE", "alone SOUP5 vs ENS5"), ("wE", "w1", "W14 shape ENS5 vs SINGLE"),
                    ("wS", "w1", "W14 shape SOUP5 vs SINGLE"), ("pE", "p1", "W13 shape ENS5 vs SINGLE"), ("aE", "a1", "alone ENS5 vs SINGLE"),
                    ("aS", "a1", "alone SOUP5 vs SINGLE")):
    print(f"D493 descriptive {lab}:", cmp(a_, b_), flush=True)
