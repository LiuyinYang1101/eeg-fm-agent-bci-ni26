#!/usr/bin/env python
"""T2-D446 (pre-registered in docs/DECISIONS_T2.md before any number): a WIDE-ADD screen on M7 from stored predictions only.
Pool = 20 members that are not in M7, fixed before the read (9 other families, 11 variants of our own families). Every add gets the same
small weight; nothing is fitted. SELECT on draws 0-5 (rank the singles at w 0.03 by typical delta vs M7), JUDGE on draws 6-11:
top-k sets for k = 1, 3, 6, 10, 20, each add at w = min(0.03, 0.30 / k), M7's members scaled to the rest.
Three bases: M7 (views on pooled nets + copies, as shipped), V8 (copies' views dropped), V1 (no views at all = the 28.9 ms path).
On V8 / V1 the CNN adds use their own plain files when they exist. Typical = non-P04 cells; causal SK 10 / 3 / 0.95.
    mem-guard python scripts/local/t2_d446_wide.py
"""
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
ALL = json.load(open(O / "t2_repro/all_members.json"))
M7 = json.load(open(O / "t2_d408/spec_m7.json"))["arms"]["M7"]
POOL = ["eegnex", "atcnet", "deep4", "eegtcnet", "fbts", "mor_p1v2", "mantis", "rocket", "cbg_a",                       # other families
        "stl_eaf", "lplh36", "vp12", "sess", "scx", "sh_us5", "fbc_us5", "fbc_tl15f", "fbc_ssl", "x_ovr", "bag_runs"]  # variants of ours
VIEW = {"M7": {}, "V8": {"fbc_tv8c": "fbc_tv8", "sh_tv8c": "sh_tv8", "ifn": "ifn_tv8"}, "V1": {"fbc_tv8c": "fbc_pl", "sh_tv8c": "sh_pl", "ifn": "ifn_pl"}}
idx = pd.read_csv(EXP / "index.csv")
SUBJ = idx.subject.str[-3:].to_numpy()
cache = {}


def path(m, proto, i, plain):
    m, _, k = m.partition("@")          # T2-D446c: "name@k" = the same member's file k draws further on (another seed)
    v = ALL[m]
    f = v[proto].format(s=v.get("offset", 0) + (i + int(k or 0)) % 12)
    if plain:
        g = f.replace("tv8cfb", "fb")
        if g != f and os.path.exists(g):
            return g
    return f


def prob(m, proto, i, split, plain=False):
    k = (m, proto, i, split, plain)
    if k not in cache:
        z = np.load(path(m, proto, i, plain))
        cache[k] = (z["test_idx"], z["prob_last"].astype(np.float64)) if split == "test" else (z["val_idx"], z["prob_val_last"].astype(np.float64))
    return cache[k]


def score(weights, draws, plain=False):
    """weights {member: w}; returns DataFrame rows (i, subj, ba) over non-P04 cells."""
    rows = []
    for proto in ("kit", "seen1"):
        for i in draws:
            for split in ("test", "val"):
                mix, ix = 0.0, None
                for m, w in weights.items():
                    ix_, p = prob(m, proto, i, split, plain and m in POOL)
                    assert ix is None or (ix == ix_).all(), m
                    ix, mix = ix_, mix + w * p
                P = causal_sinkhorn(mix, (idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]), 10, 3, 0.95).argmax(1)
                y, cells, sj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], SUBJ[ix]
                for c in np.unique(cells):
                    mk = cells == c
                    rows.append((i, sj[mk][0], np.mean([np.mean(P[mk][y[mk] == q] == q) for q in np.unique(y[mk])])))
    return pd.DataFrame(rows, columns=["i", "subj", "ba"])


def base(v):
    return {VIEW[v].get(k, k): w for k, w in M7.items()}


def with_adds(v, adds):
    w = min(0.03, 0.30 / len(adds)) if adds else 0.0
    b = {k: x * (1 - w * len(adds)) for k, x in base(v).items()}
    return {**b, **{a: w for a in adds}}


def delta(a, b):
    ty = lambda d: d[d.subj != "P04"].groupby("i").ba.mean()   # noqa: E731
    d = ty(a) - ty(b)
    u = (a.groupby(["subj", "i"]).ba.mean() - b.groupby(["subj", "i"]).ba.mean()).groupby("subj").mean()
    return d.mean(), int((d > 0).sum()), len(d), u


if len(sys.argv) > 1 and sys.argv[1] == "b":   # T2-D446b: the level-rule set (variants of our families, >= 0.70 alone), all 12 draws
    SET = POOL[9:]
    ref = score(base("M7"), range(12))
    for v in ("M7", "V8"):
        for share in (0.15, 0.30, 0.45):
            w = share / len(SET)
            wt = {**{k: x * (1 - share) for k, x in base(v).items()}, **{a: w for a in SET}}
            d, up, n, u = delta(score(wt, range(12), plain=v != "M7"), ref)
            print(f"  base {v:2s} + 11 variants at total {share:.2f}: typical {d:+.4f} ({up}/{n})  " + " ".join(f"{s_} {x:+.4f}" for s_, x in u.items())
                  + f"  | typical w/o best user {np.sort(u.drop('P04').to_numpy())[:-1].mean():+.4f}", flush=True)
    CTL = [f"fbc_tv8c@{k}" for k in (1, 2, 3)] + [f"sh_tv8c@{k}" for k in (1, 2, 3)] + ["e100f0@1", "e100f1@1", "e100f2@1", "lp10@1", "lp10@2"]
    for share in (0.15, 0.30, 0.45):
        w = share / 11
        b = {k: x * (1 - share) for k, x in base("M7").items()}
        sv, sc = score({**b, **{a: w for a in SET}}, range(12)), score({**b, **{a: w for a in CTL}}, range(12))
        d, up, n, u = delta(sc, ref)
        d2, up2, n2, u2 = delta(sv, sc)
        print(f"  CONTROL (11 other-seed nets of M7's recipes) at {share:.2f}: typical {d:+.4f} ({up}/{n})  " + " ".join(f"{s_} {x:+.4f}" for s_, x in u.items()))
        print(f"     variants - control: {d2:+.4f} ({up2}/{n2})  " + " ".join(f"{s_} {x:+.4f}" for s_, x in u2.items()), flush=True)
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "d":   # T2-D446d: where does the 'more nets' gain sit: CNN seeds or ST-L seeds?
    ref = score(base("M7"), range(12))
    ST = ["e100f0", "e100f1", "e100f2", "e100f3", "lp10"]

    def show(name, wt, plain=False):
        d, up, n, u = delta(score(wt, range(12), plain=plain), ref)
        print(f"  {name:44s} typical {d:+.4f} ({up}/{n})  " + " ".join(f"{s_} {x:+.4f}" for s_, x in u.items()), flush=True)

    b = base("M7")
    show("M7 + 6 other-seed CNN nets (0.03 each)", {**{k: x * 0.82 for k, x in b.items()}, **{f"{m}@{k}": 0.03 for m in ("fbc_tv8c", "sh_tv8c") for k in (1, 2, 3)}})
    for v in ("M7", "V8", "V1"):
        for fm in (0.35, 0.45):
            bb = base(v)
            cnn = {k: x * (1 - fm) / 0.65 for k, x in bb.items() if k not in ST}
            show(f"{v} with TEN ST-L members, FM share {fm}", {**cnn, **{m: fm / 10 for m in ST}, **{f"{m}@1": fm / 10 for m in ST}}, plain=v != "M7")
        if v != "M7":
            show(f"{v} alone (five ST-L, 0.35)", base(v))
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "e":   # T2-D446e: recipe diversity inside a 3-net slot, at equal net count
    b = base("M7")
    wf, ws = b.pop("fbc_tv8c"), b.pop("sh_tv8c")
    A = {**b, **{f"fbc_tv8c@{k}": wf / 3 for k in (0, 1, 2)}, **{f"sh_tv8c@{k}": ws / 3 for k in (0, 1, 2)}}
    B = {**b, "fbc_tv8c@0": wf / 3, "fbc_tl15f": wf / 3, "fbc_us5": wf / 3, "sh_tv8c@0": ws / 3, "sh_us5": ws / 3, "sh_eaf": ws / 3}
    sa, sb, ref = score(A, range(12)), score(B, range(12)), score(base("M7"), range(12))
    for nm, x, y in (("A (3 seeds) vs M7 (1 net per slot)", sa, ref), ("B (1 + 2 variants) vs M7", sb, ref), ("B - A", sb, sa)):
        d, up, n, u = delta(x, y)
        print(f"  {nm:36s} typical {d:+.4f} ({up}/{n})  " + " ".join(f"{s_} {v:+.4f}" for s_, v in u.items()), flush=True)
    sys.exit(0)
SEL, JUD = range(0, 6), range(6, 12)
ref_s, ref_j = score(base("M7"), SEL), score(base("M7"), JUD)
print("== singles at w 0.03 on M7 (SELECT half, draws 0-5 | judge half shown for information only)")
rank = []
for a in POOL:
    ds, us, ns, _ = delta(score(with_adds("M7", [a]), SEL), ref_s)
    dj, uj, nj, _ = delta(score(with_adds("M7", [a]), JUD), ref_j)
    alone = score({a: 1.0}, SEL)
    rank.append((ds, a))
    print(f"  {a:10s} alone {alone[alone.subj != 'P04'].groupby('i').ba.mean().mean():.3f}  select {ds:+.4f} ({us}/{ns})   judge {dj:+.4f} ({uj}/{nj})", flush=True)
order = [a for _, a in sorted(rank, reverse=True)]
print("order (by the select half):", " ".join(order))
print("\n== JUDGE half (draws 6-11): top-k adds, each at min(0.03, 0.30 / k); delta vs M7 AS SHIPPED; per user")
for v in ("M7", "V8", "V1"):
    for k in (0, 1, 3, 6, 10, 20):
        if v == "M7" and k == 0:
            continue
        d, up, n, u = delta(score(with_adds(v, order[:k]), JUD, plain=v != "M7"), ref_j)
        print(f"  base {v:2s} + top {k:2d}: typical {d:+.4f} ({up}/{n})  " + " ".join(f"{s} {x:+.4f}" for s, x in u.items())
              + f"  | typical w/o best user {np.sort(u.drop('P04').to_numpy())[:-1].mean():+.4f}", flush=True)
