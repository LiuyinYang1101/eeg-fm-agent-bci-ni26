#!/usr/bin/env python
"""T2-D377: the combination ledger — forward-greedy stacking of today's small positives on nm35 (t2_d345r/full), SELECTED on one half of
the 12 draws and JUDGED on the other, under SKλ (causal Sinkhorn 10 / n0 3 / λ 0.95), with the rescreen's groups (typical = every non-P04
cell over kit + seen1, test + val; P04; new = P05 P08 P09). Budgets: B40 = swaps only (runtime-neutral), B44 = at most one ADD (≤ +4.5 ms).
Stops when no move adds >= +0.0003 on the select half. Reports each step on the judge half (typical / P04 / new Δ vs nm35, draws up),
additivity (the stack vs the sum of its singles, judge half) and the projected ms; both half assignments.
  python t2_d377_stack.py --export E --out DIR [--jobs 8]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
SK = dict(iters=10, n0=3, lam=0.95)
M5_MS = 39.23
SLOT = ("r0", "r12", "r24", "vp12", "lp10")
G = {}


def members():
    base = json.load(open(O / "t2_d345r/full/spec.json"))
    M = dict(base["members"])
    take = {"t2_d370/spec.json": ("lplh0", "lplh12", "sw2", "mor_rc", "mor_rc5"), "t2_d358/spec.json": ("fbc_tl15",), "t2_d360r/spec.json": ("stl_ea",),
            "t2_d365/spec.json": ("fbc_us5", "sh_us5"), "t2_d346r/spec.json": ("eegnex",), "t2_d349b/spec.json": ("a0", "b0", "c0")}
    for f, ks in take.items():
        J = json.load(open(O / f))["members"]
        for k in ks:
            M[k] = J[k]
    return M, base["arms"]["nm35"]


def swap(old, new):
    def f(w):
        if old not in w or new in w:
            return None
        w = dict(w)
        w[new] = w.pop(old)
        return w
    return f


def add_st(new):
    def f(w):
        if new in w:
            return None
        slot = [k for k in w if k in SLOT or k in ("lplh0", "lplh12", "stl_ea", "sw2", "a0", "b0", "c0")]
        tot = sum(w[k] for k in slot)
        w = {k: v for k, v in w.items() if k not in slot}
        w.update({k: tot / (len(slot) + 1) for k in slot + [new]})
        return w
    return f


def split_cnn(old, twin):
    def f(w):
        if old not in w or twin in w:
            return None
        w = dict(w)
        w[twin] = w[old] / 2
        w[old] = w[old] / 2
        return w
    return f


def add_frac(new, frac, scale_keys=None):
    def f(w):
        if new in w:
            return None
        keys = [k for k in w if (scale_keys is None or k in scale_keys)]
        tot = sum(w[k] for k in keys)
        w = dict(w)
        for k in keys:
            w[k] = w[k] * (tot - frac) / tot
        w[new] = frac
        return w
    return f


MOVES = {  # name: (fn, kind, ms, exclusive group)
    "L2": (lambda w: (lambda w1: swap("r12", "lplh12")(w1) if w1 else None)(swap("r0", "lplh0")(w)), "swap", 0.0, None),
    "TLsw": (swap("fbc_tv8c", "fbc_tl15"), "swap", 0.0, "fbc"),
    "EAlp": (swap("lp10", "stl_ea"), "swap", 0.13, "ea"),
    "EAr24": (swap("r24", "stl_ea"), "swap", 0.13, "ea"),
    "SW2": (add_st("sw2"), "add", 4.46, None),
    "EA6": (add_st("stl_ea"), "add", 4.59, "ea"),
    "GRb6": (add_st("b0"), "add", 4.5, "grid"),
    "GRa6": (add_st("a0"), "add", 4.5, "grid"),
    "GRc6": (add_st("c0"), "add", 4.5, "grid"),
    "USf": (split_cnn("fbc_tv8c", "fbc_us5"), "add", 5.0, "fbc"),
    "USs": (split_cnn("sh_tv8c", "sh_us5"), "add", 5.0, None),
    "EEG": (add_frac("eegnex", 0.10, ("fbc_tv8c", "sh_tv8c", "ifn", "fbc_tl15")), "add", 3.0, None),
    "MORrc": (add_frac("mor_rc", 0.05), "add", 4.0, "mor"),
    "MORrc5": (add_frac("mor_rc5", 0.05), "add", 4.0, "mor"),
}


def init(export, M):
    d = pd.read_csv(f"{export}/index.csv")
    G.update(strm=d.stream.to_numpy(), ordr=d.order.to_numpy(), y=d.y.to_numpy(), cells=d.cell.to_numpy(), subj=d.subject.str[-3:].to_numpy(), M=M, cache={})


def load(m, proto, i, split):
    key = (m, proto, i, split)
    if key not in G["cache"]:
        z = np.load(G["M"][m][proto].format(s=G["M"][m].get("offset", 0) + i))
        G["cache"][key] = (z["test_idx"], z["prob_last"].astype(np.float64)) if split == "test" else (z["val_idx"], z["prob_val_last"].astype(np.float64))
    return G["cache"][key]


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def evaluate(w):
    """per draw: (typical, P04, new) cell-mean BA under SKλ."""
    out = []
    for i in range(12):
        rows = []
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                ix, mix = None, 0.0
                for m, wm in w.items():
                    ix_, p = load(m, proto, i, split)
                    assert ix is None or np.array_equal(ix, ix_), (m, proto, i, split)
                    ix, mix = ix_, mix + wm * p
                pr = causal_sinkhorn(mix, (G["strm"][ix], G["ordr"][ix]), SK["iters"], SK["n0"], SK["lam"]).argmax(1)
                for c in np.unique(G["cells"][ix]):
                    k = G["cells"][ix] == c
                    rows.append((G["subj"][ix][k][0], bal(G["y"][ix][k], pr[k])))
        s = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows])
        out.append((b[s != "P04"].mean(), b[s == "P04"].mean(), b[np.isin(s, ["P05", "P08", "P09"])].mean()))
    return np.array(out)


def ms_of(moves):
    return M5_MS + sum(MOVES[m][2] for m in moves)


def eligible(used, budget):
    n_add = sum(MOVES[m][1] == "add" for m in used)
    groups = {MOVES[m][3] for m in used if MOVES[m][3]}
    for m, (fn, kind, ms, grp) in MOVES.items():
        if m in used or (grp and grp in groups):
            continue
        if kind == "add" and (budget == "B40" or n_add >= 1):
            continue
        yield m


def apply(moves, nm):
    w = dict(nm)
    for m in moves:
        w2 = MOVES[m][0](w)
        if w2 is None:
            return None
        w = w2
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M, nm = members()
    with Pool(a.jobs, initializer=init, initargs=(a.export, M)) as P:
        base = P.apply(evaluate, (nm,))
        singles = dict(zip(MOVES, P.map(evaluate, [apply([m], nm) for m in MOVES])))
        D = {m: (v - base) for m, v in singles.items()}
        res = {"singles_all12": {m: dict(typical=float(d[:, 0].mean()), up=int((d[:, 0] > 0).sum()), P04=float(d[:, 1].mean()), new=float(d[:, 2].mean())) for m, d in D.items()}}
        print("singles (all 12 draws):", json.dumps({m: round(v["typical"], 4) for m, v in res["singles_all12"].items()}), flush=True)
        for sel, jud, tag in ((range(0, 6), range(6, 12), "S0-5/J6-11"), (range(6, 12), range(0, 6), "S6-11/J0-5")):
            sel, jud = list(sel), list(jud)
            for budget in ("B40", "B44"):
                used, cur, steps = [], 0.0, []
                while True:
                    cands = [m for m in eligible(used, budget) if apply(used + [m], nm) is not None]
                    if not cands:
                        break
                    vals = P.map(evaluate, [apply(used + [m], nm) for m in cands])
                    scores = {m: float((v - base)[sel, 0].mean()) for m, v in zip(cands, vals)}
                    best = max(scores, key=scores.get)
                    if scores[best] - cur < 0.0003:
                        break
                    used.append(best)
                    cur = scores[best]
                    v = dict(zip(cands, vals))[best] - base
                    sum_singles = float(sum(D[m][jud, 0].mean() for m in used))
                    st = dict(moves=list(used), select_typical=round(cur, 4), judge_typical=round(float(v[jud, 0].mean()), 4), judge_up=int((v[jud, 0] > 0).sum()),
                              judge_P04=round(float(v[jud, 1].mean()), 4), judge_new=round(float(v[jud, 2].mean()), 4), sum_singles_judge=round(sum_singles, 4),
                              additivity=round(float(v[jud, 0].mean()) / sum_singles, 2) if abs(sum_singles) > 1e-6 else None, ms=round(ms_of(used), 2))
                    steps.append(st)
                    print(f"[{tag} {budget}] +{best}: select {cur:+.4f} | judge {st['judge_typical']:+.4f} ({st['judge_up']}/6), P04 {st['judge_P04']:+.4f}, new {st['judge_new']:+.4f} | "
                          f"sum of singles {sum_singles:+.4f} (x{st['additivity']}) | {st['ms']} ms", flush=True)
                res[f"{tag} {budget}"] = steps
    (out / "d377.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
