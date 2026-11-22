#!/usr/bin/env python
"""T2-D466 (written before any number): pruning M7's ensemble. See the row. Zero GPU.
    mem-guard python scripts/local/t2_d466_prune.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d408/spec_m7.json"))
M7 = J["arms"]["M7"]
MEM = list(M7)
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
COMBOS = [(p, i, s) for p in ("kit", "seen1") for i in range(12) for s in ("test", "val")]
P = {}
for (p, i, s) in COMBOS:
    for m in MEM:
        z = np.load(J["members"][m][p].format(s=J["members"][m].get("offset", 0) + i))
        ix, pr = (z["test_idx"], z["prob_last"]) if s == "test" else (z["val_idx"], z["prob_val_last"])
        P[(m, p, i, s)] = pr.astype(np.float64)
        P[("ix", p, i, s)] = ix


def cells(w, combo):
    p, i, s = combo
    ix = P[("ix", p, i, s)]
    mix = sum(w[m] * P[(m, p, i, s)] for m in w if w[m] > 0)
    pred = causal_sinkhorn(mix, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1)
    out = []
    for c in np.unique(CELL[ix]):
        mk = CELL[ix] == c
        out.append((p, i, s, c, U[ix][mk][0], np.mean([np.mean(pred[mk][Y[ix][mk] == v] == v) for v in np.unique(Y[ix][mk])])))
    return out


def table(w):
    rows = []
    for combo in COMBOS:
        rows += cells(w, combo)
    return pd.DataFrame(rows, columns=["proto", "i", "split", "cell", "u", "ba"])


def typ(T, draws=None, users=None):
    t = T[T.u != "P04"]
    if draws is not None:
        t = t[t.i.isin(draws)]
    if users is not None:
        t = t[t.u.isin(users)]
    return t.groupby("i").ba.mean()


def norm(w):
    s = sum(w.values())
    return {k: v / s for k, v in w.items() if v > 0}


def run(job):
    kind, arg = job
    return job, table(norm(arg))


if __name__ == "__main__":
    ex = ProcessPoolExecutor(8)
    base = table(M7)
    b = typ(base)
    print(f"M7 typical {b.mean():.4f}", flush=True)
    # (1) leave one member out
    jobs = [("loo", {k: (0.0 if k == m else v) for k, v in M7.items()}) for m in MEM]
    for (kind, w), T in ex.map(run, jobs):
        m = [k for k in MEM if w[k] == 0][0]
        d = typ(T) - b
        print(f"drop {m:9s} (share {M7[m]:.3f}): typical {d.mean():+.4f} ({int((d > 0).sum())}/12)", flush=True)
    # (2) greedy forward selection with replacement, nested
    folds = [("draws 0-5 -> 6-11", dict(draws=range(6)), dict(draws=range(6, 12))), ("draws 6-11 -> 0-5", dict(draws=range(6, 12)), dict(draws=range(6))),
             ("users P05 P08 -> P09 P10", dict(users=["P05", "P08"]), dict(users=["P09", "P10"])), ("users P09 P10 -> P05 P08", dict(users=["P09", "P10"]), dict(users=["P05", "P08"]))]
    for name, sel, ev in folds:
        counts, best_path = {m: 0 for m in MEM}, []
        for step in range(10):
            cands = [("g", norm({k: counts[k] + (1 if k == m else 0) for k in MEM})) for m in MEM]
            res = list(ex.map(run, cands))
            sc = [typ(T, **sel).mean() for _, T in res]
            j = int(np.argmax(sc))
            counts[MEM[j]] += 1
            best_path.append((MEM[j], sc[j], res[j][1]))
        k = int(np.argmax([s for _, s, _ in best_path]))
        T = best_path[k][2]
        d = typ(T, **ev) - typ(base, **ev)
        chosen = {}
        for m, _, _ in best_path[:k + 1]:
            chosen[m] = chosen.get(m, 0) + 1
        print(f"greedy [{name}]: picked {k + 1} -> {chosen}; out-of-sample typical {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)}); members used {len(chosen)} of {len(MEM)}", flush=True)
