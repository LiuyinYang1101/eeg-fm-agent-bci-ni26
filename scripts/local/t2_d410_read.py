#!/usr/bin/env python
"""T2-D410 read (pre-registered 2026-11-07, before any number): the learning curve over the NUMBER OF TRAINING USERS.
Per K in {3, 5, 7, 9} x 10 balanced cyclic folds (t2p_bench lcK): FBCNet rc75 (none) + ShallowNet vr (ea), seed 400, `last`; the mix at M6's CNN
ratio (0.314 : 0.236 -> 0.571 / 0.429), causal SKλ (10 it, n0 3, λ 0.95) per stream over the fold's test windows; cell-BA per cell.
  NEW(K)  = the mean over the 10 users of each user's mean cell-BA over the folds where it is UNTRAINED (each user equally often per K);
  typical(K) = the same without P04;  RETURNING(K) = P04 / P10 sessions 2-3 in the folds where they train (descriptive, 2 users).
  slope b = OLS of the per-user-per-K NEW BA on ln K with user fixed effects; 90 % user-cluster bootstrap (2,000); D(9->10) = b ln(10/9),
  D(6->10) = b ln(10/6) (kit's 6 training users -> the --train all builds' 10).
Bar: ALIVE iff D(9->10) >= +0.003 AND b's CI lower bound > 0 AND NEW(9) - NEW(7) > 0; FLAT iff b's CI includes 0 OR D(9->10) < +0.001; else WEAK.
  python t2_d410_read.py --export E --runs DIR [DIR ...] [--out DIR]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
KS = (3, 5, 7, 9)
MODELS = {"fbc": ("fbcnet_bboth", 0.314), "sh": ("shallow", 0.236)}


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def load(runs, fold, model):
    fs = [f for r in runs for f in glob.glob(f"{r}/runs/{fold}__{model}__*__s400.npz")]
    assert len(fs) == 1, (fold, model, fs)
    z = np.load(fs[0]); return z["test_idx"], z["prob_last"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(O / "t2_d410"))
    a = ap.parse_args()
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    strm, ordr, y, cell = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy(), d.cell.to_numpy()
    sub, ses = d.subject.str[-3:].to_numpy(), d.session.astype(str).to_numpy()
    users = sorted(set(sub))
    rows = []
    for K in KS:
        for i in range(10):
            fold = f"lc{K}_{i}"
            P = {}
            for k, (m, w) in MODELS.items():
                ix, p = load(a.runs, fold, m); P[k] = (ix, p)
            ix = P["fbc"][0]; assert np.array_equal(ix, P["sh"][0])
            tot = sum(w for _, w in MODELS.values())
            arms = {"mix": sum(MODELS[k][1] / tot * P[k][1] for k in MODELS), "fbc": P["fbc"][1], "sh": P["sh"][1]}
            for arm, pr in arms.items():
                q = causal_sinkhorn(pr, (strm[ix], ordr[ix]), 10, 3, 0.95).argmax(1)
                for c in np.unique(cell[ix]):
                    k = cell[ix] == c; u = sub[ix][k][0]; s = ses[ix][k][0]
                    trained_u = (u in ("P04", "P10")) and s in ("2", "3") and not (sub[ix] == u).sum() == (sub == u).sum()
                    rows.append(dict(K=K, fold=i, arm=arm, user=u, session=s, cell=c, group="returning" if trained_u else "new",
                                     ba=bal(y[ix][k], q[k])))
    R = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    R.to_csv(Path(a.out) / "cells.csv", index=False)
    out = {}
    for arm in ("mix", "fbc", "sh"):
        N = R[(R.arm == arm) & (R.group == "new")].groupby(["K", "user", "fold"]).ba.mean().groupby(["K", "user"]).mean().unstack("user")
        Ret = R[(R.arm == arm) & (R.group == "returning")].groupby(["K", "user"]).ba.mean().unstack("user")
        curve = dict(new={int(k): float(v) for k, v in N.mean(1).items()},
                     typical={int(k): float(v) for k, v in N.drop(columns="P04").mean(1).items()},
                     returning={int(k): float(v) for k, v in Ret.mean(1).items()},
                     per_user_new={u: {int(k): float(v) for k, v in N[u].items()} for u in N.columns})

        xk = np.log(np.array(N.index, float)); xk = xk - xk.mean()

        def slope(A):                                              # A: (K, users) -> OLS slope on ln K with user fixed effects
            Ac = A - A.mean(0, keepdims=True)
            return float((Ac * xk[:, None]).sum() / (A.shape[1] * (xk ** 2).sum()))
        A = N.to_numpy(float); b = slope(A)
        rng = np.random.default_rng(0)
        boot = [slope(A[:, rng.integers(0, A.shape[1], A.shape[1])]) for _ in range(2000)]
        lo, hi = np.percentile(boot, [5, 95])
        curve.update(b=b, b_ci90=[float(lo), float(hi)], b_typical=slope(N.drop(columns="P04").to_numpy(float)),
                     d_9_10=float(b * np.log(10 / 9)), d_6_10=float(b * np.log(10 / 6)), d_7_9=float(curve["new"][9] - curve["new"][7]))
        out[arm] = curve
    m = out["mix"]
    alive = m["d_9_10"] >= 0.003 and m["b_ci90"][0] > 0 and m["d_7_9"] > 0
    flat = m["b_ci90"][0] <= 0 <= m["b_ci90"][1] or m["d_9_10"] < 0.001
    out["verdict"] = "ALIVE" if alive else ("FLAT" if flat else "WEAK")
    (Path(a.out) / "curve.json").write_text(json.dumps(out, indent=1))
    for arm in ("mix", "fbc", "sh"):
        c = out[arm]
        print(f"[D410] {arm:3s} NEW  " + "  ".join(f"K{k} {c['new'][k]:.4f}" for k in KS) + f"  | typical " +
              "  ".join(f"K{k} {c['typical'][k]:.4f}" for k in KS))
        print(f"[D410] {arm:3s} RET  " + "  ".join(f"K{k} {c['returning'].get(k, float('nan')):.4f}" for k in KS) +
              f"  | b {c['b']:+.4f} per ln K (CI90 [{c['b_ci90'][0]:+.4f}, {c['b_ci90'][1]:+.4f}]; typical {c['b_typical']:+.4f}); "
              f"D(7->9) {c['d_7_9']:+.4f}; D(9->10) {c['d_9_10']:+.4f}; D(6->10) {c['d_6_10']:+.4f}")
    for u, v in out["mix"]["per_user_new"].items():
        print(f"   {u}: " + "  ".join(f"K{k} {v[k]:.4f}" for k in KS if k in v))
    print(f"[D410] VERDICT (mix, NEW users): {out['verdict']}")


if __name__ == "__main__":
    main()
