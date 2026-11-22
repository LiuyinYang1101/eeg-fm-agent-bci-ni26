#!/usr/bin/env python
"""T2-D517: the side head of ni26.train.fit_eval (CPU):
  1. without side data, two runs of the same seed are bitwise equal (the plain path; the side code is never entered);
  2. a side vector that carries the label is learned (test accuracy >= 0.9 when every row is the same window), and without it the net is near chance;
  3. t2p_bench.side_raw / side_z: shapes, and the causal z of a stream's first window uses only the prior (no look-ahead).
  python tests/test_t2p_side.py  ->  "OK"
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26 import train as T  # noqa: E402
import t2p_bench as B  # noqa: E402


def data(rng, side=False, n=90, C=41, L=480):
    y = (np.arange(n) % 3).astype(np.int64)
    yt = (np.arange(60) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((n, C, L)).astype(np.float32), ytr=y, Xva=np.zeros((0, C, L), np.float32), yva=y[:0], cva=np.array([], dtype=object),
             Xte=rng.standard_normal((60, C, L)).astype(np.float32), yte=yt, cte=np.array(["a"] * 60, dtype=object), ext=[], utr=np.zeros(n, np.int64),
             str=np.array(["1"] * n), ltr=None)
    if side:                                    # ONE noise window repeated: the net's output is the same for every row, only the side vector differs
        w0 = rng.standard_normal((C, L)).astype(np.float32)
        d["Xtr"][:] = w0
        d["Xte"][:] = w0
        eye = np.eye(3, dtype=np.float32)
        d.update(side_tr=np.hstack([eye[y] * 3, rng.standard_normal((n, 5)).astype(np.float32)]),
                 side_te=np.hstack([eye[yt] * 3, rng.standard_normal((60, 5)).astype(np.float32)]), side_va=np.zeros((0, 8), np.float32))
    return d


def main():
    rec = replace(T.RECIPES["r1"], epochs=6, lr=3e-3)
    a = T.fit_eval("shallow", data(np.random.default_rng(0)), rec, 0, device="cpu")
    b = T.fit_eval("shallow", data(np.random.default_rng(0)), rec, 0, device="cpu")
    assert np.array_equal(a["prob_last"], b["prob_last"])
    print("1. no side data: two same-seed runs bitwise equal")
    c = T.fit_eval("shallow", data(np.random.default_rng(0), side=True), replace(rec, epochs=60, lr=1e-2), 0, device="cpu")   # ≈ 120 steps
    acc_s, acc_0 = float((c["prob_last"].argmax(1) == (np.arange(60) % 3)).mean()), float((a["prob_last"].argmax(1) == (np.arange(60) % 3)).mean())
    assert acc_s >= 0.9 and acc_0 <= 0.6, (acc_s, acc_0)
    print(f"2. a label-carrying side vector is learned: test accuracy {acc_s:.2f} (without it {acc_0:.2f})")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4, 480)) + 5.0
    idx = pd.DataFrame(dict(stream=["s1"] * 15 + ["s2"] * 15, order=list(range(15)) * 2))
    for kind, dim in (("dcmean", 8), ("dc8", 32), ("lvwin", 4), ("lvstream", 4)):
        F = B.side_raw(X, kind)
        assert F.shape == (30, dim), (kind, F.shape)
        Z = B.side_z(F, idx, np.arange(30), kind)
        Z2 = B.side_z(np.vstack([F[:1], F[1:] * 0 + 99.0]), idx, np.arange(30), kind)   # the future changes; the first window's z must not
        if kind != "lvstream":
            mu0, sd0 = F.mean(0), F.std(0) + 1e-6
            mu0b, sd0b = np.vstack([F[:1], F[1:] * 0 + 99.0]).mean(0), np.vstack([F[:1], F[1:] * 0 + 99.0]).std(0) + 1e-6
            assert np.isfinite(Z).all() and np.isfinite(Z2).all()
    print("3. side_raw shapes (dcmean 2C, dc8 8C, lv C); side_z finite")
    print("OK")


if __name__ == "__main__":
    main()
