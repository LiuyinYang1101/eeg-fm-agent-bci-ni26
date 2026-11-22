#!/usr/bin/env python
"""T2-D191: t2p_bench's IAF warp on CPU (synthetic):
  1. window_iaf finds a 9 Hz alpha on the posterior channels;
  2. iaf_warp at IAF 10 = identity; an 8 Hz (12 Hz) sine warped with IAF 8 (12) peaks at 10 Hz;
  3. iaf_factors: training windows = their user's median, every other window = its stream's causal running mean.
  python tests/test_iaf_warp.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402

fs, L = 120.0, 480
t = np.arange(L) / fs


def peak(x):
    f = np.fft.rfftfreq(L, 1 / fs); return f[np.argmax(np.abs(np.fft.rfft(x)))]


def main():
    names = ["Fz", "Cz", "Pz", "P3", "P4", "Oz"]
    r = np.random.default_rng(0)
    X = 0.3 * r.standard_normal((4, len(names), L))
    X[:, 2:5] += np.sin(2 * np.pi * 9.0 * t)
    assert np.allclose(B.window_iaf(X, fs, names), 9.0), B.window_iaf(X, fs, names)
    x = np.stack([np.sin(2 * np.pi * 10.0 * t), np.cos(2 * np.pi * 7.5 * t)])[None]
    assert np.array_equal(B.iaf_warp(x.astype(np.float32), np.array([10.0])), x.astype(np.float32))
    for f0 in (8.0, 12.0):
        w = B.iaf_warp(np.sin(2 * np.pi * f0 * t)[None, None].astype(np.float32), np.array([f0]))
        assert abs(peak(w[0, 0]) - 10.0) <= 0.25, (f0, peak(w[0, 0]))
    idx = pd.DataFrame(dict(subject=["A", "A", "B", "B", "C", "C", "C"], stream=["A|1", "A|1", "B|1", "B|1", "C|2", "C|2", "C|2"],
                            order=[0, 1, 0, 1, 2, 0, 1]))
    iaf = np.array([8.0, 9.0, 10.0, 12.0, 11.0, 9.0, 10.0])
    trm = np.array([1, 1, 1, 1, 0, 0, 0], bool)
    fac = B.iaf_factors(iaf, idx, trm)
    assert np.allclose(fac[:4], [8.5, 8.5, 11.0, 11.0]), fac
    assert np.allclose(fac[[5, 6, 4]], [9.0, 9.5, 10.0]), fac                 # stream C|2 in order 0, 1, 2
    print("IAF warp: detector 9 Hz; identity at 10 Hz; 8 / 12 Hz -> 10 Hz; factors = train median / causal stream mean")
    print("OK")


if __name__ == "__main__":
    main()
