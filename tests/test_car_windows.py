#!/usr/bin/env python
"""T2-D475: ni26.stream.car_windows. 1) every window's good channels average to 0 after; 2) a flat channel and a NaN channel are left out
of the mean (the other channels equal the CAR over the good ones); the NaN channel stays NaN; 3) idempotent on clean data; 4) 2-D input.
    python tests/test_car_windows.py -> OK"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ni26.stream import car_windows  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 8, 100)).astype(np.float32) + rng.normal(size=(5, 8, 1)).astype(np.float32) * 10
Y = car_windows(X)
assert np.abs(Y.mean(1)).max() < 1e-4
assert np.allclose(car_windows(Y), Y, atol=1e-5)
Z = X.copy()
Z[:, 2] = 3.0                      # flat
Z[:, 5] = np.nan                   # dead
W = car_windows(Z)
good = [c for c in range(8) if c not in (2, 5)]
ref = Z[:, good] - Z[:, good].mean(1, keepdims=True)
assert np.allclose(W[:, good], ref, atol=1e-5)
assert np.isnan(W[:, 5]).all()
assert np.allclose(W[:, 2], 3.0 - Z[:, good].mean(1), atol=1e-5)
assert np.allclose(car_windows(X[0]), Y[0], atol=1e-6)
print("OK")
