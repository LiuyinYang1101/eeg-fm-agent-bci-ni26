#!/usr/bin/env python
"""T2-D306: t2p_bench.softclip, CPU: 'ch' = tanh(x / q99 per window and channel), 'glob' = tanh(x / q99 per window); outputs in (-1, 1); 'glob'
keeps the between-channel amplitude ratios of small values (≈ linear below q99), 'ch' equalises the channels; dtype float32.
  python tests/test_softclip.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/t2_bci/scripts")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import t2p_bench as B  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4, 480)).astype(np.float32) * np.array([1, 2, 4, 8], np.float32)[None, :, None]
    for mode in ("ch", "glob"):
        Y = B.softclip(X, mode)
        q = np.quantile(np.abs(X), 0.99, axis=(2,) if mode == "ch" else (1, 2), keepdims=True)
        assert Y.dtype == np.float32 and np.allclose(Y, np.tanh(X / q), atol=1e-6) and np.abs(Y).max() < 1
    r_ch, r_gl = B.softclip(X, "ch").std((0, 2)), B.softclip(X, "glob").std((0, 2))
    assert r_ch.max() / r_ch.min() < 1.1 and r_gl.max() / r_gl.min() > 3, (r_ch, r_gl)
    print(f"softclip: = tanh(x / q99); 'ch' equalises the channels (std ratio {r_ch.max() / r_ch.min():.2f}), 'glob' keeps them ({r_gl.max() / r_gl.min():.2f})")
    print("OK")


if __name__ == "__main__":
    main()
