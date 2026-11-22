#!/usr/bin/env python
"""T2-D459: template W12 (development copy) = W11 + an optional decoder start vector (cfg tta.start). CPU:
  1. start absent: W12's SinkhornCorrector = W11's bitwise, window by window, at lam 1.0 and 0.95, across a reset;
  2. start = uniform: bitwise the same as absent;
  3. start given: = t2p_report.causal_sinkhorn(..., start) (1e-12) and differs from the uniform start; reset restores the start rows;
  4. t2p_report.causal_sinkhorn(start=None) = the old call bitwise; a malformed start is refused.
  python tests/test_w12_template.py  ->  "OK"
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_report as R  # noqa: E402

bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
bs.CompetSolver = object; bu.base_solver = bs
sys.modules.update({"benchmark_utils": bu, "benchmark_utils.base_solver": bs})
SUB = ROOT / "tracks/t2_bci/submission"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stream(c, P):
    return np.stack([c(p) for p in P])


def main():
    w11, w12 = load(SUB / "t2p_submission_w11.py", "w11"), load(SUB / "t2p_submission_w12.py", "w12")
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3) * 0.7, 70)
    keys = (np.zeros(70), np.arange(70))
    s = np.array([0.40, 0.23, 0.37])
    for lam in (1.0, 0.95):
        a, b = w11.SinkhornCorrector(10, 3, 3, lam), w12.SinkhornCorrector(10, 3, 3, lam)
        oa, ob = stream(a, P), stream(b, P)
        assert np.array_equal(oa, ob), "start absent must be W11 bitwise"
        a.reset(); b.reset()
        assert np.array_equal(stream(a, P[:20]), stream(b, P[:20])), "after reset"
        u = w12.SinkhornCorrector(10, 3, 3, lam, [1 / 3, 1 / 3, 1 / 3])
        assert np.array_equal(stream(u, P), ob), "uniform start = absent"
        c = w12.SinkhornCorrector(10, 3, 3, lam, s.tolist())
        oc = stream(c, P)
        ref = R.causal_sinkhorn(P, keys, 10, 3, lam, s)
        assert np.abs(oc - ref).max() < 1e-12, np.abs(oc - ref).max()
        assert np.abs(oc - ob).max() > 1e-3, "the start must change the output"
        c.reset()
        assert np.array_equal(stream(c, P), oc), "reset must restore the start rows"
        assert np.array_equal(R.causal_sinkhorn(P, keys, 10, 3, lam), R.causal_sinkhorn(P, keys, 10, 3, lam, None))
        assert np.abs(R.causal_sinkhorn(P, keys, 10, 3, lam) - ob).max() < 1e-12
    for bad in ([0.5, 0.5], [0.5, 0.6, 0.2], [1.0, 0.0, 0.0], [float("nan"), 0.5, 0.5]):
        try:
            w12.SinkhornCorrector(10, 3, 3, 0.95, bad)
        except AssertionError:
            continue
        raise SystemExit(f"a malformed start was accepted: {bad}")
    print("OK")


if __name__ == "__main__":
    main()
