#!/usr/bin/env python
"""T2-D476: template W13 (development copy) = W12 + optional per-family decoding (cfg tta.per_family). CPU:
  1. W13's SinkhornCorrector = W12's bitwise (the class is unchanged);
  2. _fam_decode over a stream = sum_f w_f x t2p_report.causal_sinkhorn(part_f, start_f) (1e-12), 3 and 4 families; a reset restores it;
  3. without per_family the decoder path is W12's (source: the single corrector stays the else branch; the deepcopy copies fam_prior).
  python tests/test_w13_template.py  ->  "OK"
"""
import copy
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


def main():
    w12, w13 = load(SUB / "t2p_submission_w12.py", "w12"), load(SUB / "t2p_submission_w13.py", "w13")
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(3) * 0.7, 60)
    for lam in (1.0, 0.95):
        a, b = w12.SinkhornCorrector(10, 3, 3, lam, [0.4, 0.25, 0.35]), w13.SinkhornCorrector(10, 3, 3, lam, [0.4, 0.25, 0.35])
        assert np.array_equal(np.stack([a(p) for p in P]), np.stack([b(p) for p in P])), "W13's corrector must be W12's"
    keys = (np.zeros(60), np.arange(60))
    starts = {"cnn": [0.4412, 0.237, 0.3218], "cnn2": [0.3526, 0.2889, 0.3585], "cnn3": [0.3524, 0.2662, 0.3814], "fm": [0.4294, 0.2426, 0.328]}
    for fams, w in ((("cnn", "cnn2", "fm"), {"cnn": 0.4, "cnn2": 0.25, "fm": 0.35}),
                    (("cnn", "cnn2", "cnn3", "fm"), {"cnn": 0.3143, "cnn2": 0.2357, "cnn3": 0.1, "fm": 0.35})):
        parts = {f: rng.dirichlet(np.ones(3) * 0.8, 60) for f in fams}
        m = w13.StreamModel.__new__(w13.StreamModel)
        m.__dict__["fam_w"] = w
        m.__dict__["fam_prior"] = {f: w13.SinkhornCorrector(10, 3, 3, 0.95, starts[f]) for f in fams}
        def run(mm):
            out = []
            for t in range(60):
                pl = [parts["cnn"][t], parts["cnn2"][t] if "cnn2" in fams else None, parts["fm"][t] if "fm" in fams else None]
                if "cnn3" in fams:
                    pl.append(parts["cnn3"][t])
                out.append(mm._fam_decode(pl))
            return np.stack(out)
        got = run(m)
        ref = sum(w[f] * R.causal_sinkhorn(parts[f], keys, 10, 3, 0.95, starts[f]) for f in fams)
        assert np.abs(got - ref).max() < 1e-12, np.abs(got - ref).max()
        assert np.allclose(got.sum(1), 1.0) if abs(sum(w.values()) - 1) < 1e-9 else True
        for c in m.fam_prior.values():
            c.reset()
        assert np.array_equal(run(m), got), "reset must restore the per-family correctors"
    src = (SUB / "t2p_submission_w13.py").read_text()
    assert src.count("if self.fam_prior is not None:") == 3 and src.count("elif self.prior is not None:") == 2, "W12's single corrector stays the else branch"
    assert '"prior", "fam_prior"):' in src, "each stream copy gets fresh per-family correctors"
    print("OK")


if __name__ == "__main__":
    main()
