#!/usr/bin/env python
"""T2-D304 (SKλ twin): the forgetting causal Sinkhorn in the reference and the template, CPU:
  1. t2p_report.causal_sinkhorn(lam 1.0) = the default call bitwise; lam 0.95 = T2-D304's sk_lambda (the read's decoder) on random streams;
  2. the template's SinkhornCorrector streamed window by window: lam absent = lam 1.0 bitwise; = causal_sinkhorn at lam 1.0 and 0.95 (1e-12);
     cfg tta {"kind": "sinkhorn", ..., "lam": 0.95} reaches the corrector.
  python tests/test_sk_lambda.py  ->  "OK"
"""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "scripts/local")); sys.path.insert(0, str(ROOT / "src"))
import t2p_report as R  # noqa: E402
from t2_d304_skl import sk_lambda  # noqa: E402

bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
bs.CompetSolver = object; bu.base_solver = bs
sys.modules.update({"benchmark_utils": bu, "benchmark_utils.base_solver": bs})


def main():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3) * 0.7, 60)
    keys = (np.zeros(60), np.arange(60))
    a, b = R.causal_sinkhorn(P, keys, 10, 3), R.causal_sinkhorn(P, keys, 10, 3, 1.0)
    assert np.array_equal(a, b), "lam 1.0 must be the default bitwise"
    for lam in (0.95, 0.9):
        ref = R.causal_sinkhorn(P, keys, 10, 3, lam)
        got = sk_lambda(P[None], 10, lam)[0]
        assert np.allclose(ref, got, atol=1e-12, rtol=0), (lam, float(np.abs(ref - got).max()))
    assert not np.allclose(R.causal_sinkhorn(P, keys, 10, 3, 0.95), a, atol=1e-6), "lam 0.95 must change the output"
    print("causal_sinkhorn: lam 1.0 = default bitwise; lam 0.95 / 0.90 = T2-D304's sk_lambda (1e-12)")

    ns = {"__name__": "t2p_sub_skl"}
    exec(compile((ROOT / "tracks/t2_bci/submission/t2p_submission.py").read_text(), "t2p_submission.py", "exec"), ns)
    SC = ns["SinkhornCorrector"]
    for lam, ref in ((None, a), (1.0, a), (0.95, R.causal_sinkhorn(P, keys, 10, 3, 0.95))):
        c = SC(10, 3, 3) if lam is None else SC(10, 3, 3, lam)
        out = np.stack([c(p) for p in P])
        assert np.allclose(out, ref, atol=1e-12, rtol=0), (lam, float(np.abs(out - ref).max()))
    c1, c2 = SC(10, 3, 3), SC(10, 3, 3, 1.0)
    assert all(np.array_equal(c1(p), c2(p)) for p in P), "lam absent = lam 1.0 bitwise"
    print("template SinkhornCorrector: lam absent = 1.0 bitwise; = causal_sinkhorn at lam 1.0 / 0.95 (1e-12)")

    import torch
    sys.path.insert(0, str(ROOT / "src"))
    import ni26.train as T
    C, L = 8, 480
    nets = [T.build_model("shallow", C, 3, L, 120.0).eval()]
    cfg = dict(ch_names=[f"E{i}" for i in range(C)], sfreq=120.0, n_times=L, n_classes=3, align=dict(mode="none"),
               tta=dict(kind="sinkhorn", iters=10, n0=3, lam=0.95))
    m = ns["StreamModel"](nets, cfg, dict(ch_names=cfg["ch_names"], sfreq=120.0), ()).eval()
    assert isinstance(m.prior, SC) and m.prior.lam == 0.95
    cfg2 = dict(cfg, tta=dict(kind="sinkhorn", iters=10, n0=3))
    assert ns["StreamModel"](nets, cfg2, dict(ch_names=cfg["ch_names"], sfreq=120.0), ()).prior.lam == 1.0
    _ = m.predict(torch.randn(1, C, L))
    print("cfg tta.lam reaches the corrector (absent = 1.0); predict runs")
    print("OK")


if __name__ == "__main__":
    main()
