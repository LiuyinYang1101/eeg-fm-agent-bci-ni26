#!/usr/bin/env python
"""T2-D584 (confidence x class-balance weighted refits on every prefix window), CPU, no data:
  1. d584_weights: label = argmax; weight = max p x min(c / (3 n_yhat), 3), mean 1; the cap 3 binds for a 1-window class among 30;
  2. the weighted head fit with all-ones weights equals the unweighted fit_head (same optimum, 1e-6), and sw=None IS the old path;
  3. zero weights drop rows exactly: weights n/m on m kept rows and 0 on the rest == fit_head on the kept rows alone (1e-6);
  4. source: the lab's fm_feedback passes the same weights to the side-head refit, the MI main-head refit and the prototype means, and
     t2_lab pairs NI26_LAB_D584=1 with --idea w20_fb_wref (one arm, never half of it).
  python tests/test_t2_d584_weights.py  ->  "OK"
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import ast  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
RSRC = (ROOT / "scripts/local/t2_d540a_stack_read.py").read_text()     # the reader runs its reads at import: exec only the functions under test
_tree = ast.parse(RSRC)
R = types.SimpleNamespace()
_ns = {"np": np, "minimize": minimize}
for _n in _tree.body:
    if isinstance(_n, ast.FunctionDef) and _n.name in ("d584_weights", "fit_head", "_fit_head_w"):
        exec(compile(ast.Module([_n], []), "t2_d540a_stack_read.py", "exec"), _ns)
R.d584_weights, R.fit_head = _ns["d584_weights"], _ns["fit_head"]


def main():
    rng = np.random.default_rng(584)
    # 1. weights
    c = 30
    pp = rng.dirichlet(np.ones(3), c)
    pp[:, 2] = 0.0
    pp[0] = [0.1, 0.1, 0.8]                                   # ONE window predicted class 2
    pp = pp / pp.sum(1, keepdims=True)
    ya, w = R.d584_weights(pp)
    assert np.array_equal(ya, pp.argmax(1)) and abs(w.mean() - 1) < 1e-12 and (w > 0).all()
    n_ = np.bincount(ya, minlength=3)
    raw = pp.max(1) * np.minimum(c / (3.0 * n_[ya]), 3.0)
    assert np.allclose(w, raw / raw.mean()) and min(c / (3.0 * n_[2]), 3.0) == 3.0, n_   # 30 / 3 = 10 -> capped at 3
    print(f"1. weights: mean 1, label = argmax, the cap binds for a 1-window class (counts {n_.tolist()})")
    # 2. all-ones weights == the unweighted fit; sw=None is the old code path
    n, D = 40, 5
    Ls, Ss = rng.standard_normal((n, 3)), rng.standard_normal((n, D))
    ys = rng.integers(0, 3, n)
    W0, b0 = 0.3 * rng.standard_normal((3, D)), 0.1 * rng.standard_normal(3)
    Wa, ba, oka = R.fit_head(Ls, Ss, ys, W0, b0, 1.0)
    Wb, bb, okb = R.fit_head(Ls, Ss, ys, W0, b0, 1.0, sw=np.ones(n))
    assert oka and okb and np.abs(Wa - Wb).max() < 1e-6 and np.abs(ba - bb).max() < 1e-6, (np.abs(Wa - Wb).max(), np.abs(ba - bb).max())
    Wc, bc, _ = R.fit_head(Ls, Ss, ys, W0, b0, 1.0, sw=None)
    assert np.array_equal(Wa, Wc) and np.array_equal(ba, bc)
    print(f"2. all-ones weights == fit_head (max |dW| {np.abs(Wa - Wb).max():.1e}); sw=None is the unweighted path")
    # 3. zero weights drop rows
    keep = rng.random(n) < 0.6
    m = int(keep.sum())
    sw = np.where(keep, n / m, 0.0)
    Wk, bk, _ = R.fit_head(Ls[keep], Ss[keep], ys[keep], W0, b0, 1.0)
    Wz, bz, _ = R.fit_head(Ls, Ss, ys, W0, b0, 1.0, sw=sw)
    assert np.abs(Wk - Wz).max() < 1e-6 and np.abs(bk - bz).max() < 1e-6, np.abs(Wk - Wz).max()
    print(f"3. weights n/m on {m} kept rows + 0 elsewhere == the fit on the kept rows (max |dW| {np.abs(Wk - Wz).max():.1e})")
    # 4. source checks
    isrc = (ROOT / "scripts/local/t2_lab_ideas.py").read_text()
    src = ast.get_source_segment(isrc, next(n for n in ast.parse(isrc).body if isinstance(n, ast.FunctionDef) and n.name == "fm_feedback"))
    assert src.count("sw=wv") == 2 and "ctx.d584_weights(q)" in src and "wv[ys == q_] @ fs_[ys == q_]" in src, "fm_feedback's D584 wiring"
    assert "float(map_tau) * Wn" in src and 'dm["Wm"] / (np.linalg.norm(dm["Wm"], axis=1, keepdims=True)' in src, "fm_feedback's D585 form"
    lab = (ROOT / "scripts/local/t2_lab.py").read_text()
    assert '(_os0.environ.get("NI26_LAB_D584") == "1") == (ARGS.get("--idea") == "w20_fb_wref")' in lab
    rsrc = ast.get_source_segment(RSRC, next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "side_prob_h"))
    assert "d584_weights(pp[fin])" in rsrc and "sw=wv" in rsrc
    print("4. one weight vector for the side heads, the MI refit and the prototype means; the reader env and the lab idea are one arm")
    print("OK")


if __name__ == "__main__":
    main()
