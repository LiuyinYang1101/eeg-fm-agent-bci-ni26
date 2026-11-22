#!/usr/bin/env python
"""T2-D569 (W21 template OPTION, prepared as code only; no build without the owner's typed W21 line) GOLDEN TEST of cfg cnn_lp in
tracks/t2_bci/submission/t2p_submission_w21lp.py (= t2p_submission_w20.py + the option), CPU:
  1. _lp_smooth == the lab's `t2_lab.py lpteach` block (copied verbatim below) on random prefixes (c 60 / 100 / 150, six 1024-d members): 0 diff;
  2. source: w21lp = w20 + ADDED lines only; _cnn_lp_post is called only inside _instream_refit (the CNN-side refits: D539b + P1), never in
     _fm_refit (the FM refits keep the plain teacher); the call is guarded by cnn_lp (absent cfg = W20 bitwise);
  3. fallbacks: fewer than k + 1 windows with features -> the plain teacher (skipped); a broken feature row -> the plain teacher (errors), no raise;
  4. runtime: _lp_smooth at c 150 on 6 x 1024-d features (median of 5) printed, asserted < 50 ms (3 refit points per stream);
  5. LAB vs TEMPLATE on real data: `t2_lab.py lpgolden --set A` replays every (fold, draw, stream, refit) of set A through
     StreamModel._cnn_lp_post with each member's raw pooled features, vs the saved LP root (t2_d571_lpw20_825): max |diff| < 1e-6.
NOT covered (Track3's F1 on d04af48a): StreamModel.predict on the GRAPH path's features (bf16 / CUDA-graph numerics ~1e-3 can flip near-tie
neighbours). Hence the build's gates with cnn_lp ON: K1 / K2 must show "D569 cnn_lp at" lines with run-level smoothed > 0 and errors 0 (an empty
fp_feat or a wiring fault would otherwise ship W20's behaviour as "W21"; F2: the run totals are printed on every refit line).
  python tests/test_t2p_w21lp_cnn_lp.py  ->  "OK"
"""
import difflib
import importlib.util
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")
T20 = ROOT / "tracks/t2_bci/submission/t2p_submission_w20.py"
T21 = ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py"
spec = importlib.util.spec_from_file_location("t2p_w21lp", T21)
TM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TM)


def lab_block(Fr_a, T0, alpha_, kk_):
    """scripts/local/t2_lab.py, MODE lpteach, the per-stream block VERBATIM (Fr[a] -> Fr_a, c = len(T0))."""
    c = len(T0)
    X = Fr_a - Fr_a.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Sm = X @ X.T
    np.fill_diagonal(Sm, -np.inf)
    nn = np.argsort(-Sm, 1, kind="stable")[:, :kk_]
    A_ = np.zeros((c, c))
    rr = np.repeat(np.arange(c), kk_)
    A_[rr, nn.ravel()] = np.maximum(Sm[rr, nn.ravel()], 0)
    A_ = np.maximum(A_, A_.T)
    dg = A_.sum(1) + 1e-9
    Sn = A_ / np.sqrt(dg[:, None] * dg[None, :])
    Fp = np.clip(np.linalg.solve(np.eye(c) - alpha_ * Sn, T0), 1e-9, None)
    return Fp / Fp.sum(1, keepdims=True)


# 1. the function vs the lab block
rng = np.random.default_rng(569)
for c in (60, 100, 150):
    F = np.hstack([x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9) for x in [rng.normal(size=(c, 1024)) for _ in range(6)]])
    T0 = rng.dirichlet(np.ones(3), size=c)
    d = float(np.abs(TM._lp_smooth(F, T0, 0.5, 10) - lab_block(F, T0, 0.5, 10)).max())
    assert d == 0.0, (c, d)
print("1. _lp_smooth == the lab's lpteach block (0 diff at c 60 / 100 / 150)")

# 2. source checks
a, b = T20.read_text().splitlines(), T21.read_text().splitlines()
ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
assert all(op[0] == "insert" for op in ops), [op for op in ops if op[0] != "insert"]
src = T21.read_text()
body = lambda name: src.split(f"    def {name}(")[1].split("\n    def ")[0]
assert body("_instream_refit").count("self._cnn_lp_post(post, c)") == 1 and 'getattr(self, "cnn_lp", None) is not None' in body("_instream_refit")
assert "_cnn_lp_post" not in body("_fm_refit") and "_lp_smooth" not in body("_fm_refit")
assert src.count("self._cnn_lp_post(") == 1, "called once, in _instream_refit"
assert body("_fm_refit") == T20.read_text().split("    def _fm_refit(")[1].split("\n    def ")[0], "the FM refit is W20's"
print(f"2. source: {sum(op[4] - op[3] for op in ops)} added lines only; LP on the CNN-side path only; _fm_refit == W20's")

# 3. fallbacks
mk = lambda fp: types.SimpleNamespace(cnn_lp=dict(alpha=0.5, k=10), cnn_lp_stats=dict(refits=0, smoothed=0, skipped=0, errors=0), fp_feat=fp)
post = rng.dirichlet(np.ones(3), size=60)
s1 = mk({i: [rng.normal(size=1024) for _ in range(6)] for i in range(10)})          # 10 windows with features <= k
assert np.array_equal(TM.StreamModel._cnn_lp_post(s1, post, 60), post) and s1.cnn_lp_stats["skipped"] == 1
fp = {i: [rng.normal(size=1024) for _ in range(6)] for i in range(60)}
fp[7] = [rng.normal(size=999)] + fp[7][1:]                                        # a broken (ragged) feature row -> the stack raises
s2 = mk(fp)
assert np.array_equal(TM.StreamModel._cnn_lp_post(s2, post, 60), post) and s2.cnn_lp_stats["errors"] == 1
s3 = mk({i: [rng.normal(size=1024) for _ in range(6)] for i in range(60) if i % 7})  # windows without features keep the plain teacher
o3 = TM.StreamModel._cnn_lp_post(s3, post, 60)
assert s3.cnn_lp_stats["smoothed"] == 1 and np.array_equal(o3[::7], post[::7]) and not np.array_equal(o3, post)
print("3. fallbacks: skipped / errors -> the plain teacher, no raise; windows without features keep the plain teacher")

# 4. runtime
F = np.hstack([x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9) for x in [rng.normal(size=(150, 1024)) for _ in range(6)]])
T0 = rng.dirichlet(np.ones(3), size=150)
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    TM._lp_smooth(F, T0, 0.5, 10)
    ts.append(time.perf_counter() - t0)
ms = 1000 * float(np.median(ts))
assert ms < 50, ms
print(f"4. runtime: _lp_smooth at c 150 (6 x 1024-d) {ms:.2f} ms median; x 3 refit points per stream (+ the per-window feature concat)")

# 5. lab vs template on real data
O = Path.home() / "workspace/outputs/neural_interface_challenge"
fb, lp = O / "t2_d571_fbw20_825/teacher_p1", O / "t2_d571_lpw20_825/teacher_p1"
assert fb.is_dir() and lp.is_dir(), "needs the D569 / D571 set-A roots"
env = {k: v for k, v in os.environ.items() if not k.startswith("NI26_")}
env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
r = subprocess.run([sys.executable, "-u", "scripts/local/t2_lab.py", "lpgolden", "--set", "A", "--fb", str(fb), "--lp", str(lp)],
                   cwd=ROOT, env=env, capture_output=True, text=True)
line = [x for x in r.stdout.splitlines() if "D569 GOLDEN" in x]
assert r.returncode == 0 and line, (r.returncode, r.stdout[-1500:], r.stderr[-1500:])
print("5. " + line[0].replace("[lab] ", ""))
print("OK")
