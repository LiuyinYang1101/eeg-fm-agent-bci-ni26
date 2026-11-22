#!/usr/bin/env python
"""T2-D594b (a set-valued loss for the low-margin MI / WORD P1 rows), CPU:
  1. set_ce: a row with the set {MI, WORD} = -log(p_MI + p_WORD); a singleton row = the plain CE with label smoothing (closed forms);
  2. finetune without yset = bitwise the pre-change copy (the PIN of tests/test_t2_p1_mix.py); with an all-singleton yset = the plain copy
     (<= 1e-6: reduction 'none' + mean vs mean); with set rows it runs, is finite and moves the copy; yset with ft_mix is refused;
  3. t2p_bench.setval_mask on hand-set posteriors: top-2 {MI, WORD} with a margin < 0.20 -> {MI, WORD}; a wider margin, or a top-2 with SUB,
     -> the row's own label.
  python tests/test_t2_p1_setval.py  ->  "OK"
"""
import hashlib
import math
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import torch  # noqa: E402
from ni26 import train as T  # noqa: E402

torch.set_num_threads(1)
PIN = "e3f7d9d1898664a3ce8d7ffbee8ba06e3153dcbe9b0f9d1c396c3282dd6a0947"   # = tests/test_t2_p1_mix.py's (the plain copy, before D589 / D594b)


def data():
    rng = np.random.default_rng(589)
    X = torch.as_tensor(rng.standard_normal((48, 41, 480)).astype(np.float32))
    y = torch.as_tensor(rng.integers(0, 3, 48))
    return X, y


def copy_probs(rec, yset=None, seed=1234):
    X, y = data()
    torch.manual_seed(0)
    net = T.build_model("shallow", 41, 3, 480, 120.0)
    m = T.finetune(net, X, y, replace(rec, epochs=2), seed, **({"yset": yset} if yset is not None else {}))
    with torch.no_grad():
        return torch.softmax(m(X).float(), 1).numpy().astype(np.float32)


def main():
    lo = torch.tensor([[1.0, 0.0, 0.5], [0.2, 1.5, -0.3]])
    y = torch.tensor([0, 1])
    ys = torch.tensor([[True, False, True], [False, True, False]])
    p0 = torch.softmax(lo[0], 0)
    want0 = -math.log(float(p0[0] + p0[2]))
    want1 = float(torch.nn.functional.cross_entropy(lo[1:], y[1:], label_smoothing=0.1))
    got = float(T.set_ce(lo, y, ys, 0.1))
    assert abs(got - (want0 + want1) / 2) < 1e-6, (got, want0, want1)
    print(f"1. set_ce: the set row -log(p_MI + p_WORD) {want0:.5f}, the singleton row the plain CE {want1:.5f}, mean {got:.5f}")
    rec = T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr"])
    p_plain = copy_probs(rec)
    assert hashlib.sha256(p_plain.tobytes()).hexdigest() == PIN, "the plain copy moved"
    X, y = data()
    one = torch.zeros((48, 3), dtype=torch.bool)
    one[torch.arange(48), y] = True
    p_one = copy_probs(rec, one)
    assert float(np.abs(p_one - p_plain).max()) < 1e-5, float(np.abs(p_one - p_plain).max())
    sets = one.clone()
    sets[:16, 0] = sets[:16, 2] = True
    sets[torch.arange(16), y[:16]] = True
    p_set = copy_probs(rec, sets)
    assert np.isfinite(p_set).all() and float(np.abs(p_set - p_plain).max()) > 1e-4
    try:
        T.finetune(T.build_model("shallow", 41, 3, 480, 120.0), X, y, replace(rec, epochs=1, ft_mix=0.4), 1, yset=sets)
        raise SystemExit("yset with ft_mix was not refused")
    except AssertionError:
        pass
    print(f"2. no yset = the PIN; all-singleton yset = the plain copy (max |dp| {float(np.abs(p_one - p_plain).max()):.1e}); 16 set rows move the "
          f"copy (max |dp| {float(np.abs(p_set - p_plain).max()):.3f}); yset + ft_mix refused")
    import t2p_bench as B
    pp = np.array([[0.45, 0.10, 0.45], [0.50, 0.05, 0.45], [0.70, 0.05, 0.25], [0.40, 0.45, 0.15], [0.30, 0.10, 0.60]])
    m = B.setval_mask(pp, pp.argmax(1), 0.20)
    assert m[0].tolist() == [True, False, True] and m[1].tolist() == [True, False, True]
    assert m[2].tolist() == [True, False, False] and m[3].tolist() == [False, True, False] and m[4].tolist() == [False, False, True]
    print("3. setval_mask: {MI, WORD} for the low-margin MI / WORD rows only; wider margins and SUB top-2 rows keep their label")
    print("OK")


if __name__ == "__main__":
    main()
