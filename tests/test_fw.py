#!/usr/bin/env python
"""T2-D229: the "virtual IAF" warp of the pooled training windows (Recipe.fw -> ni26.train.augment -> fw_bank), CPU:
  1. r1_pulx(x)_vp2_e180_vr_fw05 / _fw10 differ from the vr line in `fw` only; their per-user copies are the vr line's (fw 0); only these four
     recipes warp;
  2. fw_matrix = T2-D191's t2p_bench.iaf_warp (x @ R vs iaf_warp(x), stretched and compressed, max |diff| < 1e-5); n_new = L = the identity;
     the bank's factors are log-symmetric within [1 / (1 + fw), 1 + fw], the middle one the identity; a 10 Hz sine warped by a = 1.1 peaks at 10 / 1.1 Hz;
  3. fit_eval with fw 0 = r1 bitwise; with fw 0.1 the weights move, the BatchNorm step counter matches (FBCNet B3 and ShallowNet).
  python tests/test_fw.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks" / "t2_bci" / "scripts"))
from ni26 import train as T  # noqa: E402
import t2p_bench as B  # noqa: E402


def data(seed=0, C=41, L=480):
    rng = np.random.default_rng(seed)
    d = {}
    for k, m in dict(tr=48, va=12, te=12).items():
        d[f"X{k}"] = rng.standard_normal((m, C, L)).astype(np.float32)
        d[f"y{k}"] = rng.integers(0, 3, m)
        d[f"c{k}"] = np.array(["a"] * m)
    return d


def main():
    for f in ("pulxx", "pulx"):
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        for t, fw in (("fw05", 0.05), ("fw10", 0.10)):
            r = T.RECIPES[f"r1_{f}_vp2_e180_vr_{t}"]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "fw"} and r.fw == fw, (f, t)
            assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).fw == 0, "the copies must be unchanged"
    on = sorted(n for n, r in T.RECIPES.items() if r.fw)
    assert on == sorted(f"r1_{f}_vp2_e180_vr_{t}" for f in ("pulxx", "pulx") for t in ("fw05", "fw10")), on
    L = 480
    x = np.random.default_rng(1).standard_normal((3, 5, L)).astype(np.float32)
    for n_new in (430, 457, 480, 503, 528):
        got = np.einsum("ncl,lm->ncm", x.astype(np.float64), T.fw_matrix(L, n_new))
        want = B.iaf_warp(x, np.full(3, 10.0 * n_new / L))
        assert np.abs(got - want).max() < 1e-5, (n_new, np.abs(got - want).max())
    assert np.array_equal(T.fw_matrix(L, L), np.eye(L))
    for fw in (0.05, 0.10):
        R = T.fw_bank(L, fw, "cpu")
        a = np.exp(np.linspace(-np.log1p(fw), np.log1p(fw), T.FW_K))
        assert R.shape == (T.FW_K, L, L) and torch.equal(R[T.FW_K // 2], torch.eye(L)) and np.allclose(a * a[::-1], 1.0)
        assert a.min() >= 1 / (1 + fw) - 1e-12 and a.max() <= 1 + fw + 1e-12
        print(f"fw {fw}: {T.FW_K} factors {a.min():.4f}..{a.max():.4f}, n_new {round(L * a.min())}..{round(L * a.max())}")
    fs, t = 120.0, np.arange(L) / 120.0
    s = np.sin(2 * np.pi * 10.0 * t) @ T.fw_matrix(L, int(round(L * 1.1)))
    fr = np.fft.rfftfreq(4 * L, 1 / fs)[np.argmax(np.abs(np.fft.rfft(s * np.hanning(L), 4 * L)))]
    assert abs(fr - 10 / 1.1) < 0.1, fr
    print(f"warp a 1.1: a 10 Hz sine -> {fr:.2f} Hz (10 / 1.1 = {10 / 1.1:.2f}); fw_matrix = iaf_warp (max |diff| < 1e-5)")
    d = data()
    plain = replace(T.RECIPES["r1"], epochs=2)
    for m in ("fbcnet_bboth", "shallow"):
        a0 = T.fit_eval(m, d, plain, 0, device="cpu", return_state=True)
        b0 = T.fit_eval(m, d, replace(plain, fw=0.0), 0, device="cpu", return_state=True)
        c0 = T.fit_eval(m, d, replace(plain, fw=0.1), 0, device="cpu", return_state=True)
        sa, sb, sc = a0["state_last"], b0["state_last"], c0["state_last"]
        assert all(torch.equal(sa[k], sb[k]) for k in sa) and np.array_equal(a0["prob_last"], b0["prob_last"]), (m, "fw 0 must be r1 bitwise")
        moved = max(float((sa[k] - sc[k]).abs().max()) for k in sa if sa[k].is_floating_point())
        assert moved > 1e-4 and all(torch.equal(sa[k], sc[k]) for k in sa if "num_batches" in k), (m, moved)
        print(f"{m}: fw 0 = r1 bitwise; fw 0.1 moves the weights by up to {moved:.1e}")
    print("OK")


if __name__ == "__main__":
    main()
