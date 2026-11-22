#!/usr/bin/env python
"""T2-D580 (L2: joint mixup for the CNN side nets), CPU:
  1. PIN (Track3 F3): the plain side recipe's fixed-seed output is bitwise the one recorded BEFORE the side_mix change (train.py at b5307eff):
     ShallowNet, r1_pulx_vp2_e180_vr with epochs 3 / bs 16 / vp 0 (no VP bank in a unit test; the side branch is the same code), seed 0, on
     test_t2p_side.data(rng 580, side=True) + 0.1 noise -> sha256 of prob_last (float32) == PIN;
  2. the new recipe r1_pulx_vp2_e180_vr_smx4 differs from its base ONLY in name and side_mix (0.4); its per-user copies are unchanged; every
     other recipe has side_mix 0;
  3. side_mix > 0 runs, is finite and changes the output (the plain branch's own learning test is tests/test_t2p_side.py);
  4. source: the mixed branch mixes x and the side vector with ONE pairing j and ONE lambda, and the labels with the same pair.
  python tests/test_t2_side_mix.py  ->  "OK"
"""
import hashlib
import inspect
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # Track3 F1 on de640708: the PIN is a 1-thread result
    os.environ[_v] = "1"
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import torch  # noqa: E402
from ni26 import train as T  # noqa: E402

torch.set_num_threads(1)
import test_t2p_side as TS  # noqa: E402

PIN = "5e8b1ec86886e81caa7394d936562c6951ba97c5e2f0efc4a1d4f73bb8afc715"


def pin_data():
    rng = np.random.default_rng(580)
    d = TS.data(rng, side=True)
    d["Xtr"] += 0.1 * rng.standard_normal(d["Xtr"].shape).astype(np.float32)
    return d


def main():
    rec = replace(T.RECIPES["r1_pulx_vp2_e180_vr"], epochs=3, bs=16, vp=0)
    p = np.asarray(T.fit_eval("shallow", pin_data(), rec, 0, device="cpu")["prob_last"], np.float32)
    h = hashlib.sha256(p.tobytes()).hexdigest()
    assert h == PIN, ("the plain side recipe's output moved", h)
    print("1. the plain side recipe is bitwise the pre-change output (PIN)")
    b, r = T.RECIPES["r1_pulx_vp2_e180_vr"], T.RECIPES["r1_pulx_vp2_e180_vr_smx4"]
    assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "side_mix"} and r.side_mix == 0.4, asdict(r)
    assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).side_mix == 0
    assert [n for n, x in T.RECIPES.items() if x.side_mix] == ["r1_pulx_vp2_e180_vr_smx4"]
    print("2. r1_pulx_vp2_e180_vr_smx4 = the side line + side_mix 0.4 only; copies unchanged; no other recipe mixes")
    rm = replace(rec, side_mix=0.4)
    pm = np.asarray(T.fit_eval("shallow", pin_data(), rm, 0, device="cpu")["prob_last"], np.float32)
    assert np.isfinite(pm).all() and not np.array_equal(pm, p)
    print(f"3. side_mix 0.4 runs, is finite and moves the output (max |dp| {float(np.abs(pm - p).max()):.3e})")
    src = inspect.getsource(T.fit_eval)
    blk = src.split("if recipe.side_mix > 0:")[1].split("else:")[0]
    assert "lam * xb + (1 - lam) * xb[j]" in blk and "lam * Sb + (1 - lam) * Sb[j]" in blk and "yb[j]" in blk and blk.count("randperm") == 1, blk
    print("4. one pairing j and one lambda for x, the side vector and the labels")
    print("OK")


if __name__ == "__main__":
    main()
