#!/usr/bin/env python
"""T2-D225: the pooled batch-size recipes (CPU):
  1. r1_pulx(x)_vp2_e180_vr_bs32 / _bs128 differ from the vr line in `bs` only; their per-user copies are the vr line's (PERUSER, bs 32);
  2. no epoch of the kit / seen1 pooled training sets (1521 / 1325 windows) ends on a batch of one (BatchNorm in train mode);
  3. fit_eval runs at bs 32 and 128 for FBCNet B3 and ShallowNet, and bs 64 = the vr line's batch = r1 bitwise.
  python tests/test_bs.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402


def data(seed=0, C=41, L=480):
    rng = np.random.default_rng(seed)
    d = {}
    for k, m in dict(tr=70, va=12, te=12).items():
        d[f"X{k}"] = rng.standard_normal((m, C, L)).astype(np.float32)
        d[f"y{k}"] = rng.integers(0, 3, m)
        d[f"c{k}"] = np.array(["a"] * m)
    return d


def main():
    for f in ("pulxx", "pulx"):
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        assert b.bs == 64
        for bs in (32, 128):
            r = T.RECIPES[f"r1_{f}_vp2_e180_vr_bs{bs}"]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "bs"} and r.bs == bs, (f, bs)
            assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).bs == 32, "the copies must be unchanged"
            assert all(n % bs != 1 for n in (1521, 1325)), (bs, "a batch of one")
    d = data()
    plain = replace(T.RECIPES["r1"], epochs=2)
    a = T.fit_eval("shallow", d, plain, 0, device="cpu", return_state=True)
    b = T.fit_eval("shallow", d, replace(plain, bs=64), 0, device="cpu", return_state=True)
    assert all(torch.equal(a["state_last"][k], b["state_last"][k]) for k in a["state_last"]), "bs 64 must be r1 bitwise"
    for m in ("fbcnet_bboth", "shallow"):
        for bs in (32, 128):
            r = T.fit_eval(m, d, replace(plain, bs=bs), 0, device="cpu", return_state=True)
            nb = int(r["state_last"][[k for k in r["state_last"] if "num_batches" in k][0]])
            assert nb == 2 * int(np.ceil(70 / bs)), (m, bs, nb)        # BatchNorm counted one step per batch: the loop used bs
            print(f"{m} bs {bs}: {nb} steps in 2 epochs of 70 windows")
    print("OK")


if __name__ == "__main__":
    main()
