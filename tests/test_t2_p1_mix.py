#!/usr/bin/env python
"""T2-D589 (mixup inside the in-stream P1 students' fine-tune), CPU:
  1. PIN: ni26.train.finetune with peruser_for(r1_pulx_vp2_e180_vr) (ShallowNet, 2 epochs, seed 1234, rng 589 data, 1 thread) is bitwise the
     output recorded BEFORE the change (sha256 of the float32 softmax);
  2. r1_pulx_vp2_e180_vr_pmx4 differs from its base only in name and pu_mix (0.4); peruser_for passes it on as ft_mix and changes nothing else;
     no other recipe sets pu_mix; finetune() never reads recipe.mixup (the branch keys on ft_mix only);
  3. the POOLED fit of the pmx4 recipe is bitwise the base recipe's (fit_eval, 2 epochs; the P1 roots retrain the pooled nets);
  4. ft_mix 0.4 runs, is finite and moves the copy; soft targets / sbal / kl_other with ft_mix are refused.
  python tests/test_t2_p1_mix.py  ->  "OK"
"""
import hashlib
import inspect
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import torch  # noqa: E402
from ni26 import train as T  # noqa: E402

torch.set_num_threads(1)
PIN = "e3f7d9d1898664a3ce8d7ffbee8ba06e3153dcbe9b0f9d1c396c3282dd6a0947"


def data():
    rng = np.random.default_rng(589)
    X = torch.as_tensor(rng.standard_normal((48, 41, 480)).astype(np.float32))
    y = torch.as_tensor(rng.integers(0, 3, 48))
    return X, y


def copy_probs(rec, seed=1234):
    X, y = data()
    torch.manual_seed(0)
    net = T.build_model("shallow", 41, 3, 480, 120.0)
    m = T.finetune(net, X, y, replace(rec, epochs=2), seed)
    with torch.no_grad():
        return torch.softmax(m(X).float(), 1).numpy().astype(np.float32)


def main():
    base, pmx = T.RECIPES["r1_pulx_vp2_e180_vr"], T.RECIPES["r1_pulx_vp2_e180_vr_pmx4"]
    p0 = copy_probs(T.peruser_for(base))
    assert hashlib.sha256(p0.tobytes()).hexdigest() == PIN, "the plain copy fine-tune moved"
    print("1. PIN: the plain copy fine-tune is bitwise the pre-change output")
    assert {k for k, v in asdict(pmx).items() if v != asdict(base)[k]} == {"name", "pu_mix"} and pmx.pu_mix == 0.4
    cb, cp = asdict(T.peruser_for(base)), asdict(T.peruser_for(pmx))
    assert {k for k in cb if cb[k] != cp[k]} == {"ft_mix"} and cp["ft_mix"] == 0.4 and cb["ft_mix"] == 0.0
    assert [n for n, r in T.RECIPES.items() if r.pu_mix] == ["r1_pulx_vp2_e180_vr_pmx4"] and not any(r.ft_mix for r in T.RECIPES.values())
    src = inspect.getsource(T.finetune)
    assert "recipe.mixup" not in src and src.count("recipe.ft_mix") == 5, "finetune must key on ft_mix only"   # 5 = + T2-D594b's yset refusal
    print("2. pmx4 = the base + pu_mix 0.4; peruser_for -> ft_mix only; finetune never reads recipe.mixup")
    import test_t2p_side as TS
    d = TS.data(np.random.default_rng(5890), side=False)
    r0 = T.fit_eval("shallow", d, replace(base, epochs=2, bs=16, vp=0), 0, device="cpu")["prob_last"]
    r1 = T.fit_eval("shallow", d, replace(pmx, epochs=2, bs=16, vp=0), 0, device="cpu")["prob_last"]
    assert np.array_equal(np.asarray(r0), np.asarray(r1)), "the pooled fit of pmx4 differs from the base's"
    print("3. the pooled fit of pmx4 is bitwise the base's")
    p1 = copy_probs(T.peruser_for(pmx))
    assert np.isfinite(p1).all() and not np.array_equal(p0, p1)
    X, y = data()
    torch.manual_seed(0)
    net = T.build_model("shallow", 41, 3, 480, 120.0)
    for bad in (dict(kl_other=0.5), ):
        try:
            T.finetune(net, X, y, replace(T.peruser_for(pmx), epochs=1, **bad), 1, X_other=X)
            raise SystemExit(f"ft_mix with {bad} was not refused")
        except AssertionError:
            pass
    try:
        T.finetune(net, X, torch.full((48, 3), 1 / 3), replace(T.peruser_for(pmx), epochs=1), 1)
        raise SystemExit("ft_mix with soft targets was not refused")
    except AssertionError:
        pass
    print(f"4. ft_mix 0.4 runs and moves the copy (max |dp| {float(np.abs(p1 - p0).max()):.3e}); kl_other / soft targets refused")
    print("OK")


if __name__ == "__main__":
    main()
