#!/usr/bin/env python
"""T2-D153: ni26.train weight EMA (Recipe.ema) on CPU, FBCNet B3 + ShallowNet on small random data:
  1. recipe r1 through the edited fit_eval = the pre-D153 fit_eval exactly (state_last and prob_last bitwise; the reference
     is the committed train.py at HEAD~ loaded from git, or --ref <file>);
  2. r1_ema changes state_last and prob_last (the EMA weights are what is returned);
  3. with decay 1e-12 the EMA = the last raw weights (state_last = r1's to 1e-6), i.e. the EMA is updated after every step.
  python tests/test_train_ema.py [--ref <old train.py>]  ->  "OK"
"""

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import train as T  # noqa: E402


def data(seed=0, n=96, C=8, L=480):
    r = np.random.default_rng(seed)
    d = dict(Xtr=r.standard_normal((n, C, L)).astype(np.float32), ytr=r.integers(0, 3, n), Xva=r.standard_normal((24, C, L)).astype(np.float32),
             yva=r.integers(0, 3, 24), cva=np.array(["a"] * 12 + ["b"] * 12), Xte=r.standard_normal((24, C, L)).astype(np.float32),
             yte=r.integers(0, 3, 24), cte=np.array(["c"] * 24))
    d["ytr"][:3] = [0, 1, 2]
    return d


def run(mod, model, recipe):
    r = mod.fit_eval(model, data(), recipe, 0, device="cpu", return_state=True)
    return r["state_last"], r["prob_last"]


def maxd(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(args.ref) if args.ref else Path(tmp) / "train_ref.py"
        if not args.ref:
            ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:src/ni26/train.py"], capture_output=True, text=True,
                                          check=True).stdout)
        spec = importlib.util.spec_from_file_location("ni26.train_ref", ref)
        old = importlib.util.module_from_spec(spec)
        sys.modules["ni26.train_ref"] = old
        spec.loader.exec_module(old)
        assert not hasattr(old.Recipe(), "ema"), "the reference already has Recipe.ema"
        for model in ("fbcnet_bboth", "shallow"):
            r1 = replace(T.RECIPES["r1"], epochs=3)
            s_new, p_new = run(T, model, r1)
            s_old, p_old = run(old, model, old.Recipe("r1", epochs=3))
            assert set(s_new) == set(s_old) and maxd(s_new, s_old) == 0.0 and np.array_equal(p_new, p_old), model
            s_ema, p_ema = run(T, model, replace(T.RECIPES["r1_ema"], epochs=3))
            fk = [k for k, v in s_new.items() if v.is_floating_point()]
            d_ema = maxd({k: s_ema[k] for k in fk}, {k: s_new[k] for k in fk})
            assert d_ema > 1e-4 and not np.allclose(p_ema, p_new), (model, d_ema)
            assert all(torch.equal(s_ema[k], s_new[k]) for k in s_new if k not in fk), "integer buffers != the raw net's"
            s_0, _ = run(T, model, replace(T.RECIPES["r1"], epochs=3, ema=1e-12))
            fl = {k: v for k, v in s_0.items() if v.is_floating_point()}
            d0 = maxd(fl, {k: s_new[k] for k in fl})
            assert d0 <= 1e-6, (model, d0)
            print(f"{model}: r1 = pre-D153 bitwise; r1_ema moves the state by {d_ema:.3e}; decay 1e-12 = the raw last weights "
                  f"({d0:.1e})", flush=True)
    print("OK")


if __name__ == "__main__":
    main()
