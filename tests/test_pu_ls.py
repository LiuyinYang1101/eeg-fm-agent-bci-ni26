#!/usr/bin/env python
"""T2-D226: label smoothing of the per-user copies (Recipe.pu_ls -> ni26.train.peruser_for), CPU:
  1. r1_pulx(x)_vp2_e180_vr_pls2 / _pls3 differ from the vr line in `pu_ls` only; their copies = the vr line's with label_smoothing 0.2 / 0.3;
     every other recipe's copies keep PERUSER's 0.1 (pu_ls default = 0.1);
  2. the pooled training ignores pu_ls: fit_eval(r1 + pu_ls 0.3) = fit_eval(r1) bitwise (FBCNet B3);
  3. finetune(peruser_for(r1 + pu_ls 0.3)) = finetune(PERUSER + LS 0.3) bitwise, and differs from finetune(PERUSER).
  python tests/test_pu_ls.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from ni26 import train as T  # noqa: E402
from test_train_ema import data, maxd  # noqa: E402


def main():
    for f in ("pulxx", "pulx"):
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        for t, ls in (("pls2", 0.2), ("pls3", 0.3)):
            r = T.RECIPES[f"r1_{f}_vp2_e180_vr_{t}"]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "pu_ls"} and r.pu_ls == ls, (f, t)
            assert T.peruser_for(r) == replace(T.peruser_for(b), label_smoothing=ls), (f, t)
    others = [n for n, r in T.RECIPES.items() if r.pu_ls != 0.1]
    assert sorted(others) == sorted(f"r1_{f}_vp2_e180_vr_{t}" for f in ("pulxx", "pulx") for t in ("pls2", "pls3")), others
    assert all(T.peruser_for(r).label_smoothing == 0.1 for n, r in T.RECIPES.items() if n not in others)
    d = data()
    r1 = replace(T.RECIPES["r1"], epochs=2)
    a = T.fit_eval("fbcnet_bboth", d, r1, 0, device="cpu", return_state=True)
    b = T.fit_eval("fbcnet_bboth", d, replace(r1, pu_ls=0.3), 0, device="cpu", return_state=True)
    assert maxd(a["state_last"], b["state_last"]) == 0.0 and np.array_equal(a["prob_last"], b["prob_last"]), "the pooled net must ignore pu_ls"
    net = T.build_model("fbcnet_bboth", d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
    net.load_state_dict(a["state_last"])
    X, y = torch.as_tensor(d["Xtr"][:32]), torch.as_tensor(d["ytr"][:32], dtype=torch.long)
    got = T.finetune(net, X, y, replace(T.peruser_for(replace(T.RECIPES["r1"], pu_ls=0.3)), epochs=2), 5).state_dict()
    want = T.finetune(net, X, y, replace(T.PERUSER, label_smoothing=0.3, epochs=2), 5).state_dict()
    base = T.finetune(net, X, y, replace(T.PERUSER, epochs=2), 5).state_dict()
    assert maxd(got, want) == 0.0 and maxd(got, base) > 1e-7, (maxd(got, want), maxd(got, base))
    print(f"pu_ls: recipes = the vr line + pu_ls; pooled net bitwise unchanged; copies at LS 0.3 = PERUSER + LS 0.3 bitwise, "
          f"{maxd(got, base):.1e} from LS 0.1")
    print("OK")


if __name__ == "__main__":
    main()
