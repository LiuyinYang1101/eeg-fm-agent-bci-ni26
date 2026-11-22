#!/usr/bin/env python
"""T2-D228: partial fine-tuning of the per-user copies (Recipe.pu_train -> peruser_for -> Recipe.train_part -> finetune), CPU, FBCNet B3 and ShallowNet:
  1. r1_pulx(x)_vp2_e180_vr_phead / _pfeat differ from the vr line in `pu_train` only; their copies = the vr line's with train_part head / feat;
     every other recipe's copies train every part; the pooled fit_eval refuses train_part;
  2. finetune with train_part "all" = PERUSER bitwise;
  3. "head": every parameter outside final_layer.* = the pooled net's bitwise and final_layer moves; "feat": final_layer.* = the pooled net's
     bitwise and the rest moves; in both, BatchNorm running statistics follow the user's windows as in "all" (they differ from the pooled net's);
     the pooled net itself is never modified (requires_grad stays True on it).
  python tests/test_pu_part.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from ni26 import train as T  # noqa: E402
from test_train_ema import data, maxd  # noqa: E402


def main():
    for f in ("pulxx", "pulx"):
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        for t in ("head", "feat"):
            r = T.RECIPES[f"r1_{f}_vp2_e180_vr_p{t}"]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "pu_train"} and r.pu_train == t, (f, t)
            assert T.peruser_for(r) == replace(T.peruser_for(b), train_part=t), (f, t)
    on = sorted(n for n, r in T.RECIPES.items() if T.peruser_for(r).train_part != "all")
    assert on == sorted(f"r1_{f}_vp2_e180_vr_p{t}" for f in ("pulxx", "pulx") for t in ("head", "feat")), on
    assert all(r.train_part == "all" for r in T.RECIPES.values()), "train_part is set through pu_train only"
    d = data()
    try:
        T.fit_eval("shallow", d, replace(T.RECIPES["r1"], epochs=1, train_part="head"), 0, device="cpu")
        raise SystemExit("fit_eval must refuse train_part")
    except AssertionError:
        pass
    X, y = torch.as_tensor(d["Xtr"][:48]), torch.as_tensor(d["ytr"][:48], dtype=torch.long)
    pu = replace(T.PERUSER, epochs=3)
    for m in ("fbcnet_bboth", "shallow"):
        r1 = T.fit_eval(m, d, replace(T.RECIPES["r1"], epochs=2), 0, device="cpu", return_state=True)
        net = T.build_model(m, d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
        net.load_state_dict(r1["state_last"])
        base = {k: v.clone() for k, v in net.state_dict().items()}
        grad0 = {k: p.requires_grad for k, p in net.named_parameters()}
        plain = T.finetune(net, X, y, pu, 5).state_dict()
        assert maxd(T.finetune(net, X, y, replace(pu, train_part="all"), 5).state_dict(), plain) == 0.0
        pnames = {k for k, _ in net.named_parameters()}
        for part in ("head", "feat"):
            got = T.finetune(net, X, y, replace(pu, train_part=part), 5).state_dict()
            head = {k for k in pnames if k.startswith("final_layer.")}
            moves = head if part == "head" else {k for k in pnames - head if grad0[k]}
            keeps = pnames - moves
            assert head and moves and all(torch.equal(got[k], base[k]) for k in keeps), (m, part, "a frozen parameter moved")
            dm = max(float((got[k] - base[k]).abs().max()) for k in moves)
            stats = [k for k in got if "running_" in k]
            ds = max(float((got[k] - base[k]).abs().max()) for k in stats)
            assert dm > 1e-5 and ds > 1e-5, (m, part, dm, ds)
            print(f"{m} {part}: {len(keeps)} parameter tensors frozen bitwise, the {len(moves)} trained ones move up to {dm:.1e}; "
                  f"BN statistics move {ds:.1e}")
        assert maxd(net.state_dict(), base) == 0.0 and grad0 == {k: p.requires_grad for k, p in net.named_parameters()}, "the pooled net was modified"
    print("OK")


if __name__ == "__main__":
    main()
