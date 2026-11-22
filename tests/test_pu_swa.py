#!/usr/bin/env python
"""T2-D227: weight averaging (SWA) of the per-user copies (Recipe.pu_swa -> peruser_for -> Recipe.swa -> finetune), CPU, FBCNet B3 (BatchNorm):
  1. r1_pulx(x)_vp2_e180_vr_psw25 / _psw50 differ from the vr line in `pu_swa` only; their copies = the vr line's with swa 0.25 / 0.5; every other
     recipe's copies have swa 0; the pooled fit_eval refuses swa;
  2. finetune with swa 0 = PERUSER bitwise; swa covering one epoch = no averaging, bitwise;
  3. finetune with swa 0.5 over 4 epochs = a hand-made reference: the same loop, the end-of-epoch states of epochs 3 and 4 averaged
     (float tensors; BatchNorm's step counter = the last epoch's), and it differs from the plain copy.
  python tests/test_pu_swa.py  ->  "OK"
"""
import copy
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from ni26 import train as T  # noqa: E402
from test_train_ema import data, maxd  # noqa: E402


def reference(net, X, y, recipe, seed, keep):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    net = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.lr, weight_decay=recipe.wd)
    total = recipe.epochs * int(math.ceil(len(X) / recipe.bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: T._cosine(s, total, int(recipe.warmup_frac * total)))
    states = []
    for _ in range(recipe.epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), recipe.bs):
            b = perm[i:i + recipe.bs]
            loss = F.cross_entropy(net(X[b]), y[b], label_smoothing=recipe.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        states.append({k: v.detach().clone() for k, v in net.state_dict().items()})
    last = states[-keep:]
    return {k: torch.stack([s[k] for s in last]).mean(0) if v.is_floating_point() else states[-1][k] for k, v in states[-1].items()}


def main():
    for f in ("pulxx", "pulx"):
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        for t, a in (("psw25", 0.25), ("psw50", 0.5)):
            r = T.RECIPES[f"r1_{f}_vp2_e180_vr_{t}"]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "pu_swa"} and r.pu_swa == a, (f, t)
            assert T.peruser_for(r) == replace(T.peruser_for(b), swa=a), (f, t)
    on = sorted(n for n, r in T.RECIPES.items() if T.peruser_for(r).swa)
    assert on == sorted(f"r1_{f}_vp2_e180_vr_{t}" for f in ("pulxx", "pulx") for t in ("psw25", "psw50")), on
    assert not any(r.swa for r in T.RECIPES.values()), "swa is set through pu_swa only"
    d = data()
    try:
        T.fit_eval("shallow", d, replace(T.RECIPES["r1"], epochs=1, swa=0.5), 0, device="cpu")
        raise SystemExit("fit_eval must refuse swa")
    except AssertionError:
        pass
    r1 = T.fit_eval("fbcnet_bboth", d, replace(T.RECIPES["r1"], epochs=2), 0, device="cpu", return_state=True)
    net = T.build_model("fbcnet_bboth", d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
    net.load_state_dict(r1["state_last"])
    X, y = torch.as_tensor(d["Xtr"][:48]), torch.as_tensor(d["ytr"][:48], dtype=torch.long)
    pu = replace(T.PERUSER, epochs=4)
    plain = T.finetune(net, X, y, pu, 5).state_dict()
    assert maxd(T.finetune(net, X, y, replace(pu, swa=0.0), 5).state_dict(), plain) == 0.0
    assert maxd(T.finetune(net, X, y, replace(pu, swa=0.25), 5).state_dict(), plain) == 0.0, "one averaged epoch = the last epoch"
    got = T.finetune(net, X, y, replace(pu, swa=0.5), 5).state_dict()
    want = reference(net, X, y, pu, 5, keep=2)
    dref, dplain = maxd(got, want), maxd(got, plain)
    assert dref < 1e-6 and dplain > 1e-5, (dref, dplain)
    assert all(torch.equal(got[k], plain[k]) for k in got if not got[k].is_floating_point()), "integer buffers = the last epoch's"
    print(f"finetune swa 0.5: = the hand-made 2-epoch average (max |diff| {dref:.1e}); {dplain:.1e} from the plain copy; swa 0 / one epoch bitwise")
    print("OK")


if __name__ == "__main__":
    main()
