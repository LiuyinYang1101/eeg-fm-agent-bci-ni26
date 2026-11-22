#!/usr/bin/env python
"""T2-D224: ni26.train.sam_grad and Recipe.sam on CPU (FBCNet B3 and ShallowNet, both with BatchNorm; ShallowNet with dropout):
  1. the SAM gradient = a hand-made reference (a copy perturbed by rho g / ||g||, the same dropout draw), weights restored, BatchNorm running
     statistics bitwise those of the first pass;
  2. fit_eval with sam = 0 (the vr line) = the vr line bitwise; with sam > 0 the weights move away from it; the recipes differ from the vr line in
     `sam` only, and their per-user copies are the vr line's.
  python tests/test_sam.py  ->  "OK"
"""
import copy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402


def bn_state(net):
    return {k: v.clone() for k, v in net.state_dict().items() if "running" in k or "num_batches" in k}


def one_step(model):
    torch.manual_seed(0)
    C, L, K = 41, 480, 3
    net = T.build_model(model, C, K, L, 120.0).train()
    x, y = torch.randn(16, C, L), torch.randint(K, (16,))
    rec = replace(T.RECIPES["r1_pulx_vp2_e180_vr_sam2"])
    ref = copy.deepcopy(net)
    torch.manual_seed(1)
    F.cross_entropy(net(x), y, label_smoothing=rec.label_smoothing).backward()
    use = [p.grad is not None for p in net.parameters()]           # FBCNet holds an unused parameter (no grad); sam_grad skips it too
    P = lambda m: [p for p, u in zip(m.parameters(), use) if u]
    g = [p.grad.clone() for p in P(net)]
    w0 = [p.detach().clone() for p in net.parameters()]
    bn1 = bn_state(net)
    torch.manual_seed(2)
    T.sam_grad(net, x, y, rec)
    assert all(torch.allclose(p, w, atol=1e-6, rtol=0) for p, w in zip(net.parameters(), w0)), "weights must be restored"
    assert all(torch.equal(bn1[k], v) for k, v in bn_state(net).items()), "BN running stats must not move in the second pass"
    torch.manual_seed(1)
    F.cross_entropy(ref(x), y, label_smoothing=rec.label_smoothing)     # the same first-pass BN update
    gn = torch.norm(torch.stack([a.norm(2) for a in g]), 2)
    with torch.no_grad():
        for p, a in zip(P(ref), g):
            p.add_(rec.sam * a / (gn + 1e-12))
    ref.zero_grad(set_to_none=True)
    torch.manual_seed(2)
    F.cross_entropy(ref(x), y, label_smoothing=rec.label_smoothing).backward()
    d = max(float((p.grad - q.grad).abs().max()) for p, q in zip(P(net), P(ref)))
    dg = max(float((p.grad - a).abs().max()) for p, a in zip(P(net), g))
    assert d < 1e-5 and dg > 1e-4, (model, d, dg)
    print(f"{model}: SAM grad = reference (max |diff| {d:.1e}; moved {dg:.1e} from the plain grad); weights restored; BN stats frozen", flush=True)


def data(seed=0, C=41, L=480):
    rng = np.random.default_rng(seed)
    n = dict(tr=48, va=12, te=12)
    d = {}
    for k, m in n.items():
        d[f"X{k}"] = rng.standard_normal((m, C, L)).astype(np.float32)
        d[f"y{k}"] = rng.integers(0, 3, m)
        d[f"c{k}"] = np.array(["a"] * m)
    return d


def main():
    for m in ("fbcnet_bboth", "shallow"):
        one_step(m)
    for f in ("pulxx", "pulx"):
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        for t, rho in (("sam05", 0.05), ("sam2", 0.2)):
            r = T.RECIPES[f"r1_{f}_vp2_e180_vr_{t}"]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "sam"} and r.sam == rho, (f, t)
            assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    d = data()
    plain = replace(T.RECIPES["r1"], epochs=2)
    a = T.fit_eval("shallow", d, plain, 0, device="cpu", return_state=True)
    b = T.fit_eval("shallow", d, replace(plain, sam=0.0), 0, device="cpu", return_state=True)
    c = T.fit_eval("shallow", d, replace(plain, sam=0.2), 0, device="cpu", return_state=True)
    same = all(torch.equal(a["state_last"][k], b["state_last"][k]) for k in a["state_last"])
    moved = max(float((a["state_last"][k] - c["state_last"][k]).abs().max()) for k in a["state_last"] if a["state_last"][k].is_floating_point())
    assert all(torch.equal(a["state_last"][k], c["state_last"][k]) for k in a["state_last"] if "num_batches" in k), "the BN step counter must match"
    assert same and np.array_equal(a["prob_last"], b["prob_last"]) and moved > 1e-4, (same, moved)
    print(f"fit_eval: sam 0 = r1 bitwise; sam 0.2 moves the weights by up to {moved:.1e}; recipes = the vr line + sam; copies unchanged")
    print("OK")


if __name__ == "__main__":
    main()
