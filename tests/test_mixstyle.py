#!/usr/bin/env python
"""Backlog 70: MixStyle across training users in the pooled CNNs (Recipe.mixstyle -> ni26.train.mixstyle_hook, fit_eval), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_ms5 / r1_pulx_vp2_e180_vr_ms5 differ from the FBCNet rc75 / ShallowNet vr lines in mixstyle (0.5) only; copies
     unchanged; only these two set mixstyle;
  2. the hook: eval mode or no users -> the map unchanged; train mode, p 1: every window's per-channel time mean / std = lam : 1 - lam of its own
     and its partner's, the partner is always another training user with the SAME class, and a window with no such partner is left unchanged;
  3. fit_eval with mixstyle runs for both families, moves the weights away from mixstyle 0, and leaves no hook behind (eval = the plain net).
  python tests/test_mixstyle.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ni26.train as T  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    for base, n in (("r1_pulxx_vp2_e180_vr_rc75", "r1_pulxx_vp2_e180_vr_rc75_ms5"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e180_vr_ms5")):
        b, r = T.RECIPES[base], T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "mixstyle"} and r.mixstyle == 0.5, n
        assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    assert sorted(n for n, x in T.RECIPES.items() if x.mixstyle) == ["r1_pulx_vp2_e180_vr_ms5", "r1_pulxx_vp2_e180_vr_rc75_ms5"]
    print("recipes: the two ms5 recipes differ from their lines in mixstyle only; copies unchanged")

    torch.manual_seed(0)
    mod = torch.nn.Identity()
    out = torch.randn(12, 5, 1, 64) * torch.rand(12, 5, 1, 1) * 3 + torch.randn(12, 5, 1, 1)
    st = {}
    h = T.mixstyle_hook(1.0, st)
    mod.train()
    assert h(mod, None, out) is None, "no users -> unchanged"
    st["u"] = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    st["y"] = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1])     # every window has a same-class partner of another user
    mod.eval()
    assert h(mod, None, out) is None, "eval -> unchanged"
    mod.train()
    torch.manual_seed(1)
    got = h(mod, None, out)
    torch.manual_seed(1)                               # replay the hook's draws to recover the partners and lam
    float(torch.rand(()))
    sc = torch.rand(12, 12)
    u = st["u"]
    y = st["y"]
    sc[(u[:, None] == u[None, :]) | (y[:, None] != y[None, :])] = -1.0
    j = sc.argmax(1)
    lam = torch.distributions.Beta(0.1, 0.1).sample((12, 1, 1, 1))
    assert (u[j] != u).all() and (y[j] == y).all(), "the partner must be another training user with the same class"
    mu, sd = out.mean((2, 3), keepdim=True), (out.var((2, 3), keepdim=True) + 1e-6).sqrt()
    gm, gs = got.mean((2, 3), keepdim=True), got.var((2, 3), keepdim=True).sqrt()
    assert torch.allclose(gm, lam * mu + (1 - lam) * mu[j], atol=1e-4)
    assert torch.allclose(gs, lam * sd + (1 - lam) * sd[j], atol=1e-3), float((gs - (lam * sd + (1 - lam) * sd[j])).abs().max())
    st["u"] = torch.zeros(12, dtype=torch.long)
    assert torch.allclose(h(mod, None, out), out, atol=1e-5), "one user in the batch -> unchanged"
    st["u"], st["y"] = torch.arange(12) % 3, torch.arange(12) % 3      # every same-class window is the same user -> no partner
    assert torch.allclose(h(mod, None, out), out, atol=1e-5), "no same-class partner of another user -> unchanged"
    print("hook: eval / no users / no same-class other-user partner -> unchanged; else stats = lam-mix with such a partner (replayed)")

    d = data()
    for model in ("fbcnet_bboth", "shallow"):
        plain = replace(T.RECIPES["r1"], epochs=2)
        a0 = T.fit_eval(model, d, plain, 0, device="cpu", return_state=True)
        a1 = T.fit_eval(model, d, replace(plain, mixstyle=1.0), 0, device="cpu", return_state=True)
        sa, sb = a0["state_last"], a1["state_last"]
        moved = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
        assert moved > 1e-5, (model, moved)
        net = T.build_model(model, d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
        net.load_state_dict(sb)
        assert not getattr(net, T.MIXSTYLE_AT[model])._forward_hooks
        assert np.allclose(T.predict_proba(net, torch.as_tensor(d["Xte"])), a1["prob_last"], atol=1e-5), "eval = the plain net"
        print(f"fit_eval {model}: mixstyle runs, weights move by up to {moved:.1e}, eval = the plain net with the trained weights")
    print("OK")


if __name__ == "__main__":
    main()
