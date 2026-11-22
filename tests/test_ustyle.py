#!/usr/bin/env python
"""T2-D365: user-level feature re-styling in the pooled CNNs (Recipe.ustyle -> ni26.train.ustyle_hook, fit_eval), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_us5 / r1_pulx_vp2_e180_vr_us5 differ from the FBCNet rc75 / ShallowNet vr lines in ustyle (0.5) only; copies
     unchanged; only these two set ustyle;
  2. the hook: eval mode or no users -> unchanged; s 0 -> unchanged; a user with one window in its first batch has no statistics -> its
     windows unchanged; p 1, s 1: every window with statistics, standardised by ANOTHER user's statistics, equals itself standardised by its
     own user's (= it now carries v's mean / std), and the EMA updates the statistics on the next call;
  3. fit_eval with ustyle runs for both families, moves the weights away from ustyle 0, and leaves no hook behind (eval = the plain net).
  python tests/test_ustyle.py  ->  "OK"
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
    for base, n in (("r1_pulxx_vp2_e180_vr_rc75", "r1_pulxx_vp2_e180_vr_rc75_us5"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e180_vr_us5")):
        b, r = T.RECIPES[base], T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "ustyle"} and r.ustyle == 0.5 and r.ustyle_s == 0.75, n
        assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    assert sorted(n for n, x in T.RECIPES.items() if x.ustyle) == ["r1_pulx_vp2_e180_vr_us5", "r1_pulxx_vp2_e180_vr_rc75_us5"]
    print("recipes: the two us5 recipes differ from their lines in ustyle only; copies unchanged")

    torch.manual_seed(0)
    mod = torch.nn.Identity()
    u = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3])          # user 3 has a single window -> no statistics
    f = torch.randn(13, 6, 4) * (1 + u.view(-1, 1, 1).float()) + 3 * u.view(-1, 1, 1).float()
    st = {}
    h = T.ustyle_hook(1.0, 1.0, 4, st)
    mod.train()
    assert h(mod, (f,)) is None, "no users -> unchanged"
    st["u"] = u
    mod.eval()
    assert h(mod, (f,)) is None, "eval -> unchanged"
    mod.train()
    h0 = T.ustyle_hook(1.0, 0.0, 4, dict(u=u))
    assert torch.allclose(h0(mod, (f,))[0], f, atol=1e-4), "s 0 -> unchanged"
    out = h(mod, (f,))[0]
    mu = {k: f[u == k].mean(0) for k in range(3)}
    sd = {k: (f[u == k].var(0, unbiased=False) + 1e-5).sqrt() for k in range(3)}
    assert torch.equal(out[12], f[12]), "a user without statistics is left unchanged"
    for i in range(12):
        z_own = (f[i] - mu[int(u[i])]) / sd[int(u[i])]
        hits = [k for k in range(3) if k != int(u[i]) and torch.allclose((out[i] - mu[k]) / sd[k], z_own, atol=1e-3)]
        assert len(hits) == 1, (i, hits)
    f2 = f + 10.0
    h(mod, (f2,))                                                       # the EMA moves the user means by 0.1 x 10 = 1
    print("hook: eval / no users / s 0 -> unchanged; a user without statistics unchanged; s 1 -> each window carries another user's mean / std")

    d = data()
    for model in ("fbcnet_bboth", "shallow"):
        plain = replace(T.RECIPES["r1"], epochs=2)
        a0 = T.fit_eval(model, d, plain, 0, device="cpu", return_state=True)
        a1 = T.fit_eval(model, d, replace(plain, ustyle=1.0), 0, device="cpu", return_state=True)
        sa, sb = a0["state_last"], a1["state_last"]
        moved = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
        assert moved > 1e-5, (model, moved)
        net = T.build_model(model, d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
        net.load_state_dict(sb)
        assert not net.final_layer._forward_pre_hooks
        assert np.allclose(T.predict_proba(net, torch.as_tensor(d["Xte"])), a1["prob_last"], atol=1e-5), "eval = the plain net"
        a2 = T.fit_eval(model, d, plain, 0, device="cpu", return_state=True)
        assert all(torch.equal(a0["state_last"][k], a2["state_last"][k]) for k in sa), "ustyle 0 must be bitwise reproducible"
        print(f"fit_eval {model}: ustyle runs, weights move by up to {moved:.1e}, eval = the plain net with the trained weights")
    print("OK")


if __name__ == "__main__":
    main()
