#!/usr/bin/env python
"""Backlog 73: same-class cross-user input mixing in the pooled CNNs (Recipe.xmix -> ni26.train.xmix_batch, fit_eval), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_xm5 / r1_pulx_vp2_e180_vr_xm5 differ from the FBCNet rc75 / ShallowNet vr lines in xmix (0.5) only; copies
     unchanged; only these two set xmix;
  2. xmix_batch: p 0 -> the batch unchanged; p 1 -> every window with a partner = (lam x + (1 - lam) x_b) / sqrt(lam^2 + (1 - lam)^2) with x_b of
     the SAME label and ANOTHER user (replayed), windows without one unchanged; the power of independent unit-power windows is kept (≈ 1);
  3. fit_eval with xmix runs for both families and moves the weights away from xmix 0.
  python tests/test_xmix.py  ->  "OK"
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
    for base, n in (("r1_pulxx_vp2_e180_vr_rc75", "r1_pulxx_vp2_e180_vr_rc75_xm5"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e180_vr_xm5")):
        b, r = T.RECIPES[base], T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "xmix"} and r.xmix == 0.5, n
        assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    assert sorted(n for n, x in T.RECIPES.items() if x.xmix) == ["r1_pulx_vp2_e180_vr_xm5", "r1_pulxx_vp2_e180_vr_rc75_xm5"]
    print("recipes: the two xm5 recipes differ from their lines in xmix only; copies unchanged")

    torch.manual_seed(0)
    x = torch.randn(13, 5, 400)
    u = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 3])
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1, 0])
    y[12] = 3                                          # window 12: a label nobody else has -> no partner
    assert torch.equal(T.xmix_batch(x, y, u, 0.0), x), "p 0 -> unchanged"
    torch.manual_seed(1)
    got = T.xmix_batch(x, y, u, 1.0)
    torch.manual_seed(1)                               # replay the draws
    sc = torch.rand(13, 13)
    sc[(u[:, None] == u[None, :]) | (y[:, None] != y[None, :])] = -1.0
    j = sc.argmax(1)
    on = (sc.max(1).values >= 0) & (torch.rand(13) < 1.0)
    lam = torch.rand(13, 1, 1)
    want = torch.where(on.view(13, 1, 1), (lam * x + (1 - lam) * x[j]) / torch.sqrt(lam ** 2 + (1 - lam) ** 2), x)
    assert torch.allclose(got, want, atol=1e-6)
    assert ((u[j] != u) & (y[j] == y))[on].all() and not on[12] and on[:12].all()
    assert torch.equal(got[12], x[12]), "no same-label partner of another user -> unchanged"
    pw = float(got[:12].pow(2).mean())
    assert abs(pw - 1.0) < 0.1, pw
    print(f"xmix_batch: p 0 unchanged; p 1 = the replayed same-label / other-user mix, the partnerless window unchanged; power {pw:.3f} (≈ 1)")

    d = data()
    for model in ("fbcnet_bboth", "shallow"):
        plain = replace(T.RECIPES["r1"], epochs=2)
        a0 = T.fit_eval(model, d, plain, 0, device="cpu", return_state=True)
        a1 = T.fit_eval(model, d, replace(plain, xmix=1.0), 0, device="cpu", return_state=True)
        sa, sb = a0["state_last"], a1["state_last"]
        moved = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
        assert moved > 1e-5, (model, moved)
        print(f"fit_eval {model}: xmix runs, weights move by up to {moved:.1e}")
    print("OK")


if __name__ == "__main__":
    main()
