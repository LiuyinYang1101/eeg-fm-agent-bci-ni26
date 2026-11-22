#!/usr/bin/env python
"""T2-D413: the MeanFlow generator (tracks/t2_bci/scripts/t2p_meanflow.py), CPU, tiny shapes:
  1. the JVP du/dt along (v, 0, 1) matches a central finite difference of u(z + h v, r, t + h) (float64);
  2. the primal u from torch.func.jvp equals the direct call, and the loss's gradients reach the parameters (after the zero-init
     output layers have taken one step, every parameter tensor gets a gradient);
  3. one-step sampling = xhat(eps, r 0, t 1), de-normalised; interpolated conditions: a != b, alpha in [0.25, 0.75], the dominant
     parent is the bank's u; (t, r) sampling: 0 < r <= t <= 1, r == t on about 75 %.
  python tests/test_meanflow.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks" / "t2_bci" / "scripts"))
import t2p_meanflow as M  # noqa: E402


def main():
    torch.manual_seed(0)
    C, L, B, nu = 8, 480, 4, 3
    net = M.UNet1D(C, nu, ch=(16, 32, 32)).double()
    for p in net.parameters():                                   # leave the zero-init: perturb so the JVP test is non-trivial
        p.data.add_(0.05 * torch.randn_like(p))
    x, eps = torch.randn(B, C, L, dtype=torch.float64), torch.randn(B, C, L, dtype=torch.float64)
    t, r = torch.tensor([0.9, 0.5, 0.3, 0.7], dtype=torch.float64), torch.tensor([0.2, 0.5, 0.1, 0.0], dtype=torch.float64)
    c, U = torch.tensor([0, 1, 2, 1]), torch.tensor([0, 2, 1, 0])
    ue = net.uemb(U).detach()
    z, v = (1 - t.view(B, 1, 1)) * x + t.view(B, 1, 1) * eps, eps - x

    def u_fn(z_, r_, t_):
        return (z_ - net(z_, t_, r_, c, ue)) / t_.view(-1, 1, 1)

    u, dudt = torch.func.jvp(u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    h = 1e-6                                                    # the 1000 x t sinusoids: O(h^2 1e9) truncation
    fd = (u_fn(z + h * v, r, t + h) - u_fn(z - h * v, r, t - h)) / (2 * h)
    err = float((dudt - fd).abs().max() / fd.abs().max())
    assert err < 1e-5, err
    assert torch.allclose(u, u_fn(z, r, t), atol=1e-10)
    print(f"jvp: du/dt matches the finite difference (rel err {err:.1e}); primal = the direct call")
    net = M.UNet1D(C, nu, ch=(16, 32, 32))
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(B, C, L)
    for _ in range(2):
        opt.zero_grad()
        loss, raw = M.mf_loss(net, xs, c, net.uemb(U), g)
        loss.backward()
        opt.step()
    opt.zero_grad()
    M.mf_loss(net, xs, c, net.uemb(U), g)[0].backward()
    dead = [n for n, p in net.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
    assert not dead, dead
    print(f"mf_loss: finite ({float(loss):.3f}), every parameter tensor gets a gradient after the zero-init layers moved")
    mu, sd = torch.zeros(C, 1), 2 * torch.ones(C, 1)
    gs1, gs2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    Xg = M.generate(net, c, net.uemb(U).detach(), mu, sd, gs1)
    e = torch.randn((B, C, L), generator=gs2)
    with torch.no_grad():
        want = net(e, torch.ones(B), torch.zeros(B), c, net.uemb(U)) * sd + mu
    assert Xg.shape == (B, C, L) and np.allclose(Xg, want.numpy(), atol=1e-5)
    E = net.uemb.weight.detach()
    cc, ue2, a, b, al = M.interp_conds(2000, nu, torch.Generator().manual_seed(1), "cpu", E)
    assert (a != b).all() and al.min() >= 0.25 and al.max() <= 0.75 and set(cc.tolist()) == {0, 1, 2}
    assert torch.allclose(ue2, torch.as_tensor(al)[:, None] * E[a] + (1 - torch.as_tensor(al)[:, None]) * E[b], atol=1e-6)
    tt, rr = M.sample_tr(20000, torch.Generator().manual_seed(2), "cpu")
    assert (rr <= tt).all() and (tt >= 1e-3).all() and (tt <= 1).all() and (rr > 0).all()
    eq = float((tt == rr).float().mean())
    assert 0.73 < eq < 0.77, eq
    print(f"sampling: one step = xhat(eps, 0, 1) de-normalised; interp conds valid; (t, r): r = t on {eq:.3f}")
    # T2-D413b knobs: generate_k(1 step) = generate bitwise; WhiteNorm inverts; zero calibration gains = identity; p_eq 1 = flow matching
    Xk = M.generate_k(net, c, net.uemb(U).detach(), mu, sd, torch.Generator().manual_seed(5), steps=1)
    assert np.allclose(Xk, Xg, atol=1e-4), "generate_k(steps 1) must equal generate (float rounding: z - (z - xhat))"
    Xw = torch.randn(40, C, L) * torch.linspace(0.5, 3.0, C)[None, :, None] + 2.0
    wn = M.WhiteNorm(Xw)
    rt = float((wn.inv(wn.fwd(Xw)) - Xw).abs().max())
    assert rt < 1e-4, rt
    Z = wn.fwd(Xw)
    assert 0.5 < float(Z.pow(2).mean(-1).sqrt().median()) < 2.0
    wn2 = M.WhiteNorm(state=wn.state())
    assert torch.equal(wn2.fwd(Xw), Z)
    Xa = Xw.numpy()
    assert np.allclose(M.apply_gains(Xa, np.zeros((len(Xa), C, L // 2 + 1))), Xa, atol=1e-4)
    gfm = torch.Generator().manual_seed(3)
    net2 = M.UNet1D(C, nu, ch=(16, 32, 32))
    lf, _ = M.mf_loss(net2, xs, c, net2.uemb(U), gfm, p_eq=1.0)
    gfm = torch.Generator().manual_seed(3)
    t_, r_ = M.sample_tr(B, gfm, "cpu", 1.0)
    e_ = torch.randn(xs.shape, generator=gfm)
    z_ = (1 - t_.view(B, 1, 1)) * xs + t_.view(B, 1, 1) * e_
    l_ = ((z_ - net2(z_, t_, t_, c, net2.uemb(U))) / t_.view(B, 1, 1) - (e_ - xs)).pow(2).mean((1, 2))
    assert torch.equal(t_, r_) and torch.allclose(lf, (l_ / (l_.detach() + 1e-3)).mean(), atol=1e-6)
    print(f"D413b: generate_k(1) = generate (rounding); WhiteNorm round trip {rt:.1e}; zero gains = identity; p_eq 1 = the FM loss")
    print("OK")


if __name__ == "__main__":
    main()
