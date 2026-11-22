#!/usr/bin/env python
"""T2-D318: ni26.heegnet geometry + layers, CPU, fp64.
  1. exp0 / log0, expmap / logmap invert; points stay on the sheet; dist(o, exp0(v)) = |v|;
  2. gyroadd(gyroinv(mu), mu) = o, and the gyro-translation is an isometry (pairwise distances kept);
  3. the Frechet mean is a stationary point (mean log_mu x ~ 0) and the DSMDBN output has its mean at o, variance ~ 1;
  4. LorentzMLR: logits = signed distance x |z| (sign = the side of the hyperplane), shapes;
  5. HEEGNet / EEGNetCtl shapes on a (41 x 480) window; train step backprops through DSMDBN + HSW; predict_stream is causal
     (window t's output does not change when windows > t change) and at t = N equals the refit on the whole stream;
  6. adamom_eta: ~1 at epoch 0, ~0.2 after the schedule.
  python tests/test_heegnet.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ni26 import heegnet as H  # noqa: E402

torch.set_default_dtype(torch.float64)


def on_sheet(x, tol=1e-9):
    return bool(((H.ldot(x, x, keepdim=False) + 1).abs() < tol * (1 + x[..., 0] ** 2)).all() and (x[..., 0] > 0).all())


def main():
    g = torch.Generator().manual_seed(0)
    v = 0.4 * torch.randn(50, 16, generator=g)                # |v| ~ 1.6: the radii the BN sees (unit-norm lift, then 1 / sigma)
    x = H.exp0(v)
    assert on_sheet(x) and torch.allclose(H.log0(x), v, atol=1e-10)
    o = torch.zeros(17); o[0] = 1
    assert torch.allclose(H.dist(o.expand_as(x), x), v.norm(dim=-1), atol=1e-9)
    y = H.exp0(0.5 * torch.randn(50, 16, generator=g))
    u = H.logmap(x, y)
    sc = x[:, :1].abs() * (1 + y[:, :1].abs())                # fp64 cancellation grows with cosh(r): relative tolerances
    assert (H.ldot(u, x).abs() / sc).max() < 1e-10 and ((H.expmap(x, u) - y).abs() / sc).max() < 1e-8

    mu = H.frechet_mean(x, iters=20)
    assert on_sheet(mu)
    assert H.logmap(mu.expand_as(x), x).mean(0).abs().max() < 1e-6, "Karcher mean not stationary"
    t = H.gyroadd(H.gyroinv(mu), x)
    assert on_sheet(t, 1e-8)
    assert torch.allclose(H.gyroadd(H.gyroinv(mu), mu[None])[0], o, atol=1e-9)
    D0 = H.dist(x[:, None], x[None]); D1 = H.dist(t[:, None], t[None])
    assert ((D0 - D1).abs() / (1 + D0)).max() < 1e-6, "gyro-translation must be an isometry (fp64 cancellation at r ~ 10)"

    bn = H.DSMDBN(17, 2)
    X = x.reshape(5, 1, 10, 17)
    m2, v2 = bn.stats(X.reshape(-1, 17))
    Xn = bn.normalise(X, m2, v2).reshape(-1, 17)
    mn = H.frechet_mean(Xn, iters=20)
    assert H.dist(mn, o) < 1e-4, f"normalised mean not at o: {float(H.dist(mn, o))}"
    assert abs(float(H.frechet_var(Xn, mn)) - 1.0) < 0.05, float(H.frechet_var(Xn, mn))

    mlr = H.LorentzMLR(17, 3)
    lo = mlr(x)
    assert lo.shape == (50, 3)
    with torch.no_grad():                                   # a = 0: the hyperplane through o normal to z -> sign = sign(z . xs)
        mlr.a.zero_()
    assert torch.equal(torch.sign(mlr(x)), torch.sign(x[:, 1:] @ mlr.z.t()))

    net = H.HEEGNet(41, 480, 3, n_domains=3)
    assert net.n_pos == 121 and net.d_feat == 241, (net.n_pos, net.d_feat)
    xb = torch.randn(20, 41, 480, generator=g)
    dom = torch.tensor([0] * 10 + [2] * 10)
    net.train()
    net.bn.eta = H.adamom_eta(0)
    lo, f = net(xb, dom)
    assert lo.shape == (20, 3) and f.shape == (20, 241) and on_sheet(f, 1e-7)
    loss = torch.nn.functional.cross_entropy(lo, torch.zeros(20, dtype=torch.long)) + 0.01 * H.hsw_loss(f, dom, n_proj=50, gen=g)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
    assert not torch.equal(net.bn.run_mean[0], o) and torch.equal(net.bn.run_mean[1], o)   # only the batch's domains moved
    net.bn.eta = H.adamom_eta(50)
    net(xb, dom)                                            # eta < 1: the geodesic interpolation path

    net.eval()
    s = torch.randn(12, 41, 480, generator=g)
    a = net.predict_stream(s)
    s2 = s.clone(); s2[7:] = torch.randn(5, 41, 480, generator=g)
    b = net.predict_stream(s2)
    assert torch.allclose(a[:7], b[:7], atol=1e-10) and not torch.allclose(a[7:], b[7:]), "predict_stream must be causal"
    assert torch.allclose(a[-1:], net.forward_refit(s)[0][-1:], atol=1e-8), "t = N must equal the whole-stream refit"

    c = H.EEGNetCtl(41, 480, 3)
    assert c(xb)[0].shape == (20, 3) and c.head.in_features == 240

    e0, e1 = H.adamom_eta(0), H.adamom_eta(90)
    assert abs(e0 - 1) < 0.01 and abs(e1 - 0.2) < 1e-9, (e0, e1)
    bb = H.domain_batches(np.array([0] * 23 + [1] * 31 + [2] * 9), rng=np.random.default_rng(0))
    assert sorted(np.concatenate(bb).tolist()) == list(range(63)) and max(len(b) for b in bb) <= 50
    print("OK")


if __name__ == "__main__":
    main()
