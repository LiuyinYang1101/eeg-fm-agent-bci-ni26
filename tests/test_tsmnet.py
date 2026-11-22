#!/usr/bin/env python
"""T2-D491: ni26.tsmnet checks (CPU, seconds): (1) SymFn's Daleckii-Krein backward = a finite-difference gradient (log / exp / reeig);
(2) the per-stream decode is CAUSAL (changing windows > t leaves windows <= t unchanged); (3) the batch-1 StreamState = stream_logits;
(4) a training forward / backward on domain groups is finite."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ni26.tsmnet import StreamState, TSMNet, symfn  # noqa: E402


def spd(n, d, g):
    A = torch.randn(n, d, d, generator=g, dtype=torch.float64)
    return A @ A.transpose(1, 2) / d + 0.5 * torch.eye(d, dtype=torch.float64)


def main():
    g = torch.Generator().manual_seed(0)
    for kind in ("log", "exp", "reeig"):
        S = spd(3, 5, g)
        if kind == "exp":
            S = 0.3 * S
        S.requires_grad_(True)
        Gw = torch.randn(3, 5, 5, generator=g, dtype=torch.float64)
        eps = 0.3 if kind == "reeig" else 1e-6
        (symfn(S, kind, eps) * Gw).sum().backward()
        E = torch.randn(3, 5, 5, generator=g, dtype=torch.float64)
        E = 0.5 * (E + E.transpose(1, 2))
        h = 1e-6
        with torch.no_grad():
            fd = ((symfn(S + h * E, kind, eps) - symfn(S - h * E, kind, eps)) * Gw).sum() / (2 * h)
        an = (S.grad * E).sum()
        assert abs(float(fd - an)) < 1e-5 * max(1.0, abs(float(fd))), (kind, float(fd), float(an))
    torch.manual_seed(0)
    net = TSMNet(6, F_t=2, k_t=9, D_s=6, D_b=4).double().eval()
    net.temp.double(), net.spat.double()
    X = torch.randn(12, 6, 120, dtype=torch.float64)
    net.set_prior(X.float().double(), torch.tensor([0] * 6 + [1] * 6))
    a = net.stream_logits(X)
    X2 = X.clone()
    X2[7:] = torch.randn(5, 6, 120, dtype=torch.float64) * 3
    b = net.stream_logits(X2)
    assert torch.allclose(a[:7], b[:7], atol=1e-10) and not torch.allclose(a[7:], b[7:]), "stream decode is not causal"
    st = StreamState(net)
    p1 = torch.cat([st.step(X[t:t + 1]) for t in range(len(X))])
    assert torch.allclose(p1, torch.softmax(a, 1), atol=1e-8), (p1 - torch.softmax(a, 1)).abs().max()
    net.train()
    lo = net.forward_groups(X, torch.tensor([0] * 4 + [1] * 4 + [2] * 4))
    lo.logsumexp(1).sum().backward()
    assert torch.isfinite(lo).all() and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    print("test_tsmnet OK")


if __name__ == "__main__":
    main()
