#!/usr/bin/env python
"""Self-test (T2-D283): t2p_bench.mae_tta_ln_probs, the LayerNorm-affine test-time MAE arms P / K, on a toy ST-shaped classifier (CPU).
The MAE loss is monkeypatched to a toy reconstruction through the same LayerNorm, so only the control flow is under test:
  1. causality: arm P predicts windows 1..30 of a stream with the start weights (= the frozen net), later windows differ;
     arm K predicts windows 1..8 with the start weights;
  2. reset per stream: the first window of every stream = the frozen net's output;
  3. isolation: the caller's net (every parameter) is unchanged after the call; only LayerNorm parameters move (drift > 0);
  4. determinism: two calls give identical outputs; stream 2 alone == stream 2 after stream 1;
  5. restoration: K's drift with restore 0.99 < with restore 1.0.
  python t2p_tta_ln_selftest.py -> "OK"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_cpt  # noqa: E402


class Vit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.proj = nn.Linear(c, c)


class Enc(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.vit = Vit(c)
        self.register_buffer("chan_idx", torch.arange(c))


class Toy(nn.Module):
    def __init__(self, c=6):
        super().__init__()
        self.enc = Enc(c)
        self.head = nn.Linear(c, 3)

    def forward(self, x):
        return self.head(self.enc.vit.proj(self.enc.vit.norm(x.float().std(-1))))


def toy_loss(enc, dec, x, chan, ratio, noise):
    h = enc.norm(x.float().std(-1))
    return ((enc.proj(h) - dec.w) ** 2).mean()


def main():
    torch.manual_seed(0)
    t2p_cpt.mae_loss_split = toy_loss
    net, mae = Toy().eval(), nn.Module()
    mae.w = nn.Parameter(torch.randn(6))
    rng = np.random.default_rng(0)
    n1, n2 = 70, 45                                              # two streams
    XA = rng.standard_normal((n1 + n2, 6, 480)).astype(np.float32) * np.linspace(0.5, 2, 6)[None, :, None]
    idx = pd.DataFrame(dict(i=np.arange(n1 + n2), stream=["a"] * n1 + ["b"] * n2, order=list(range(n1)) + list(range(n2))))
    rows = np.arange(n1 + n2)
    with torch.no_grad():
        plain = torch.softmax(net(torch.as_tensor(XA)), 1).numpy()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    for arm, pre in (("P", 30), ("K", 8)):
        p1, _, _, d1 = B.mae_tta_ln_probs(net, mae, XA, idx, rows, 1e-2, "cpu", arm)
        p2, _, _, _ = B.mae_tta_ln_probs(net, mae, XA, idx, rows, 1e-2, "cpu", arm)
        assert np.array_equal(p1, p2), f"{arm}: not deterministic"
        for s0, n in ((0, n1), (n1, n2)):
            assert np.allclose(p1[s0:s0 + pre], plain[s0:s0 + pre], atol=1e-6), f"{arm}: a window was predicted after a step that used it"
            assert not np.allclose(p1[s0 + pre:s0 + n], plain[s0 + pre:s0 + n], atol=1e-6), f"{arm}: no adaptation after the first step"
        pb, _, _, _ = B.mae_tta_ln_probs(net, mae, XA, idx, rows[n1:], 1e-2, "cpu", arm)
        assert np.array_equal(pb, p1[n1:]), f"{arm}: stream 2 alone != stream 2 after stream 1"
        assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items()), f"{arm}: the caller's net changed"
        assert (d1 > 0).all(), d1
        print(f"arm {arm}: causal (first {pre} windows = frozen net), reset per stream, deterministic, caller untouched; drift {d1.round(5).tolist()}")
    _, _, _, dk = B.mae_tta_ln_probs(net, mae, XA, idx, rows, 1e-2, "cpu", "K", restore=1.0)
    _, _, _, dr = B.mae_tta_ln_probs(net, mae, XA, idx, rows, 1e-2, "cpu", "K")
    assert (dr < dk).all(), (dr, dk)
    print(f"restoration: K drift {dr.round(5).tolist()} (0.99) < {dk.round(5).tolist()} (1.0)")
    print("OK")


if __name__ == "__main__":
    main()
