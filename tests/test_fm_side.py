#!/usr/bin/env python
"""T2-X1: the amplitude side input (t2p_fm.AmpSide, t2p_bench --fm-side amp) on CPU:
  1. MOR's front end (FFT resample to 256 Hz x 4 s, then zwin) gives the same input for --aligns wz and --aligns none on real windows
     (zwin deletes the per-channel scale), so the X1 arm (none + side) and its control (wz) differ only by the side head;
  2. a zero-initialised AmpSide adds exactly 0: FMClf with it = FMClf without it, bitwise;
  3. the side head learns (one optimiser step moves its weights) and a flat channel stays finite (clip +-5).
  python tests/test_fm_side.py  ->  "OK"
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_fm as FM  # noqa: E402
from ni26 import stream  # noqa: E402


class Dummy(nn.Module):
    ALWAYS_USED = ()
    dim, n_layers = 8, 1

    def __init__(self, C):
        super().__init__()
        self.lin = nn.Linear(1, self.dim)

    def layer_of(self, n):
        return 0

    def prep(self, x):
        return FM.t1.zwin(x, 1e9)

    def forward(self, x, layer):
        return self.lin(x.mean(-1, keepdim=True)).unsqueeze(2)          # (B, C, 1, D)


def main():
    E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
    X = np.load(E / "X.npy", mmap_mode="r")
    C = len(json.load(open(E / "meta.json"))["ch_names"])
    xs = np.asarray(X[:64], np.float64)
    scale = float(np.median(xs.std(-1)))
    a_wz, a_none = stream.align_stream(xs, "wz"), stream.align_stream(xs, "none", scale=scale)
    f = lambda a: FM.t1.zwin(FM.t1.fft_resample(torch.as_tensor(a), 1024), 1e9)   # noqa: E731
    d = float((f(a_wz) - f(a_none)).abs().max())
    assert d < 1e-3, d
    print(f"MOR front end: wz vs none max |d| {d:.1e} (scale {scale:.3g})")
    torch.manual_seed(0)
    net = FM.FMClf(Dummy(C), 1, "mean", 3)
    x = torch.as_tensor(a_none[:16])
    y0 = net(x)
    ls_ = FM.AmpSide.logstd(torch.as_tensor(a_none)).numpy()
    net.amp_side = FM.AmpSide(ls_.mean(0), ls_.std(0) + 1e-3, 3)
    assert torch.equal(net(x), y0), "zero-initialised side != no side"
    opt = torch.optim.SGD(net.amp_side.parameters(), lr=0.1)
    nn.functional.cross_entropy(net(x), torch.randint(3, (16,))).backward()
    opt.step()
    assert float(net.amp_side.lin.weight.abs().max()) > 0 and not torch.equal(net(x), y0)
    xf = x.clone(); xf[:, 3] = 0
    assert torch.isfinite(net(xf)).all()
    print("AmpSide: zero init = FMClf bitwise; it learns; a flat channel stays finite")
    print("OK")


if __name__ == "__main__":
    main()
