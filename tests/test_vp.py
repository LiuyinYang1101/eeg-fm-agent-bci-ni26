#!/usr/bin/env python
"""T2-D193: virtual participants (t2p_bench.vp_bank + ni26.train.augment with recipe.vp) on CPU:
  1. sigma 0 -> every map is ~ the identity (max |M - I| < 0.05); sigma 5 mm -> near identity (mean |M - I| < 0.05, mean diagonal > 0.9) and rows sum to ~1 (a constant field stays);
  2. an identity bank (I, gain 1, shift 0) -> augment = the input exactly; a real bank -> finite, same shape, different;
  3. recipe.vp = 0 (r1) never touches VP_BANK;
  4. T2-D195: sigma 10 mm = the same draws as 5 mm with the jitter doubled (gains / shifts equal; |M - I| larger);
  5. T2-D200: vp_a 1.0 = the bank bitwise; vp_a 0.5 with unit gains and no shift = exactly half the map's change;
  6. T2-D215: vref 0 = the bank bitwise; vref 0.5 keeps gains / shifts, and each map changes by one row repeated (a common reference), row sum in [-0.5, 0].
  python tests/test_vp.py  ->  "OK"
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402
from ni26 import train as T  # noqa: E402


def main():
    names = json.load(open(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json"))["ch_names"]
    C = len(names)
    b0 = B.vp_bank(names, K=4, sigma=0.0)
    d0 = float((b0["mats"] - torch.eye(C)).abs().max())
    assert d0 < 0.05, d0
    b = B.vp_bank(names, K=16)
    E = b["mats"] - torch.eye(C)
    d, dg, rs = float(E.abs().mean()), float(b["mats"].diagonal(dim1=1, dim2=2).mean()), float((b["mats"].sum(2) - 1).abs().max())
    assert d < 0.05 and dg > 0.9 and rs < 0.05, (d, dg, rs)
    assert set(b["shifts"].tolist()) <= {-1, 0, 1} and float((b["gains"] - 1).abs().max()) <= 0.1 + 1e-6
    print(f"vp_bank: sigma 0 -> max |M - I| {d0:.3f}; sigma 5 mm -> mean |M - I| {d:.4f}, diagonal {dg:.3f}, rows sum to 1 within {rs:.1e}", flush=True)
    b64, b256 = B.vp_bank(names, K=64, sigma=0.010), B.vp_bank(names, K=256, sigma=0.010)     # T2-D202: K = 256 extends the K = 64 draws
    assert all(torch.equal(b256[k][:64], b64[k]) for k in ("mats", "gains", "shifts")) and len(b256["mats"]) == 256
    bg, bs6 = B.vp_bank(names, K=64, sigma=0.010, gain=0.2), B.vp_bank(names, K=64, sigma=0.010, shift=6)   # T2-D214: maps unchanged
    assert torch.equal(bg['mats'], b64['mats']) and torch.equal(bs6['mats'], b64['mats']) and torch.equal(bs6['gains'], b64['gains'])
    assert torch.allclose(bg['gains'] - 1, 2 * (b64['gains'] - 1), atol=1e-6) and int(bs6['shifts'].abs().max()) <= 6 and int(bs6['shifts'].abs().max()) > 1
    bv0, bv = B.vp_bank(names, K=64, sigma=0.010, vref=0.0), B.vp_bank(names, K=64, sigma=0.010, vref=0.5)   # T2-D215: virtual references
    assert all(torch.equal(bv0[k], b64[k]) for k in ("mats", "gains", "shifts"))
    assert torch.equal(bv['gains'], b64['gains']) and torch.equal(bv['shifts'], b64['shifts'])
    D = (bv['mats'] - b64['mats']).double()
    assert float((D - D[:, :1, :]).abs().max()) < 1e-5 and float(D.abs().max()) > 1e-3, "one common row per map"
    rsum = D[:, 0, :].sum(1)
    assert float(rsum.max()) <= 1e-4 and float(rsum.min()) >= -0.5 - 0.05, (float(rsum.min()), float(rsum.max()))
    print(f"vp_bank: vref 0.5 -> a common-row change per map, row sums {float(rsum.min()):+.3f} .. {float(rsum.max()):+.3f}", flush=True)
    b2 = B.vp_bank(names, K=16, sigma=0.010)                          # T2-D195: vp 2 = the same draws, jitter x 2; gains / shifts unchanged
    d2, dg2 = float((b2["mats"] - torch.eye(C)).abs().mean()), float(b2["mats"].diagonal(dim1=1, dim2=2).mean())
    assert torch.equal(b2["gains"], b["gains"]) and torch.equal(b2["shifts"], b["shifts"]) and d2 > 1.5 * d and dg2 > 0.75, (d2, dg2)
    print(f"vp_bank: sigma 10 mm -> mean |M - I| {d2:.4f}, diagonal {dg2:.3f} (same gains / shifts as 5 mm)", flush=True)
    x = torch.randn(8, C, 480)
    rec = replace(T.RECIPES["r1"], vp=1)
    T.VP_BANK.clear(); T.VP_BANK.update(mats=torch.eye(C).repeat(3, 1, 1), gains=torch.ones(3, C), shifts=torch.zeros(3, dtype=torch.long))
    assert torch.equal(T.augment(x, rec), x)
    T.VP_BANK.clear(); T.VP_BANK.update(b)
    y = T.augment(x, rec)
    assert y.shape == x.shape and torch.isfinite(y).all() and not torch.allclose(y, x)
    torch.manual_seed(1); ya = T.augment(x, replace(rec, vp_a=1.0)); torch.manual_seed(1); yb = T.augment(x, rec)
    assert torch.equal(ya, yb)                                    # T2-D200: vp_a 1.0 = the bank as drawn, bitwise
    T.VP_BANK.clear(); T.VP_BANK.update(mats=b["mats"], gains=torch.ones_like(b["gains"]), shifts=torch.zeros_like(b["shifts"]))   # unit gains: the map part is exactly linear in a
    torch.manual_seed(2); y1 = T.augment(x, rec); torch.manual_seed(2); yh = T.augment(x, replace(rec, vp_a=0.5))
    assert torch.allclose(yh - x, 0.5 * (y1 - x), atol=1e-4, rtol=1e-3), float(((yh - x) - 0.5 * (y1 - x)).abs().max())  # half strength = half the change
    T.VP_BANK.clear()
    assert torch.equal(T.augment(x, T.RECIPES["r1"]), x)
    print("augment: identity bank = input exactly; a real bank changes the input; r1 needs no bank")
    print("OK")


if __name__ == "__main__":
    main()
