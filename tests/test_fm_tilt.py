#!/usr/bin/env python
"""Backlog 94 / T2-D399: the CNN recipes' spectral tilt on the FM's training windows, CPU:
  1. t2p_fm.fit_eval_fm takes tilt (default 0.0 = off: no tilt Recipe is built) and applies it AFTER the vp step on target batches only;
     t2p_bench --fm-tilt (default 0) reaches both fit_eval_fm kwargs and fm_args, and the stem gains tl<x> only when set;
  2. ni26.train.augment with a tilt-only Recipe changes nothing but the tilt: tilt 0 -> bitwise unchanged; a 10 Hz sinusoid keeps its
     amplitude at beta's pivot ((10 + 1) / 11 = 1), and the map is linear.
  python tests/test_fm_tilt.py  ->  "OK"
"""
import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import ni26.train as T  # noqa: E402
import t2p_fm  # noqa: E402


def main():
    sig = inspect.signature(t2p_fm.fit_eval_fm).parameters
    assert sig["tilt"].default == 0.0, "tilt must default to off"
    src = inspect.getsource(t2p_fm.fit_eval_fm)
    assert 'rtl = TV.Recipe("fm_tilt", tilt=float(tilt)) if tilt > 0 else None' in src
    i_vp, i_tl = src.index("xb = TV.augment(xb, rvp)"), src.index("xb = TV.augment(xb, rtl)")
    assert i_vp < i_tl and "if rtl is not None and k == 0:" in src, "tilt after vp, target batches only"
    b = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert 'ap.add_argument("--fm-tilt", type=float, default=0.0,' in b and b.count("tilt=args.fm_tilt") == 2
    assert '(f"tl{args.fm_tilt:g}" if getattr(args, "fm_tilt", 0) else "")' in b
    print("fit_eval_fm: tilt off by default, after vp; t2p_bench: --fm-tilt reaches the kwargs + fm_args; tag only when set")
    torch.manual_seed(0)
    x = torch.randn(8, 5, 480)
    assert torch.equal(T.augment(x, T.Recipe("t0", tilt=0.0)), x), "tilt 0 -> unchanged"
    t = torch.arange(480) / T.TILT_FS
    s = torch.sin(2 * torch.pi * 10.0 * t).repeat(8, 5, 1)
    out = T.augment(s, T.Recipe("t1", tilt=0.5))
    assert torch.allclose(out.std(-1), s.std(-1), rtol=1e-3), "a 10 Hz sinusoid keeps its amplitude"
    torch.manual_seed(1); a = T.augment(x + 2 * s, T.Recipe("t1", tilt=0.5))
    torch.manual_seed(1); b1 = T.augment(x, T.Recipe("t1", tilt=0.5))
    torch.manual_seed(1); b2 = T.augment(s, T.Recipe("t1", tilt=0.5))
    assert torch.allclose(a, b1 + 2 * b2, atol=1e-4), "linear"
    print("augment: a tilt-only Recipe = the CNN tilt (off -> bitwise; 10 Hz kept; linear)")
    print("OK")


if __name__ == "__main__":
    main()
