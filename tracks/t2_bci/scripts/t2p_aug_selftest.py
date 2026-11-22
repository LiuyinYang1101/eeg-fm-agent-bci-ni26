#!/usr/bin/env python
"""Self-test (T2-D257 / D260): gain jitter + time shift (+ channel dropout) on the FM's target training windows (--fm-aug g:s[:c]).

  1. ni26.train.augment with Recipe(gain_jit=g): each window x channel is the input times one factor in [exp(-g), exp(g)];
     Recipe(tshift=s): each window is a circular roll of its input by one shift in [-s, s] samples; both: the roll of a gained window;
     Recipe(chdrop=c): each window x channel is the input or all zeros, zeroed at a rate near c (T2-D260);
  2. t2p_bench.fm_aug / task_rec: '' -> None and the adopted ST-L stem (T2-D247r's stl_rcll files) unchanged; '0.2:30' -> (0.2, 30) and
     tag aug0.2-30 after rc0.75, before cpt; '0:0:0.1' -> (0.0, 0, 0.1) and tag aug0-0-0.1; malformed strings refused;
  3. fit_eval_fm end to end (CPU, st_small, the adopted rc + mixup + LLRD path, 1 epoch): aug None twice = bitwise equal (off = the code
     path of T2-D247r); aug (0.2, 30) and aug (0, 0, 0.1) each change the result.
  python t2p_aug_selftest.py -> "OK"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402
from ni26 import train as TV  # noqa: E402

EXPORT = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
RCLL = "fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cptstlarge-d89b"     # T2-D247r's stl_rcll kit files


def test_augment():
    torch.manual_seed(0)
    x = torch.randn(16, 41, 480) + 0.1
    y = TV.augment(x, TV.Recipe("g", gain_jit=0.2))
    r = (y / x)
    assert torch.allclose(r, r[:, :, :1].expand_as(r), rtol=1e-4, atol=1e-5), "gain: one factor per window x channel"
    assert float(r.min()) >= np.exp(-0.2) - 1e-5 and float(r.max()) <= np.exp(0.2) + 1e-5 and float(r.std()) > 0.05
    y = TV.augment(x, TV.Recipe("t", tshift=30))
    sh = []
    for b in range(len(x)):
        hit = [s for s in range(-30, 31) if torch.equal(y[b], torch.roll(x[b], s, -1))]
        assert len(hit) == 1, (b, hit)
        sh += hit
    assert len(set(sh)) > 4, sh
    y = TV.augment(x, TV.Recipe("gt", gain_jit=0.2, tshift=30))
    for b in range(len(x)):
        r = [y[b] / torch.roll(x[b], s, -1) for s in range(-30, 31)]
        assert sum(bool(torch.allclose(q, q[:, :1].expand_as(q), rtol=1e-4, atol=1e-5)) for q in r) == 1, b
    assert torch.equal(TV.augment(x, TV.Recipe("off")), x)
    y = TV.augment(x, TV.Recipe("c", chdrop=0.1))
    z = (y == 0).all(-1)
    assert bool(((y == x).all(-1) | z).all()), "chdrop: a channel is kept or zeroed whole"
    assert 0.05 < float(z.float().mean()) < 0.15, float(z.float().mean())
    print(f"augment: gain one factor per window x channel in exp(+-0.2); shift one roll in [-30, 30] ({len(set(sh))} distinct); "
          f"both = the roll of a gained window; chdrop 0.1 zeroes whole channels at {float(z.float().mean()):.3f}; off = identity")


def test_tag():
    a = argparse.Namespace(fm_epochs=30, fm_bs=64, fm_lr_enc=1e-4, fm_lr_head=1e-3, fm_wd=0.05, fm_llrd=0.75, fm_ema=0.0,
                           fm_drop_path=0.0, fm_layer=-1, fm_lora=0, fm_mixup=0.4, fm_calib="gain", fm_vp=0, fm_rc=0.75,
                           fm_ckpt="/x/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth", fm_fuse="", fm_band="", fm_supcon=0.0,
                           fm_qkv_lr_mult=1.0, fm_layer_clip=False, fm_lora_in=0, fm_mae_retain=0, fm_side="", fm_sw=0.0, fm_aug="")
    assert B.fm_aug("") is None
    assert B.task_rec(a, "fm:st_large:mean") == RCLL, B.task_rec(a, "fm:st_large:mean")
    a.fm_aug = "0.2:30"
    assert B.fm_aug(a.fm_aug) == (0.2, 30)
    want = RCLL.replace("rc0.75cpt", "rc0.75aug0.2-30cpt")
    assert B.task_rec(a, "fm:st_large:mean") == want, B.task_rec(a, "fm:st_large:mean")
    a.fm_aug = "0:0:0.1"
    assert B.fm_aug(a.fm_aug) == (0.0, 0, 0.1)
    want3 = RCLL.replace("rc0.75cpt", "rc0.75aug0-0-0.1cpt")
    assert B.task_rec(a, "fm:st_large:mean") == want3, B.task_rec(a, "fm:st_large:mean")
    for bad in ("0.2", "0.2:30:1", "-0.2:30", "0:0", "0.20:30", "0.2:3.5", "0:0:0", "0.2:30:0", "0:0:0.10", "0:0:1.5", "0:0:0.1:1"):
        try:
            B.fm_aug(bad)
            raise RuntimeError(f"fm_aug accepted {bad!r}")
        except (AssertionError, ValueError):
            pass
    print(f"tag: off = the stl_rcll stem; 0.2:30 -> {want}; 0:0:0.1 -> {want3}; malformed refused")


def test_fit(idx, names):
    X = np.load(EXPORT / "X.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    tr = np.sort(np.concatenate([rng.choice(np.flatnonzero((idx["subject"] == u).to_numpy()), 12, replace=False)
                                 for u in ("P01", "P02")]))
    te = rng.choice(np.flatnonzero((idx["subject"] == "P04").to_numpy()), 8, replace=False)
    y = idx["y"].to_numpy().astype(int)
    Xa = np.asarray(X[np.concatenate([tr, te])], np.float32)
    Xa = Xa / np.median(Xa[:len(tr)].std(-1))
    cells = idx["stream"].to_numpy()
    data = dict(Xtr=Xa[:len(tr)], ytr=y[tr], Xva=Xa[:0], yva=y[:0], cva=cells[:0], Xte=Xa[len(tr):], yte=y[te], cte=cells[te],
                utr=(idx.loc[tr, "subject"] == "P02").to_numpy().astype(int), str=idx.loc[tr, "session"].astype(str).to_numpy())
    kw = dict(epochs=1, bs=8, llrd=0.75, mixup=0.4, rc=0.75, calib="gain", device="cpu", log=lambda m: None)
    p0 = t2p_fm.fit_eval_fm("st_small", data, 0, names, **kw)["prob_last"]
    p1 = t2p_fm.fit_eval_fm("st_small", data, 0, names, aug=None, **kw)["prob_last"]
    p2 = t2p_fm.fit_eval_fm("st_small", data, 0, names, aug=(0.2, 30), **kw)["prob_last"]
    p3 = t2p_fm.fit_eval_fm("st_small", data, 0, names, aug=(0.0, 0, 0.1), **kw)["prob_last"]
    d2, d3 = float(np.abs(p2 - p0).max()), float(np.abs(p3 - p0).max())
    assert np.array_equal(p0, p1) and d2 > 1e-3 and d3 > 1e-3, (float(np.abs(p1 - p0).max()), d2, d3)
    print(f"fit_eval_fm (st_small, rc + mixup + LLRD, CPU): aug None twice bitwise equal; aug (0.2, 30) max |d prob| {d2:.1e}; "
          f"aug (0, 0, 0.1) {d3:.1e}")


def main():
    _, idx, meta = B.load_export(EXPORT)
    test_augment()
    test_tag()
    test_fit(idx, meta["ch_names"])
    print("OK")


if __name__ == "__main__":
    main()
