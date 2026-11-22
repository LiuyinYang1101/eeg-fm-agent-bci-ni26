#!/usr/bin/env python
"""Self-test (T2-D261): label smoothing of the FM fine-tune (--fm-ls).

  1. t2p_bench.task_rec: --fm-ls 0.1 (the default) and a Namespace without fm_ls -> the adopted ST-L stem (T2-D247r's stl_rcll files)
     unchanged; 0.2 -> tag ls0.2 after rc0.75 (and after aug, when given), before cpt; the argument parser's default is 0.1;
  2. fit_eval_fm end to end (CPU, st_small, the adopted rc + mixup + LLRD path, 1 epoch): no ls and ls 0.1 = bitwise equal (the bench
     now passes ls=args.fm_ls = 0.1, the code path of T2-D247r); ls 0.2 changes the result.
  python t2p_ls_selftest.py -> "OK"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402
from t2p_aug_selftest import EXPORT, RCLL  # noqa: E402

NS = dict(fm_epochs=30, fm_bs=64, fm_lr_enc=1e-4, fm_lr_head=1e-3, fm_wd=0.05, fm_llrd=0.75, fm_ema=0.0, fm_drop_path=0.0, fm_layer=-1,
          fm_lora=0, fm_mixup=0.4, fm_calib="gain", fm_vp=0, fm_rc=0.75, fm_ckpt="/x/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth",
          fm_fuse="", fm_band="", fm_supcon=0.0, fm_qkv_lr_mult=1.0, fm_layer_clip=False, fm_lora_in=0, fm_mae_retain=0, fm_side="",
          fm_sw=0.0, fm_aug="")


def test_tag():
    m = "fm:st_large:mean"
    assert B.task_rec(argparse.Namespace(**NS), m) == RCLL, "no fm_ls attribute -> unchanged"
    assert B.task_rec(argparse.Namespace(**NS, fm_ls=0.1), m) == RCLL, "ls 0.1 -> unchanged"
    want = RCLL.replace("rc0.75cpt", "rc0.75ls0.2cpt")
    assert B.task_rec(argparse.Namespace(**NS, fm_ls=0.2), m) == want, B.task_rec(argparse.Namespace(**NS, fm_ls=0.2), m)
    both = B.task_rec(argparse.Namespace(**dict(NS, fm_aug="0:30"), fm_ls=0.2), m)
    assert both == RCLL.replace("rc0.75cpt", "rc0.75aug0-30ls0.2cpt"), both
    import inspect
    assert "\"--fm-ls\", type=float, default=0.1" in inspect.getsource(B), "the parser default is 0.1"
    print(f"tag: no fm_ls / 0.1 = the stl_rcll stem; 0.2 -> {want}; with aug 0:30 -> ...aug0-30ls0.2cpt...")


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
    p1 = t2p_fm.fit_eval_fm("st_small", data, 0, names, ls=0.1, **kw)["prob_last"]
    p2 = t2p_fm.fit_eval_fm("st_small", data, 0, names, ls=0.2, **kw)["prob_last"]
    d2 = float(np.abs(p2 - p0).max())
    assert np.array_equal(p0, p1) and d2 > 1e-4, (float(np.abs(p1 - p0).max()), d2)
    print(f"fit_eval_fm (st_small, rc + mixup + LLRD, CPU): no ls / ls 0.1 bitwise equal; ls 0.2 max |d prob| {d2:.1e}")


def main():
    _, idx, meta = B.load_export(EXPORT)
    test_tag()
    test_fit(idx, meta["ch_names"])
    print("OK")


if __name__ == "__main__":
    main()
