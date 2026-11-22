#!/usr/bin/env python
"""T2-D524: the FM path's window-mean side head (t2p_fm.fit_eval_fm side='dcmean'), CPU, a random-init ST-base cut to 1 layer:
  every row is the SAME window, so only the side vector can carry the label; in 36 optimiser steps (CPU budget) the side head lifts the test
  accuracy to >= 0.6 on the plain and the mixup path, while without the side input the net stays at chance (<= 0.4). Measured 2026-11-15:
  0.65 / 0.68 vs 0.33 (a 0.9 bar was too strict for 36 steps; the head learns, the path is wired).
  python tests/test_t2p_fm_side.py  ->  "OK"
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def data(rng, n=96, nt=60):
    C, L = len(CH), 480
    y, yt = (np.arange(n) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    w0 = rng.standard_normal((C, L)).astype(np.float32)
    eye = np.eye(3, dtype=np.float32)
    S = lambda yy: np.hstack([eye[yy] * 3, rng.standard_normal((len(yy), 79)).astype(np.float32)])  # noqa: E731  (82 dims)
    return dict(Xtr=np.repeat(w0[None], n, 0), ytr=y, Xva=np.zeros((0, C, L), np.float32), yva=y[:0], cva=np.array([], dtype=object),
                Xte=np.repeat(w0[None], nt, 0), yte=yt, cte=np.array(["a"] * nt, dtype=object), ext=[], utr=np.zeros(n, np.int64),
                side_tr=S(y), side_te=S(yt), side_va=np.zeros((0, 82), np.float32))


def main():
    torch.set_num_threads(4)
    kw = dict(epochs=12, bs=32, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", log=lambda m: None, layer=1, calib="gain", ls=0.0)
    acc = lambda r, d: float((r["prob_last"].argmax(1) == d["yte"]).mean())  # noqa: E731
    d = data(np.random.default_rng(0))
    r1 = FM.fit_eval_fm("rand_st_base", d, 0, CH, side="dcmean", **kw)
    r2 = FM.fit_eval_fm("rand_st_base", d, 0, CH, side="dcmean", mixup=0.4, **kw)
    r0 = FM.fit_eval_fm("rand_st_base", d, 0, CH, side="", **kw)
    a1, a2, a0 = acc(r1, d), acc(r2, d), acc(r0, d)
    assert a1 >= 0.6 and a2 >= 0.6 and a0 <= 0.4, (a1, a2, a0)
    print(f"1. side head learned: plain {a1:.2f}, mixup {a2:.2f}; without the side input {a0:.2f}")
    print("OK")


if __name__ == "__main__":
    main()
