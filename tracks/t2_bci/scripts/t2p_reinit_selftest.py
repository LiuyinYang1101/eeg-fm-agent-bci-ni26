#!/usr/bin/env python
"""Self-test (T2-D264): the ST member's top encoder block(s) re-initialised before its fine-tune (--fm-reinit-top).

  1. t2p_bench.task_rec: --fm-reinit-top 0 (the default) and a Namespace without fm_reinit_top -> the adopted ST-L stem (T2-D247r's
     stl_rcll files) unchanged; 1 -> tag ri1 after rc0.75 (and after lp, when given), before cpt;
  2. fit_eval_fm end to end (CPU, st_small, the adopted rc + mixup + LLRD path, 1 epoch): no reinit_top and reinit_top 0 = bitwise equal
     (the bench now passes reinit_top = 0, the code path of T2-D247r); reinit_top 1 changes the result; the model as the optimiser first
     sees it (before any update) shows, against the reinit_top 0 run: the top block's and the final norm's LayerNorms at weight 1 / bias 0,
     its Linear biases 0, its Linear weights inside the xavier-uniform bound with the uniform's sd (within 5 %) and none equal to before;
     every other tensor (the blocks below, the embeddings, the head) bitwise equal = the re-init's own generator leaves every other draw
     as the baseline's.
  python t2p_reinit_selftest.py -> "OK"
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402
from t2p_aug_selftest import EXPORT, RCLL  # noqa: E402
from t2p_ls_selftest import NS  # noqa: E402

NETS, SNAP = [], []


class RecFMClf(t2p_fm.FMClf):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        NETS.append(self)


class RecAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        if len(SNAP) < len(NETS):                                 # the first step of this run: the model before any update
            SNAP.append({k: v.detach().clone() for k, v in NETS[-1].state_dict().items()})
        return super().step(closure)


def test_tag():
    m = "fm:st_large:mean"
    assert B.task_rec(argparse.Namespace(**NS), m) == RCLL, "no fm_reinit_top attribute -> unchanged"
    assert B.task_rec(argparse.Namespace(**NS, fm_reinit_top=0), m) == RCLL, "ri 0 -> unchanged"
    want = RCLL.replace("rc0.75cpt", "rc0.75ri1cpt")
    assert B.task_rec(argparse.Namespace(**NS, fm_reinit_top=1), m) == want, B.task_rec(argparse.Namespace(**NS, fm_reinit_top=1), m)
    both = B.task_rec(argparse.Namespace(**NS, fm_lp_epochs=5, fm_reinit_top=1), m)
    assert both == RCLL.replace("rc0.75cpt", "rc0.75lp5ri1cpt"), both
    import inspect
    assert "\"--fm-reinit-top\", type=int, default=0" in inspect.getsource(B), "the parser default is 0"
    print(f"tag: no fm_reinit_top / 0 = the stl_rcll stem; 1 -> {want}; with lp 5 -> ...lp5ri1cpt...")


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
    o_cls, o_opt = t2p_fm.FMClf, torch.optim.AdamW
    t2p_fm.FMClf, torch.optim.AdamW = RecFMClf, RecAdamW
    msgs = []
    try:
        p0 = t2p_fm.fit_eval_fm("st_small", data, 0, names, **kw)["prob_last"]
        p1 = t2p_fm.fit_eval_fm("st_small", data, 0, names, reinit_top=0, **kw)["prob_last"]
        p2 = t2p_fm.fit_eval_fm("st_small", data, 0, names, reinit_top=1, **dict(kw, log=msgs.append))["prob_last"]
    finally:
        t2p_fm.FMClf, torch.optim.AdamW = o_cls, o_opt
    d2 = float(np.abs(p2 - p0).max())
    assert np.array_equal(p0, p1) and d2 > 1e-4, (float(np.abs(p1 - p0).max()), d2)
    assert len(SNAP) == 3 and all(torch.equal(SNAP[0][k], SNAP[1][k]) for k in SNAP[0]), "reinit 0 = the default, before any update"
    n = NETS[-1].enc.n_layers
    top = (f"enc.vit.blocks.{n - 1}.", "enc.vit.norm.")
    a_, b_ = SNAP[0], SNAP[2]
    assert set(a_) == set(b_)
    moved = sorted(k for k in a_ if k.startswith(top))
    assert moved and all(not torch.equal(a_[k], b_[k]) for k in moved if not (k.endswith("bias") and not a_[k].abs().any())), "top block re-drawn"
    for k in moved:
        w = b_[k]
        if "norm" in k:
            assert torch.equal(w, torch.ones_like(w) if k.endswith("weight") else torch.zeros_like(w)), k
        elif k.endswith("bias"):
            assert not w.abs().any(), k
        else:
            bound = math.sqrt(6.0 / (w.shape[0] + w.shape[1]))
            assert float(w.abs().max()) <= bound + 1e-6 and abs(float(w.std()) / (bound / math.sqrt(3)) - 1) < 0.05, (k, float(w.std()), bound)
    same = [k for k in a_ if k not in moved]
    assert same and all(torch.equal(a_[k], b_[k]) for k in same), [k for k in same if not torch.equal(a_[k], b_[k])][:5]
    assert any(m.startswith(f"re-initialised the top 1 of {n} encoder blocks") for m in msgs), msgs
    print(f"fit_eval_fm (st_small, rc + mixup + LLRD, CPU): no reinit / reinit 0 bitwise equal; reinit 1 max |d prob| {d2:.1e}; "
          f"block {n} of {n} + the final norm re-drawn ({len(moved)} tensors: LN 1 / 0, biases 0, xavier-uniform weights); "
          f"the other {len(same)} tensors bitwise equal to the baseline's")


def main():
    _, idx, meta = B.load_export(EXPORT)
    test_tag()
    test_fit(idx, meta["ch_names"])
    print("OK")


if __name__ == "__main__":
    main()
