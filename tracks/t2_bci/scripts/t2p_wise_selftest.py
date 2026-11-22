#!/usr/bin/env python
"""Self-test (T2-D270): WiSE-FT on LP-FT for the FM member (--fm-wise).

  1. t2p_bench.task_rec: --fm-wise 0 (the default) and a Namespace without fm_wise -> the adopted ST-L stem (T2-D247r's stl_rcll
     files) unchanged; lp 10 + wise 0.5 -> tag ws0.5 after lp10 (and after ri, when given), before cpt;
  2. fit_eval_fm end to end (CPU, st_small, the adopted rc + mixup + LLRD path, LP-FT lp 1 of 2 epochs, 3 steps each): lp 1 without wise
     and with wise 0 = bitwise equal (the bench now passes wise = 0, T2-D263 / D267's code path); lp 1 + wise 0.5 trains exactly as lp 1
     (a recorder sees the same model at every step) and its evaluated model = the lerp of the fine-tune's end (the wise-0 run's last model)
     and the probe's end (the model at step 3, before its update): its encoder = the midpoint of the CPT'd encoder and the fine-tuned one,
     its head = the midpoint of the probed head and the fine-tuned head; prob_last = that model's probabilities, and differs from lp 1's.
  python t2p_wise_selftest.py -> "OK"
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
from t2p_aug_selftest import EXPORT, RCLL  # noqa: E402
from t2p_ls_selftest import NS  # noqa: E402

NETS, SNAPS = [], []


class RecFMClf(t2p_fm.FMClf):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        NETS.append(self)
        SNAPS.append([])


class RecAdamW(torch.optim.AdamW):
    def step(self, closure=None):                                 # the model before each update
        SNAPS[-1].append({k: v.detach().clone() for k, v in NETS[-1].state_dict().items()})
        return super().step(closure)


def test_tag():
    m = "fm:st_large:mean"
    assert B.task_rec(argparse.Namespace(**NS), m) == RCLL, "no fm_wise attribute -> unchanged"
    assert B.task_rec(argparse.Namespace(**NS, fm_wise=0.0), m) == RCLL, "wise 0 -> unchanged"
    want = RCLL.replace("rc0.75cpt", "rc0.75lp10ws0.5cpt")
    got = B.task_rec(argparse.Namespace(**NS, fm_lp_epochs=10, fm_wise=0.5), m)
    assert got == want, got
    both = B.task_rec(argparse.Namespace(**NS, fm_lp_epochs=5, fm_reinit_top=1, fm_wise=0.5), m)
    assert both == RCLL.replace("rc0.75cpt", "rc0.75lp5ri1ws0.5cpt"), both
    import inspect
    assert "\"--fm-wise\", type=float, default=0.0" in inspect.getsource(B), "the parser default is 0"
    print(f"tag: no fm_wise / 0 = the stl_rcll stem; lp 10 + wise 0.5 -> {want}; with ri 1 -> ...lp5ri1ws0.5cpt...")


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
    kw = dict(epochs=2, bs=8, llrd=0.75, mixup=0.4, rc=0.75, calib="gain", device="cpu", log=lambda m: None, lp_epochs=1, return_model=True)
    o_cls, o_opt = t2p_fm.FMClf, torch.optim.AdamW
    t2p_fm.FMClf, torch.optim.AdamW = RecFMClf, RecAdamW
    msgs = []
    try:
        r0 = t2p_fm.fit_eval_fm("st_small", data, 0, names, **kw)
        r1 = t2p_fm.fit_eval_fm("st_small", data, 0, names, wise=0.0, **kw)
        r2 = t2p_fm.fit_eval_fm("st_small", data, 0, names, wise=0.5, **dict(kw, log=msgs.append))
    finally:
        t2p_fm.FMClf, torch.optim.AdamW = o_cls, o_opt
    p0, p1, p2 = r0["prob_last"], r1["prob_last"], r2["prob_last"]
    d2 = float(np.abs(p2 - p0).max())
    assert np.array_equal(p0, p1) and d2 > 1e-4, (float(np.abs(p1 - p0).max()), d2)
    spe = 3                                                       # 24 windows / batch 8
    assert [len(s) for s in SNAPS] == [2 * spe] * 3, [len(s) for s in SNAPS]
    for a_, b_ in zip(SNAPS[0], SNAPS[2]):                        # wise trains exactly as lp 1
        assert all(torch.equal(a_[k], b_[k]) for k in a_), "wise must not change the training"
    lp_end, start = SNAPS[0][spe], SNAPS[0][0]
    ft_end = {k: v for k, v in r0["net_last"].state_dict().items()}
    got = r2["net_last"].state_dict()
    enc_keys = [k for k in got if k.startswith("enc.")]
    assert enc_keys and all(torch.equal(lp_end[k], start[k]) for k in enc_keys), "the probe's end keeps the CPT'd encoder"
    n_mid = 0
    for k, v in got.items():
        if v.is_floating_point():
            want = ft_end[k].clone().lerp_(lp_end[k], 0.5)
            assert torch.allclose(v, want, atol=1e-7, rtol=0), k
            n_mid += int(not torch.equal(ft_end[k], lp_end[k]))
        else:
            assert torch.equal(v, ft_end[k]), k
    assert n_mid > 0
    p_mid = t2p_fm.predict_proba(r2["net_last"], torch.as_tensor(data["Xte"]))
    assert np.allclose(p_mid, p2, atol=1e-6), float(np.abs(p_mid - p2).max())
    assert any(m.startswith("WiSE-FT: the evaluated weights = 0.5 x the fine-tune's end + 0.5 x the probe's end (step 3") for m in msgs), msgs
    assert any(m.startswith("WiSE-FT 0.5: test ") for m in msgs), msgs
    print(f"fit_eval_fm (st_small, rc + mixup + LLRD, LP-FT lp 1 of 2, CPU): no wise / wise 0 bitwise equal; wise 0.5 trains as lp 1 "
          f"(all {2 * spe} steps' models equal), its model = the midpoint of the probe's end (CPT'd encoder + probed head) and the "
          f"fine-tune's end ({n_mid} tensors moved, {len(enc_keys)} encoder tensors); max |d prob| vs lp 1 {d2:.1e}")


def main():
    _, idx, meta = B.load_export(EXPORT)
    test_tag()
    test_fit(idx, meta["ch_names"])
    print("OK")


if __name__ == "__main__":
    main()
