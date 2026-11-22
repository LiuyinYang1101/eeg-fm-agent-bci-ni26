#!/usr/bin/env python
"""Self-test (T2-D263): linear probe, then fine-tune (LP-FT) for the FM member (--fm-lp-epochs).

  1. t2p_bench.task_rec: --fm-lp-epochs 0 (the default) and a Namespace without fm_lp_epochs -> the adopted ST-L stem (T2-D247r's
     stl_rcll files) unchanged; 5 -> tag lp5 after rc0.75 (and after ls, when given), before cpt;
  2. fit_eval_fm end to end (CPU, st_small, the adopted rc + mixup + LLRD path, 2 epochs, 3 steps each): no lp_epochs and lp_epochs 0 =
     bitwise equal (the bench now passes lp_epochs = 0, the code path of T2-D247r); lp_epochs 1 changes the result, and an AdamW recorder
     shows, for the probe's 3 steps: every encoder gradient None, the encoder lr 0, the encoder weights unchanged, the head trained at its
     own schedule; from step 4: encoder gradients present, the encoder lr re-warmed from its own step 0 (no warm-up step at 3 FT steps:
     int(0.1 x 3) = 0, so the cosine starts at 1) and the encoder weights moving; every trainable tensor trainable again at the end.
  python t2p_lp_selftest.py -> "OK"
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

REC = []


class RecAdamW(torch.optim.AdamW):
    """AdamW that records, before each step: (encoder lr max, encoder grads all None, head lr max, head grads present, encoder checksum)."""

    def step(self, closure=None):
        enc = [g for g in self.param_groups if str(g.get("name", "")).startswith("enc_")]
        head = [g for g in self.param_groups if str(g.get("name", "")).startswith("head")]
        REC.append((max(g["lr"] for g in enc), all(p.grad is None for g in enc for p in g["params"]),
                    max(g["lr"] for g in head), all(p.grad is not None for g in head for p in g["params"]),
                    float(sum(p.detach().double().sum() for g in enc for p in g["params"]))))
        return super().step(closure)


def test_tag():
    m = "fm:st_large:mean"
    assert B.task_rec(argparse.Namespace(**NS), m) == RCLL, "no fm_lp_epochs attribute -> unchanged"
    assert B.task_rec(argparse.Namespace(**NS, fm_lp_epochs=0), m) == RCLL, "lp 0 -> unchanged"
    want = RCLL.replace("rc0.75cpt", "rc0.75lp5cpt")
    assert B.task_rec(argparse.Namespace(**NS, fm_lp_epochs=5), m) == want, B.task_rec(argparse.Namespace(**NS, fm_lp_epochs=5), m)
    both = B.task_rec(argparse.Namespace(**NS, fm_ls=0.2, fm_lp_epochs=5), m)
    assert both == RCLL.replace("rc0.75cpt", "rc0.75ls0.2lp5cpt"), both
    import inspect
    assert "\"--fm-lp-epochs\", type=int, default=0" in inspect.getsource(B), "the parser default is 0"
    print(f"tag: no fm_lp_epochs / 0 = the stl_rcll stem; 5 -> {want}; with ls 0.2 -> ...ls0.2lp5cpt...")


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
    kw = dict(epochs=2, bs=8, llrd=0.75, mixup=0.4, rc=0.75, calib="gain", device="cpu", log=lambda m: None)
    p0 = t2p_fm.fit_eval_fm("st_small", data, 0, names, **kw)["prob_last"]
    p1 = t2p_fm.fit_eval_fm("st_small", data, 0, names, lp_epochs=0, **kw)["prob_last"]
    msgs = []
    orig = torch.optim.AdamW
    torch.optim.AdamW = RecAdamW
    try:
        r2 = t2p_fm.fit_eval_fm("st_small", data, 0, names, lp_epochs=1, return_model=True, **dict(kw, log=msgs.append))
    finally:
        torch.optim.AdamW = orig
    p2 = r2["prob_last"]
    d2 = float(np.abs(p2 - p0).max())
    assert np.array_equal(p0, p1) and d2 > 1e-4, (float(np.abs(p1 - p0).max()), d2)
    spe = 3                                                       # 24 windows / batch 8
    assert len(REC) == 2 * spe, len(REC)
    lp, ft = REC[:spe], REC[spe:]
    assert all(e_lr == 0.0 and e_none and h_lr > 0 and h_grad for e_lr, e_none, h_lr, h_grad, _ in lp), lp
    assert len({c for *_, c in lp} | {ft[0][4]}) == 1, "the encoder must not move during the probe"
    assert all(e_lr > 0 and not e_none and h_grad for e_lr, e_none, _, h_grad, _ in ft), ft
    assert ft[1][4] != ft[0][4], "the encoder moves once the fine-tune starts"
    lr_enc_top = 1e-4                                             # fit_eval_fm's default lr_enc; the top group at llrd ** 0
    want = [lr_enc_top * 0.5 * (1 + math.cos(math.pi * s / 3)) for s in range(3)]   # w2 = int(0.1 x 3) = 0: cosine from its step 0
    assert np.allclose([e for e, *_ in ft], want, rtol=1e-6), ([e for e, *_ in ft], want)
    wsteps, total = int(0.1 * 6), 6
    want_h = [1e-3 * ((s + 1) / max(wsteps, 1) if s < wsteps else 0.5 * (1 + math.cos(math.pi * (s - wsteps) / max(total - wsteps, 1))))
              for s in range(6)]
    assert np.allclose([h for _, _, h, _, _ in REC], want_h, rtol=1e-6), "the head keeps the whole run's schedule"
    lm = [m for m in msgs if m.startswith("linear probe: epochs 1-1 of 2 train the head only (")]
    assert len(lm) == 1, msgs
    n_frozen = int(lm[0].split("(")[1].split()[0])
    assert n_frozen > 0 and sum(q.requires_grad for q in r2["net_last"].enc.parameters()) == n_frozen, "trainable again at the end"
    print(f"fit_eval_fm (st_small, rc + mixup + LLRD, CPU): no lp / lp 0 bitwise equal; lp 1 max |d prob| {d2:.1e}; probe steps: encoder "
          f"grads None, lr 0, weights unchanged; FT steps: encoder lr {[f'{e:.2e}' for e, *_ in ft]} (cosine from its own step 0), "
          f"weights moving; head lr = the 6-step schedule")


def main():
    _, idx, meta = B.load_export(EXPORT)
    test_tag()
    test_fit(idx, meta["ch_names"])
    print("OK")


if __name__ == "__main__":
    main()
