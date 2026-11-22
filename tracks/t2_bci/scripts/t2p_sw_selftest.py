#!/usr/bin/env python
"""Self-test (T2-D253): the out-of-fold CNN folds, their read-back and the FM's per-window loss weights.

  1. folds kitoof / seen1oof: per training user one fold; the fold tests partition the base protocol's training rows; each
     fold trains on the other training users only (no val, no test row of the base protocol);
  2. t2p_bench.oof_pcnn on synthetic result files = FBCNet 4/7 + ShallowNet 3/7 of the seed means; a missing fold, a row
     twice or a second recipe fail;
  3. t2p_fm.wce: w = None = F.cross_entropy (label smoothing); w = 1 equal; other w = the weighted mean;
  4. fit_eval_fm end to end (CPU, st_small, the adopted rc + mixup + LLRD path, 1 epoch): sw = 1 = sw None (prob max |d| <
     1e-4); a non-uniform sw changes the result.
  python t2p_sw_selftest.py -> "OK"
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402

EXPORT = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def test_folds(idx):
    for proto, n in (("kit", 6), ("seen1", 7)):
        _, trn, _, _ = B.folds(idx, proto)[0]
        fo = B.folds(idx, proto + "oof")
        assert len(fo) == n, (proto, len(fo))
        seen = np.zeros(len(idx), int)
        for fname, tr, va, te in fo:
            assert not va.any() and te.any() and tr.any(), fname
            assert np.array_equal(tr | te, trn) and not (tr & te).any(), fname
            assert idx.loc[te, "subject"].nunique() == 1 and fname.endswith(idx.loc[te, "subject"].iloc[0]), fname
            seen += te
        assert np.array_equal(seen, trn.astype(int)), proto
        print(f"folds {proto}oof: {n} folds, the tests partition {int(trn.sum())} {proto} training rows")


def test_oof_pcnn(idx):
    rng = np.random.default_rng(0)
    y = idx["y"].to_numpy().astype(int)
    fo = B.folds(idx, "kitoof")
    rows = np.flatnonzero(B.folds(idx, "kit")[0][1])
    with tempfile.TemporaryDirectory() as d:
        truth = {}
        for m, rec in (("fbcnet_bboth", "none__n00__rA"), ("shallow", "ea__n00__rB")):
            for s in (5, 6):
                a = np.zeros((len(idx), 3))
                for fname, _, _, te in fo:
                    p = rng.dirichlet(np.ones(3), te.sum())
                    a[te] = p
                    np.savez(Path(d) / f"{fname}__{m}__{rec}__s{s}.npz", test_idx=np.flatnonzero(te), prob_last=p.astype(np.float32))
                truth[(m, s)] = a
        want = sum(w * (truth[(m, 5)] + truth[(m, 6)]) / 2 for m, w in B.SW_W.items())[rows, y[rows]]
        got = B.oof_pcnn(d, "kit", y, rows)
        assert np.abs(got - want).max() < 1e-6, np.abs(got - want).max()
        f0 = Path(d) / f"{fo[0][0]}__shallow__ea__n00__rB__s6.npz"
        for bad in ("missing", "twice", "recipe"):
            with tempfile.TemporaryDirectory() as d2:
                for f in Path(d).glob("*.npz"):
                    if not (bad == "missing" and f == f0):
                        (Path(d2) / f.name).write_bytes(f.read_bytes())
                if bad == "twice":       # the same rows again under another fold name
                    (Path(d2) / f0.name.replace(fo[0][0], fo[1][0] + "x")).write_bytes(f0.read_bytes())
                if bad == "recipe":
                    (Path(d2) / f0.name.replace("rB", "rC").replace("s6", "s7")).write_bytes(f0.read_bytes())
                try:
                    B.oof_pcnn(d2, "kit", y, rows)
                    raise RuntimeError(f"oof_pcnn accepted a {bad} file")
                except AssertionError:
                    pass
    print(f"oof_pcnn: = the 4/7 + 3/7 blend of the seed means on {len(rows)} rows; missing / twice / second recipe refused")


def test_wce():
    g = torch.Generator().manual_seed(0)
    lo, y = torch.randn(16, 3, generator=g), torch.randint(0, 3, (16,), generator=g)
    a = t2p_fm.wce(lo, y, 0.1)
    assert torch.equal(a, F.cross_entropy(lo, y, label_smoothing=0.1))
    assert abs(float(t2p_fm.wce(lo, y, 0.1, torch.ones(16)) - a)) < 1e-6
    w = torch.rand(16, generator=g) * 2
    m = (F.cross_entropy(lo, y, label_smoothing=0.1, reduction="none") * w).mean()
    assert abs(float(t2p_fm.wce(lo, y, 0.1, w) - m)) < 1e-6
    print("wce: None = F.cross_entropy, w = 1 equal, other w = the weighted mean")


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
    p1 = t2p_fm.fit_eval_fm("st_small", data, 0, names, sw=np.ones(len(tr), np.float32), **kw)["prob_last"]
    w = np.where(np.arange(len(tr)) % 2, 1.9, 0.1).astype(np.float32)
    p2 = t2p_fm.fit_eval_fm("st_small", data, 0, names, sw=w, **kw)["prob_last"]
    d1, d2 = float(np.abs(p1 - p0).max()), float(np.abs(p2 - p0).max())
    assert d1 < 1e-4 and d2 > 1e-3, (d1, d2)
    print(f"fit_eval_fm (st_small, rc + mixup + LLRD, CPU): sw = 1 vs None max |d prob| {d1:.1e}; a 0.1 / 1.9 sw {d2:.1e}")


def main():
    _, idx, meta = B.load_export(EXPORT)
    test_folds(idx)
    test_oof_pcnn(idx)
    test_wce()
    test_fit(idx, meta["ch_names"])
    print("OK")


if __name__ == "__main__":
    main()
