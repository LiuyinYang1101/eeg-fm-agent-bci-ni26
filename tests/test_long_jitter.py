#!/usr/bin/env python
"""T2-D486 (real out-of-window training data): t2p_bench --long-export / --long-jitter + ni26.train.long_crop, CPU, synthetic:
  1. long_aligned recovers each window's alignment operator from its contract window ('none' and 'ea', two streams) and a crop at the
     contract offset reproduces the aligned contract window (relative error < 1e-4);
  2. long_crop draws starts in [lo, hi] and demeans each crop;
  3. long_check passes a row-aligned long export whose contract slice is the export bitwise, and refuses one changed sample / a shuffled row;
  4. fit_eval refuses an ltr of the wrong length and trains with one (1 epoch, shallow, CPU); finetune takes Lx.
  python tests/test_long_jitter.py  ->  "OK"
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402
from ni26 import train as T  # noqa: E402

CH = ['C2', 'CP2', 'P2', 'Pz', 'CPz', 'Cz', 'C1', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'TP7', 'CP5', 'CP3', 'C3']


def export(d, X, idx, start):
    d.mkdir(parents=True)
    np.save(d / "X.npy", X.astype(np.float32))
    idx.to_csv(d / "index.csv", index=False)
    (d / "meta.json").write_text(json.dumps(dict(sfreq=120.0, n_times=X.shape[-1], ch_names=CH, window_start=start)))


def main():
    rng = np.random.default_rng(0)
    N, C, TL, Tw, off = 24, len(CH), 1320, 480, 480
    XL = (rng.standard_normal((N, C, TL)) * 10 + rng.standard_normal((N, C, 1)) * 3).astype(np.float32)
    Xc = np.ascontiguousarray(XL[:, :, off:off + Tw])
    idx = pd.DataFrame(dict(i=np.arange(N), subject=["Dreyer2026Proteus/P01"] * 12 + ["Dreyer2026Proteus/P02"] * 12, session="1",
                            order=list(range(12)) * 2, y=rng.integers(0, 3, N), split="train"))
    with tempfile.TemporaryDirectory() as tmp:
        Tp = Path(tmp)
        export(Tp / "contract", Xc, idx, 1.0)
        export(Tp / "long", XL, idx, -3.0)
        X, ix, meta = B.load_export(Tp / "contract")
        XLm, o_, ls_ = B.long_check(Tp / "long", X, ix, meta)
        assert (o_, ls_) == (480, -3.0), (o_, ls_)
        bad = XL.copy(); bad[5, 3, off + 10] += 1e-3
        export(Tp / "long_bad", bad, idx, -3.0)
        try:
            B.long_check(Tp / "long_bad", X, ix, meta); raise AssertionError("a changed sample must be refused")
        except AssertionError as e:
            assert "contract slice differs" in str(e), e
        export(Tp / "long_shuf", XL, idx.assign(order=idx.order[::-1].to_numpy()), -3.0)
        try:
            B.long_check(Tp / "long_shuf", X, ix, meta); raise AssertionError("a shuffled row order must be refused")
        except AssertionError as e:
            assert "not row-aligned" in str(e), e
        tm = np.ones(N, bool)
        rows = np.arange(N)
        for mode, scale in (("none", 7.0), ("ea", 1.0)):
            XA = B.aligned_all(X, ix, mode, tm, 0.0, 0.05, scale)
            LA, res = B.long_aligned(X, XA, XLm, rows)
            assert res < 1e-5, (mode, res)
            T.LONG.update(lo=off, hi=off)
            g = torch.Generator().manual_seed(0)
            cr = T.long_crop(torch.as_tensor(LA), Tw, off, off, g).numpy()
            err = np.abs(cr - XA).max() / np.abs(XA).max()
            assert err < 1e-4, (mode, err)
        g = torch.Generator().manual_seed(1)
        cr = T.long_crop(torch.as_tensor(LA), Tw, 420, 660, g)
        assert cr.shape == (N, C, Tw) and float(cr.mean(-1).abs().max()) < 1e-3
        st = torch.randint(420, 661, (N,), generator=torch.Generator().manual_seed(1))
        ref = np.stack([LA[k][:, s:s + Tw] - LA[k][:, s:s + Tw].mean(-1, keepdims=True) for k, s in enumerate(st.tolist())])
        assert np.allclose(cr.numpy(), ref, atol=1e-4), "crop k starts at the generator's k-th draw"
        T.LONG.update(lo=420, hi=660)
        rec = T.RECIPES["r1"]
        from dataclasses import replace
        rec = replace(rec, epochs=1)
        data = dict(Xtr=XA, ytr=idx.y.to_numpy(), Xva=XA[:0], yva=idx.y.to_numpy()[:0], cva=np.array([]), Xte=XA[:6], yte=idx.y.to_numpy()[:6],
                    cte=np.array(["c"] * 6), utr=np.repeat([0, 1], 12), str=np.array(["1"] * N), ltr=LA[:-1])
        try:
            T.fit_eval("shallow", data, rec, 0, device="cpu", log_every=1); raise AssertionError("a short ltr must be refused")
        except AssertionError as e:
            assert "one long window per training window" in str(e), e
        data["ltr"] = LA
        r = T.fit_eval("shallow", data, rec, 0, device="cpu", log_every=1, return_state=True)
        assert r["prob_last"].shape == (6, 3)
        net = T.build_model("shallow", C, 3, Tw, 120.0).eval()
        net.load_state_dict(r["state_last"])
        m = T.finetune(net, torch.as_tensor(XA[:12]), torch.as_tensor(idx.y.to_numpy()[:12]), replace(T.peruser_for(T.RECIPES["r1_pulx"]), epochs=1), 3,
                       Lx=torch.as_tensor(LA[:12]))
        assert m is not None
        T.LONG.clear()
    print("OK")


if __name__ == "__main__":
    main()
