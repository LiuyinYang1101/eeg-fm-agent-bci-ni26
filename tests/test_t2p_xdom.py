#!/usr/bin/env python
"""T2-D507: the external-domain options of the pooled CNN (ni26.train.XDOM + data["dtr"]) and the CAR of external windows (CPU):
  1. DomainBN routes the masked rows through bn1 and the others through bn0; xdom_unwrap puts bn0 back (plain state_dict keys);
  2. XDOM set but no data["dtr"] = the plain fit bitwise (same seed);
  3. an XDOM fit (own head, weight 0.25, domain BN) runs, and its state_last has exactly the plain net's keys;
  4. aligned_extra(car=True) gives windows whose channel mean is 0 before alignment ("none" only removes each channel's mean in time and scales).
  python tests/test_t2p_xdom.py  ->  "OK"
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
from ni26 import train as T  # noqa: E402


def data(rng, n=24, C=41, L=480, ext=0):
    X = rng.standard_normal((n + ext, C, L)).astype(np.float32)
    y = (np.arange(n + ext) % 3).astype(np.int64)
    Xte = rng.standard_normal((9, C, L)).astype(np.float32)
    d = dict(Xtr=X, ytr=y, Xva=Xte[:0], yva=y[:0], cva=np.array([], dtype=object), Xte=Xte, yte=(np.arange(9) % 3).astype(np.int64),
             cte=np.array(["a"] * 9, dtype=object), ext=[], utr=np.zeros(n, np.int64), str=np.array(["1"] * n), ltr=None)
    return d


def main():
    # 1. DomainBN
    st = {}
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.BatchNorm2d(4))
    ref = {k: v.clone() for k, v in net.state_dict().items()}
    assert T.xdom_wrap(net, st) == ["1"]
    net.train()
    x = torch.randn(6, 2, 5, 1)
    st["m"] = torch.tensor([True, False, True, False, False, False])
    net(x)
    bn = net[1]
    assert not torch.allclose(bn.bn0.running_mean, bn.bn1.running_mean), "each domain updates its own statistics"
    st["m"] = None
    T.xdom_unwrap(net)
    assert isinstance(net[1], torch.nn.BatchNorm2d) and set(net.state_dict()) == set(ref), "unwrap = the plain keys"
    # 2. XDOM set, no dtr -> bitwise the plain fit
    rec = replace(T.RECIPES["r1"], epochs=1)
    T.XDOM.clear()
    a = T.fit_eval("shallow", data(np.random.default_rng(0)), rec, 0, device="cpu", return_state=True)
    T.XDOM.update(head=True, w=0.25, dsbn=True)
    b = T.fit_eval("shallow", data(np.random.default_rng(0)), rec, 0, device="cpu", return_state=True)
    assert np.array_equal(a["prob_last"], b["prob_last"]), "no dtr = off"
    # 3. an XDOM fit
    d = data(np.random.default_rng(0), ext=12)
    d["dtr"] = np.r_[np.zeros(24, np.int64), np.ones(12, np.int64)]
    c = T.fit_eval("shallow", d, rec, 0, device="cpu", return_state=True)
    assert set(c["state_last"]) == set(a["state_last"]), "XDOM fit: plain state_dict keys"
    assert not np.array_equal(a["prob_last"], c["prob_last"])
    T.XDOM.clear()
    # 4. CAR of external windows
    import t2p_bench as B
    rng = np.random.default_rng(1)
    idx = pd.DataFrame(dict(stream=["A|1"] * 10, order=np.arange(10), y=np.arange(10) % 3))
    idx["i"] = np.arange(10)
    X = (rng.standard_normal((10, 4, 32)) + 5.0).astype(np.float32)
    ext = dict(name="export_syn", X=X, idx=idx, keep=np.arange(10), names=["C3", "Cz", "C4", "Pz"], sha=None)
    Xc, _ = B.aligned_extra(ext, "none", 0.05, car=True)
    X0, _ = B.aligned_extra(ext, "none", 0.05)
    assert np.abs(Xc.mean(1)).max() < 1e-5 and np.abs(X0.mean(1)).max() > 1e-3, "car: zero channel mean"
    print("DomainBN routes / unwraps; no dtr = bitwise; XDOM fit keeps the plain keys; external CAR zero-mean")
    print("OK")


if __name__ == "__main__":
    main()
