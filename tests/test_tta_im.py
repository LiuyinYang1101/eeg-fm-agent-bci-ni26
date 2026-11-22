#!/usr/bin/env python
"""T2-D405: the IM variants of t2p_bench.pl_stream_probs (T-TIME-style test-time adaptation), CPU:
  1. TTA_PL holds IM1 / IM2 = ('im', 1e-4 / 1e-5, buffer 16) and IM_MIN = 8; D113's V1..V4 are unchanged;
  2. streamed per stream from a fresh copy: a stream's first IM_MIN windows are predicted by the UNADAPTED net (bitwise = no TTA), later
     windows by adapted weights (they differ), each stream restarts from the trained net, and the source net is never modified;
  3. rows are probabilities (sum 1).
  python tests/test_tta_im.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402
import ni26.train as T  # noqa: E402


def main():
    assert B.TTA_PL["IM1"] == ("im", 1e-4, 16) and B.TTA_PL["IM2"] == ("im", 1e-5, 16) and B.IM_MIN == 8
    assert B.TTA_PL["V1"] == ("ent", 1e-4, 0.0) and B.TTA_PL["V4"] == ("pl", 1e-3, 0.6), "D113's variants unchanged"
    torch.manual_seed(0)
    net = T.build_model("fbcnet_bboth", 41, 3, 480, 120.0).eval()
    sd0 = {k: v.clone() for k, v in net.state_dict().items()}
    n = 24
    XA = np.random.default_rng(0).standard_normal((2 * n, 41, 480)).astype(np.float32)
    idx = pd.DataFrame(dict(i=np.arange(2 * n), stream=np.repeat(["a", "b"], n), order=np.tile(np.arange(n), 2)))
    rows = np.arange(2 * n)
    base = B.pl_stream_probs(net, XA, idx, rows, ("im", 0.0, 16), "cpu")          # lr 0: Adam moves nothing = the unadapted net
    with torch.no_grad():
        ref = torch.softmax(net(torch.as_tensor(XA)).float(), 1).numpy()
    assert np.allclose(base, ref, atol=1e-5), "lr 0 = the plain net"
    P = B.pl_stream_probs(net, XA, idx, rows, B.TTA_PL["IM1"], "cpu")
    assert np.allclose(P.sum(1), 1, atol=1e-5)
    for s0 in (0, n):                                                               # each stream: first IM_MIN windows unadapted
        assert np.allclose(P[s0:s0 + B.IM_MIN], ref[s0:s0 + B.IM_MIN], atol=1e-5), "no update before IM_MIN windows"
        assert not np.allclose(P[s0 + B.IM_MIN + 1:s0 + n], ref[s0 + B.IM_MIN + 1:s0 + n], atol=1e-5), "later windows adapted"
    assert all(torch.equal(v, sd0[k]) for k, v in net.state_dict().items()), "the source net must not change"
    print(f"IM1: first {B.IM_MIN} windows per stream = the plain net; later windows adapted (max |dp| {np.abs(P - ref).max():.4f}); source unchanged")
    print("OK")


if __name__ == "__main__":
    main()
