#!/usr/bin/env python
"""T2-D420: the EATA (EA1 / EA2) and anchored-TTT (AT1 / AT2) variants of t2p_bench.pl_stream_probs, CPU:
  1. TTA_PL / EATA_E0 / EATA_MIN / ANC hold the a-priori constants; D113's and D405's variants are unchanged;
  2. lr 0 = the plain net for both kinds (EATA moves nothing; the anchored teacher stays the source);
  3. a stream's first IM_MIN windows are the plain net; with a step size that moves, later windows differ; the source net is never modified;
  4. EATA with E0 = 0 (no reliable window) = the plain net; the anchored collapse guard at 0 (reset on every buffer) = the plain net;
  5. rows are probabilities.
  python tests/test_tta_d420.py  ->  "OK"
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
    assert B.TTA_PL["EA1"] == ("eata", 1e-5, 16) and B.TTA_PL["EA2"] == ("eata", 3e-5, 16)
    assert B.TTA_PL["AT1"] == ("anc", 1e-5, 16) and B.TTA_PL["AT2"] == ("anc", 3e-5, 16)
    assert abs(B.EATA_E0 - 0.4 * np.log(3)) < 1e-12 and B.EATA_MIN == 4 and B.ANC == dict(steps=2, ema=0.99, lam=1.0, sk=3, guard=0.8)
    assert B.TTA_PL["IM2"] == ("im", 1e-5, 16) and B.TTA_PL["V4"] == ("pl", 1e-3, 0.6) and B.IM_MIN == 8, "earlier variants unchanged"
    torch.manual_seed(0)
    net = T.build_model("fbcnet_bboth", 41, 3, 480, 120.0).eval()
    sd0 = {k: v.clone() for k, v in net.state_dict().items()}
    n = 24
    XA = np.random.default_rng(0).standard_normal((2 * n, 41, 480)).astype(np.float32)
    idx = pd.DataFrame(dict(i=np.arange(2 * n), stream=np.repeat(["a", "b"], n), order=np.tile(np.arange(n), 2)))
    rows = np.arange(2 * n)
    with torch.no_grad():
        ref = torch.softmax(net(torch.as_tensor(XA)).float(), 1).numpy()
    for kind in ("eata", "anc"):
        assert np.allclose(B.pl_stream_probs(net, XA, idx, rows, (kind, 0.0, 16), "cpu"), ref, atol=1e-5), f"{kind} lr 0 = the plain net"
    e0, g0 = B.EATA_E0, B.ANC["guard"]
    try:
        B.EATA_E0 = 10.0                                                       # every window reliable -> EATA adapts
        B.ANC["guard"] = 1.01                                                  # the guard never fires -> the anchored net adapts
        for kind, lr in (("eata", 1e-3), ("anc", 1e-2)):
            P = B.pl_stream_probs(net, XA, idx, rows, (kind, lr, 16), "cpu")
            assert np.allclose(P.sum(1), 1, atol=1e-5)
            for s0 in (0, n):
                assert np.allclose(P[s0:s0 + B.IM_MIN], ref[s0:s0 + B.IM_MIN], atol=1e-5), f"{kind}: no update before IM_MIN"
                assert not np.allclose(P[s0 + B.IM_MIN + 1:s0 + n], ref[s0 + B.IM_MIN + 1:s0 + n], atol=1e-5), f"{kind}: later windows adapted"
            print(f"{kind} lr {lr:g}: first {B.IM_MIN} windows per stream = the plain net; later adapted (max |dp| {np.abs(P - ref).max():.4f})")
        B.EATA_E0 = 0.0                                                        # no reliable window -> no step
        assert np.allclose(B.pl_stream_probs(net, XA, idx, rows, ("eata", 1e-3, 16), "cpu"), ref, atol=1e-5), "EATA without reliable windows = plain"
        B.ANC["guard"] = 0.0                                                   # reset on every buffer -> never adapts
        assert np.allclose(B.pl_stream_probs(net, XA, idx, rows, ("anc", 1e-2, 16), "cpu"), ref, atol=1e-5), "guard 0 = plain"
    finally:
        B.EATA_E0, B.ANC["guard"] = e0, g0
    assert all(torch.equal(v, sd0[k]) for k, v in net.state_dict().items()), "the source net must not change"
    print("OK")


if __name__ == "__main__":
    main()
