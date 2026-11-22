#!/usr/bin/env python
"""Self-test (T2-D464): the 'attnres' readout of FMClf (a learned per-token softmax over all layer outputs). CPU, st_small:
  1. with the last layer's bias pushed to +1e4 the features equal the 'mean' readout's (same norm weights) to 1e-5;
  2. at its init (b = 0 .. 0, 6) the last layer's weight is exp(6) / (exp(6) + L - 1) for every token (q = 0);
  3. a backward pass gives gradients on ar_q, ar_b and the encoder; t1.param_groups puts ar_q / ar_b in the head group.
  python t2p_attnres_selftest.py -> "OK"
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_fm  # noqa: E402
from t2p_aug_selftest import EXPORT  # noqa: E402

t1 = t2p_fm.t1


def main():
    torch.manual_seed(0)
    names = json.loads((Path(EXPORT) / "meta.json").read_text())["ch_names"]
    enc = t2p_fm.make_encoder("st_small", names, lambda m: None, 0.0, "gain")
    L = enc.n_layers
    a, b = t2p_fm.FMClf(enc, L, "mean", 3), t2p_fm.FMClf(enc, L, "attnres", 3)
    b.norm.load_state_dict(a.norm.state_dict())
    x = torch.randn(2, len(names), 480)
    a.eval(); b.eval()
    with torch.no_grad():
        assert abs(torch.softmax(b.ar_b, 0)[-1].item() - math.exp(6) / (math.exp(6) + L - 1)) < 1e-6
        f0 = b.features(x)
        b.ar_b[-1] = 1e4
        d = (a.features(x) - b.features(x)).abs().max().item()
        assert d < 1e-5, d
        assert (f0 - a.features(x)).abs().max().item() > 1e-6, "the init must mix in earlier layers"
        b.ar_b[-1] = 6.0
    b.train()
    b(x).sum().backward()
    assert b.ar_q.grad is not None and b.ar_q.grad.abs().sum() > 0 and b.ar_b.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    g = t1.param_groups(b, 1e-4, 1e-3, 0.75, 0.05, lambda m: None)
    head = {id(p) for gr in g if str(gr.get("name", "")).startswith("head") for p in gr["params"]}
    assert id(b.ar_q) in head and id(b.ar_b) in head
    print("OK")


if __name__ == "__main__":
    main()
