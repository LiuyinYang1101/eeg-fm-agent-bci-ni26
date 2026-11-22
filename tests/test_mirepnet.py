#!/usr/bin/env python
"""Row 77 / T2-D305: the vendored MIRepNet member (src/ni26/fm/mirepnet.py), CPU:
  1. the encoder strict-loads MIRepNet.pth (weights_only; decoder / mask / head dropped) and holds the checkpoint's tensors;
  2. pad_map on our 41-channel montage: (45, 41); present template channels copied (one-hot rows), missing ones = weights summing to 1;
  3. prep: a 10 Hz sine at 120 Hz -> 1000 samples at 250 Hz with the same amplitude and frequency; the 8-30 Hz arm removes a 3 Hz sine, the
     broadband arm keeps it;
  4. build_model("mirepnet" / "mirepnet_bb") with ni26.train.MIREP_CH set -> (B, 3) logits; fit_eval with mirep_ft (1 epoch) runs on CPU.
  python tests/test_mirepnet.py  ->  "OK"
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import ni26.train as T  # noqa: E402
from ni26.fm import mirepnet as M  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    names = json.load(open(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json"))["ch_names"]
    sd = torch.load(M.WEIGHTS, map_location="cpu", weights_only=True)
    net = M.load_pretrained(M.MIRepNet())
    for k in ("embedding.conv2.weight", "transformer.5.1.fn.1.3.weight", "embedding.projection.0.weight"):
        assert torch.equal(net.state_dict()[k], sd[k]), k
    print(f"encoder: MIRepNet.pth strict-loads ({len(sd)} keys in the file), tensors equal")

    W = M.pad_map(names)
    tm = [c.upper() for c in M.TEMPLATE["template"]]
    ours = [c.upper() for c in names]
    assert W.shape == (45, 41) and torch.allclose(W.sum(1), torch.ones(45))
    for i, c in enumerate(tm):
        if c in ours:
            assert W[i, ours.index(c)] == 1.0 and W[i].sum() == 1.0
    print(f"pad_map: (45, 41); {sum(c in ours for c in tm)} copied, {sum(c not in ours for c in tm)} inverse-distance rows (sum 1)")

    t = np.arange(480) / 120.0
    x10 = torch.tensor(np.sin(2 * np.pi * 10 * t), dtype=torch.float32).expand(1, 41, 480).clone()
    x3 = torch.tensor(np.sin(2 * np.pi * 3 * t), dtype=torch.float32).expand(1, 41, 480).clone()
    for band, keep3 in (((8.0, 30.0), False), (None, True)):
        m = M.MIRepMember(names, 480, 120.0, 3, band=band, pretrained=False)
        y = m.prep(x10)
        assert y.shape == (1, 45, 1000)
        ref = np.sin(2 * np.pi * 10 * np.arange(1000) / 250.0)
        assert np.allclose(y[0, 0].numpy(), ref, atol=1e-3), float(np.abs(y[0, 0].numpy() - ref).max())
        r3 = float(m.prep(x3).std())
        assert (r3 > 0.5) == keep3, (band, r3)
    print("prep: 10 Hz sine resampled to 250 Hz exactly (1e-3); 8-30 Hz removes 3 Hz, broadband keeps it")

    T.MIREP_CH[:] = list(names)
    d = data(C=41)
    for name in ("mirepnet", "mirepnet_bb"):
        mdl = T.build_model(name, 41, 3, 480, 120.0).eval()
        with torch.no_grad():
            assert mdl(torch.as_tensor(d["Xte"][:4])).shape == (4, 3)
        r = T.fit_eval(name, d, replace(T.RECIPES["mirep_ft"], epochs=1), 0, device="cpu")
        assert r["prob_last"].shape == (len(d["Xte"]), 3) and np.isfinite(r["prob_last"]).all()
        print(f"{name}: build_model -> (B, 3); fit_eval (mirep_ft, 1 epoch) runs")
    print("OK")


if __name__ == "__main__":
    main()
