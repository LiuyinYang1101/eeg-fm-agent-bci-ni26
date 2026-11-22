#!/usr/bin/env python
"""Backlog 74: a random spectral tilt per training window (Recipe.tilt -> ni26.train.augment), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_tl15 / r1_pulx_vp2_e180_vr_tl15 differ from the FBCNet rc75 / ShallowNet vr lines in tilt (0.15) only; copies
     unchanged; only these two set tilt; the export the CNNs train on is at TILT_FS (120 Hz);
  2. augment: tilt 0 -> bitwise unchanged; tilt > 0 = the replayed rfft x ((f + 1) / 11)^beta; a 10 Hz sinusoid keeps its amplitude; linear in x;
  3. fit_eval with tilt runs for both families and moves the weights; a window rate other than TILT_FS is refused.
  python tests/test_tilt.py  ->  "OK"
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ni26.train as T  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    for base, n in (("r1_pulxx_vp2_e180_vr_rc75", "r1_pulxx_vp2_e180_vr_rc75_tl15"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e180_vr_tl15")):
        b, r = T.RECIPES[base], T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "tilt"} and r.tilt == 0.15, n
        assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    assert sorted(n for n, x in T.RECIPES.items() if x.tilt) == ["r1_pulx_vp2_e180_vr_tl15", "r1_pulxx_vp2_e180_vr_rc75_tl15", "r1_pulxx_vp2_e180_vr_rc75_tl15_mf20", "r1_pulxx_vp2_e180_vr_rc75_tl15_mf50", "r1_pulxx_vp2_e180_vr_rc75_tl15_pv10", "r1_pulxx_vp2_e180_vr_rc75cc_tl15", "r1_pulxx_vp2_e360_vr_rc75_tl15"]   # + T2-D382, D388, D412, D561, D413
    meta = json.load(open(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json"))
    fs = float(meta.get("sfreq", meta.get("fs", 0)))
    assert fs == T.TILT_FS, (fs, T.TILT_FS)
    print(f"recipes: the two tl15 recipes differ from their lines in tilt only; copies unchanged; export at {fs:g} Hz = TILT_FS")

    torch.manual_seed(0)
    B, C, L = 6, 4, 480
    x = torch.randn(B, C, L)
    plain = replace(T.RECIPES["r1"])
    assert torch.equal(T.augment(x.clone(), plain), x), "tilt 0 -> unchanged"
    rec = replace(plain, tilt=0.5)
    torch.manual_seed(3)
    got = T.augment(x.clone(), rec)
    torch.manual_seed(3)
    beta = torch.randn(B, 1, 1) * 0.5
    f = torch.fft.rfftfreq(L, 1.0 / T.TILT_FS)
    want = torch.fft.irfft(torch.fft.rfft(x, dim=2) * ((f + 1.0) / 11.0) ** beta, n=L, dim=2)
    assert torch.allclose(got, want, atol=1e-5)
    t = torch.arange(L) / T.TILT_FS
    s10 = torch.sin(2 * np.pi * 10.0 * t).expand(B, C, L).clone()
    torch.manual_seed(4)
    a10 = T.augment(s10, rec)
    assert torch.allclose(a10.std(-1), s10.std(-1), rtol=1e-3), "10 Hz keeps its amplitude"
    y = torch.randn(B, C, L)
    torch.manual_seed(5); l1 = T.augment((2 * x + 3 * y).clone(), rec)
    torch.manual_seed(5); l2 = 2 * T.augment(x.clone(), rec)
    torch.manual_seed(5); l3 = 3 * T.augment(y.clone(), rec)
    assert torch.allclose(l1, l2 + l3, atol=1e-4), "linear in x"
    print(f"augment: tilt 0 bitwise; tilt = the replayed rfft x ((f + 1) / 11)^beta; 10 Hz amplitude kept; linear")

    d = data()
    for model in ("fbcnet_bboth", "shallow"):
        p2 = replace(T.RECIPES["r1"], epochs=2)
        a0 = T.fit_eval(model, d, p2, 0, device="cpu", return_state=True)
        a1 = T.fit_eval(model, d, replace(p2, tilt=0.15), 0, device="cpu", return_state=True)
        sa, sb = a0["state_last"], a1["state_last"]
        moved = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
        assert moved > 1e-5, (model, moved)
        print(f"fit_eval {model}: tilt runs, weights move by up to {moved:.1e}")
    try:
        T.fit_eval("shallow", d, replace(T.RECIPES["r1"], epochs=1, tilt=0.15), 0, device="cpu", sfreq=128.0)
        raise SystemExit("a rate other than TILT_FS must be refused")
    except AssertionError as e:
        assert "tilt" in str(e), e
    print("fit_eval: tilt at 128 Hz refused")
    print("OK")


if __name__ == "__main__":
    main()
