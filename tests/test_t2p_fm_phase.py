#!/usr/bin/env python
"""T2-D586: the shared-phase Fourier surrogate of the FM target windows (t2p_fm.phase_setup / phase_batch; fit_eval_fm(phase=(p, amax))), CPU:
  1. with p 1 every window is rotated: each channel's amplitude spectrum and every cross-spectrum X_c conj(X_d) are kept (1e-4 relative), DC
     and Nyquist are untouched, and the rotation angle of each bin is the SAME for all channels of a window and lies in [-amax, +amax];
     the waveform moves;
  2. p ~0 leaves the batch untouched; p 0.25 rotates ~25 % of the windows (counted);
  3. OFF = bitwise: fit_eval_fm on a random-init ST-base (1 layer) with phase=None equals the call without the argument; phase (0.25, pi/4)
     runs, moves the output and logs a rotated-window count; it composes with the dcmean side head.
  python tests/test_t2p_fm_phase.py  ->  "OK"
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(586)
    n, C, L = 64, 6, 480
    X = torch.as_tensor(rng.standard_normal((n, C, L)).astype(np.float32))
    # 1. p 1: spectra kept, one shared angle per bin, DC / Nyquist untouched
    ph = FM.phase_setup(1.0, math.pi / 4, 0, log=lambda m: None)
    Y = FM.phase_batch(X, ph)
    assert ph["count"] == n and Y.shape == X.shape and Y.dtype == X.dtype
    Fx, Fy = torch.fft.rfft(X.double(), dim=-1), torch.fft.rfft(Y.double(), dim=-1)
    amp = (Fy.abs() - Fx.abs()).abs().amax() / Fx.abs().amax()
    assert float(amp) < 1e-4, float(amp)
    cx = Fx[:, :, None, :] * Fx[:, None, :, :].conj()
    cy = Fy[:, :, None, :] * Fy[:, None, :, :].conj()
    assert float((cy - cx).abs().amax() / cx.abs().amax()) < 1e-4, "a cross-spectrum moved"
    assert float((Fy[..., 0] - Fx[..., 0]).abs().amax()) < 1e-3 and float((Fy[..., -1] - Fx[..., -1]).abs().amax()) < 1e-3, "DC / Nyquist moved"
    ang = torch.angle(Fy[..., 1:-1] / Fx[..., 1:-1])                     # (n, C, bins)
    spread = (torch.remainder(ang - ang[:, :1] + math.pi, 2 * math.pi) - math.pi).abs().amax()
    assert float(spread) < 1e-3, ("the angle differs across channels", float(spread))
    assert float(ang.abs().amax()) <= math.pi / 4 + 1e-4 and float(ang.abs().amax()) > 0.6, float(ang.abs().amax())
    assert float((Y - X).abs().amax()) > 0.1
    print(f"1. p 1: amplitude / cross-spectra kept (rel {float(amp):.1e}), DC / Nyquist untouched, ONE angle per bin for all channels, "
          f"|angle| <= pi/4 (max {float(ang.abs().amax()):.3f})")
    # 2. p ~0 / p 0.25
    ph0 = FM.phase_setup(1e-12, math.pi / 4, 0, log=lambda m: None)
    assert torch.equal(FM.phase_batch(X, ph0), X) and ph0["count"] == 0
    ph25 = FM.phase_setup(0.25, math.pi / 4, 1, log=lambda m: None)
    for _ in range(20):
        Z = FM.phase_batch(X, ph25)
    frac = ph25["count"] / ph25["seen"]
    same = (Z - X).abs().amax((1, 2)) == 0
    assert 0.18 < frac < 0.32 and ph25["seen"] == 20 * n and 0 < int((~same).sum()) < n, frac
    print(f"2. p ~0 leaves the batch untouched; p 0.25 rotated {ph25['count']} / {ph25['seen']} windows ({frac:.3f})")
    # 3. off = bitwise; on = runs, moves, counts; with the dcmean side head
    nn, nt = 96, 30
    yy, yt = (np.arange(nn) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn) % 2).astype(np.int64))
    kw = dict(epochs=2, bs=32, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", layer=1, calib="gain", ls=0.0)
    r_a = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    r_b = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, phase=None, **kw)
    assert np.array_equal(r_a["prob_last"], r_b["prob_last"]), "phase=None is not bitwise"
    logs = []
    r_c = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, phase=(0.25, math.pi / 4), **kw)
    assert not np.array_equal(r_a["prob_last"], r_c["prob_last"]) and np.isfinite(r_c["prob_last"]).all()
    cnt = [m for m in logs if "phase-rotated" in m]
    assert cnt and any("shared-phase surrogate" in m for m in logs), logs[-3:]
    d2 = dict(d, side_tr=rng.standard_normal((nn, 82)).astype(np.float32), side_va=np.zeros((0, 82), np.float32),
              side_te=rng.standard_normal((nt, 82)).astype(np.float32))
    r_s = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=lambda m: None, side="dcmean", phase=(0.25, math.pi / 4), **kw)
    assert np.isfinite(r_s["prob_last"]).all()
    print(f"3. phase=None bitwise; phase (0.25, pi/4) runs and moves the output ({cnt[-1].split(';')[-1].strip()}); with the side head too")
    print("OK")


if __name__ == "__main__":
    main()
