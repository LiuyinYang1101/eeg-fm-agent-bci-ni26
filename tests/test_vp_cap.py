#!/usr/bin/env python
"""Backlog 65 (b): the rigid cap shift in the virtual-participant bank (Recipe.vp_cap -> t2p_bench.vp_bank cap), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_cap10 / r1_pulx_vp2_e180_vr_cap10 differ from the FBCNet rc75 / ShallowNet vr lines in vp_cap (0.010) only;
     their per-user copies are the lines'; only these two set vp_cap;
  2. cap 0 = the line's bank (sigma 10 mm, vref 0.5) bitwise; cap 10 mm keeps the gains / shifts and changes the maps;
  3. cap_rotations: orthogonal, det +1; over the montage the common displacement's RMS is ~cap (within 10 %);
  4. sigma 0 + cap: map k = the spherical spline from the real positions to the positions rotated by cap_rotations' k-th matrix (one rigid
     displacement for every electrode), and with sigma 10 mm the map differs from both the jitter-only and the rotation-only maps.
  python tests/test_vp_cap.py  ->  "OK"
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402
from ni26 import train as T  # noqa: E402


def main():
    import mne
    from mne.channels.interpolation import _make_interpolation_matrix
    for base, n in (("r1_pulxx_vp2_e180_vr_rc75", "r1_pulxx_vp2_e180_vr_rc75_cap10"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e180_vr_cap10")):
        b, r = T.RECIPES[base], T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "vp_cap"} and r.vp_cap == 0.010, n
        assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    assert sorted(n for n, x in T.RECIPES.items() if x.vp_cap) == ["r1_pulx_vp2_e180_vr_cap10", "r1_pulxx_vp2_e180_vr_rc75_cap10"]
    print("recipes: the two cap10 recipes differ from their lines in vp_cap only; copies unchanged")

    names = json.load(open(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json"))["ch_names"]
    line = dict(K=64, sigma=0.010, vref=0.5)
    b0, bz, bc = B.vp_bank(names, **line), B.vp_bank(names, **line, cap=0.0), B.vp_bank(names, **line, cap=0.010)
    assert all(torch.equal(b0[k], bz[k]) for k in b0), "cap 0 must be the line's bank bitwise"
    assert torch.equal(b0["gains"], bc["gains"]) and torch.equal(b0["shifts"], bc["shifts"])
    dm = float((bc["mats"] - b0["mats"]).abs().mean())
    assert dm > 1e-3, dm
    print(f"vp_bank: cap 0 = the line's bank bitwise; cap 10 mm keeps gains / shifts, mean |M_cap - M| {dm:.4f}")

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array([pos[c] for c in names], np.float64)
    sol = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
    Q = P - sol[:3]
    R = float(np.linalg.norm(Q, axis=1).mean())
    cap = 0.010
    rots = B.cap_rotations(2000, cap / (np.sqrt(2) * R), 3)
    assert np.allclose(np.einsum("kij,kil->kjl", rots, rots), np.eye(3), atol=1e-12) and np.allclose(np.linalg.det(rots), 1.0)
    rms = float(np.sqrt(((np.einsum("kij,cj->kci", rots, Q) - Q) ** 2).sum(-1).mean()))
    assert abs(rms - cap) < 0.1 * cap, rms
    print(f"cap_rotations: orthogonal, det +1; RMS scalp displacement {1000 * rms:.2f} mm for cap 10 mm")

    bs = B.vp_bank(names, K=8, sigma=0.0, cap=cap, seed=0)
    rk = B.cap_rotations(8, cap / (np.sqrt(2) * R), 3)
    err = max(float(np.abs(bs["mats"][k].numpy() - _make_interpolation_matrix(Q, Q @ rk[k].T, alpha=1e-8)).max()) for k in range(8))
    assert err < 1e-5, err
    bj, bjc = B.vp_bank(names, K=8, sigma=0.010), B.vp_bank(names, K=8, sigma=0.010, cap=cap)
    d1, d2 = float((bjc["mats"] - bj["mats"]).abs().mean()), float((bjc["mats"] - bs["mats"]).abs().mean())
    assert d1 > 1e-3 and d2 > 1e-3, (d1, d2)
    print(f"vp_bank: sigma 0 + cap = the spline to the rigidly rotated positions (max err {err:.1e}); jitter + cap differs from either alone")
    print("OK")


if __name__ == "__main__":
    main()
