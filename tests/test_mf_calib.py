#!/usr/bin/env python
"""T2-D413b calib 2 (tracks/t2_bci/scripts/t2p_mf_calib.py), CPU:
  1. the bands partition the DFT axis: sum of split(X) = X;
  2. bures: T Cs T = Cr (symmetric T);
  3. apply_maps with identity maps = X; fit_maps + apply_maps moves every (user, class, band) synthetic mean covariance onto the real one
     (up to the 1e-3 shrinkage);
  python tests/test_mf_calib.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks" / "t2_bci" / "scripts"))
import t2p_mf_calib as K  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    C, n = 6, 60
    mixr, mixs = rng.standard_normal((C, C)) * 0.5 + np.eye(C), rng.standard_normal((C, C)) * 0.5 + np.eye(C)
    Xr = np.einsum("cd,ndl->ncl", mixr, rng.standard_normal((n, C, K.L))).astype(np.float32)
    Xs = np.einsum("cd,ndl->ncl", mixs, rng.standard_normal((n, C, K.L))).astype(np.float32)
    y, u = np.arange(n) % 3, (np.arange(n) // 3) % 2
    B = K.split(Xr)
    assert B.shape[0] == len(K.EDGES) - 1 and np.allclose(B.sum(0).numpy(), Xr, atol=1e-4)
    Cs, Cr = K.mean_cov(K.split(Xs)[2]), K.mean_cov(B[2])
    T = K.bures(Cs, Cr)
    assert torch.allclose(T, T.T, atol=1e-8) and torch.allclose(T @ Cs @ T, Cr, rtol=1e-6, atol=1e-8)
    I = np.broadcast_to(np.eye(C), (n, len(K.EDGES) - 1, C, C)).copy()
    assert np.allclose(K.apply_maps(Xs, I), Xs, atol=1e-4)
    M = K.fit_maps(Xr, y, u, Xs, y, u, 2)
    Xc = K.apply_maps(Xs, M[u, y].numpy())
    err = 0.0
    for j in range(2):
        for c in range(3):
            k = (u == j) & (y == c)
            Br_, Bc_ = K.split(Xr[k]), K.split(Xc[k])
            for b in range(len(K.EDGES) - 1):
                A, Bm = K.mean_cov(Br_[b]), K.mean_cov(Bc_[b])
                err = max(err, float((A - Bm).abs().max() / A.abs().max()))
    assert err < 0.02, err
    print(f"calib 2: bands partition the DFT axis; T Cs T = Cr; identity maps = X; re-coloured band covariances match real (max rel err {err:.1e})")
    print("OK")


if __name__ == "__main__":
    main()
