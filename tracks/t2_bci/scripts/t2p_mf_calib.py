"""T2-D413b calibration levers for the MeanFlow / flow-matching generator's samples (t2p_meanflow.py --calib 2 / 3); all fitted on a SEPARATE
own-user calibration draw against the protocol's REAL TRAINING windows (never a val / test window), then applied to the evaluated sets:

  calib 2 = FILTER-BANK SPATIAL RE-COLOURING: the DFT axis is partitioned exactly into bands EDGES (Hz); per (user, class, band) the Bures / OT
            map T with T C_syn T = C_real (C = the mean per-window-demeaned band covariance, shrunk 1e-3 tr / C) re-colours that band of every
            synthetic window; the bands sum back exactly. Interpolated users: T = alpha T_a + (1 - alpha) T_b per band (a linear mix of maps).
  calib 3 = calib 2, then calib 1's per (user, class, channel) spectral gains re-fitted on the re-coloured calibration draw.
Diagnosis (Phase A / batch 1): after spectral gains the own-user log-band-power AUC passes (0.757) but the 8-30 Hz covariance stays ~1.3x the
user spread: the remaining gap is spatial, per band.
"""
import numpy as np
import torch

EDGES = (0.0, 4.0, 8.0, 13.0, 20.0, 30.0, 61.0)
FS, L = 120.0, 480


def band_masks():
    f = np.fft.rfftfreq(L, 1.0 / FS)
    m = [(f >= lo) & (f < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    assert np.array_equal(np.sum(m, 0), np.ones(len(f))), "the bands must partition the DFT axis"
    return [torch.as_tensor(k) for k in m]


def split(X):
    """(n, C, L) -> (K, n, C, L) float64, the exact band parts (sum = X)."""
    Fx = torch.fft.rfft(torch.as_tensor(X, dtype=torch.float64), dim=-1)
    return torch.stack([torch.fft.irfft(Fx * k, n=L, dim=-1) for k in band_masks()])


def mean_cov(Xb):
    Z = Xb - Xb.mean(-1, keepdim=True)
    C = torch.einsum("nct,ndt->cd", Z, Z) / (Z.shape[0] * Z.shape[-1])
    return C + 1e-3 * torch.trace(C) / len(C) * torch.eye(len(C), dtype=C.dtype)


def _sqrtm(C):
    w, V = torch.linalg.eigh(C)
    return (V * w.clamp_min(0).sqrt()) @ V.T


def bures(Cs, Cr):
    """T (symmetric PD) with T Cs T = Cr."""
    Ss = _sqrtm(Cs)
    w, V = torch.linalg.eigh(Cs)
    Si = (V / w.clamp_min(1e-12).sqrt()) @ V.T
    return Si @ _sqrtm(Ss @ Cr @ Ss) @ Si


def fit_maps(Xr, yr, ur, Xs, ys, us, n_users, bs=512):
    """-> (n_users, 3, K, C, C) float64 band maps."""
    K = len(EDGES) - 1
    out = torch.zeros(n_users, 3, K, Xr.shape[1], Xr.shape[1], dtype=torch.float64)
    for j in range(n_users):
        for c in range(3):
            kr, ks = (yr == c) & (ur == j), (ys == c) & (us == j)
            assert kr.any() and ks.any(), (j, c)
            Br, Bs = split(Xr[kr]), split(Xs[ks])
            for b in range(K):
                out[j, c, b] = bures(mean_cov(Bs[b]), mean_cov(Br[b]))
    return out


def apply_maps_idx(X, M, cls, ua, ub=None, al=None, bs=512):
    """X (n, C, L); M (n_users, 3, K, C, C); per window T = M[ua, cls] (own user) or al M[ua, cls] + (1 - al) M[ub, cls] (interpolated)."""
    out = []
    for i in range(0, len(X), bs):
        s = slice(i, i + bs)
        T = M[ua[s], cls[s]]
        if ub is not None:
            w = torch.as_tensor(al[s], dtype=torch.float64)[:, None, None, None]
            T = w * T + (1 - w) * M[ub[s], cls[s]]
        out.append(apply_maps(X[s], T.numpy(), bs=bs))
    return np.concatenate(out)


def apply_maps(X, T, bs=512):
    """X (n, C, L) numpy, T (n, K, C, C) per-window band maps -> sum_b T_b @ band_b(X) (float32)."""
    out = []
    for i in range(0, len(X), bs):
        B = split(X[i:i + bs])                                           # (K, m, C, L)
        out.append(torch.einsum("mbcd,bmdt->mct", torch.as_tensor(T[i:i + bs]), B).float().numpy())
    return np.concatenate(out)
