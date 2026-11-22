"""Shared, dependency-light spectral features (numpy only — usable inside a submission.py too)."""

import numpy as np

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12.5, 30), "gamma": (30, 50)}


def psd(X, fs):
    """Hann-tapered periodogram over the last axis. X (..., T) -> (freqs, P (..., F))."""
    T = X.shape[-1]
    w = np.hanning(T)
    P = np.abs(np.fft.rfft(np.asarray(X, dtype=np.float64) * w, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    return np.fft.rfftfreq(T, 1 / fs), P


def log_band_power(X, fs, bands=BANDS):
    """X (N, C, T) -> log band power (N, C, B)."""
    f, P = psd(X, fs)
    return np.stack([np.log(P[..., (f >= lo) & (f < hi)].mean(-1) + 1e-12) for lo, hi in bands.values()], -1)


def batched(fn, X, *a, chunk=2048, **k):
    return np.concatenate([fn(np.asarray(X[i:i + chunk]), *a, **k) for i in range(0, len(X), chunk)])
