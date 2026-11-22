"""Causal, label-free per-session input normalisation for the Track 2 streaming contract (kit >= 442edcd).

Codabench feeds ``predict`` ONE window at a time, chronologically within a participant x session stream, and
resets the model at every new session; statistics may use the current and earlier windows of the current session
only (scientific FAQ, 2026-10-13). Every mode below is therefore computed causally: window t of a stream is
normalised with statistics of windows 1..t. Training sessions go through the SAME causal recursion in their
acquisition order, so training sees exactly the early-session noise the test stream will have.

Modes (``align``):
  none  global scale only (x / scale; scale = a training constant)            stateless
  wz    per-window, per-channel z-score                                         stateless
  sz    running per-channel mean / sd of the stream (windows 1..t)              causal
  ea    causal Euclidean alignment: x <- R_t^{-1/2} x with R_t the shrunk mean   causal
        of the window covariances 1..t, plus an optional population prior of
        strength n0 windows (trace-matched) so the first windows are not
        whitened by their own covariance alone
  eafull  EA with the WHOLE stream's mean covariance (non-causal; oracle / analysis only, never at test)
  fbea  T2-D424 filter-bank EA (bench only: no StreamAligner twin yet, so a package needs one = a template change):
        each window is split into the bands FB_EDGES by rectangular rfft masks (they partition [0, fs / 2], so the bands sum
        back to the window and their sample cross-covariances are exactly 0); per band, the EA recursion above (own running
        covariance, own prior); the band outputs are re-weighted by sqrt(the stream's running power share of that band) and
        summed. Spatial whitening per band, the stream's spectral shape kept; with ONE band it is exactly "ea"
  eab   T2-D426 (bench only): ONE EA matrix from the stream's running 8-30 Hz covariance, trace-matched to the broadband one, applied to the
        broadband window (exactly "ea" when the band covariance is proportional to the broadband one)
  eal   T2-D427 (bench only): as eab with the 0-8 Hz band
  eaf   T2-D428 (bench only): broadband EA with a FORGETTING running covariance (gamma EAF_GAMMA; gamma 1 = ea)

Every window is first demeaned per channel (window-local, legal). ``StreamAligner`` is the one-window-at-a-time
form used inside ``predict``; ``align_stream`` is the vectorised form for training/evaluation. They agree to float
precision (tested in ``tracks/t2_bci/scripts/t2p_selftest.py``).
"""

import numpy as np

MODES = ("none", "wz", "sz", "ea", "eafull")
FB_FS, FB_T = 120.0, 480                     # fbea: the contract's rate and window length (asserted)
FB_EDGES = (0.0, 8.0, 13.0, 20.0, 30.0)      # fbea band lower edges in Hz; the last band runs to Nyquist
EAB_BAND = (8.0, 30.0)                       # eab (T2-D426): the band whose covariance sets the one EA matrix
EAL_BAND = (0.0, 8.0)                        # eal (T2-D427): the same with the < 8 Hz band
EAF_GAMMA = 0.98                             # eaf (T2-D428): the forgetting factor of the running covariance (≈ 50-window memory)


def band_split(X, fs=FB_FS, edges=FB_EDGES):
    """(N, C, T) -> (B, N, C, T) float64: rectangular rfft masks [edge_b, edge_b+1) that partition [0, fs / 2]."""
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[-1]
    F = np.fft.rfft(X, axis=-1)
    f = np.fft.rfftfreq(T, 1.0 / fs)
    e = list(edges) + [np.inf]
    assert e[0] <= 0.0, edges
    return np.stack([np.fft.irfft(F * ((f >= lo) & (f < hi)), n=T, axis=-1) for lo, hi in zip(e[:-1], e[1:])])


def fb_prior(X, fs=FB_FS, edges=FB_EDGES):
    """Demeaned training windows (N, C, T) -> (B, C, C): per band, the trace-normalised mean covariance (each band's EA prior)."""
    return np.stack([trace_normalised_mean(window_covs(xb)) for xb in band_split(X, fs, edges)])


def _demean(X):
    return X - X.mean(axis=-1, keepdims=True)


def window_covs(X):
    """(N, C, T) demeaned windows -> (N, C, C) sample covariances (float64)."""
    X = np.asarray(X, dtype=np.float64)
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]


def trace_normalised_mean(covs):
    tr = np.trace(covs, axis1=1, axis2=2)
    return (covs / tr[:, None, None]).mean(0)


def _inv_sqrt(R, shrink):
    C = R.shape[-1]
    tr = np.trace(R) / C
    R = (1 - shrink) * R + shrink * tr * np.eye(C)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, w.max() * 1e-8)
    return (V * w ** -0.5) @ V.T


def lap_neighbours(ch_names, k=4):
    """T2-D480: the k nearest electrodes of each channel on the standard_1005 positions (3-D distance; ties broken by channel order).
    Returns an int array (C, k) of channel indices. A fixed function of the montage: a package can store it in its config."""
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array([pos[c] for c in ch_names], dtype=np.float64)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def lap_windows(X, ch_names, k=4, flat_tol=1e-6, chunk=256):
    """T2-D480: nearest-neighbour (Hjorth) surface Laplacian per window, from the window alone (predict-safe): x_c <- x_c - mean of x over
    the c's k nearest GOOD neighbours (finite everywhere and not flat, as car_windows). A channel with no good neighbour is left as it is;
    a dead channel never enters a neighbour's mean. It removes what is common to a channel and its neighbours (far-field sources: the
    reference, the ocular field) and keeps the local. X: (N, C, T) or (C, T)."""
    X = np.asarray(X, dtype=np.float32)
    one = X.ndim == 2
    Xa = X[None] if one else X
    nb = lap_neighbours(ch_names, k)
    out = np.empty_like(Xa)
    for a in range(0, len(Xa), chunk):
        x = Xa[a:a + chunk]
        good = np.isfinite(x).all(-1) & (np.nan_to_num(x).std(-1) > flat_tol)                # (n, C)
        z = np.where(good[..., None], np.nan_to_num(x), 0.0).astype(np.float32)
        g = good[:, nb]                                                                       # (n, C, k)
        s = np.einsum("nckt,nck->nct", z[:, nb], g.astype(np.float32))
        cnt = g.sum(2)
        out[a:a + chunk] = np.where((cnt > 0)[..., None], x - s / np.maximum(cnt, 1)[..., None], x)
    return out[0] if one else out


def car_windows(X, flat_tol=1e-6):
    """T2-D475: common average reference per window, from the window alone (predict-safe): x_c <- x_c - mean over the window's GOOD
    channels (finite everywhere and not flat: std > flat_tol). A dead (flat / NaN) channel never leaks into the others through the mean;
    it is itself re-referenced if finite, and left as it is (NaN) if not. X: (N, C, T) or (C, T)."""
    X = np.asarray(X, dtype=np.float32)
    one = X.ndim == 2
    Xa = X[None] if one else X
    good = np.isfinite(Xa).all(-1) & (np.nan_to_num(Xa).std(-1) > flat_tol)                   # (N, C)
    m = np.where(good[..., None], np.nan_to_num(Xa), 0.0).sum(1, keepdims=True) / np.maximum(good.sum(1), 1)[:, None, None]
    out = (Xa - m).astype(np.float32)
    return out[0] if one else out


KEEP_MEAN = {}   # T2-D517: {"on": True} -> the alignment's statistics stay on the de-meaned windows, but the transform is applied to the window WITH
                 # its per-channel time mean (the DC level every member's _demean removed); modes none / sz / ea only; {} = off (bitwise)


def align_stream(X, mode, scale=1.0, prior=None, n0=0.0, shrink=0.05, edges=FB_EDGES, band=EAB_BAND, gamma=EAF_GAMMA):
    """Causally normalise ONE stream's windows, given in stream (chronological) order. Returns float32 (N, C, T)."""
    X0 = np.asarray(X, dtype=np.float64)
    X = _demean(X0)
    Xa = X0 if KEEP_MEAN.get("on") else X                   # T2-D517: what the transform is applied to
    assert not KEEP_MEAN.get("on") or mode in ("none", "sz", "ea"), f"T2-D517: keep-mean is wired for none / sz / ea, not {mode}"
    N, C, T = X.shape
    if mode == "eaf":
        # T2-D428 (bench only): broadband EA whose running covariance forgets: R_t = sum_k gamma^(t-k) C_k / sum_k gamma^(t-k); gamma 1 = "ea".
        assert prior is None or n0 == 0, "eaf: no population prior (the shipped ShallowNet line runs n0 0)"
        covs = window_covs(X)
        out = np.empty_like(X)
        S, w = np.zeros((C, C)), 0.0
        for t in range(N):
            S, w = gamma * S + covs[t], gamma * w + 1.0
            out[t] = _inv_sqrt(S / w, shrink) @ X[t]
        return out.astype(np.float32)
    if mode in ("eab", "eal"):
        band = band if mode == "eab" else EAL_BAND
        # T2-D426 (bench only, as fbea): ONE matrix W_t = sqrt(tr R_b / tr R) R_b^{-1/2} applied to the broadband window, R_b = the stream's causal
        # running covariance of the `band` band (rfft mask), R = the broadband one; when R_b is proportional to R this is exactly "ea".
        assert T == FB_T, f"eab assumes the contract's {FB_T} samples at {FB_FS:g} Hz, got T={T}"
        assert prior is None or n0 == 0, "eab: no population prior (the shipped ShallowNet line runs n0 0)"
        Xb = band_split(X, FB_FS, (0.0,) + tuple(band))[1]             # the [lo, hi) band
        cb, ca = np.cumsum(window_covs(Xb), 0), np.cumsum(window_covs(X), 0)
        out = np.empty_like(X)
        for t in range(N):
            Rb, Ra = cb[t] / (t + 1.0), ca[t] / (t + 1.0)
            out[t] = np.sqrt(np.trace(Rb) / max(np.trace(Ra), 1e-300)) * (_inv_sqrt(Rb, shrink) @ X[t])
        return out.astype(np.float32)
    if mode == "fbea":
        assert T == FB_T, f"fbea assumes the contract's {FB_T} samples at {FB_FS:g} Hz, got T={T}"
        Xb = band_split(X, FB_FS, edges)                          # (B, N, C, T)
        covs = np.stack([window_covs(xb) for xb in Xb])           # (B, N, C, C)
        cum = np.cumsum(covs, 1)
        trs = np.cumsum(np.trace(covs, axis1=2, axis2=3), 1)      # (B, N)
        if prior is not None:
            assert prior.shape == (len(Xb), C, C), prior.shape
        out = np.zeros_like(X)
        for t in range(N):
            R, k = cum[:, t], t + 1.0
            if prior is not None and n0 > 0:
                R = R + n0 * (trs[:, t] / k)[:, None, None] * prior   # per band, as "ea"
                k += n0
            R = R / k
            tb = np.trace(R, axis1=1, axis2=2)
            share = tb / max(tb.sum(), 1e-300)
            for b in range(len(Xb)):
                if share[b] > 0:
                    out[t] += np.sqrt(share[b]) * (_inv_sqrt(R[b], shrink) @ Xb[b, t])
        return out.astype(np.float32)
    if mode == "none":
        return (Xa / scale).astype(np.float32)
    if mode == "wz":
        return (X / (X.std(axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
    if mode == "sz":
        ms = (X ** 2).mean(-1)                       # (N, C) per-window channel power (window mean is 0)
        cum = np.cumsum(ms, 0) / np.arange(1, N + 1)[:, None]
        return (Xa / np.sqrt(cum + 1e-12)[:, :, None]).astype(np.float32)
    if mode in ("ea", "eafull"):
        covs = window_covs(X)
        out = np.empty_like(X)
        if mode == "eafull":
            W = _inv_sqrt(covs.mean(0), shrink)
            return np.einsum("cd,ndt->nct", W, X).astype(np.float32)
        cum = np.cumsum(covs, 0)
        trs = np.cumsum(np.trace(covs, axis1=1, axis2=2))
        for t in range(N):
            R = cum[t]
            k = t + 1.0
            if prior is not None and n0 > 0:
                R = R + n0 * (trs[t] / k) * prior    # prior is trace-normalised: scale-matched to the stream
                k += n0
            out[t] = _inv_sqrt(R / k, shrink) @ Xa[t]
        return out.astype(np.float32)
    raise ValueError(f"unknown align mode {mode!r}")


class StreamAligner:
    """One-window-at-a-time twin of ``align_stream`` (what ``predict`` runs). ``reset()`` at every session start.

    Guards (hub LESSONS B24; inert on every public PROTEUS stream, so a clean stream is bit-identical to the plain
    recursion): non-finite samples -> 0 and |x| <= CLIP before anything else; a window that had non-finite samples or
    has zero power is normalised with the current statistics but never updates them; "sz": a channel's window power
    above SPIKE x the running median of that channel's window powers enters the running mean as that median (largest
    public ratio 1.2e4 (P06 s1); a x1e4-amplitude spike is 1e8), so one spike cannot squash the rest of a session;
    "ea" (T2-D116): likewise per window, a covariance whose trace is above SPIKE x the running median trace enters the
    running mean rescaled to that median trace; shrinkage + the eigenvalue floor of _inv_sqrt cover flat channels and a
    near-singular first window."""
    CLIP, SPIKE = 1e6, 1e6

    def __init__(self, mode, scale=1.0, prior=None, n0=0.0, shrink=0.05):
        assert mode in ("none", "wz", "sz", "ea"), mode
        self.mode, self.scale, self.prior, self.n0, self.shrink = mode, scale, prior, n0, shrink
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_ms = None
        self.hist = []
        self.sum_cov = None
        self.sum_tr = 0.0
        self.covs = []

    def _mean_ms(self):
        H = np.asarray(self.hist)                                     # (n, C) window powers that entered the sum
        med = np.median(H, 0)
        out = (H > self.SPIKE * med) & (med > 0)
        if not out.any():
            return self.sum_ms / self.n                               # the plain recursion, bit for bit
        return (self.sum_ms - np.where(out, H - med, 0.0).sum(0)) / self.n

    def __call__(self, x):
        """x: (C, T) one window -> (C, T) float32."""
        x = np.asarray(x, dtype=np.float64)
        bad = not np.isfinite(x).all()
        if bad:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -self.CLIP, self.CLIP)
        x = x - x.mean(-1, keepdims=True)
        if self.mode == "none":
            return (x / self.scale).astype(np.float32)
        if self.mode == "wz":
            return (x / (x.std(-1, keepdims=True) + 1e-6)).astype(np.float32)
        if self.mode == "sz":
            ms = (x ** 2).mean(-1)
            if not bad and ms.any():
                self.n += 1
                self.sum_ms = ms if self.sum_ms is None else self.sum_ms + ms
                self.hist.append(ms)
            if self.n == 0:                                           # nothing usable yet: the window's own power
                return (x / np.sqrt(ms + 1e-12)[:, None]).astype(np.float32)
            return (x / np.sqrt(self._mean_ms() + 1e-12)[:, None]).astype(np.float32)
        cov = x @ x.T / x.shape[-1]
        if not bad and cov.any():
            self.n += 1
            self.sum_cov = cov if self.sum_cov is None else self.sum_cov + cov
            self.sum_tr += float(np.trace(cov))
            self.covs.append(cov)
            self.hist.append(float(np.trace(cov)))
        if self.n == 0:
            return np.zeros_like(x, dtype=np.float32)
        R, k, s_tr = self.sum_cov, float(self.n), self.sum_tr
        H = np.asarray(self.hist)
        med = float(np.median(H))
        out = np.flatnonzero((H > self.SPIKE * med) & (med > 0))
        if len(out):                                                  # else the plain recursion, bit for bit
            R = R - sum(self.covs[i] * (1.0 - med / H[i]) for i in out)
            s_tr = s_tr - float(sum(H[i] - med for i in out))         # the prior's scale gets the same correction
        if self.prior is not None and self.n0 > 0:
            R = R + self.n0 * (s_tr / self.n) * self.prior
            k += self.n0
        return (_inv_sqrt(R / k, self.shrink) @ x).astype(np.float32)
