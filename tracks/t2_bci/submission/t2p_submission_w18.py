"""Track 2 (PROTEUS streaming contract, kit >= 442edcd) submission: causal per-session alignment + CNN ensemble
(+ optional causal output-prior correction). github.com/LiuyinYang1101/eeg-fm-agent-bci-ni26, tracks/t2_bci.

Self-contained by contract (no extra modules): imports only numpy, torch, braindecode (all in the worker image).
Files read from meta["submission_dir"]: config.json (training contract), weights.pt (list of CNN state dicts) and, for
ST-EEGFormer members (cfg["fm_members"]), one fp16 state dict per member. The pure-torch ST-EEGFormer
(src/ni26/fm/steeg_pt.py; parity with the timm path: steeg_pt_check.py) is inlined at the marker below by t2p_final.py.
Output = (1 - fm_weight) * mean CNN probability + fm_weight * mean FM probability, then the optional prior correction
(T2-D352, template W11 = W9 + this: an optional THIRD CNN family cfg["cnn3"] takes cfg["cnn3"]["weight"] of the output out of the CNN
share, with its own causal aligner and, with cfg["user_blend3"], per-user copies routed by the main family's fingerprint; without cnn3 every
path runs W9's arithmetic exactly.)
(T2-D116: an optional second CNN family cfg["cnn2"] takes cfg["cnn2"]["weight"] of the output out of the CNN share;
T2-D161: with cfg["user_blend2"], that family is blended toward its own per-user copies of the SAME routed user);
every member has its own causal aligner (the CNN and FM inputs may use different alignment modes). FM members run under
bf16 autocast on CUDA (their training / research-scoring precision); the CNNs stay fp32.

Codabench streams ONE window (1, 41, 480) at a time, chronologically within a participant x session, from a fresh
``copy.deepcopy`` of this model whose ``reset_state()`` is called first (benchmark_utils/streaming.py). Every piece of
state here (aligner sums, prior sums) lives on the model instance and is cleared by ``reset_state()``; nothing is
shared across copies (numpy arrays are deep-copied with the model). The aligner is a verbatim copy of
``src/ni26/stream.py::StreamAligner`` (the class the models were trained through; equality tested in
``tracks/t2_bci/scripts/t2p_selftest.py``).
"""

# T2-D525 / D524 (template W18 = W17 + TWO SEPARABLE SWITCHES; PREPARED 2026-11-16, NOT APPROVED: it ships only after the user's own line in
# T2's shell, naming exactly the switches the W18 fresh-draw stack read keeps). With neither switch every line below is W17's (bitwise).
# (1) cfg["imcoh"] = {"file": "imcoh.npz", "weight": w}: D487's lagged-coupling family. Per window: 18 nodes, 153 pairs x theta / mu / beta, the
#     imaginary coherency from 1-s Hann segments (50 % overlap, segment-mean removed) of the DELIVERED window = scripts/local/t2_d487_wpli.features
#     exactly; per stream a causal re-centring (minus the mean of windows 1 .. t-1; none at t = 1); z with the training mu / sd; an L2 logistic
#     (coef, intercept: softmax); its OWN Sinkhorn corrector (file start); output = (1 - w) * W17's pool + w * that, AFTER the window-mean family.
# (2) cfg["fm_side"] = true: every FM member whose config entry carries "side" = {"kind": "dcmean", "n0", "file"} adds the window-mean side head
#     at its logits: p_j <- softmax(log p_j + W_j s + b_j), s = the side nets' causal-z window means (W17's _side_z; the head file's mu / sd / n0
#     must equal side.pt's). Exact, since softmax(o + h) = softmax(log softmax(o) + h); applied after the (graph or eager) FM forward, so the
#     CUDA graph is unchanged. The head was trained with the member (t2p_fm side='dcmean'; t2p_final --fm-side). Members without "side" are W17's.
# T2-D517 (template W17 = W16 + window-mean SIDE NETS; PREPARED 2026-11-15, NOT APPROVED: it ships only after the user's own line in T2's shell):
# cfg["side"] = {"file": "side.pt", "kind": "dcmean", "r": 1/3, "families": ["cnn2", "cnn3"]}. Per window: the same 82 window means as the
# window-mean family (delivered + CAR view), z-scored causally per stream with the side nets' OWN training prior (side.pt mu / sd, n0) =
# t2p_bench.side_z; per family f, the side nets (pooled nets of f's architecture trained on all users with that z vector entering a linear head
# added at the logits, tracks/t2_bci/scripts/t2p_side_final.py) take f's own aligned input; their mean softmax(net(z_f) + head(s)) is blended
# into f's part AFTER the in-stream blend, BEFORE f's corrector: part = (1 - r) * part + r * side. Per-stream state only (reset_state); the side
# nets are frozen and shared by the kit's per-stream copies. Without cfg["side"] every line below is W16's (bitwise).
# T2-D516 (template W16 = W15 + a window-mean family; PREPARED 2026-11-15, NOT APPROVED: it ships only after the user's own line in T2's shell):
# cfg["dcfam"] = {"file": "dcfam.npz", "weight": w, "seeds": [1, 2, 3], "n0": 10}. Per window: the 82 features = the per-channel means of the
# DELIVERED window (the members' aligners remove exactly these) and of its CAR view, z-scored causally per stream (running mean / sd of the windows
# <= t, shrunk to the training prior with n0 pseudo-windows); a bag of LightGBM models evaluated from padded tree arrays (numpy, bitwise LightGBM's
# predict_proba in scripts/local/t2_d516_final.py's check); the family's probability through its OWN Sinkhorn corrector (cfg start); the output =
# (1 - w) * W15's per-family pool + w * that. Per-stream state only (reset_state). Without cfg["dcfam"] every line below is W15's (bitwise).
# T2-D510 / D511 (template W15 = W14 + in-stream adaptation; PREPARED 2026-11-15, NOT APPROVED: it ships only after the user's own line in T2's
# shell): cfg["instream"] = {"families": {"cnn" | "cnn2" | "cnn3": {"member": i, "epochs": E, "lr": lr}}, "refits": [60, 100, 150], "r": 0.5,
# "labels": "conf" | "confR", "bs": 32, "wd": 0.01, "label_smoothing": 0.1, "seed": S, "max_stream_s": .., "max_run_s": ..}. Per stream, at each
# refit point c (c windows delivered), a COPY of the family's member net i is fine-tuned from the shipped weights (the per-user copies' own recipe:
# AdamW, cosine, no warm-up, last epoch = ni26.train.finetune) on the windows delivered so far, labelled by the package's OWN decoded posterior
# ("conf": each window's causal per-family decode of the unadapted parts; "confR": the unadapted parts of windows 1 .. c re-decoded over the whole
# prefix), the confident half of each predicted class kept; every later window's family part = (1 - r) * the part + r * the copy's probability,
# BEFORE that family's corrector. Every piece of it is per-stream state (reset_state; the kit's deepcopy per stream): the shipped members are
# never modified. A stream shorter than the first refit point, or a run past max_run_s, never adapts = W14's output (bitwise). Without
# cfg["instream"] every line below is W14's (bitwise).
# T2-D482 (template W14 = W13 + a per-member input reference; PREPARED 2026-11-14, NOT APPROVED: it ships only after the user's own line in T2's
# shell): an fm_members entry may carry "ref": "car" | "lap" (or t2p_final's "car": true) -> that member's aligner gets the window re-referenced
# by car_windows / lap_windows below (verbatim from src/ni26/stream.py, the functions its training input went through; "lap" reads the member's
# "lap_nb" neighbour table). Without any such key every line below is W13's (bitwise).
# T2-D476 (template W13 = W12 + per-family decoding; the user's "go W13" in T2's shell, 2026-11-13): cfg["tta"]["per_family"] =
# {"cnn": start, "cnn2": start, "cnn3": start, "fm": start} (one class vector per family present) -> each family's output part goes through
# ITS OWN causal Sinkhorn corrector (same iters / n0 / lam, its own start vector) BEFORE the pool; the output = the sum over families of
# (the family's share of the output x its corrected part). Without "per_family" every line below is W12's (bitwise).
import inspect
import json
import os

import numpy as np
_PARTS = __import__("os").environ.get("NI26_PARTS") == "1"   # T2-D210: expose the pre-mix parts (diagnostics only; off on the worker)
import torch
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver


# ------------------------------------------------------------------- verbatim from src/ni26/stream.py (keep in sync)
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


def lap_windows(X, nb, flat_tol=1e-6, chunk=256):
    """T2-D480: nearest-neighbour (Hjorth) surface Laplacian per window, from the window alone (predict-safe): x_c <- x_c - mean of x over
    the c's k nearest GOOD neighbours (finite everywhere and not flat, as car_windows). A channel with no good neighbour is left as it is;
    a dead channel never enters a neighbour's mean. It removes what is common to a channel and its neighbours (far-field sources: the
    reference, the ocular field) and keeps the local. X: (N, C, T) or (C, T)."""
    X = np.asarray(X, dtype=np.float32)
    one = X.ndim == 2
    Xa = X[None] if one else X
    nb = np.asarray(nb, dtype=np.int64)            # W14: the member's neighbour table (stream.lap_neighbours at merge time)
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


def _inv_sqrt(R, shrink):
    C = R.shape[-1]
    tr = np.trace(R) / C
    R = (1 - shrink) * R + shrink * tr * np.eye(C)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, w.max() * 1e-8)
    return (V * w ** -0.5) @ V.T


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


# ------------------------------------------------------------------- end verbatim


# ------------------------------------------------------------------- verbatim from tracks/t2_bci/scripts/t2p_route.py (keep in sync)
def fingerprint(X, fs, names):
    """T2-D124 session-stable fingerprints per window: IAF (peak of the mean posterior-channel PSD in 7-13 Hz), 1/f slope
    (log-log fit of the channel-mean PSD over 2-40 Hz without 7-13 Hz), channel-normalised relative-band topography (log
    relative power per channel and band minus its channel mean), trace-normalised covariance shape (upper triangle of
    logm(C / tr C))."""
    from scipy.signal import welch
    X = np.asarray(X, np.float64)
    X = X - X.mean(-1, keepdims=True)
    f, P = welch(X, fs=fs, nperseg=min(240, X.shape[-1]), axis=-1)            # (N, C, F)
    post = [i for i, c in enumerate(names) if c.upper().startswith("P")]
    a = (f >= 7) & (f <= 13)
    iaf = f[a][np.argmax(np.log(P[:, post][:, :, a].mean(1) + 1e-20), 1)]
    fit = (f >= 2) & (f <= 40) & ~a
    lf, lp = np.log10(f[fit]), np.log10(P[:, :, fit].mean(1) + 1e-20)
    slope = np.polyfit(lf, lp.T, 1)[0]
    tot = P[:, :, (f >= 1) & (f <= 45)].sum(-1) + 1e-20
    topo = []
    for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        r = np.log(P[:, :, (f >= lo) & (f < hi)].sum(-1) / tot + 1e-20)
        topo.append(r - r.mean(1, keepdims=True))
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    C = C / np.trace(C, axis1=1, axis2=2)[:, None, None]
    C = C + 1e-6 * np.eye(C.shape[1])
    w, V = np.linalg.eigh(C)
    L = np.einsum("nij,nj,nkj->nik", V, np.log(np.maximum(w, 1e-12)), V)
    iu = np.triu_indices(C.shape[1])
    return np.concatenate([iaf[:, None], slope[:, None]] + topo + [L[:, iu[0], iu[1]]], 1)


# ------------------------------------------------------------------- end verbatim fingerprint

# @@INLINE_STEEG_PT@@ (t2p_final.py replaces this line with src/ni26/fm/steeg_pt.py when the package has ST members)

# T2-D140: the in-house MOR member is not part of this public release (see docs/REPRODUCE.md).


def _fft_resample(x, n_new):
    """= tracks/t1_image/scripts/t1_fm.py::fft_resample (the MOR member's training input path)."""
    n = x.shape[-1]
    if n == n_new:
        return x
    Xf = torch.fft.rfft(x.float(), dim=-1)
    k = n_new // 2 + 1
    Xf = Xf[..., :k] if Xf.shape[-1] >= k else F.pad(Xf, (0, k - Xf.shape[-1]))
    return torch.fft.irfft(Xf, n=n_new, dim=-1) * (n_new / n)




class MORMemberT2(torch.nn.Module):
    """The in-house MOR member. Its model code is unpublished and is not part of this public release."""

    def __init__(self, m, n_classes):
        raise NotImplementedError("the in-house MOR model is not part of this public release")


class PriorCorrector:
    """Causal output re-balancing = t2p_report.causal_prior_correction for one stream: q_t = p_t / m_t^alpha with
    m_t = (n0 / K + sum_{i<=t} p_i) / (n0 + t)."""

    def __init__(self, alpha, n0, k):
        self.alpha, self.n0, self.k = float(alpha), float(n0), int(k)
        self.reset()

    def reset(self):
        self.n = 0
        self.cum = np.zeros(self.k)

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float64)
        if np.isfinite(p).all() and p.sum() > 0:                  # B24: a broken output never enters the running mean
            self.n += 1
            self.cum += p
        else:
            p = np.full(self.k, 1.0 / self.k)
        if self.n0 + self.n > 0:
            m = (self.n0 / self.k + self.cum) / (self.n0 + self.n)
        else:                                                      # n0 = 0 and nothing usable yet: no correction
            m = np.full(self.k, 1.0 / self.k)
        q = p / np.maximum(m, 1e-12) ** self.alpha                 # n0 = 0: a class whose running mean underflowed
        return q / q.sum()


class SinkhornCorrector:
    """T2-D285 / D286 (the user approved SK10, 2026-10-27): causal Sinkhorn re-balancing of one stream. At window t, the matrix
    [n0 uniform pseudo-rows; p_1 .. p_t] (the package mix, before any correction) gets `iters` rounds of column scaling to equal
    class mass and row normalisation; the output is its last row. It assumes only what PriorCorrector assumes (balanced classes
    within a session): iters = 1 is PriorCorrector(alpha 1, n0) exactly. = t2p_report.causal_sinkhorn for one stream."""

    def __init__(self, iters, n0, k, lam=1.0, start=None):
        self.iters, self.n0, self.k, self.lam = int(iters), int(n0), int(k), float(lam)
        assert self.iters >= 1 and self.n0 >= 0 and 0.0 < self.lam <= 1.0, (iters, n0, lam)
        # T2-D459 (template W12; the user's template line 'go', T2 shell 2026-11-12 17:5x): the n0 pseudo-rows hold
        # cfg tta.start (a fixed class vector, the package mix's known bias) instead of the uniform row; absent = W11 bitwise
        self.start = np.full(self.k, 1.0 / self.k) if start is None else np.asarray(start, dtype=np.float64)
        assert self.start.shape == (self.k,) and np.isfinite(self.start).all() and (self.start > 0).all() \
            and abs(self.start.sum() - 1.0) < 1e-6, start
        self.reset()

    def reset(self):
        self.rows = [self.start.copy() for _ in range(self.n0)]

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float64)
        if not (np.isfinite(p).all() and p.sum() > 0):           # B24: a broken output enters as uniform
            p = np.full(self.k, 1.0 / self.k)
        self.rows.append(p)
        M = np.stack(self.rows)
        if self.lam == 1.0:
            for _ in range(self.iters):
                M = M * (M.shape[0] / self.k) / np.maximum(M.sum(0, keepdims=True), 1e-300)
                M = M / M.sum(1, keepdims=True)
            return M[-1]
        w = self.lam ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)     # T2-D304: SK-lambda, row weight lam^age in the column balance
        for _ in range(self.iters):
            M = M * (w.sum() / self.k) / np.maximum((M * w[:, None]).sum(0, keepdims=True), 1e-300)
            M = M / M.sum(1, keepdims=True)
        return M[-1]


def _norm(name):
    s = str(name).strip().upper()
    s = s[4:] if s.startswith("EEG ") else s
    return s.split("-")[0].strip()


def _prefix_decode(rows, iters, lam, k, n0):
    """T2-D511 (W15): the corrector's Sinkhorn on [n0 start rows; p_1 .. p_c] (rows as SinkhornCorrector holds them), EVERY data row returned
    (the corrector returns only the last). = scripts/local/t2_d511_teacher.py prefix_decode."""
    M = np.stack(rows).astype(np.float64)
    if lam == 1.0:
        for _ in range(int(iters)):
            M = M * (M.shape[0] / k) / np.maximum(M.sum(0, keepdims=True), 1e-300)
            M = M / M.sum(1, keepdims=True)
        return M[int(n0):]
    w = float(lam) ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
    for _ in range(int(iters)):
        M = M * (w.sum() / k) / np.maximum((M * w[:, None]).sum(0, keepdims=True), 1e-300)
        M = M / M.sum(1, keepdims=True)
    return M[int(n0):]


def _conf_half(post):
    """T2-D510 (W15): -> (kept row indices, their labels): the confident half (ceil) of each predicted class (= t2p_bench's 'conf')."""
    ya, cf = post.argmax(1), post.max(1)
    keep = np.zeros(len(post), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    ka = np.flatnonzero(keep)
    return ka, ya[ka]


def _w15_cosine(step, total, warm):
    """= ni26.train._cosine."""
    import math
    if step < warm:
        return (step + 1) / max(warm, 1)
    p = (step - warm) / max(total - warm, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def _w15_finetune(net, X, y, epochs, lr, wd, bs, ls, seed):
    """T2-D510 (W15): = ni26.train.finetune(net, X, y, peruser_for(recipe), seed) for the copies' recipe (PERUSER: no warm-up, no swa / kl / sbal /
    vp, every part trains): a COPY of net, AdamW, cosine per step, the last epoch, eval mode. X (N, C, T) / y (N,) tensors on the net's device."""
    import copy
    import math
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    net = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    total = epochs * int(math.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: _w15_cosine(s_, total, 0))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g).to(X.device)
        for b in [perm[i:i + bs] for i in range(0, len(X), bs)]:
            loss = F.cross_entropy(net(X[b]), y[b], label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


def _dc_trees_eval(z, feat, thr, left, right, dleft, leaf, K=3):
    """T2-D516 (W16): raw class scores of padded LightGBM trees for ONE row z (F,); tree t belongs to class t % K; NaN goes to the default side.
    = scripts/local/t2_d516_final.py eval_arrays for one row."""
    T = len(feat)
    tt = np.arange(T)
    node = np.zeros(T, np.int64)
    done = np.zeros(T, bool)
    out = np.zeros(T)
    for _ in range(64):
        x = z[feat[tt, node]]
        go = np.where(np.isnan(x), dleft[tt, node], x <= thr[tt, node])
        nxt = np.where(go, left[tt, node], right[tt, node])
        hit = (nxt < 0) & ~done
        out[hit] = leaf[tt[hit], -(nxt[hit]) - 1]
        done |= nxt < 0
        node = np.where(nxt < 0, 0, nxt)
        if done.all():
            break
    return np.array([out[k::K].sum() for k in range(K)])


class DCFamily:
    """T2-D516 (W16): the window-mean family for one stream (state: the features' running sums; reset per stream)."""

    def __init__(self, arrs, seeds, n0, k):
        self.mu, self.sd = np.asarray(arrs["mu"], np.float64), np.asarray(arrs["sd"], np.float64)
        self.models = [tuple(np.asarray(arrs[f"s{s}_{nm}"]) for nm in ("feat", "thr", "left", "right", "dleft", "leaf")) for s in seeds]
        self.n0, self.k = float(n0), int(k)
        self.reset()

    def reset(self):
        self.s1, self.s2, self.n = np.zeros_like(self.mu), np.zeros_like(self.mu), 0

    def __call__(self, x):
        """x: the delivered window (C, T) in the training channel order -> the family's class probability (bag mean)."""
        m = np.asarray(x, np.float64).mean(-1)
        f = np.concatenate([m, m - m.mean()])
        f = np.where(np.isfinite(f), f, self.mu)
        self.s1, self.s2, self.n = self.s1 + f, self.s2 + f * f, self.n + 1
        nn = self.n0 + self.n
        mm = (self.n0 * self.mu + self.s1) / nn
        vv = (self.n0 * (self.sd ** 2 + self.mu ** 2) + self.s2) / nn - mm ** 2
        z = (f - mm) / np.sqrt(np.maximum(vv, 1e-12))
        ps = []
        for a in self.models:
            sc = _dc_trees_eval(z, *a, K=self.k)
            e = np.exp(sc - sc.max())
            ps.append(e / e.sum())
        return np.mean(ps, 0)


IC_NODES = ("Fz", "FCz", "Cz", "C3", "C4", "CP3", "CP4", "F5", "FC5", "F6", "FC6", "TP7", "TP8", "F3", "F4", "Pz", "P3", "P4")
IC_BANDS = ((4, 8), (8, 13), (13, 30))


def imcoh_features(x, ci, fs, seg=120, hop=60):
    """T2-D525 (W18): one window (C, T) -> the 459 imaginary-coherency features of scripts/local/t2_d487_wpli.features (same arithmetic, N = 1)."""
    xx = np.asarray(x, np.float64)[list(ci)]
    C, T = xx.shape
    w = np.hanning(seg)
    Fs_ = np.stack([np.fft.rfft((xx[:, s:s + seg] - xx[:, s:s + seg].mean(-1, keepdims=True)) * w, axis=-1) for s in range(0, T - seg + 1, hop)], 0)
    freqs = np.fft.rfftfreq(seg, 1 / fs)
    iu = np.triu_indices(C, 1)
    out = []
    for lo, hi in IC_BANDS:
        Fb = Fs_[..., (freqs >= lo) & (freqs < hi)]                        # (S, C, nb)
        Sm = np.einsum("sif,sjf->sfij", Fb, Fb.conj()).mean((0, 1))         # (C, C)
        p = np.real(np.diagonal(Sm))
        out.append((np.imag(Sm) / np.sqrt(p[:, None] * p[None, :] + 1e-30))[iu[0], iu[1]])
    return np.concatenate(out).astype(np.float32).astype(np.float64)       # float32 as the training features, then float64


class ICFamily:
    """T2-D525 (W18): the imcoh family for one stream (state: the running sum of the stream's earlier windows' features; reset per stream)."""

    def __init__(self, arrs, ci, fs):
        self.ci, self.fs = tuple(int(v) for v in ci), float(fs)
        self.mu, self.sd = np.asarray(arrs["mu"], np.float64), np.asarray(arrs["sd"], np.float64)
        self.coef, self.icpt = np.asarray(arrs["coef"], np.float64), np.asarray(arrs["intercept"], np.float64)
        assert self.coef.shape == (len(self.icpt), len(self.mu)) == (len(self.icpt), 459), (self.coef.shape, self.mu.shape)
        self.reset()

    def reset(self):
        self.s, self.n = np.zeros_like(self.mu), 0

    def __call__(self, x):
        f = imcoh_features(x, self.ci, self.fs)
        f = np.where(np.isfinite(f), f, 0.0)
        r = f - self.s / self.n if self.n else f                         # causal re-centring: minus the mean of windows 1 .. t-1
        self.s, self.n = self.s + f, self.n + 1
        sc = self.coef @ ((r - self.mu) / self.sd) + self.icpt
        e = np.exp(sc - sc.max())
        return e / e.sum()


def build_net(cfg):
    import braindecode.models as M
    cls = getattr(M, cfg["braindecode_class"])
    kw = dict(n_chans=len(cfg["ch_names"]), n_outputs=int(cfg["n_classes"]), n_times=int(cfg["n_times"]))
    for k, v in (cfg.get("model_kwargs") or {}).items():    # T2-D122: e.g. FBCNet's 13-band bank (bands as [lo, hi] lists)
        kw[k] = [tuple(b) for b in v] if k == "n_bands" and isinstance(v, list) else v
    if "sfreq" in inspect.signature(cls.__init__).parameters:
        kw["sfreq"] = float(cfg["sfreq"])
    return cls(**kw)


def load_net(net, sd):
    """strict load. Review 2026-10-16 (T2 P1a): FBCNet's FIR taps (spectral_filtering.fir_list / fir_a_list) are frozen
    Parameters whose LENGTH comes from mne's filter design when the net is built, so another braindecode / mne in the
    worker image could build other lengths; the forward reads the tensors themselves, so a length mismatch there adopts
    the trained taps (the values are overwritten by the strict load anyway). Anything else still fails loudly."""
    own = net.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape and ".fir_" in k and not k.endswith(".weight"):
            mod, _, i = k.rpartition(".")
            net.get_submodule(mod)[int(i)] = torch.nn.Parameter(v.detach().clone(), requires_grad=False)
            print(f"[ni26] {k}: built {tuple(own[k].shape)} != trained {tuple(v.shape)}: trained taps adopted", flush=True)
    net.load_state_dict(sd, strict=True)
    return net


def versions():
    import importlib.metadata as md
    out = {}
    for d in ("torch", "braindecode", "numpy", "scipy", "mne"):
        try:
            out[d] = md.version(d)
        except Exception:
            out[d] = None
    return out


class Stacked(torch.nn.Module):
    """Runtime (hub LESSONS E13): same-architecture members in ONE vmapped forward (torch.func.stack_module_state +
    functional_call), instead of one launch chain per member; -> (n_members, B, K) logits. The per-member modules stay in
    the model as the reference and the fallback (StreamModel checks stacked == loop at load)."""

    def __init__(self, nets):
        super().__init__()
        import copy
        from torch.func import stack_module_state
        pr, bu = stack_module_state(list(nets))
        self.names_p, self.names_b = list(pr), list(bu)
        for i, v in enumerate(pr.values()):
            self.register_buffer(f"p{i}", v.detach().clone())
        for i, v in enumerate(bu.values()):
            self.register_buffer(f"b{i}", v.detach().clone())
        self.base = [copy.deepcopy(nets[0]).to("meta")]      # a list: not a submodule (stays on meta)

    def forward(self, x):
        from torch.func import functional_call
        pr = {k: getattr(self, f"p{i}") for i, k in enumerate(self.names_p)}
        bu = {k: getattr(self, f"b{i}") for i, k in enumerate(self.names_b)}
        return torch.vmap(lambda a, b, z: functional_call(self.base[0], (a, b), (z,)), in_dims=(0, 0, None))(pr, bu, x)


def _stack_or_none(nets, n_chans, n_times, tol=1e-4):
    """Stacked(nets) if it reproduces the per-member loop on a probe input within tol (logits), else None (loop)."""
    if len(nets) < 2:
        return None
    try:
        st = Stacked(nets)
        dev = next(nets[0].parameters()).device
        st = st.to(dev)
        z = torch.randn(1, n_chans, n_times, generator=torch.Generator().manual_seed(0)).to(dev) * 20
        with torch.no_grad():
            d = float((st(z) - torch.stack([n(z) for n in nets])).abs().max())
        if d <= tol:
            return st
        print(f"[ni26] stacked members differ from the loop by {d:.2e}: loop kept")
    except Exception as e:                              # e.g. an op vmap cannot batch: the loop is always correct
        print(f"[ni26] member stacking unavailable ({type(e).__name__}): loop kept")
    return None


class STLinearD(torch.nn.Module):
    """T2-D188 / D189: an ST-EEGFormer Linear plus an optional per-user low-rank delta y += s (x A^T) B^T, with A (r x in) and B (out x r)
    sliced from the flat vector `d` that STRouted sets before each call (d = None -> the plain Linear). The weight / bias keep their
    state-dict names, so the member's file loads strict before the swap."""

    def __init__(self, lin, oa, ob, oc, r, scale):
        super().__init__()
        self.weight, self.bias = lin.weight, lin.bias
        self.inf, self.outf = lin.in_features, lin.out_features
        self.oa, self.ob, self.oc, self.r, self.s = int(oa), int(ob), int(oc), int(r), float(scale)
        self.d = None

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        d = self.d
        if d is None:
            return y
        a = d[self.oa:self.ob].view(self.r, self.inf).to(x.dtype)
        b = d[self.ob:self.oc].view(self.outf, self.r).to(x.dtype)
        return y + self.s * F.linear(F.linear(x, a), b)


class STRouted(torch.nn.Module):
    """T2-D188 / D189: an ST member with per-user deltas applied INSIDE its one forward. bank (K, N) = every training user's flat
    delta (A then B per layout entry); forward(x, d) points every STLinearD at d (a graph input, so a CUDA-graph replay reads the
    routed user's delta that _run copies in); delta(None) = zeros = the pooled member."""

    def __init__(self, net, bank, layout, r, scale):
        super().__init__()
        self.net, off = net, 0
        for name, i, o in layout:
            parent_name, attr = name.rsplit(".", 1)
            parent = net.get_submodule(parent_name)
            lin = getattr(parent, attr)
            assert isinstance(lin, torch.nn.Linear) and (lin.in_features, lin.out_features) == (int(i), int(o)), name
            setattr(parent, attr, STLinearD(lin, off, off + r * i, off + r * i + o * r, r, scale))
            off += r * i + o * r
        assert bank.shape[1] == off, (bank.shape, off)
        self.register_buffer("bank", bank.float(), persistent=False)
        self.register_buffer("zero", torch.zeros(off), persistent=False)

    def delta(self, k):
        return self.zero if k is None else self.bank[k]

    def forward(self, x, d):
        for m in self.net.modules():
            if isinstance(m, STLinearD):
                m.d = d
        return self.net(x)


class StreamModel(torch.nn.Module):
    def __init__(self, nets, cfg, meta, fm_nets=(), nets2=(), pu_nets=(), pu_nets2=(), nets3=(), pu_nets3=()):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        # T2-D124c: per-user copies of the main CNN family (pu_nets[k] = the seeds' copies for training user k), blended
        # into the main family's output with weight r toward the user whose fingerprint centroid is nearest to the
        # session's running-mean fingerprint (LDA space of the training users); no gate
        self.pu_nets = torch.nn.ModuleList([torch.nn.ModuleList(v) for v in pu_nets])
        self.stk, self.stk_pu = [None, None], [None] * len(self.pu_nets)     # loops until stack_members() (load_model)
        ub = cfg.get("user_blend") or {}
        if len(self.pu_nets):
            self.ub_r = float(ub["r"])
            self.ub_mean, self.ub_scale = np.asarray(ub["mean"], np.float64), np.asarray(ub["scale"], np.float64)
            self.ub_W, self.ub_mu = np.asarray(ub["W"], np.float64), np.asarray(ub["mu"], np.float64)
            assert len(self.pu_nets) == len(self.ub_mu)
        self.ub_sum, self.ub_n = None, 0
        self.ub_every, self.ub_t = int(ub.get("every", 1)), 0
        # T2-D116: an optional second CNN family (cfg["cnn2"]) with its own causal aligner; output share cfg["cnn2"]["weight"]
        self.nets2 = torch.nn.ModuleList(nets2)
        c2 = cfg.get("cnn2") or {}
        if len(self.nets2):
            a2 = c2["align"]
            self.aligner2 = StreamAligner(a2["mode"], scale=float(a2.get("scale", 1.0)),
                                          prior=np.asarray(a2["prior"], dtype=np.float64) if a2.get("prior") is not None else None,
                                          n0=float(a2.get("n0", 0.0)), shrink=float(a2.get("shrink", 0.05)))
        self.w2 = float(c2.get("weight", 0.0)) if len(self.nets2) else 0.0
        # T2-D161: per-user copies of the SECOND family (pu_nets2[k] = its seeds' copies for training user k), blended into the
        # second family's output with weight cfg["user_blend2"]["r"] toward the user the main family's fingerprint routes to
        self.pu_nets2 = torch.nn.ModuleList([torch.nn.ModuleList(v) for v in pu_nets2])
        self.stk_pu2 = [None] * len(self.pu_nets2)
        if len(self.pu_nets2):
            assert len(self.pu_nets2) == len(self.pu_nets) and len(self.nets2), "user_blend2 needs user_blend (the route) and cnn2"
            self.ub2_r = float(cfg["user_blend2"]["r"])
        # T2-D352 (W11): an optional THIRD CNN family (cfg["cnn3"]), its own causal aligner, output share cfg["cnn3"]["weight"]; its per-user
        # copies (cfg["user_blend3"]) are routed by the main family's fingerprint, as the second family's
        self.nets3 = torch.nn.ModuleList(nets3)
        c3 = cfg.get("cnn3") or {}
        if len(self.nets3):
            assert len(self.nets2), "cnn3 needs cnn2"
            a3 = c3["align"]
            self.aligner3 = StreamAligner(a3["mode"], scale=float(a3.get("scale", 1.0)),
                                          prior=np.asarray(a3["prior"], dtype=np.float64) if a3.get("prior") is not None else None,
                                          n0=float(a3.get("n0", 0.0)), shrink=float(a3.get("shrink", 0.05)))
        self.w3 = float(c3.get("weight", 0.0)) if len(self.nets3) else 0.0
        self.pu_nets3 = torch.nn.ModuleList([torch.nn.ModuleList(v) for v in pu_nets3])
        self.stk_pu3 = [None] * len(self.pu_nets3)
        if len(self.pu_nets3):
            assert len(self.pu_nets3) == len(self.pu_nets) and len(self.nets3), "user_blend3 needs user_blend (the route) and cnn3"
            self.ub3_r = float(cfg["user_blend3"]["r"])
        self.cfg = cfg
        names = list(meta.get("ch_names") or [])
        train = list(cfg["ch_names"])
        if names:
            lookup = {_norm(n): i for i, n in reversed(list(enumerate(names)))}
            missing = [c for c in train if _norm(c) not in lookup]
            if missing and len(names) == len(train):          # e.g. the kit's Simulated contract dataset
                print(f"[ni26] channel names differ from training ({len(missing)} unknown): positional mapping")
                self.idx = list(range(len(train)))
            elif missing:
                raise ValueError(f"evaluation montage lacks training channels {missing}")
            else:
                self.idx = [lookup[_norm(c)] for c in train]
        elif int(meta.get("n_chans") or -1) == len(train):
            self.idx = list(range(len(train)))
        else:
            raise ValueError("no channel names and a channel count different from training")
        if abs(float(meta.get("sfreq") or cfg["sfreq"]) - float(cfg["sfreq"])) > 1e-6:
            raise ValueError(f"evaluation sfreq {meta.get('sfreq')} != training {cfg['sfreq']}")
        a = cfg["align"]
        prior = np.asarray(a["prior"], dtype=np.float64) if a.get("prior") is not None else None
        self.aligner = StreamAligner(a["mode"], scale=float(a.get("scale", 1.0)), prior=prior,
                                     n0=float(a.get("n0", 0.0)), shrink=float(a.get("shrink", 0.05)))
        self.fm_nets = torch.nn.ModuleList(fm_nets)
        self.fm_aligners = [StreamAligner(m["align"]["mode"], scale=float(m["align"].get("scale", 1.0)),
                                          shrink=float(m["align"].get("shrink", 0.05))) for m in cfg.get("fm_members", [])]
        assert len(self.fm_aligners) == len(self.fm_nets)
        # T2-D482 (W14): each FM member's input reference (None = the window as delivered, W13's path)
        self.fm_ref = [m.get("ref") or ("car" if m.get("car") else None) for m in cfg.get("fm_members", [])]
        assert all(r in (None, "car", "lap") for r in self.fm_ref), self.fm_ref
        self.fm_lapnb = [np.asarray(m["lap_nb"], dtype=np.int64) if r == "lap" else None for m, r in zip(cfg.get("fm_members", []), self.fm_ref)]
        assert all(nb is None or nb.shape[0] == len(cfg["ch_names"]) for nb in self.fm_lapnb)
        self.fm_weight = float(cfg.get("fm_weight", 0.0)) if len(self.fm_nets) else 0.0
        # T2-D140: per-member output weights (e.g. ST-L 0.2 + MOR 0.1); their sum is the FM share. Without them every FM
        # member gets an equal part of fm_weight (the T2-D95..D136 packages).
        mw = [m.get("weight") for m in cfg.get("fm_members", [])]
        if len(self.fm_nets) and any(w is not None for w in mw):
            assert all(w is not None and w > 0 for w in mw), mw
            assert abs(sum(mw) - self.fm_weight) < 1e-9, (mw, self.fm_weight)
            self.fm_w = np.asarray(mw, np.float64) / sum(mw)
        else:
            self.fm_w = None                              # legacy: the plain fp32 mean (bit-identical to the D135 template)
        t = cfg.get("tta") or {}
        if t.get("kind") == "sinkhorn":                  # T2-D286: SK10 (tta = {"kind": "sinkhorn", "iters": 10, "n0": 3}); T2-D304: + "lam" (absent = 1.0)
            self.prior = SinkhornCorrector(t["iters"], t["n0"], cfg["n_classes"], float(t.get("lam", 1.0)), t.get("start"))
        else:
            self.prior = PriorCorrector(t["alpha"], t["n0"], cfg["n_classes"]) if t.get("kind") == "prior" else None
        self.fam_prior, self.fam_w = None, None         # T2-D476 (W13): per-family correctors; None = W12's single corrector
        if t.get("per_family"):
            assert t.get("kind") == "sinkhorn", "W13: per_family needs the sinkhorn decoder"
            fams = ["cnn"] + (["cnn2"] if len(self.nets2) else []) + (["cnn3"] if len(self.nets3) else []) + (["fm"] if len(self.fm_nets) else [])
            assert sorted(t["per_family"]) == sorted(fams), (sorted(t["per_family"]), fams)
            self.fam_w = {"cnn": 1.0 - self.fm_weight - self.w2 - self.w3, "cnn2": self.w2, "cnn3": self.w3, "fm": self.fm_weight}
            assert all(self.fam_w[f] > 0 for f in fams) and abs(sum(self.fam_w[f] for f in fams) - 1.0) < 1e-9, self.fam_w
            self.fam_prior = {f: SinkhornCorrector(t["iters"], t["n0"], cfg["n_classes"], float(t.get("lam", 1.0)), t["per_family"][f]) for f in fams}
        self.is_cfg = cfg.get("instream") or None       # T2-D510 / D511 (W15): in-stream adaptation; None = W14 (bitwise)
        if self.is_cfg is not None:
            assert self.fam_prior is not None, "W15: instream needs the per-family decode (W13)"
            fam_nets = {"cnn": self.nets, "cnn2": self.nets2, "cnn3": self.nets3}
            self.is_fams = sorted(self.is_cfg["families"])
            assert set(self.is_fams) <= {"cnn", "cnn2", "cnn3"} and all(len(fam_nets[f]) for f in self.is_fams), self.is_fams
            self.is_base = {f: fam_nets[f][int(self.is_cfg["families"][f].get("member", 0))] for f in self.is_fams}
            self.is_refits = [int(c) for c in self.is_cfg["refits"]]
            assert self.is_refits == sorted(self.is_refits) and self.is_refits[0] >= 2, self.is_refits
            self.is_r = float(self.is_cfg["r"])
            assert 0.0 < self.is_r <= 1.0 and self.is_cfg.get("labels", "conf") in ("conf", "confR"), self.is_cfg
            self.is_tcfg = (int(t["iters"]), int(t["n0"]), float(t.get("lam", 1.0)), {f: t["per_family"][f] for f in fams})
            self.reset_state()
        self.dc, self.dc_w, self.dc_prior = None, 0.0, None   # T2-D516 (W16): set by set_dcfam (load_model); None = W15 bitwise
        self.ic, self.ic_w, self.ic_prior = None, 0.0, None   # T2-D525 (W18): set by set_imcoh; None = W17 bitwise
        self.fm_heads = None                                  # T2-D524 (W18): set by set_fm_side; None = W17 bitwise
        self.side, self.side_r = None, 0.0               # T2-D517 (W17): set by set_side (load_model); None = W16 bitwise
        self.n_times = int(cfg["n_times"])
        # T2-D140: one CUDA graph for every GPU member of a window (B = 1 is kernel-launch bound), T3-D40 / D50 pattern; ON only
        # for packages with a MOR member (every earlier package keeps the eager path, bit-identical); NI26_EAGER=1 forces eager.
        # `gshared` is shared by the per-stream copies (__deepcopy__ keeps the reference): one capture per run.
        self.gpu_graph = (any(m.get("family") == "mor" for m in cfg.get("fm_members", [])) or bool(cfg.get("gpu_graph"))) \
            and os.environ.get("NI26_EAGER") != "1"
        self.gshared = {"g": {}}
        # row 75 (T2-D296 / D302): test-time averaging of the CNN families over K views of the window (the identity + the first K - 1
        # draws of the training vp bank: gain x remap, then the circular shift, as training's augment applies a draw); cfg["cnn_tta_vp"] =
        # {"k": K, "copies": bool, "file": the draws}; the per-user copies take the views only with "copies"; absent / k 1 = today's path
        tv = cfg.get("cnn_tta_vp") or {}
        self.tv_k, self.tv_copies, self.tv_bank = int(tv.get("k", 1)), bool(tv.get("copies", False)), None

    def set_dcfam(self, arrs):
        """T2-D516 (W16): attach the window-mean family (its tree arrays, z prior and start) and its own corrector."""
        c = self.cfg["dcfam"]
        assert self.fam_prior is not None, "W16: dcfam needs the per-family decode (W13)"
        self.dc_w = float(c["weight"])
        assert 0.0 < self.dc_w < 1.0, self.dc_w
        self.dc = DCFamily(arrs, [int(v) for v in c.get("seeds", [1, 2, 3])], float(c.get("n0", 10.0)), int(self.cfg["n_classes"]))
        t = self.cfg["tta"]
        self.dc_prior = SinkhornCorrector(t["iters"], t["n0"], self.cfg["n_classes"], float(t.get("lam", 1.0)), np.asarray(arrs["start"], np.float64))
        return self

    def set_imcoh(self, arrs):
        """T2-D525 (W18): attach the imcoh family (LR coef / intercept, z prior, start) and its own corrector."""
        c = self.cfg["imcoh"]
        assert self.fam_prior is not None, "W18: imcoh needs the per-family decode (W13)"
        self.ic_w = float(c["weight"])
        assert 0.0 < self.ic_w < 1.0, self.ic_w
        ch = list(self.cfg["ch_names"])
        assert all(n in ch for n in IC_NODES), [n for n in IC_NODES if n not in ch]
        self.ic = ICFamily(arrs, [ch.index(n) for n in IC_NODES], float(self.cfg["sfreq"]))
        t = self.cfg["tta"]
        self.ic_prior = SinkhornCorrector(t["iters"], t["n0"], self.cfg["n_classes"], float(t.get("lam", 1.0)), np.asarray(arrs["start"], np.float64))
        return self

    def set_fm_side(self, heads):
        """T2-D524 (W18): per FM member None or its side-head blob {kind, n0, mu, sd, weight (K, 82), bias (K)}; needs W17's side vector."""
        assert self.cfg.get("fm_side") and self.side is not None, "W18 fm_side needs cfg['fm_side'] and the side nets' z (cfg['side'])"
        assert len(heads) == len(self.fm_nets) and any(h is not None for h in heads), (len(heads), len(self.fm_nets))
        out = []
        for h in heads:
            if h is None:
                out.append(None)
                continue
            assert h["kind"] == "dcmean" and float(h["n0"]) == self.side_n0, (h["kind"], h["n0"], self.side_n0)
            assert np.allclose(np.asarray(h["mu"], np.float64), self.side_mu, rtol=0, atol=1e-9) and \
                np.allclose(np.asarray(h["sd"], np.float64), self.side_sd, rtol=0, atol=1e-9), "W18 fm_side: the head's z prior != side.pt's"
            W_, b_ = np.asarray(h["weight"], np.float64), np.asarray(h["bias"], np.float64)
            assert W_.shape == (int(self.cfg["n_classes"]), self.side_mu.shape[0]) and b_.shape == (W_.shape[0],), (W_.shape, b_.shape)
            out.append((W_, b_))
        self.fm_heads = out
        return self

    def _fm_head_apply(self, ps):
        """T2-D524 (W18): FM member probabilities (list of (K,) arrays) -> softmax(log p + W s + b) for the members with a head (float64)."""
        s_ = np.asarray(self._side_now, np.float64)
        out = []
        for p, h in zip(ps, self.fm_heads):
            p = np.asarray(p, np.float64)
            if h is not None:
                lg = np.log(np.maximum(p, 1e-30)) + h[0] @ s_ + h[1]
                e = np.exp(lg - lg.max())
                p = e / e.sum()
            out.append(p)
        return out

    def set_side(self, blob, dev):
        """T2-D517 (W17): attach the side nets (frozen; a plain list, not submodules, so the module tree and every W16 path stay as they were)."""
        c = self.cfg["side"]
        assert self.fam_prior is not None, "W17: side nets need the per-family decode (W13)"
        assert c.get("kind", "dcmean") == blob["kind"] == "dcmean", (c.get("kind"), blob["kind"])
        self.side_r = float(c["r"])
        assert 0.0 < self.side_r < 1.0, self.side_r
        fams = list(c.get("families", ["cnn2", "cnn3"]))
        have = {"cnn2": len(self.nets2) > 0, "cnn3": len(self.nets3) > 0}
        assert fams and all(f in have and have[f] for f in fams), (fams, have)
        self.side_mu = np.asarray(blob["mu"], np.float64)
        self.side_sd = np.asarray(blob["sd"], np.float64)
        assert self.side_mu.shape == self.side_sd.shape == (2 * len(self.cfg["ch_names"]),), self.side_mu.shape
        self.side_n0 = float(blob.get("n0", c.get("n0", 10.0)))
        K = int(self.cfg["n_classes"])
        self.side = {}
        for f in fams:
            fc = self.cfg[f]
            mem = []
            for e in blob[f]:
                net = load_net(build_net(dict(self.cfg, braindecode_class=fc["braindecode_class"], model_kwargs=fc.get("model_kwargs") or {})), e["net"])
                head = torch.nn.Linear(self.side_mu.shape[0], K)
                head.load_state_dict({k: v.float() for k, v in e["head"].items()}, strict=True)
                mem.append((net.to(dev).eval(), head.to(dev).eval()))
            assert mem, f
            self.side[f] = mem
        # one stacked forward per family (the members' own Stacked / vmap path; None = the loop) + the heads as one batched matmul
        C_, L_ = len(self.cfg["ch_names"]), int(self.cfg["n_times"])
        self.side_stk = {f: _stack_or_none([n for n, _ in m], C_, L_) for f, m in self.side.items()}
        self.side_hw = {f: (torch.stack([h.weight.detach() for _, h in m]), torch.stack([h.bias.detach() for _, h in m])) for f, m in self.side.items()}
        self.reset_side()
        return self

    def reset_side(self):
        if getattr(self, "side", None) is not None:
            self.sd_s1, self.sd_s2, self.sd_n = np.zeros_like(self.side_mu), np.zeros_like(self.side_mu), 0
            self._side_now = None

    def _side_z(self, x):
        """T2-D517 (W17): the delivered window (C, T) -> its 82-vector z (float32) = t2p_bench.side_z 'dcmean' (causal, prior n0 windows)."""
        m = np.asarray(x, np.float64).mean(-1)
        f = np.concatenate([m, m - m.mean()])
        f = np.where(np.isfinite(f), f, self.side_mu)
        self.sd_s1, self.sd_s2, self.sd_n = self.sd_s1 + f, self.sd_s2 + f * f, self.sd_n + 1
        nn = self.side_n0 + self.sd_n
        mm = (self.side_n0 * self.side_mu + self.sd_s1) / nn
        vv = (self.side_n0 * (self.side_sd ** 2 + self.side_mu ** 2) + self.sd_s2) / nn - mm ** 2
        return ((f - mm) / np.sqrt(np.maximum(vv, 1e-12))).astype(np.float32)

    def _side_probs(self, f, z, s, dev):
        """T2-D517 (W17): family f's side nets on its aligned input z (C, T) and the side vector s -> mean softmax(net(z) + head(s)) (float64)."""
        with torch.no_grad():
            zt = torch.from_numpy(np.asarray(z, np.float32))[None].to(dev)
            st = torch.from_numpy(np.asarray(s, np.float32))[None].to(dev)
            stk = self.side_stk.get(f)
            if stk is not None:                          # (n, 1, K) logits + (n, K) head logits
                Wh, bh = self.side_hw[f]
                lg = stk(zt) + (torch.einsum("nkd,d->nk", Wh, st[0]) + bh)[:, None, :]
                return F.softmax(lg.float(), dim=-1)[:, 0].mean(0).cpu().numpy().astype(np.float64)
            ps = [F.softmax((net(zt) + head(st)).float(), dim=1)[0] for net, head in self.side[f]]
            return torch.stack(ps).mean(0).cpu().numpy().astype(np.float64)

    def _side_blend(self, parts, zs, dev):
        """T2-D517 (W17): blend each side family into its part in place (after the in-stream blend, before the correctors)."""
        idx_ = {"cnn2": 1, "cnn3": 3}
        if self._side_now is None or not np.isfinite(self._side_now).all():
            return
        for f in self.side:
            if zs[f] is None or not np.isfinite(zs[f]).all():
                continue
            ps = self._side_probs(f, zs[f], self._side_now, dev)
            if np.isfinite(ps).all() and parts[idx_[f]] is not None:
                parts[idx_[f]] = (1.0 - self.side_r) * np.asarray(parts[idx_[f]], np.float64) + self.side_r * ps

    def set_tta_bank(self, bank, dev):
        """row 75: the K - 1 stored vp draws (load_model)."""
        assert self.tv_k > 1 and bank["mats"].shape[0] >= self.tv_k - 1, (self.tv_k, tuple(bank["mats"].shape))
        n = self.tv_k - 1
        self.tv_bank = (bank["mats"][:n].float().to(dev), bank["gains"][:n].float().to(dev), [int(s) for s in bank["shifts"][:n]])
        return self

    def _v(self, z, copies=False):
        """row 75: window z (1, C, T) -> its K views (K, C, T); z itself when the averaging is off (or for the copies without "copies")."""
        if self.tv_k <= 1 or (copies and not self.tv_copies):
            return z
        M, G, S = self.tv_bank
        return torch.cat([z] + [torch.roll(G[k][:, None] * (M[k] @ z[0]), shifts=S[k], dims=-1)[None] for k in range(self.tv_k - 1)])

    def reset_state(self):
        self.aligner.reset()
        self.ub_sum, self.ub_n, self.ub_t = None, 0, 0
        if len(self.nets2):
            self.aligner2.reset()
        if len(self.nets3):
            self.aligner3.reset()
        for al in self.fm_aligners:
            al.reset()
        if self.prior is not None:
            self.prior.reset()
        if self.fam_prior is not None:                  # T2-D476 (W13)
            for c in self.fam_prior.values():
                c.reset()
        if getattr(self, "dc", None) is not None:        # T2-D516 (W16)
            self.dc.reset()
            self.dc_prior.reset()
        if getattr(self, "ic", None) is not None:        # T2-D525 (W18)
            self.ic.reset()
            self.ic_prior.reset()
        self.reset_side()                                # T2-D517 (W17): no-op without side nets
        if getattr(self, "is_cfg", None) is not None:   # T2-D510 (W15): fresh per-stream adaptation state
            it, n0, lam, st = self.is_tcfg
            self.is_tprior = {f: SinkhornCorrector(it, n0, int(self.cfg["n_classes"]), lam, st[f]) for f in self.fam_w if f in st}
            self.is_hist_z = {f: [] for f in self.is_fams}       # the family's aligned input of every delivered window
            self.is_hist_ok = []                                 # the window may train a copy (finite input and parts)
            self.is_hist_post = []                               # 'conf': the causal teacher posterior of each window
            self.is_net = {f: None for f in self.is_fams}        # the current adapted copy per family (None = not yet)
            self.is_n, self.is_s = 0, 0.0                        # windows delivered, refit seconds spent in this stream

    def _fam_decode(self, parts):
        """T2-D476 (W13): each family's part (after its per-user copy blend) through its own corrector, then the weighted pool.
        A non-finite part enters its corrector as uniform (the corrector's B24 guard)."""
        P = {"cnn": parts[0], "cnn2": parts[1], "fm": parts[2], "cnn3": parts[3] if len(parts) > 3 else None}
        out = sum(self.fam_w[f] * c(P[f]) for f, c in self.fam_prior.items())
        if self.dc is not None:                         # T2-D516 (W16): the window-mean family through its own corrector
            out = (1.0 - self.dc_w) * out + self.dc_w * self.dc_prior(self._dc_now)
        if self.ic is not None:                         # T2-D525 (W18): the imcoh family through its own corrector, after the window means
            out = (1.0 - self.ic_w) * out + self.ic_w * self.ic_prior(self._ic_now)
        return out

    def __deepcopy__(self, memo):
        """Runtime (T3-D40 transfer): the kit deep-copies the loaded model for every stream. Members are frozen at
        inference, so a copy SHARES them (no per-stream copy of the ST-L weights) and gets fresh streaming state only:
        aligners, prior corrector, user-blend fingerprint mean. NI26_FULL_DEEPCOPY=1 restores a full copy (A/B checks)."""
        import copy
        import os
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        if os.environ.get("NI26_FULL_DEEPCOPY") == "1":
            for k, v in self.__dict__.items():
                new.__dict__[k] = {"g": {}} if k == "gshared" else copy.deepcopy(v, memo)   # a graph is not copyable
            return new
        new.__dict__.update(self.__dict__)
        for k in ("_parameters", "_buffers", "_modules"):
            new.__dict__[k] = self.__dict__[k].copy()
        for k in ("aligner", "aligner2", "aligner3", "fm_aligners", "prior", "fam_prior", "dc", "dc_prior", "ic", "ic_prior"):
            if k in self.__dict__:
                new.__dict__[k] = copy.deepcopy(self.__dict__[k], memo)
        new.reset_state()
        return new

    @staticmethod
    def _fam(nets, stk, z):
        """mean member probability of one family for window z (stacked forward when available); row 75: z may hold K views of the
        window, averaged too (one view: the same value as before)."""
        if stk is not None:
            return F.softmax(stk(z).float(), dim=-1).mean(0).mean(0).cpu().numpy()
        return torch.stack([F.softmax(n(z).float(), dim=1) for n in nets]).mean(0).mean(0).cpu().numpy()

    @staticmethod
    def _fam_t(nets, stk, z):
        """= _fam, kept on the device (the graph path)."""
        if stk is not None:
            return F.softmax(stk(z).float(), dim=-1).mean(0).mean(0)
        return torch.stack([F.softmax(n(z).float(), dim=1) for n in nets]).mean(0).mean(0)

    def _gpu_eager(self, zs, cache=True):
        """the GPU members for one window except the per-user copies: zs = [z (main aligner), z2 (second family) | dummy,
        zf per FM member] -> [main family, second family (if any), FM member 0..M-1] probability tensors."""
        o3 = 1 if len(self.nets3) else 0                 # T2-D352: zs = [z, z2, z3 (cnn3 only), zf ..., routed deltas ...]
        z, z2, zfs = zs[0], zs[1], zs[2 + o3:2 + o3 + len(self.fm_nets)]
        out = [self._fam_t(self.nets, self.stk[0], self._v(z))]
        if len(self.nets2):
            out.append(self._fam_t(self.nets2, self.stk[1], self._v(z2)))
        if len(self.nets3):
            out.append(self._fam_t(self.nets3, self.stk[2], self._v(zs[2])))
        for net, zf in zip(self.fm_nets, zfs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=zf.is_cuda, cache_enabled=cache):
                o = net(zf, zs[-1]) if isinstance(net, STRouted) else net(zf)     # T2-D188: the routed delta = the last input
                out.append(F.softmax(o.float(), dim=1)[0])
        return out

    def _capture(self, fn, zs, slot):
        """capture fn(zs) -> list of tensors in a CUDA graph; check it against eager on the capture window. slot = the
        gshared key ('graph' | ('pu', k)) whose error / first-window record this writes."""
        sh = self.gshared
        try:
            st = [z.clone() for z in zs]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn(st, cache=False)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs = fn(st, cache=False)
            for s_, z in zip(st, zs):
                s_.copy_(z)
            g.replay()
            ref = fn(zs)
            d = max(float((a - b).abs().max()) for a, b in zip(outs, ref))
            sh[("check", slot)], sh[("x_first", slot)] = d, [z.clone() for z in zs]
            if not d < 1e-3:
                sh[("error", slot)] = f"capture-window check {d:.1e}"
                return False
            return (g, st, outs)
        except Exception as e:  # noqa: BLE001  (any capture problem -> eager, never a crash)
            sh[("error", slot)] = repr(e)[:300]
            return False

    def _run(self, fn, zs, slot):
        """graph replay of fn(zs) (captured on first use, then checked once on the first DISTINCT window), else eager."""
        sh = self.gshared
        if sh.get("off"):
            return fn(zs)
        if slot not in sh["g"]:
            sh["g"][slot] = self._capture(fn, zs, slot)
            if slot == "main" or not sh["g"][slot]:
                print(f"[ni26] cuda graph {slot}: " + (f"used (capture-window check {sh[('check', slot)]:.1e})" if sh["g"][slot]
                      else f"fallback ({sh.get(('error', slot))}): eager for the rest of the run"), flush=True)
            if not sh["g"][slot]:
                sh["off"] = True
                return fn(zs)
        g, st, outs = sh["g"][slot]
        for s_, z in zip(st, zs):
            s_.copy_(z)
        g.replay()
        out = [o.clone() for o in outs]
        if not sh.get(("validated", slot)) and not all(torch.equal(z, x0) for z, x0 in zip(zs, sh[("x_first", slot)])):
            ref = fn(zs)                                  # the first DISTINCT window after this graph's capture window
            d = max(float((a - b).abs().max()) for a, b in zip(out, ref))
            sh[("validated", slot)] = True
            if not d < 1e-3:
                sh["off"] = True
                print(f"[ni26] cuda graph OFF ({slot} second-window check {d:.1e}): eager for the rest of the run", flush=True)
                return ref
            if slot == "main":
                print(f"[ni26] cuda graph second-window check {d:.1e}", flush=True)
        return out

    def _fm_x(self, j, xi):
        """T2-D482 (W14): FM member j's input: the window as delivered (W13), or re-referenced exactly as that member's training input was."""
        r = self.fm_ref[j]
        if r is None:
            return xi
        return car_windows(xi) if r == "car" else lap_windows(xi, self.fm_lapnb[j])

    def _predict_graph(self, xi, dev):
        """T2-D140: = the eager body of predict() for one window; the GPU members in CUDA graphs: ONE for the main family +
        second family + FM members, one per training user for that user's per-user copies (only the routed user's runs:
        the route comes from the CPU fingerprint, before any GPU work)."""
        a = [self.aligner(xi)] + ([self.aligner2(xi)] if len(self.nets2) else []) + ([self.aligner3(xi)] if len(self.nets3) else []) + \
            [al(self._fm_x(j, xi)) for j, al in enumerate(self.fm_aligners)]
        za = torch.from_numpy(np.stack(a).astype(np.float32, copy=False))[:, None].to(dev)   # ONE host -> device copy
        z = za[0]
        z2, zfs = (za[1], list(za[2:])) if len(self.nets2) else (torch.zeros(1, device=dev), list(za[1:]))
        z3 = []
        if len(self.nets3):                               # T2-D352: [z, z2, z3, zf ...]
            z3, zfs = [za[2]], list(za[3:])
        k = self._user(xi) if len(self.pu_nets) else None
        rd = [n.delta(k) for n in self.fm_nets if isinstance(n, STRouted)]      # T2-D188: [] unless a member is routed
        outs = self._run(self._gpu_eager, [z, z2] + z3 + zfs + rd, "main")
        pu = None
        if k is not None:
            fam = lambda zs, cache=True, k=k: [self._fam_t(self.pu_nets[k], self.stk_pu[k], self._v(zs[0], copies=True))]  # noqa: E731
            pu = self._run(fam, [z], ("pu", k))[0].cpu().numpy()
        pu2 = None
        if k is not None and len(self.pu_nets2):          # T2-D161: the second family's copies of the same routed user
            fam2 = lambda zs, cache=True, k=k: [self._fam_t(self.pu_nets2[k], self.stk_pu2[k], self._v(zs[0], copies=True))]  # noqa: E731
            pu2 = self._run(fam2, [z2], ("pu2", k))[0].cpu().numpy()
        pu3 = None
        if k is not None and len(self.pu_nets3):          # T2-D352: the third family's copies of the same routed user
            fam3 = lambda zs, cache=True, k=k: [self._fam_t(self.pu_nets3[k], self.stk_pu3[k], self._v(zs[0], copies=True))]  # noqa: E731
            pu3 = self._run(fam3, z3, ("pu3", k))[0].cpu().numpy()
        P = [q.cpu().numpy() for q in outs]
        p = P[0]
        if pu is not None:
            p = (1.0 - self.ub_r) * p.astype(np.float64) + self.ub_r * pu.astype(np.float64)
        j = 1
        parts = [np.asarray(p, np.float64), None, None]            # T2-D210 diagnostics (NI26_PARTS=1): the pre-mix parts
        if len(self.nets2):
            w2n = self.w2 / (1.0 - self.fm_weight - self.w3)    # T2-D352: = W9's w2 / (1 - fm) when there is no cnn3 (w3 0.0)
            p2 = P[j]
            if pu2 is not None:
                p2 = (1.0 - self.ub2_r) * p2.astype(np.float64) + self.ub2_r * pu2.astype(np.float64)
            parts[1] = np.asarray(p2, np.float64)
            p = (1.0 - w2n) * p.astype(np.float64) + w2n * p2.astype(np.float64)
            j += 1
        if len(self.nets3):                               # T2-D352: the CNN share 1 - fm as (1 - fm - w2 - w3) : w2 : w3
            w3n = self.w3 / (1.0 - self.fm_weight)
            p3 = P[j]
            if pu3 is not None:
                p3 = (1.0 - self.ub3_r) * p3.astype(np.float64) + self.ub3_r * pu3.astype(np.float64)
            parts.append(np.asarray(p3, np.float64))          # parts[3]
            p = (1.0 - w3n) * p.astype(np.float64) + w3n * p3.astype(np.float64)
            j += 1
        if len(self.fm_nets):
            pf = np.stack(P[j:] if self.fm_heads is None else self._fm_head_apply(P[j:])).astype(np.float64)   # T2-D524 (W18)
            pf = pf.mean(0) if self.fm_w is None else (pf * self.fm_w[:, None]).sum(0)
            parts[2] = pf
            p = (1.0 - self.fm_weight) * p.astype(np.float64) + self.fm_weight * pf
        if _PARTS:
            self.last_parts = parts
        self._parts_now = parts                          # T2-D476 (W13): the parts for the per-family correctors
        if self.is_cfg is not None:                      # T2-D510 (W15): the adapted copies blended into the parts, in place
            self._instream(parts, {"cnn": a[0], "cnn2": a[1] if len(self.nets2) else None, "cnn3": a[2] if len(self.nets3) else None}, dev)
        if self.side is not None:                        # T2-D517 (W17): the side nets blended into the parts, in place
            self._side_blend(parts, {"cnn2": a[1] if len(self.nets2) else None, "cnn3": a[2] if len(self.nets3) else None}, dev)
        return p

    def _instream(self, parts, zs, dev):
        """T2-D510 / D511 (W15): one window. parts = the unadapted per-family parts (modified in place: the adapted copies blended in); zs = {family:
        its aligned input (C, T) numpy}. Order: (1) at a refit point, re-fit each family's copy on the windows delivered BEFORE this one; (2) record
        this window (input, teacher posterior); (3) blend the current copies into this window's parts."""
        import time
        P = {"cnn": parts[0], "cnn2": parts[1], "fm": parts[2], "cnn3": parts[3] if len(parts) > 3 else None}
        idx_ = {"cnn": 0, "cnn2": 1, "fm": 2, "cnn3": 3}
        sh = self.gshared
        if self.is_n in self.is_refits and not sh.get("is_off"):
            t0 = time.time()
            self._instream_refit(dev)
            dt = time.time() - t0
            self.is_s += dt
            sh["is_s"] = sh.get("is_s", 0.0) + dt
            if sh["is_s"] > float(self.is_cfg.get("max_run_s", 1e18)):
                sh["is_off"] = True                       # the run's adaptation budget is spent: later refits never fire
                print(f"[ni26] instream: run budget {self.is_cfg.get('max_run_s')} s spent; no further refits", flush=True)
            if self.is_s > float(self.is_cfg.get("max_stream_s", 1e18)):
                self.is_refits = [c for c in self.is_refits if c <= self.is_n]   # this stream's later refits never fire
        ok = all(np.isfinite(zs[f]).all() for f in self.is_fams) and all(P[f] is None or np.isfinite(P[f]).all() for f in self.fam_w)
        for f in self.is_fams:
            self.is_hist_z[f].append(np.asarray(zs[f], np.float32))
        self.is_hist_ok.append(bool(ok))
        if self.is_cfg.get("labels", "conf") == "conf":   # the causal teacher: the unadapted parts through their own correctors, pooled
            self.is_hist_post.append(sum(self.fam_w[f] * c(P[f]) for f, c in self.is_tprior.items()))
        else:                                             # 'confR' keeps the unadapted parts in its correctors' rows only
            for f, c in self.is_tprior.items():
                q = np.asarray(P[f], dtype=np.float64)
                c.rows.append(q if (np.isfinite(q).all() and q.sum() > 0) else np.full(c.k, 1.0 / c.k))
        self.is_n += 1
        for f in self.is_fams:
            if self.is_net[f] is not None:
                with torch.no_grad():
                    z = torch.from_numpy(np.asarray(zs[f], np.float32))[None].to(dev)
                    pa = F.softmax(self.is_net[f](z).float(), dim=1)[0].cpu().numpy().astype(np.float64)
                if np.isfinite(pa).all():
                    parts[idx_[f]] = (1.0 - self.is_r) * np.asarray(P[f], np.float64) + self.is_r * pa

    def _instream_refit(self, dev):
        """T2-D510 / D511 (W15): the labels of the delivered windows, then one copy per adapted family from its SHIPPED member."""
        c = self.is_n
        if self.is_cfg.get("labels", "conf") == "conf":
            post = np.stack(self.is_hist_post)
        else:                                             # T2-D511: re-decoded over the whole delivered prefix, per family, pooled
            it, n0, lam, _ = self.is_tcfg
            post = sum(self.fam_w[f] * _prefix_decode(cr.rows, it, lam, cr.k, n0) for f, cr in self.is_tprior.items())
            post = post / post.sum(1, keepdims=True)
        assert len(post) == c, (len(post), c)
        ok = np.flatnonzero(np.asarray(self.is_hist_ok, bool))
        if len(ok) < 2:
            return
        ka, ya = _conf_half(post[ok])
        rows = ok[ka]
        det0, bench0 = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False   # a re-run gives the same copy (the audit's re-run)
        try:
            self._instream_fit(rows, ya, dev)
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det0, bench0

    def _instream_fit(self, rows, ya, dev):
        """T2-D513 (W15): one copy per adapted family from its SHIPPED member, on the delivered windows `rows` with labels `ya`."""
        k_ = self.is_refits.index(self.is_n) if self.is_n in self.is_refits else 0
        with torch.inference_mode(False), torch.enable_grad():
            y = torch.as_tensor(np.asarray(ya, np.int64), device=dev)
            for f in self.is_fams:
                fc = self.is_cfg["families"][f]
                X = torch.as_tensor(np.stack([self.is_hist_z[f][i] for i in rows]), dtype=torch.float32, device=dev)
                self.is_net[f] = _w15_finetune(self.is_base[f], X, y, int(fc.get("epochs", 40)), float(fc.get("lr", 1e-3)),
                                               float(self.is_cfg.get("wd", 0.01)), int(self.is_cfg.get("bs", 32)),
                                               float(self.is_cfg.get("label_smoothing", 0.1)), int(self.is_cfg.get("seed", 0)) + 100000 * (k_ + 1))

    def stack_members(self):
        """build the stacked forwards (call after the model is on its device); each falls back to the loop on mismatch."""
        C, L = len(self.cfg["ch_names"]), int(self.cfg["n_times"])
        self.stk = [_stack_or_none(self.nets, C, L), _stack_or_none(self.nets2, C, L) if len(self.nets2) else None] + \
            ([_stack_or_none(self.nets3, C, L)] if len(self.nets3) else [])
        self.stk_pu = [_stack_or_none(v, C, L) for v in self.pu_nets]
        self.stk_pu2 = [_stack_or_none(v, C, L) for v in self.pu_nets2]
        self.stk_pu3 = [_stack_or_none(v, C, L) for v in self.pu_nets3]
        return self

    def _user(self, xi):
        """T2-D124c: update the session's running-mean projected fingerprint with window xi (B24: a window with non-finite
        samples, zero power or a non-finite fingerprint never enters it); -> the nearest user's index, or None before
        the first usable window."""
        t, self.ub_t = self.ub_t, self.ub_t + 1
        if t % self.ub_every:                              # runtime: the fingerprint is updated every `every` windows
            return None if self.ub_n == 0 else int(np.argmin(np.linalg.norm(self.ub_sum / self.ub_n - self.ub_mu, axis=1)))
        x = np.asarray(xi, dtype=np.float64)
        ok = bool(np.isfinite(x).all())
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -StreamAligner.CLIP, StreamAligner.CLIP)
        if ok and (x - x.mean(-1, keepdims=True)).any():
            with np.errstate(all="ignore"):
                f = fingerprint(x[None], float(self.cfg["sfreq"]), list(self.cfg["ch_names"]))[0]
                zf = ((f - self.ub_mean) / self.ub_scale) @ self.ub_W
            if np.isfinite(zf).all():
                self.ub_sum = zf if self.ub_sum is None else self.ub_sum + zf
                self.ub_n += 1
        if self.ub_n == 0:
            return None
        return int(np.argmin(np.linalg.norm(self.ub_sum / self.ub_n - self.ub_mu, axis=1)))

    @torch.inference_mode()
    def predict(self, X):
        if self.training:                                # T2-D140: eval() walks every module (~2k in a MOR package): once
            self.eval()
        X = torch.as_tensor(X)
        dev = next(self.parameters()).device
        x = X.detach().float().cpu().numpy()[:, self.idx, :]
        if x.shape[-1] != self.n_times:
            raise ValueError(f"window has {x.shape[-1]} samples, the model was trained on {self.n_times}")
        out = []
        for xi in x:                                     # B = 1 on Codabench; a batch is treated as consecutive windows
            if self.dc is not None:                      # T2-D516 (W16): the window-mean family sees the DELIVERED window
                self._dc_now = self.dc(xi)
            if self.ic is not None:                      # T2-D525 (W18): the imcoh family sees the DELIVERED window
                self._ic_now = self.ic(xi)
            if self.side is not None:                    # T2-D517 (W17): the side vector from the DELIVERED window
                self._side_now = self._side_z(xi)
            if self.gpu_graph and dev.type == "cuda":    # T2-D140: the GPU members in one CUDA graph (MOR packages)
                p = self._predict_graph(xi, dev)
                if not np.isfinite(p).all():
                    p = np.full(len(p), 1.0 / len(p))
                if self.fam_prior is not None:           # T2-D476 (W13)
                    p = self._fam_decode(self._parts_now)
                elif self.prior is not None:
                    p = self.prior(p)
                self.last_p = p
                out.append(int(np.argmax(p)))
                continue
            a0, a2, a3 = self.aligner(xi), None, None
            z = torch.from_numpy(a0)[None].to(dev)
            p = self._fam(self.nets, self.stk[0], self._v(z))
            k = None
            if len(self.pu_nets):
                k = self._user(xi)
                if k is not None:
                    pu = self._fam(self.pu_nets[k], self.stk_pu[k], self._v(z, copies=True))
                    p = (1.0 - self.ub_r) * p.astype(np.float64) + self.ub_r * pu.astype(np.float64)
            parts = [np.asarray(p, np.float64), None, None]        # T2-D210 diagnostics (NI26_PARTS=1)
            if len(self.nets2):                          # CNN share 1 - fm_weight split into (1 - fm - w2) : w2
                a2 = self.aligner2(xi)
                z2 = torch.from_numpy(a2)[None].to(dev)
                p2 = self._fam(self.nets2, self.stk[1], self._v(z2))
                if k is not None and len(self.pu_nets2):     # T2-D161: the same routed user's second-family copies
                    pu2 = self._fam(self.pu_nets2[k], self.stk_pu2[k], self._v(z2, copies=True))
                    p2 = (1.0 - self.ub2_r) * p2.astype(np.float64) + self.ub2_r * pu2.astype(np.float64)
                w2n = self.w2 / (1.0 - self.fm_weight - self.w3)    # T2-D352: = W9's when there is no cnn3
                parts[1] = np.asarray(p2, np.float64)
                p = (1.0 - w2n) * p.astype(np.float64) + w2n * p2.astype(np.float64)
            if len(self.nets3):                          # T2-D352: the third family, then its share w3 / (1 - fm)
                a3 = self.aligner3(xi)
                z3 = torch.from_numpy(a3)[None].to(dev)
                p3 = self._fam(self.nets3, self.stk[2], self._v(z3))
                if k is not None and len(self.pu_nets3):
                    pu3 = self._fam(self.pu_nets3[k], self.stk_pu3[k], self._v(z3, copies=True))
                    p3 = (1.0 - self.ub3_r) * p3.astype(np.float64) + self.ub3_r * pu3.astype(np.float64)
                w3n = self.w3 / (1.0 - self.fm_weight)
                parts.append(np.asarray(p3, np.float64))      # parts[3]
                p = (1.0 - w3n) * p.astype(np.float64) + w3n * p3.astype(np.float64)
            if len(self.fm_nets):
                pf = []
                for j, (net, al) in enumerate(zip(self.fm_nets, self.fm_aligners)):
                    zf = torch.from_numpy(al(self._fm_x(j, xi)))[None].to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=zf.is_cuda):
                        o = net(zf, net.delta(k)) if isinstance(net, STRouted) else net(zf)
                        pf.append(F.softmax(o.float(), dim=1)[0])
                if self.fm_heads is not None:            # T2-D524 (W18): the side heads on the member probabilities
                    ph = np.stack(self._fm_head_apply([q.float().cpu().numpy() for q in pf]))
                    pf = ph.mean(0) if self.fm_w is None else (ph * self.fm_w[:, None]).sum(0)
                elif self.fm_w is None:
                    pf = torch.stack(pf).mean(0).cpu().numpy().astype(np.float64)
                else:
                    pf = (torch.stack(pf).double().cpu().numpy() * self.fm_w[:, None]).sum(0)
                parts[2] = np.asarray(pf, np.float64)
                p = (1.0 - self.fm_weight) * p.astype(np.float64) + self.fm_weight * pf
            if _PARTS:
                self.last_parts = parts
            if not np.isfinite(p).all():                         # B24: never argmax a NaN (it would read as class 0)
                p = np.full(len(p), 1.0 / len(p))
            if self.is_cfg is not None:                          # T2-D510 (W15): the adapted copies blended into the parts, in place
                self._instream(parts, {"cnn": a0, "cnn2": a2, "cnn3": a3}, dev)
            if self.side is not None:                            # T2-D517 (W17): the side nets blended into the parts, in place
                self._side_blend(parts, {"cnn2": a2, "cnn3": a3}, dev)
            if self.fam_prior is not None:                       # T2-D476 (W13): per-family correctors, then the pool
                p = self._fam_decode(parts)
            elif self.prior is not None:
                p = self.prior(p)
            self.last_p = p                                       # diagnostics (t2p_repack.py robustness cases)
            out.append(int(np.argmax(p)))
        return torch.as_tensor(out, dtype=torch.long, device=X.device)


class Solver(CompetSolver):
    name = "ni26_t2p_stream"

    def load_model(self, meta):
        torch.backends.cudnn.allow_tf32 = False            # exact fp32 inference (tiny models: no cost)
        torch.backends.cuda.matmul.allow_tf32 = False
        d = meta["submission_dir"]
        cfg = json.loads((d / "config.json").read_text())
        vers = versions()                                  # review T2 P1a: the worker image pins neither torch nor braindecode
        trained = cfg.get("versions") or {}
        print(f"[ni26] worker {vers}; trained with {trained or 'unrecorded'}", flush=True)
        for k in ("torch", "braindecode"):
            if trained.get(k) and vers.get(k) and trained[k] != vers[k]:
                print(f"[ni26] NOTE {k} {vers[k]} here != {trained[k]} at training", flush=True)
        if int(meta.get("n_classes") or cfg["n_classes"]) != int(cfg["n_classes"]):
            raise ValueError(f"model has {cfg['n_classes']} classes, task wants {meta.get('n_classes')}")
        states = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
        nets = []
        for sd in states:
            nets.append(load_net(build_net(cfg), sd).eval())
        pu_nets = []
        if cfg.get("user_blend"):                          # T2-D124c: per user, the main family's per-seed copies
            for per_user in torch.load(d / cfg["user_blend"]["file"], map_location="cpu", weights_only=True):
                nets_u = []
                for sd in per_user:
                    nets_u.append(load_net(build_net(cfg), sd).eval())
                pu_nets.append(nets_u)
        nets2 = []
        if cfg.get("cnn2"):
            for sd in torch.load(d / cfg["cnn2"]["file"], map_location="cpu", weights_only=True):
                net = build_net(dict(cfg, braindecode_class=cfg["cnn2"]["braindecode_class"],
                                     model_kwargs=cfg["cnn2"].get("model_kwargs") or {}))
                nets2.append(load_net(net, sd).eval())
        pu_nets2 = []
        if cfg.get("user_blend2"):                         # T2-D161: per user, the second family's per-seed copies
            for per_user in torch.load(d / cfg["user_blend2"]["file"], map_location="cpu", weights_only=True):
                pu_nets2.append([load_net(build_net(dict(cfg, braindecode_class=cfg["cnn2"]["braindecode_class"],
                                                         model_kwargs=cfg["cnn2"].get("model_kwargs") or {})), sd).eval()
                                 for sd in per_user])
        nets3, pu_nets3 = [], []                           # T2-D352 (W11): the third family + its per-user copies
        if cfg.get("cnn3"):
            b3 = lambda: build_net(dict(cfg, braindecode_class=cfg["cnn3"]["braindecode_class"], model_kwargs=cfg["cnn3"].get("model_kwargs") or {}))  # noqa: E731
            nets3 = [load_net(b3(), sd).eval() for sd in torch.load(d / cfg["cnn3"]["file"], map_location="cpu", weights_only=True)]
            if cfg.get("user_blend3"):
                pu_nets3 = [[load_net(b3(), sd).eval() for sd in per_user]
                            for per_user in torch.load(d / cfg["user_blend3"]["file"], map_location="cpu", weights_only=True)]
        fm_nets, fm_heads = [], []
        for m in cfg.get("fm_members", []):
            fm_heads.append(torch.load(d / m["side"]["file"], map_location="cpu", weights_only=True) if m.get("side") else None)   # T2-D524 (W18)
            assert m["family"] in ("st", "mor"), m["family"]
            if m["family"] == "mor":                          # T2-D140: our MOR, role-LoRA merged (t2p_mor_pkg.merge)
                net = MORMemberT2(m, int(cfg["n_classes"]))
                sd = {k: v.float() if v.is_floating_point() else v
                      for k, v in torch.load(d / m["file"], map_location="cpu", weights_only=True).items()}
                net.load_state_dict(sd, strict=True)
                fm_nets.append(net.eval())
                continue
            net = STClassifier(m["size"], m["chan_idx"], m["n_samples"], m["calib"], m["layer"], int(cfg["n_classes"]))
            sd = {k: v.float() for k, v in torch.load(d / m["file"], map_location="cpu", weights_only=True).items()}
            own = net.state_dict()
            for k in [k for k in sd if k.endswith("enc_temporal_emd.pe")]:   # the fixed sinusoidal table: the exact fp32
                assert sd[k].shape == own[k].shape and float((sd[k] - own[k]).abs().max()) < 1e-2, k   # one the member
                sd[k] = own[k].clone()                                        # trained with, not its fp16-rounded copy
            net.load_state_dict(sd, strict=True)
            if m.get("st_lora"):                              # T2-D188 / D189: per-user deltas inside the one forward
                assert cfg.get("user_blend"), "st_lora needs the user_blend route"
                L = m["st_lora"]
                bank = torch.load(d / L["file"], map_location="cpu", weights_only=True)
                assert bank.shape[0] == len(cfg["user_blend"]["users"]), (tuple(bank.shape), len(cfg["user_blend"]["users"]))
                net = STRouted(net, bank, [tuple(e) for e in L["layout"]], int(L["r"]), float(L["scale"]))
            fm_nets.append(net.eval())
        model = StreamModel(nets, cfg, meta, fm_nets, nets2, pu_nets, pu_nets2, nets3, pu_nets3).to(meta["device"]).eval().stack_members()
        if cfg.get("dcfam"):                              # T2-D516 (W16): the window-mean family's tree arrays
            z = np.load(d / cfg["dcfam"]["file"])
            model.set_dcfam({k: z[k] for k in z.files})
        if cfg.get("side"):                               # T2-D517 (W17): the window-mean side nets
            model.set_side(torch.load(d / cfg["side"]["file"], map_location="cpu", weights_only=True), meta["device"])
        if cfg.get("imcoh"):                              # T2-D525 (W18): the imcoh family's LR arrays
            z = np.load(d / cfg["imcoh"]["file"])
            model.set_imcoh({k: z[k] for k in z.files})
        assert bool(cfg.get("fm_side")) == any(h is not None for h in fm_heads), \
            "W18: cfg['fm_side'] and the members' side heads must agree (no head is dropped or applied silently)"
        if cfg.get("fm_side"):                            # T2-D524 (W18): the FM members' window-mean side heads
            model.set_fm_side(fm_heads)
        tv = cfg.get("cnn_tta_vp") or {}
        if int(tv.get("k", 1)) > 1:                       # row 75: the stored vp draws of the CNN test-time averaging
            model.set_tta_bank(torch.load(d / tv["file"], map_location="cpu", weights_only=True), meta["device"])
        return model
