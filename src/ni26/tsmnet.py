"""T2-D491 (seat G10 item 5): a deep SPD member with DOMAIN-SPECIFIC SPD re-centring (the TSMNet / SPDDSMBN form,
Kobler et al., NeurIPS 2022), re-implemented in plain torch for the PROTEUS contract (41 ch x 480 samples at 120 Hz).

    x (B, C, T) -> temporal conv (F_t filters, k_t taps) -> spatial conv (D_s virtual channels) -> covariance (+ shrinkage)
      -> BiMap W (D_b x D_s, orthonormal rows) -> ReEig (eigenvalues >= eps) -> S (B, D_b, D_b)
      -> re-centring per DOMAIN d: S_c = M_d^{-1/2} S M_d^{-1/2}, M_d = the log-Euclidean mean of the domain's S (affine congruence)
      -> L = logm(S_c) -> dispersion normalisation L * gamma / sqrt(disp_d), disp_d = mean ||L||_F^2 of the domain
      -> upper triangle (off-diagonal x sqrt 2) -> linear.

Training: domain = one training stream (user x session); every batch is a few same-domain groups, each group re-centred by its
own batch mean (detached). Test: CAUSAL per stream (one participant x session in its delivered order): window t is re-centred by
the log-Euclidean mean of windows 1..t blended with the training prior at pseudo-count n0, and its dispersion likewise; nothing
crosses a stream. Eigen-functions use the Daleckii-Krein divided-difference backward (stable for close eigenvalues), in float64.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fn(lam, kind, eps):
    """eps: a float or a (..., 1) tensor (a per-matrix floor, so no matrix's output depends on another matrix of the batch)."""
    if kind == "log":
        lc = torch.clamp(lam, min=1e-12)
        return torch.log(lc), 1.0 / lc
    if kind == "reeig":
        e_ = eps if torch.is_tensor(eps) else torch.full_like(lam[..., :1], eps)
        return torch.maximum(lam, e_), (lam > e_).to(lam.dtype)
    if kind == "exp":
        e = torch.exp(lam)
        return e, e
    raise ValueError(kind)


class SymFn(torch.autograd.Function):
    """f(S) = U f(Lambda) U^T for a batch of symmetric matrices; backward by the Daleckii-Krein formula."""

    @staticmethod
    def forward(ctx, S, kind, eps):
        lam, U = torch.linalg.eigh(S)
        f, df = _fn(lam, kind, eps)
        ctx.save_for_backward(lam, U, f, df)
        return (U * f.unsqueeze(-2)) @ U.transpose(-1, -2)

    @staticmethod
    def backward(ctx, G):
        lam, U, f, df = ctx.saved_tensors
        G = 0.5 * (G + G.transpose(-1, -2))
        dl = lam.unsqueeze(-1) - lam.unsqueeze(-2)
        dfv = f.unsqueeze(-1) - f.unsqueeze(-2)
        scale = lam.abs().amax(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-12)
        close = dl.abs() <= 1e-7 * scale
        K = torch.where(close, 0.5 * (df.unsqueeze(-1) + df.unsqueeze(-2)), dfv / torch.where(close, torch.ones_like(dl), dl))
        X = U.transpose(-1, -2) @ G @ U
        return U @ (K * X) @ U.transpose(-1, -2), None, None


def symfn(S, kind, eps=1e-6):
    return SymFn.apply(S, kind, eps)


def triu_vec(L):
    D = L.shape[-1]
    iu = torch.triu_indices(D, D, device=L.device)
    w = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).to(L.dtype)
    return L[..., iu[0], iu[1]] * w


class TSMNet(nn.Module):
    def __init__(self, n_chans, n_classes=3, F_t=8, k_t=61, D_s=32, D_b=16, shrink=0.01, eps=1e-4):
        super().__init__()
        self.D_s, self.D_b, self.shrink, self.eps = D_s, D_b, shrink, eps
        self.temp = nn.Conv2d(1, F_t, (1, k_t), padding=(0, k_t // 2), bias=False)
        self.spat = nn.Conv2d(F_t, D_s, (n_chans, 1), bias=False)
        self.bimap = nn.utils.parametrizations.orthogonal(nn.Linear(D_s, D_b, bias=False))
        self.gamma = nn.Parameter(torch.ones(()))
        self.head = nn.Linear(D_b * (D_b + 1) // 2, n_classes)
        self.register_buffer("prior_log", torch.zeros(D_b, D_b, dtype=torch.float64))   # training log-Euclidean mean of S
        self.register_buffer("prior_disp", torch.ones((), dtype=torch.float64))         # training mean domain dispersion

    def spd(self, x):
        """(B, C, T) -> ReEig'd BiMap'd covariance (B, D_b, D_b), float64."""
        h = self.spat(self.temp(x.unsqueeze(1))).squeeze(2)                             # (B, D_s, T)
        h = (h - h.mean(-1, keepdim=True)).double()
        S = h @ h.transpose(1, 2) / (h.shape[-1] - 1)
        tr = S.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
        I = torch.eye(self.D_s, dtype=S.dtype, device=S.device)
        S = (1 - self.shrink) * S + self.shrink * tr * I
        W = self.bimap.weight.double()                                                   # (D_b, D_s), orthonormal rows
        S = W @ S @ W.t()
        floor = self.eps * S.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).detach().clamp_min(1e-12)   # per window (causal-safe)
        return symfn(S, "reeig", floor)

    def head_from(self, Lc, disp):
        f = triu_vec(Lc * (self.gamma.double() / disp.clamp_min(1e-8).sqrt())[:, None, None])
        return self.head(f.to(self.head.weight.dtype))

    def forward_groups(self, x, grp):
        """training forward: grp (B,) group ids; each group re-centred by its own batch log-Euclidean mean (detached)."""
        S = self.spd(x)
        L = symfn(S, "log")
        Lc = torch.empty_like(S)
        disp = torch.empty(len(S), dtype=S.dtype, device=S.device)
        for g in torch.unique(grp):
            m = grp == g
            with torch.no_grad():
                Wm = symfn(-0.5 * L[m].mean(0, keepdim=True), "exp")                     # M^{-1/2} = expm(-1/2 mean log S)
            Lg = symfn(Wm @ S[m] @ Wm, "log")
            Lc[m] = Lg
            disp[m] = (Lg.detach() ** 2).sum((-1, -2)).mean()
        return self.head_from(Lc, disp)

    @torch.no_grad()
    def set_prior(self, X, grp, bs=256):
        """the training prior: log-Euclidean mean of S over all training windows, and the mean over training domains of their
        dispersion after re-centring by their own full mean."""
        Ls = torch.cat([symfn(self.spd(X[s:s + bs]), "log") for s in range(0, len(X), bs)])
        self.prior_log.copy_(Ls.mean(0))
        ds = []
        for g in torch.unique(grp):
            m = grp == g
            Wm = symfn(-0.5 * Ls[m].mean(0, keepdim=True), "exp")
            S = symfn(Ls[m], "exp")
            ds.append(float((symfn(Wm @ S @ Wm, "log") ** 2).sum((-1, -2)).mean()))
        self.prior_disp.fill_(sum(ds) / len(ds))

    @torch.no_grad()
    def stream_logits(self, Xs, n0=5.0, bs=256):
        """CAUSAL decode of ONE stream (windows in delivered order): window t uses only windows 1..t (+ the prior at pseudo-count n0)."""
        S = torch.cat([self.spd(Xs[s:s + bs]) for s in range(0, len(Xs), bs)])
        L = symfn(S, "log")
        n = torch.arange(1, len(S) + 1, dtype=S.dtype, device=S.device)[:, None, None]
        Lbar = (n0 * self.prior_log + torch.cumsum(L, 0)) / (n0 + n)
        Wt = symfn(-0.5 * Lbar, "exp")
        Lc = symfn(Wt @ S @ Wt, "log")
        d = (Lc ** 2).sum((-1, -2))
        disp = (n0 * self.prior_disp + torch.cumsum(d, 0)) / (n0 + n[:, 0, 0])
        return self.head_from(Lc, disp)


class StreamState:
    """batch-1 causal state for one stream (what a template's predict(X) would carry; reset per session)."""

    def __init__(self, net, n0=5.0):
        self.net, self.n0, self.n, self.Lsum, self.dsum = net, n0, 0, None, 0.0

    @torch.no_grad()
    def step(self, x):
        S = self.net.spd(x)
        L = symfn(S, "log")[0]
        self.Lsum = L if self.Lsum is None else self.Lsum + L
        self.n += 1
        Lbar = (self.n0 * self.net.prior_log + self.Lsum) / (self.n0 + self.n)
        W = symfn(-0.5 * Lbar[None], "exp")
        Lc = symfn(W @ S @ W, "log")
        self.dsum += float((Lc ** 2).sum())
        disp = (self.n0 * self.net.prior_disp + self.dsum) / (self.n0 + self.n)
        return F.softmax(self.net.head_from(Lc, disp.reshape(1)), 1)
