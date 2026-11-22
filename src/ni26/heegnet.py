"""T2-D318: HEEGNet (Li et al., ICLR 2026, arXiv 2601.03322) re-implemented for the PROTEUS bench, plus its Lorentz MLR head.

Re-derived from the paper and the reference code's MATH (github.com/fightlesliefigt/HEEGNet, CC-BY-NC-4.0: read, not vendored).
Lorentz model of curvature -1: x0^2 - |xs|^2 = 1, x0 > 0, origin o = (1, 0, ..., 0); <u, v>_L = -u0 v0 + us . vs.

  HEEGNet   EEGNet block1 -> depthwise temporal conv -> per-position unit norm -> lift -> Lorentz 1x1 conv -> DSMDBN -> Lorentz ELU
            -> Lorentz avg-pool -> Lorentz flatten -> Lorentz MLR   (the reference nets/model.py; block2 only sizes the head there)
  DSMDBN    domain-specific batch norm: centre at the domain's Frechet mean (gyro-translation of the mean to o), scale by
            1 / sigma (gyro scalar product), sigma^2 = the Frechet variance; no learned shift; training: per-domain batch
            statistics through the AdaMom schedule; predict(): the statistics of the stream's windows 1..t (causal)
  EEGNetCtl the reference block1 + block2 + a linear head (the control, same training)
  LorentzHeadNet  T2-D318 H-b: a CNN's feature vector f lifted to (sqrt(1 + |f|^2), f) -> Lorentz MLR (replaces the linear head)
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------- geometry


def ldot(u, v, keepdim=True):
    p = u[..., 1:] * v[..., 1:]
    out = p.sum(-1, keepdim=True) - u[..., :1] * v[..., :1]
    return out if keepdim else out.squeeze(-1)


def add_time(xs):
    return torch.cat([torch.sqrt(1.0 + (xs * xs).sum(-1, keepdim=True)), xs], -1)


def _norm(xs, eps):
    return torch.sqrt((xs * xs).sum(-1, keepdim=True).clamp_min(eps))


def dist(x, y):
    return torch.acosh((-ldot(x, y, keepdim=False)).clamp_min(1.0 + 1e-15))


def log0(x, eps=1e-24):
    """the space part of log_o(x) (the time part is 0)."""
    xs = x[..., 1:]
    n = _norm(xs, eps)
    return xs * (torch.asinh(n) / n)


def exp0(vs, eps=1e-24):
    """exp_o((0, vs))."""
    n = _norm(vs, eps)
    return torch.cat([torch.cosh(n), torch.sinh(n) * vs / n], -1)


def expmap(x, u, eps=1e-24):
    """exp_x(u), u tangent at x (full ambient vector)."""
    n = torch.sqrt(ldot(u, u).clamp_min(eps))
    return torch.cosh(n) * x + torch.sinh(n) * u / n


def logmap(x, y):
    """log_x(y) (full ambient vector, tangent at x)."""
    c = (-ldot(x, y)).clamp_min(1.0 + 1e-15)               # cosh d
    d = torch.acosh(c)
    u = y - c * x
    s = torch.sqrt((c * c - 1.0).clamp_min(1e-30))          # sinh d
    return torch.where(d > 1e-10, d / s, torch.ones_like(d)) * u


def transp0(a, vs):
    """parallel transport of (0, vs) from o to a."""
    v = F.pad(vs, (1, 0))
    o = torch.zeros_like(a)
    o[..., 0] = 1.0
    coef = (a[..., 1:] * vs).sum(-1, keepdim=True) / (1.0 + a[..., :1])
    return v + coef * (o + a)


def gyroinv(a):
    return torch.cat([a[..., :1], -a[..., 1:]], -1)


def gyroadd(a, y):
    """a (+) y = exp_a(PT_{o->a}(log_o y)) (the reference Hyperboloid._gyroadd_via_riem)."""
    return expmap(a.expand_as(y), transp0(a.expand_as(y), log0(y)))


def gyroscale(x, r):
    return exp0(r * log0(x))


def centroid(x):
    """the Lorentzian centroid of points x (N, D) -> (D,)."""
    m = x.mean(0)
    return m / torch.sqrt((-ldot(m, m)).abs().clamp_min(1e-30))


def frechet_mean(x, iters=5):
    """the Frechet mean of x (N, D) -> (D,): Lou et al. 2020's hyperboloid fixed point (the reference's FrechetMeanHyperboloid,
    'Alg. 3'), y <- normalise(sum_i w_i x_i), w_i = 2 acosh(c_i) / sqrt(c_i^2 - 1), c_i = -<x_i, y>, from the centroid.
    Unrolled (differentiable). A plain step-1 Karcher flow diverges on widely spread points (tests/test_heegnet.py)."""
    y = centroid(x)
    for _ in range(iters):
        c = (-ldot(x, y.expand_as(x))).clamp_min(1.0 + 1e-15)
        w = torch.where(c > 1.0 + 1e-12, 2.0 * torch.acosh(c) / torch.sqrt((c * c - 1.0).clamp_min(1e-30)), torch.full_like(c, 2.0))
        u = (w * x).sum(0)
        y = u / torch.sqrt((-ldot(u, u)).clamp_min(1e-30))
        y = add_time(y[1:])                                 # back onto the sheet (round-off)
    return y


def frechet_var(x, mu):
    return (dist(x, mu.expand_as(x)) ** 2).mean()


def geodesic(x, y, t):
    return add_time(expmap(x, t * logmap(x, y))[..., 1:])


# ---------------------------------------------------------------------------------------------------------------- layers


class LorentzLinear(nn.Module):
    """the reference LorentzFullyConnected without its internal normalisation: Linear on the full vector, then the time is
    recomputed from the space part (a 1x1 Lorentz conv = this per position)."""

    def __init__(self, d_in, d_out, bias=False, init_std=0.02):
        super().__init__()
        self.weight = nn.Linear(d_in, d_out, bias=bias)
        nn.init.uniform_(self.weight.weight, -init_std, init_std)
        if bias:
            nn.init.zeros_(self.weight.bias)

    def forward(self, x):
        return add_time(self.weight(x)[..., 1:])


class LorentzMLR(nn.Module):
    """the reference LorentzMLR (Bdeir et al. 2024), curvature -1: logit_c = sign(alpha) beta |asinh(alpha / beta)|."""

    def __init__(self, d, n_classes):
        super().__init__()
        stdv = 1.0 / math.sqrt(d - 1)
        self.a = nn.Parameter(torch.empty(n_classes).uniform_(-stdv, stdv))
        self.z = nn.Parameter(torch.empty(n_classes, d - 1).uniform_(-stdv, stdv))

    def forward(self, x):
        nz = torch.norm(self.z, dim=-1)
        w_t = torch.sinh(self.a) * nz
        w_s = torch.cosh(self.a)[:, None] * self.z
        beta = torch.sqrt((-w_t ** 2 + (w_s ** 2).sum(-1)).clamp_min(1e-30))
        alpha = -w_t * x[..., :1] + torch.cosh(self.a) * (x[..., 1:] @ self.z.t())
        return torch.sign(alpha) * beta * torch.asinh(alpha / beta).abs()


def lorentz_elu(x):
    return add_time(F.elu(x[..., 1:]))


def lorentz_avgpool_w(x, k):
    """(B, H, W, D) -> (B, H, W', D): the Lorentz centroid over non-overlapping width windows of k (remainder dropped)."""
    B, H, W, D = x.shape
    Wo = (W - k) // k + 1
    m = x[:, :, :Wo * k].reshape(B, H, Wo, k, D).mean(3)
    return m / torch.sqrt((-ldot(m, m)).abs().clamp_min(1e-30))


def lorentz_flatten(x):
    """(B, H, W, D) -> (B, 1 + H W (D - 1)): the direct concatenation (Qu et al. 2022)."""
    B = x.shape[0]
    return add_time(x[..., 1:].reshape(B, -1))


def adamom_eta(epoch, epochs=90, bs=10, bs0=50):
    """the reference MomentumBatchNormScheduler's training eta at an epoch (1 at the start -> ~ 1 - (rho - bs/bs0) at the end)."""
    rho = (bs / bs0) ** (1.0 / epochs)
    return 1.0 - (rho ** (epochs * max(epochs - epoch, 0) / (epochs - 1)) - rho ** epochs)


class DSMDBN(nn.Module):
    """domain-specific Lorentz batch norm (scalar dispersion, no learned shift)."""

    def __init__(self, d, n_domains, eps=1e-5, karcher=5):
        super().__init__()
        o = torch.zeros(max(int(n_domains), 1), d)
        o[:, 0] = 1.0
        self.register_buffer("run_mean", o)
        self.register_buffer("run_var", torch.ones(max(int(n_domains), 1)))
        self.eps, self.karcher, self.eta = eps, karcher, 1.0

    def normalise(self, X, mu, var):
        Xn = gyroadd(gyroinv(mu), X)
        return gyroscale(Xn, 1.0 / torch.sqrt(var + self.eps))

    def stats(self, P):
        mu = frechet_mean(P, self.karcher)
        return mu, frechet_var(P, mu)

    def forward_train(self, X, dom):
        """X (B, H, W, D), dom (B,) training-domain ids: each domain by its own batch statistics through eta."""
        out = torch.empty_like(X)
        eta = min(float(self.eta), 1.0)                     # the reference's epoch-0 eta is 1.004; capped at the batch statistics
        for d in torch.unique(dom):
            m = dom == d
            P = X[m].reshape(-1, X.shape[-1])
            bm, bv = self.stats(P)
            i = int(d)
            r_m, r_v = self.run_mean[i].clone(), self.run_var[i].clone()   # copies: the buffers are updated in place below
            rm = geodesic(r_m, bm, eta) if eta < 1.0 else bm
            rv = (1.0 - eta) * r_v + eta * bv
            out[m] = self.normalise(X[m], rm, rv)
            with torch.no_grad():
                self.run_mean[i] = rm.detach()
                self.run_var[i] = rv.detach()
        return out


class HEEGNet(nn.Module):
    """forward(x) needs the BN statistics: train -> forward(x, dom); test -> predict_stream / forward_refit."""

    def __init__(self, n_chans, n_times, n_classes, n_domains, F1=8, D=2, F2=16, k1=64, k2=16, dropout=0.25, karcher=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(dropout))
        self.ec1 = nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2), groups=F1 * D, bias=False)
        self.lc1 = LorentzLinear(F1 * D + 1, F2 + 1, bias=False)
        self.bn = DSMDBN(F2 + 1, n_domains, karcher=karcher)
        with torch.no_grad():
            w = self._pre_bn(torch.zeros(1, n_chans, n_times)).shape[2]
        self.n_pos = w
        self.d_feat = F2 * ((w - 8) // 8 + 1) + 1
        self.mlr = LorentzMLR(self.d_feat, n_classes)

    def _pre_bn(self, x):
        h = self.ec1(self.block1(x.unsqueeze(1))).permute(0, 2, 3, 1)       # (B, 1, W, 16)
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        return self.lc1(add_time(h))                                         # (B, 1, W, 17)

    def _post_bn(self, h):
        f = lorentz_flatten(lorentz_avgpool_w(lorentz_elu(h), 8))
        return self.mlr(f), f

    def forward(self, x, dom):
        return self._post_bn(self.bn.forward_train(self._pre_bn(x), dom))

    def forward_refit(self, x):
        """all of x = one domain: its own statistics (the reference REFIT; internal validation only)."""
        h = self._pre_bn(x)
        mu, var = self.bn.stats(h.reshape(-1, h.shape[-1]))
        return self._post_bn(self.bn.normalise(h, mu, var))

    @torch.no_grad()
    def predict_stream(self, x, bs=64):
        """x (N, C, T) = ONE stream in kit order -> logits (N, K); window t is normalised with the Frechet mean / variance of
        the BN-input points of windows 1..t (causal, label-free, no domain id)."""
        h = torch.cat([self._pre_bn(x[i:i + bs]) for i in range(0, len(x), bs)])     # (N, 1, W, D)
        N, D = h.shape[0], h.shape[-1]
        out = []
        for t in range(N):
            mu, var = self.bn.stats(h[:t + 1].reshape(-1, D))
            out.append(self._post_bn(self.bn.normalise(h[t:t + 1], mu, var))[0])
        return torch.cat(out)


class EEGNetCtl(nn.Module):
    """the reference EEGNet (block1 + block2) with a linear head: T2-D318's control."""

    def __init__(self, n_chans, n_times, n_classes, F1=8, D=2, F2=16, k1=64, k2=16, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8), nn.Dropout(dropout))
        with torch.no_grad():
            d = self.block2(self.block1(torch.zeros(1, 1, n_chans, n_times))).numel()
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, dom=None):
        f = self.block2(self.block1(x.unsqueeze(1))).flatten(1)
        return self.head(f), f


class LorentzHeadNet(nn.Module):
    """T2-D318 H-b: a feature extractor feats(x) -> f (B, d) and a Lorentz MLR on (sqrt(1 + |f|^2), f)."""

    def __init__(self, base, feats, d, n_classes):
        super().__init__()
        self.base, self._feats = base, feats
        self.mlr = LorentzMLR(d + 1, n_classes)

    def forward(self, x):
        return self.mlr(add_time(self._feats(self.base, x).flatten(1)))


# ---------------------------------------------------------------------------------------------------------------- loss


def hsw_loss(f, dom, n_proj=1000, gen=None):
    """the reference swd_loss: per training domain in the batch, the horospherical sliced Wasserstein (p 2) between its
    features and as many exp_o(unit Gaussian direction) points, Busemann coordinates log(x0 - u . xs) on n_proj directions."""
    D = f.shape[-1]
    loss = f.new_zeros(())
    for d in torch.unique(dom):
        x = f[dom == d]
        g = torch.randn(x.shape[0], D - 1, generator=gen, dtype=x.dtype, device="cpu").to(x.device)
        tgt = exp0(g / (g.norm(dim=-1, keepdim=True) + 1e-8))
        u = torch.randn(n_proj, D - 1, generator=gen, dtype=x.dtype, device="cpu").to(x.device)
        u = u / u.norm(dim=-1, keepdim=True)
        bx = torch.log((x[:, :1] - x[:, 1:] @ u.t()).clamp_min(1e-30))     # (n, P)
        bt = torch.log((tgt[:, :1] - tgt[:, 1:] @ u.t()).clamp_min(1e-30))
        loss = loss + ((torch.sort(bx, 0)[0] - torch.sort(bt, 0)[0]) ** 2).mean()
    return loss


# ---------------------------------------------------------------------------------------------------------------- training


def domain_batches(dom, per=10, k=5, rng=None):
    """the reference stratified domain loader: per-domain shuffled chunks of `per`, shuffled, k chunks per batch."""
    rng = rng or np.random.default_rng(0)
    chunks = []
    for d in np.unique(dom):
        ix = rng.permutation(np.flatnonzero(dom == d))
        chunks += [ix[i:i + per] for i in range(0, len(ix), per)]
    order = rng.permutation(len(chunks))
    return [np.concatenate([chunks[j] for j in order[i:i + k]]) for i in range(0, len(order), k)]


def fit(kind, Xtr, ytr, dtr, n_classes, seed, device="cuda", epochs=100, min_epochs=70, patience=20, lr=1e-3, wd=1e-4,
        swd_w=0.01, val_frac=0.2, log=print):
    """kind 'heeg' | 'eegnet' -> (net at its best internal-val-loss epoch, info). dtr = training-domain ids 0..D-1.
    The internal split is 20 % of each training domain (the kit val is never read)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    dt = torch.float64
    va = np.concatenate([rng.permutation(np.flatnonzero(dtr == d))[:int(round(val_frac * (dtr == d).sum()))] for d in np.unique(dtr)])
    trm = np.ones(len(ytr), bool)
    trm[va] = False
    tr = np.flatnonzero(trm)
    X = torch.as_tensor(np.ascontiguousarray(Xtr), dtype=dt, device=device)
    y = torch.as_tensor(ytr, dtype=torch.long, device=device)
    dd = torch.as_tensor(dtr, dtype=torch.long, device=device)
    C, T = X.shape[1:]
    nd = int(dtr.max()) + 1
    net = (HEEGNet(C, T, n_classes, nd) if kind == "heeg" else EEGNetCtl(C, T, n_classes)).to(device, dt)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        net.train()
        if kind == "heeg":
            net.bn.eta = adamom_eta(ep, epochs=epochs - 10)
        tl, n = 0.0, 0
        for b in domain_batches(dtr[tr], rng=rng):
            b = torch.as_tensor(tr[b], device=device)
            logits, f = net(X[b], dd[b])
            loss = F.cross_entropy(logits, y[b])
            if kind == "heeg" and swd_w:
                loss = loss + swd_w * hsw_loss(f, dd[b], gen=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss) * len(b)
            n += len(b)
        net.eval()
        with torch.no_grad():                               # internal val: each training domain's held-out windows by their own stats
            vl, vn, vc = 0.0, 0, 0
            for d in np.unique(dtr[va]):
                v = torch.as_tensor(va[dtr[va] == d], device=device)
                lo = net.forward_refit(X[v])[0] if kind == "heeg" else net(X[v])[0]
                vl += float(F.cross_entropy(lo, y[v], reduction="sum"))
                vc += int((lo.argmax(1) == y[v]).sum())
                vn += len(v)
        vl /= vn
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_loss=vl, val_acc=vc / vn))
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
        if (ep + 1) % 10 == 0:
            log(f"  ep {ep + 1:3d} train {tl / n:.4f} ival {vl:.4f} acc {vc / vn:.3f}")
        if bad >= patience and ep > min_epochs:
            break
    net.load_state_dict(best_state)
    net.eval()
    return net, dict(best_val_loss=best, epochs_run=len(hist), best_epoch=int(np.argmin([h["val_loss"] for h in hist]) + 1), curves=hist)


@torch.no_grad()
def predict_streams(net, kind, XA, streams, device="cuda"):
    """streams = list of row arrays (each one stream in kit order) -> {row: prob (K,)}; HEEGNet causal per stream."""
    out = {}
    for rows in streams:
        x = torch.as_tensor(np.ascontiguousarray(XA[rows]), dtype=torch.float64, device=device)
        lo = net.predict_stream(x) if kind == "heeg" else net(x)[0]
        p = F.softmax(lo, 1).float().cpu().numpy()
        out.update({int(r): p[k] for k, r in enumerate(rows)})
    return out
