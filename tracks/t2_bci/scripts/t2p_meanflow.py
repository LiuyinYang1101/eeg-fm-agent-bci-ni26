#!/usr/bin/env python
"""T2-D413: a ONE-STEP MeanFlow generator of PROTEUS windows, conditional on class (3) and training user, per protocol.

Pre-registered in docs/DECISIONS_T2.md row T2-D413 (2026-11-09 10:08), before any number:
  - trained ONLY on the protocol's training windows (t2p_bench.folds: kit / seen1), align none = the FBCNet member's input
    (export windows 41 x 480 at 120 Hz), per-channel z-score by the training windows' channel mean / std (inverted at sampling);
  - net: 1D U-Net over time (41 -> 96 / 192 / 192 ch, 3 levels, 2 residual blocks per level, GroupNorm + SiLU, no attention; the
    lowest level adds 4 dilated blocks (1, 2, 4, 8) for a ~2 s receptive field), FiLM on emb(t) + emb(t - r) + class + user embeddings;
  - pMF-style x-prediction: u = (z_t - xhat(z_t, r, t)) / t with z_t = (1 - t) x + t eps; MeanFlow loss
    || u - sg(v - (t - r) du/dt) ||^2, du/dt = the JVP along (v, 0, 1), v = eps - x; adaptive weight 1 / (||err||^2 + 1e-3);
  - (t, r) logit-normal(-0.4, 1), sorted, r = t on 75 % (r != t on 25 %); AdamW 2e-4, batch 64, 12k steps, EMA 0.999, seed 0;
  - one-step sampling x = xhat(eps, 0, 1) (EMA weights);
  - synthetic NEW people: user embedding alpha E[a] + (1 - alpha) E[b], (a, b) a uniform distinct pair, alpha ~ U(0.25, 0.75).
Implementation details not in the row (fixed before any number): 500-step linear lr warm-up; gradient-norm clip 1.0; t clamped to
>= 1e-3 at sampling of (t, r) (before the JVP).

Writes into --out:
  gen_<proto>_g<seed>.pt                      EMA state_dict + normalisation + users + config + the training log
  own_<proto>_g<seed>.npz                     one sample per training window with that window's (user, class) (the gate's realism set)
  interp_<proto>_g<seed>.npz                  n_train interpolated new-people windows, class uniform (the gate's TSTR set)
  bank_<proto>_g<seed>.npz                    8 x n_train interpolated windows, fp16 (Step 1's bank), with train_idx / users for checks
    mem-guard / gpu-submit python tracks/t2_bci/scripts/t2p_meanflow.py --export E --protocol kit --out outputs/t2_d413
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2p_bench import folds, load_export  # noqa: E402
import t2p_mf_calib as KC  # noqa: E402  (T2-D413b calib 2 / 3)

EMB = 256


def sincos(x, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
    a = x[:, None] * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):
    def __init__(self, ci, co, dil=1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ci), nn.Conv1d(ci, co, 3, padding=dil, dilation=dil)
        self.n2, self.c2 = nn.GroupNorm(8, co), nn.Conv1d(co, co, 3, padding=dil, dilation=dil)
        self.film = nn.Linear(EMB, 2 * co)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.film(e).unsqueeze(-1).chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + s) + b))
        return self.skip(x) + h


class UNet1D(nn.Module):
    def __init__(self, C, n_users, n_classes=3, ch=(96, 192, 192)):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(128, EMB), nn.SiLU(), nn.Linear(EMB, EMB))
        self.remb = nn.Sequential(nn.Linear(128, EMB), nn.SiLU(), nn.Linear(EMB, EMB))
        self.cemb, self.uemb = nn.Embedding(n_classes, EMB), nn.Embedding(n_users, EMB)
        self.stem = nn.Conv1d(C, ch[0], 3, padding=1)
        self.down, self.ds = nn.ModuleList(), nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.down.append(nn.ModuleList([Res(c, co), Res(co, co)])); c = co
            self.ds.append(nn.Conv1d(c, c, 4, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity())
        self.mid = nn.ModuleList([Res(c, c, d) for d in (1, 2, 4, 8)])
        self.up, self.us = nn.ModuleList(), nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList([Res(c + co, co), Res(co, co)])); c = co
            self.us.append(nn.ConvTranspose1d(c, c, 4, stride=2, padding=1) if i > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, C, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, z, t, r, c, ue):
        e = F.silu(self.temb(sincos(t)) + self.remb(sincos(t - r)) + self.cemb(c) + ue)
        h, hs = self.stem(z), []
        for blocks, ds in zip(self.down, self.ds):
            for b in blocks:
                h = b(h, e)
            hs.append(h); h = ds(h)
        for b in self.mid:
            h = b(h, e)
        for blocks, us in zip(self.up, self.us):
            h = torch.cat([h, hs.pop()], 1)
            for b in blocks:
                h = b(h, e)
            h = us(h)
        return self.out(h)


def sample_tr(B, g, device, p_eq=0.75):
    t = torch.sigmoid(torch.randn(B, generator=g) - 0.4)
    r = torch.sigmoid(torch.randn(B, generator=g) - 0.4)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    eq = torch.rand(B, generator=g) < p_eq                           # T2-D413b: p_eq 1 = flow matching (r = t always)
    r = torch.where(eq, t, r)
    t = t.clamp_min(1e-3); r = torch.minimum(r, t)
    return t.to(device), r.to(device)


def mf_loss(net, x, c, ue, g, p_eq=0.75):
    B = len(x)
    t, r = sample_tr(B, g, x.device, p_eq)
    eps = torch.randn(x.shape, generator=g).to(x.device)
    z = (1 - t.view(B, 1, 1)) * x + t.view(B, 1, 1) * eps
    v = eps - x

    def u_fn(z_, r_, t_):
        return (z_ - net(z_, t_, r_, c, ue)) / t_.view(-1, 1, 1)

    if p_eq >= 1.0:                                                  # T2-D413b: flow matching (r = t: the JVP term is 0, skip it)
        u, tgt = u_fn(z, t, t), v.detach()
    else:
        u, dudt = torch.func.jvp(u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        tgt = (v - (t - r).view(B, 1, 1) * dudt).detach()
    l = (u - tgt).pow(2).mean((1, 2))
    w = 1.0 / (l.detach() + 1e-3)
    return (w * l).mean(), float(l.mean())


@torch.no_grad()
def generate(net, c, ue, mu, sd, g, bs=256):
    out = []
    for i in range(0, len(c), bs):
        eps = torch.randn((len(c[i:i + bs]), len(mu), 480), generator=g).to(mu.device)
        one, zero = torch.ones(len(eps), device=eps.device), torch.zeros(len(eps), device=eps.device)
        x = net(eps, one, zero, c[i:i + bs], ue[i:i + bs])           # x = xhat(eps, r 0, t 1): the one-step sample
        out.append((x * sd + mu).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def generate_k(net, c, ue, mu, sd, g, steps=1, euler=False, bs=256, L=480, inv=None):
    """T2-D413b: k-step MeanFlow sampling on the grid t_i = 1 - i / k (euler False: the average velocity u(z, t_{i+1}, t_i) per step) or k Euler
    steps of the instantaneous velocity u(z, t, t) (euler True: the multi-step flow-matching ceiling of the same weights). steps 1, euler False
    = generate() up to float rounding (the same noise draw, z - (z - xhat)); the D413 path (mf1, norm ch) still calls generate()."""
    out, ts = [], torch.linspace(1, 0, steps + 1).tolist()
    for i in range(0, len(c), bs):
        cc, uu = c[i:i + bs], ue[i:i + bs]
        z = torch.randn((len(cc), len(mu), L), generator=g).to(mu.device)
        for k in range(steps):
            t, r = ts[k], ts[k + 1]
            tt = torch.full((len(cc),), t, device=z.device)
            xh = net(z, tt, tt if euler else torch.full_like(tt, r), cc, uu)
            z = z - (t - r) * (z - xh) / t
        out.append((inv(z) if inv is not None else z * sd + mu).float().cpu())     # T2-D413b: inv = WhiteNorm.inv
    return torch.cat(out).numpy()


def interp_conds(n, n_users, g, device, E):
    a = torch.randint(n_users, (n,), generator=g)
    b = (a + torch.randint(1, n_users, (n,), generator=g)) % n_users
    al = 0.25 + 0.5 * torch.rand(n, generator=g)
    c = torch.randint(3, (n,), generator=g)
    ue = al[:, None].to(device) * E[a.to(device)] + (1 - al[:, None]).to(device) * E[b.to(device)]
    return c.to(device), ue, a.numpy(), b.numpy(), al.numpy()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _smooth(A, k=5):
    return F.avg_pool1d(A[None].float(), k, stride=1, padding=k // 2, count_include_pad=False)[0].to(A.dtype)


class WhiteNorm:
    """T2-D413b --norm white: x -> irfft(rfft(x - mu) / A) / s, A (C, F) = the training windows' MEDIAN periodogram amplitude per channel
    (5-bin moving average; robust to the artifact windows), s (C, 1) = the median whitened-window RMS per channel. A diagonal operator in the
    DFT basis with non-zero gains, so inv(fwd(x)) = x exactly (float rounding)."""

    def __init__(self, Xt=None, L=480, state=None):
        self.L = L
        if state is not None:
            self.mu, self.A, self.s = state["mu"], state["A"], state["s"]
            return
        self.mu = Xt.mean((0, 2), keepdim=True)[0]
        Fx = torch.fft.rfft(Xt - self.mu, dim=-1)
        A = _smooth((Fx.abs() ** 2).median(0).values.sqrt())
        self.A = A.clamp_min(1e-6 * float(A.mean()))
        self.s = torch.fft.irfft(Fx / self.A, n=L, dim=-1).pow(2).mean(-1).sqrt().median(0).values[:, None]

    def to(self, dev):
        self.mu, self.A, self.s = self.mu.to(dev), self.A.to(dev), self.s.to(dev)
        return self

    def state(self):
        return dict(mu=self.mu.cpu(), A=self.A.cpu(), s=self.s.cpu())

    def fwd(self, x):
        return torch.fft.irfft(torch.fft.rfft(x - self.mu, dim=-1) / self.A, n=self.L, dim=-1) / self.s

    def inv(self, z):
        return torch.fft.irfft(torch.fft.rfft(z * self.s, dim=-1) * self.A, n=self.L, dim=-1) + self.mu


def _logspec(X):
    return torch.log(torch.fft.rfft(torch.as_tensor(X, dtype=torch.float64), dim=-1).abs() ** 2 + 1e-12)


def calib_gains(Xr, yr, ur, Xs, ys, us, n_users):
    """T2-D413b --calib: the log AMPLITUDE gain per (user, class, channel, DFT bin) = (mean log periodogram real - mean log periodogram syn) / 2,
    5-bin smoothed; fitted on a SEPARATE own-user draw (not the evaluated sets). -> (n_users, 3, C, F) float64."""
    G = []
    for j in range(n_users):
        Gc = []
        for c in range(3):
            kr, ks = (yr == c) & (ur == j), (ys == c) & (us == j)
            assert kr.any() and ks.any(), (j, c)
            Gc.append(_smooth((_logspec(Xr[kr]).mean(0) - _logspec(Xs[ks]).mean(0)) / 2))
        G.append(torch.stack(Gc))
    return torch.stack(G)


def apply_gains(X, lg, L=480, bs=1024):
    """X (n, C, L) numpy, lg (n, C, F) log amplitude gains -> the filtered windows (float32)."""
    out = []
    for i in range(0, len(X), bs):
        Fx = torch.fft.rfft(torch.as_tensor(X[i:i + bs], dtype=torch.float64), dim=-1) * torch.exp(torch.as_tensor(lg[i:i + bs]))
        out.append(torch.fft.irfft(Fx, n=L, dim=-1).float().numpy())
    return np.concatenate(out)


SAMPLERS = {"mf1": (1, False), "mf2": (2, False), "eu50": (50, True), "eu100": (100, True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--protocol", required=True, choices=["kit", "seen1"] + [f"lc9_{k}" for k in range(10)])   # T2-D576: fold-clean lc9 generators
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bank-mult", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    # T2-D413b knobs (defaults = T2-D413 bitwise)
    ap.add_argument("--norm", default="ch", choices=["ch", "white"])
    ap.add_argument("--p-eq", type=float, default=0.75, help="share of r = t draws; 1 = flow matching")
    ap.add_argument("--sample", default="mf1", choices=sorted(SAMPLERS))
    ap.add_argument("--calib", type=int, default=0, choices=[0, 1, 2, 3], help="1 = per (user, class, channel) spectral gains; 2 = filter-bank "
                    "spatial re-colouring per (user, class, band) (t2p_mf_calib); 3 = 2 then 1; all fitted on a separate own-user draw")
    ap.add_argument("--tag", default="", help="output stem tag (default g<seed>)")
    ap.add_argument("--from-ckpt", default="", help="skip training: sample from this gen_*.pt (its norm / users / train_idx)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, idx, meta = load_export(a.export)
    (name, trm, _, _), = folds(idx, a.protocol)
    tr = np.flatnonzero(trm)
    users = sorted(idx.iloc[tr]["subject"].unique())
    u = idx.iloc[tr]["subject"].map({s: k for k, s in enumerate(users)}).to_numpy()
    y = idx["y"].to_numpy()[tr]
    Xt = torch.as_tensor(np.asarray(X[tr], dtype=np.float32))
    U, Y = torch.as_tensor(u, device=dev), torch.as_tensor(y, device=dev)
    C, L = Xt.shape[1], Xt.shape[2]
    assert L == 480 and len(np.unique(y)) == 3, (L, np.unique(y))
    stem = f"{a.protocol}_{a.tag or f'g{a.seed}'}"
    t0, log = time.time(), []
    if a.from_ckpt:
        ck = torch.load(a.from_ckpt, map_location="cpu", weights_only=False)
        assert np.array_equal(ck["train_idx"], tr) and list(ck["users"]) == users and ck["protocol"] == a.protocol, a.from_ckpt
        norm = ck["config"].get("norm", "ch")
        ema = UNet1D(C, len(users)).to(dev); ema.load_state_dict(ck["state"])
        mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
        wn = WhiteNorm(state=ck["white"]).to(dev) if norm == "white" else None
        n_par, log = ck["n_params"], ck["log"]
        print(f"[t2p_meanflow] {a.protocol}: sampling {a.from_ckpt} (norm {norm}) as {stem}", flush=True)
    else:
        norm = a.norm
        mu, sd = Xt.mean((0, 2), keepdim=True)[0], Xt.std((0, 2), keepdim=True)[0]        # (C, 1)
        if norm == "white":
            wn = WhiteNorm(Xt)
            Xn = wn.fwd(Xt).to(dev)
            assert float((wn.inv(Xn.cpu()) - Xt).abs().max()) < 1e-2 * float(Xt.abs().max()), "WhiteNorm must invert"
            wn.to(dev)
        else:
            wn = None
            Xn = ((Xt - mu) / sd).to(dev)
        mu, sd = mu.to(dev), sd.to(dev)
        net = UNet1D(C, len(users)).to(dev)
        ema = UNet1D(C, len(users)).to(dev); ema.load_state_dict(net.state_dict()); ema.requires_grad_(False)
        n_par = sum(p.numel() for p in net.parameters())
        print(f"[t2p_meanflow] {a.protocol}: {len(tr)} training windows, users {users}, classes {np.bincount(y).tolist()}, "
              f"params {n_par / 1e6:.2f} M, steps {a.steps}, bs {a.bs}, lr {a.lr}, norm {norm}, p_eq {a.p_eq}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
        g = torch.Generator().manual_seed(a.seed)
        run = []
        for step in range(a.steps):
            j = torch.randint(len(Xn), (a.bs,), generator=g).to(dev)
            loss, raw = mf_loss(net, Xn[j], Y[j], net.uemb(U[j]), g, a.p_eq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            opt.step(); sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1 - 0.999)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
            run.append((float(loss), raw, gn))
            if (step + 1) % 500 == 0 or step == 0:
                m = np.mean(run, 0); run = []
                log.append(dict(step=step + 1, wloss=float(m[0]), mse=float(m[1]), gnorm=float(m[2]), sec=round(time.time() - t0, 1)))
                print(f"[t2p_meanflow] {a.protocol} step {step + 1}: weighted {m[0]:.4f}, raw u-mse {m[1]:.4f}, grad norm {m[2]:.3f}, "
                      f"{time.time() - t0:.0f} s", flush=True)
                assert np.isfinite(m).all(), "non-finite loss"
        torch.save(dict(state=ema.state_dict(), mu=mu.cpu(), sd=sd.cpu(), users=users, train_idx=tr, protocol=a.protocol,
                        config=dict(vars(a), norm=norm), log=log, n_params=n_par, white=wn.state() if wn is not None else None),
                   out / f"gen_{stem}.pt")
    ema.eval()
    E = ema.uemb.weight.detach()
    k_, eu_ = SAMPLERS[a.sample]

    def gen(c_, ue_, g_):
        if norm == "ch" and a.sample == "mf1":
            return generate(ema, c_, ue_, mu, sd, g_)                                    # T2-D413 bitwise
        return generate_k(ema, c_, ue_, wn.mu if wn is not None else mu, sd, g_, steps=k_, euler=eu_, inv=wn.inv if wn is not None else None)

    Gc = Mc = None
    if a.calib:
        Xc = gen(Y, E[U], torch.Generator().manual_seed(20_000 + a.seed))                # the calibration draw (never evaluated)
        if a.calib in (2, 3):
            Mc = KC.fit_maps(Xt.numpy(), y, u, Xc, y, u, len(users))
            Xc = KC.apply_maps_idx(Xc, Mc, y, u)
        if a.calib in (1, 3):
            Gc = calib_gains(Xt.numpy(), y, u, Xc, y, u, len(users))
        torch.save(dict(G=Gc, M=Mc, calib=a.calib, sample=a.sample), out / f"calib_{stem}.pt")
    gs = torch.Generator().manual_seed(10_000 + a.seed)
    Xo = gen(Y, E[U], gs)                                                                # own-user: each training window's (u, c)
    if Mc is not None:
        Xo = KC.apply_maps_idx(Xo, Mc, y, u)
    if Gc is not None:
        Xo = apply_gains(Xo, Gc[u, y])
    np.savez(out / f"own_{stem}.npz", X=Xo, y=y, u=u, train_idx=tr, users=np.array(users))

    def interp_set(n):
        c, ue, ua, ub, al = interp_conds(n, len(users), gs, dev, E)
        Xs = gen(c, ue, gs)
        cc = c.cpu().numpy()
        if Mc is not None:
            Xs = KC.apply_maps_idx(Xs, Mc, cc, ua, ub, al)
        if Gc is not None:
            w = torch.as_tensor(al, dtype=torch.float64)[:, None, None]
            Xs = apply_gains(Xs, (w * Gc[ua, cc] + (1 - w) * Gc[ub, cc]).numpy())
        return Xs, c.cpu().numpy(), ua, ub, al

    Xi, ci, ua, ub, al = interp_set(len(tr))
    np.savez(out / f"interp_{stem}.npz", X=Xi, y=ci, ua=ua, ub=ub, alpha=al, u=np.where(al >= 0.5, ua, ub),
             train_idx=tr, users=np.array(users))
    Xb, cb, ua, ub, al = interp_set(a.bank_mult * len(tr))
    assert np.isfinite(Xb).all() and np.abs(Xb).max() < 6e4, "bank out of fp16 range"
    np.savez(out / f"bank_{stem}.npz", X=Xb.astype(np.float16), y=cb, ua=ua, ub=ub, alpha=al,
             u=np.where(al >= 0.5, ua, ub), train_idx=tr, users=np.array(users), protocol=a.protocol)
    files = [f for f in ("gen", "own", "interp", "bank") if (out / f"{f}_{stem}.{'pt' if f == 'gen' else 'npz'}").exists()]
    info = {f: sha(out / f"{f}_{stem}.{'pt' if f == 'gen' else 'npz'}") for f in files}
    json.dump(dict(protocol=a.protocol, seed=a.seed, n_train=len(tr), users=users, sha256=info, log=log, sec=round(time.time() - t0, 1),
                   config=dict(vars(a), norm=norm)), open(out / f"info_{stem}.json", "w"), indent=1)
    print(f"[t2p_meanflow] {a.protocol}: wrote {stem} (own {Xo.shape}, interp {Xi.shape}, bank {Xb.shape}); sha256 bank {info['bank'][:12]}; "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
