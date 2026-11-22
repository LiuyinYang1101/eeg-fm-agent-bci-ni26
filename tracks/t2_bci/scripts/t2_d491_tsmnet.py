#!/usr/bin/env python
"""T2-D491 (seat G10 item 5; registered before any number): train / evaluate the TSMNet-form SPD member (src/ni26/tsmnet.py)
on the bench's kit / seen1 protocols and write result files in t2p_bench's format, so the A13 harness reads them as a member.

Input: the bench's EA alignment (aligned_all 'ea', n0 0, shrink 0.05, global scale = median std of the protocol's training
windows[::7]) = what the two EA families (ShallowNet / IFNet) see. Training: domain = one training stream (user x session);
an epoch = every training window once, in same-domain groups of 16 (each group re-centred by its own batch mean), 4 groups per
batch; one virtual participant (T2-D193 / D215 bank, vp 2 = 10 mm jitter, gains 0.1, shift 1, virtual references 0.5) drawn PER
GROUP, so a group stays one (virtual) person; AdamW lr 1e-3, wd 1e-2 (conv + head; not the orthogonal BiMap or gamma), 10 %
warm-up + cosine, label smoothing 0.1, gradient clip 5, 180 epochs (today's CNN recipe where it applies). No per-user copies,
no test-time views. Test / val: per stream in delivered order, the CAUSAL re-centring (prior = the training log-Euclidean mean
at pseudo-count n0 5). Files: <out>/<proto>__tsmnet__ea__s<seed>.npz (test_idx, prob_last, prob_best, val_idx, prob_val_last, res).
    gpu-submit python tracks/t2_bci/scripts/t2_d491_tsmnet.py --out <dir> --seeds 360,...,371 [--shard i --nshards n] [--epochs E]
    ... --time : ms / window of the batch-1 causal step on CPU and on the GPU (no training)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
from ni26.tsmnet import StreamState, TSMNet, symfn  # noqa: E402


class TSMNetNoRe(TSMNet):
    """T2-D491b: the same net WITHOUT SPD re-centring (the input is already EA-aligned per stream, as ShallowNet / IFNet see it):
    features = log S directly, scaled by gamma / sqrt(dispersion); dispersion = the batch mean of ||log S||^2 in training, the training
    mean at test (a fixed constant: each window is decoded on its own, so the member carries no stream state)."""

    def forward_groups(self, x, grp):
        L = symfn(self.spd(x), "log")
        return self.head_from(L, (L.detach() ** 2).sum((-1, -2)).mean().expand(len(L)))

    @torch.no_grad()
    def set_prior(self, X, grp, bs=256):
        Ls = torch.cat([symfn(self.spd(X[s:s + bs]), "log") for s in range(0, len(X), bs)])
        self.prior_log.zero_()
        self.prior_disp.fill_(float((Ls ** 2).sum((-1, -2)).mean()))

    @torch.no_grad()
    def stream_logits(self, Xs, n0=5.0, bs=256):
        L = torch.cat([symfn(self.spd(Xs[s:s + bs]), "log") for s in range(0, len(Xs), bs)])
        return self.head_from(L, self.prior_disp.expand(len(L)))

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
GROUP, NGROUP = 16, 4


def cosine(step, total, warm):
    if step < warm:
        return (step + 1) / max(warm, 1)
    return 0.5 * (1 + math.cos(math.pi * min((step - warm) / max(total - warm, 1), 1.0)))


def epoch_batches(dom, rng):
    """same-domain groups of GROUP windows (each domain shuffled, a short tail merged into its last group), NGROUP groups per batch."""
    groups = []
    for d in np.unique(dom):
        r = rng.permutation(np.flatnonzero(dom == d))
        k = max(1, len(r) // GROUP)
        groups += [g for g in np.array_split(r, k)]
    order = rng.permutation(len(groups))
    return [[groups[j] for j in order[b:b + NGROUP]] for b in range(0, len(order), NGROUP)]


def vp_group(x, bank, k):
    M, G, S = bank["mats"][k], bank["gains"][k], int(bank["shifts"][k])
    x = G[None, :, None].to(x.dtype) * torch.einsum("ij,bjt->bit", M.to(x.dtype), x)
    return torch.roll(x, S, dims=-1)


def stream_probs(net, XA, idx, rows, dev, n0):
    out = np.zeros((len(rows), 3), np.float32)
    if not len(rows):
        return out
    pos = {int(r): k for k, r in enumerate(rows)}
    sub = idx.iloc[rows]
    for _, g in sub.groupby("stream", sort=False):
        rr = g.sort_values("order")["i"].to_numpy()
        lo = net.stream_logits(torch.as_tensor(np.ascontiguousarray(XA[rr]), device=dev), n0=n0)
        p = F.softmax(lo.float(), 1).cpu().numpy()
        for k, r in enumerate(rr):
            out[pos[int(r)]] = p[k]
    return out


def fit(XA, y, dom, tr, seed, epochs, dev, bank, log, cls=TSMNet):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = cls(XA.shape[1]).to(dev)
    decay = [p for n, p in net.named_parameters() if n.startswith(("temp.", "spat.", "head.weight"))]
    other = [p for n, p in net.named_parameters() if not n.startswith(("temp.", "spat.", "head.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2}, {"params": other, "weight_decay": 0.0}], lr=1e-3)
    Xt = torch.as_tensor(np.ascontiguousarray(XA[tr]), device=dev)
    yt = torch.as_tensor(y[tr], device=dev, dtype=torch.long)
    dt = dom[tr]
    nb = len(epoch_batches(dt, np.random.default_rng(0)))
    total, warm, step = epochs * nb, int(0.1 * epochs * nb), 0
    net.train()
    for ep in range(epochs):
        for batch in epoch_batches(dt, rng):
            xs, ys, gs = [], [], []
            for j, gi in enumerate(batch):
                xg = Xt[gi]
                if bank is not None:
                    xg = vp_group(xg, bank, int(rng.integers(len(bank["mats"]))))
                xs.append(xg)
                ys.append(yt[gi])
                gs.append(torch.full((len(gi),), j, device=dev, dtype=torch.long))
            for pg in opt.param_groups:
                pg["lr"] = 1e-3 * cosine(step, total, warm)
            lo = net.forward_groups(torch.cat(xs), torch.cat(gs))
            loss = F.cross_entropy(lo, torch.cat(ys), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
        if not np.isfinite(float(loss)):
            raise SystemExit(f"non-finite loss at epoch {ep}")
        if ep % 30 == 0 or ep == epochs - 1:
            log(f"seed {seed} epoch {ep} loss {float(loss):.3f}")
    net.eval()
    net.set_prior(Xt, torch.as_tensor(dt, device=dev))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(EXP))
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocols", default="kit,seen1")
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(360, 372)))
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--n0", type=float, default=5.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--nore", action="store_true", help="T2-D491b: no SPD re-centring (TSMNetNoRe); files <proto>__tsmnetnore__ea__s<seed>.npz")
    a = ap.parse_args()
    cls, stem = (TSMNetNoRe, "tsmnetnore") if a.nore else (TSMNet, "tsmnet")
    log = lambda s: print(f"[d491] {time.strftime('%H:%M:%S')} {s}", flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, idx, meta = B.load_export(a.export)
    assert (idx["i"].to_numpy() == np.arange(len(idx))).all()
    y = idx["y"].to_numpy().astype(int)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = {k: v.to(dev) for k, v in B.vp_bank(list(meta["ch_names"]), K=64, sigma=0.01, gain=0.1, shift=1, vref=0.5).items()}
    tasks = [(p, int(s)) for p in a.protocols.split(",") for s in a.seeds.split(",")][a.shard::a.nshards]
    cache = {}
    for proto, seed in tasks:
        fp = out / f"{proto}__{stem}__ea__s{seed}.npz"
        if fp.exists() and not a.time:
            continue
        (_, trm, vam, tem), = B.folds(idx, proto)
        if proto not in cache:
            scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
            cache[proto] = B.aligned_all(X, idx, "ea", trm, 0.0, 0.05, scale)
        XA = cache[proto]
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        dom = np.unique(idx["stream"].to_numpy(), return_inverse=True)[1]
        t0 = time.time()
        if a.time:                                   # ms / window of the batch-1 causal step (an untrained net: same cost)
            net = TSMNet(XA.shape[1]).eval()
            for d_ in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
                net = net.to(d_)
                st = StreamState(net)
                xs = [torch.as_tensor(np.asarray(XA[r:r + 1]), device=d_) for r in te[:220]]
                for x in xs[:20]:
                    st.step(x)
                if d_ == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                for x in xs[20:]:
                    st.step(x)
                if d_ == "cuda":
                    torch.cuda.synchronize()
                log(f"ms / window (batch 1, causal state, {d_}): {(time.perf_counter() - t1) / 200 * 1e3:.2f}")
            return
        net = fit(XA, y, dom, tr, seed, a.epochs, dev, bank, log, cls)
        pte = stream_probs(net, XA, idx, te, dev, a.n0)
        pva = stream_probs(net, XA, idx, va, dev, a.n0)
        assert np.isfinite(pte).all() and np.isfinite(pva).all()
        res = dict(row="T2-D491b" if a.nore else "T2-D491", recentre=not a.nore, protocol=proto, seed=seed, epochs=a.epochs, n0=a.n0, align="ea", group=GROUP, ngroup=NGROUP,
                   n_train=len(tr), seconds=round(time.time() - t0, 1))
        np.savez_compressed(str(fp) + ".tmp.npz", test_idx=te, prob_last=pte, prob_best=pte, val_idx=va, prob_val_last=pva,
                            res=json.dumps(res))
        Path(str(fp) + ".tmp.npz").replace(fp)
        log(f"{fp.name} written ({res['seconds']} s)")


if __name__ == "__main__":
    main()
