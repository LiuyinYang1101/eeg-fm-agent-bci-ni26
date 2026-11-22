#!/usr/bin/env python
"""T2-D491 DIAGNOSIS of the kill (written after the kill; TRAINING windows only, no test or val number is computed): does the SPD
member fit its own training set? Clean (unaugmented) training accuracy every 20 epochs, each training stream re-centred by its own
full mean (the training-time normalisation at its best), for the registered recipe and three one-knob variants:
  base     the registered recipe              lr3      lr 3e-3                 nodisp   no dispersion normalisation (gamma only)
  nore     no re-centring (every group's mean = the identity; dispersion = the global one)
    CUDA_VISIBLE_DEVICES= mem-guard python tracks/t2_bci/scripts/t2_d491_diag.py --variant base --epochs 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
import t2_d491_tsmnet as D  # noqa: E402
from ni26.tsmnet import TSMNet, symfn, triu_vec  # noqa: E402


class Var(TSMNet):
    def __init__(self, C, variant):
        super().__init__(C)
        self.variant = variant

    def forward_groups(self, x, grp):
        if self.variant not in ("nodisp", "nore"):
            return super().forward_groups(x, grp)
        S = self.spd(x)
        if self.variant == "nore":
            Lc = symfn(S, "log")
            return self.head(triu_vec(Lc * self.gamma.double() / (Lc.detach() ** 2).sum((-1, -2)).mean().sqrt()).float())
        L = symfn(S, "log")
        Lc = torch.empty_like(S)
        for g in torch.unique(grp):
            m = grp == g
            with torch.no_grad():
                Wm = symfn(-0.5 * L[m].mean(0, keepdim=True), "exp")
            Lc[m] = symfn(Wm @ S[m] @ Wm, "log")
        return self.head(triu_vec(Lc * self.gamma.double()).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=("base", "lr3", "nodisp", "nore"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=360)
    a = ap.parse_args()
    X, idx, meta = B.load_export(D.EXP)
    y = idx["y"].to_numpy().astype(int)
    (_, trm, _, _), = B.folds(idx, "kit")
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    XA = B.aligned_all(X, idx, "ea", trm, 0.0, 0.05, scale)
    tr = np.flatnonzero(trm)
    dom = np.unique(idx["stream"].to_numpy(), return_inverse=True)[1][tr]
    bank = B.vp_bank(list(meta["ch_names"]), K=64, sigma=0.01, gain=0.1, shift=1, vref=0.5)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = Var(XA.shape[1], a.variant)
    lr = 3e-3 if a.variant == "lr3" else 1e-3
    dec = [p for n, p in net.named_parameters() if n.startswith(("temp.", "spat.", "head.weight"))]
    oth = [p for n, p in net.named_parameters() if not n.startswith(("temp.", "spat.", "head.weight"))]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 1e-2}, {"params": oth, "weight_decay": 0.0}], lr=lr)
    Xt, yt = torch.as_tensor(np.ascontiguousarray(XA[tr])), torch.as_tensor(y[tr])
    nb = len(D.epoch_batches(dom, np.random.default_rng(0)))
    total, warm, step = a.epochs * nb, int(0.1 * a.epochs * nb), 0
    for ep in range(a.epochs):
        net.train()
        for batch in D.epoch_batches(dom, rng):
            xs = [D.vp_group(Xt[gi], bank, int(rng.integers(64))) for gi in batch]
            gs = [torch.full((len(gi),), j) for j, gi in enumerate(batch)]
            for pg in opt.param_groups:
                pg["lr"] = lr * D.cosine(step, total, warm)
            loss = F.cross_entropy(net.forward_groups(torch.cat(xs), torch.cat(gs)), torch.cat([yt[gi] for gi in batch]), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
        if ep % 20 == 19 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                lo = net.forward_groups(Xt, torch.as_tensor(dom))      # each stream re-centred by its full mean, no augmentation
            print(f"[d491 diag] {a.variant} epoch {ep + 1}: clean TRAINING accuracy {(lo.argmax(1) == yt).float().mean():.3f}, "
                  f"gamma {float(net.gamma):.2f}", flush=True)


if __name__ == "__main__":
    main()
