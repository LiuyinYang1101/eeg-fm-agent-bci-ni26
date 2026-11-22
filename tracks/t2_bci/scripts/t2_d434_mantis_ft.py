#!/usr/bin/env python
"""T2-D434 (pre-registered in docs/DECISIONS_T2.md before any D434 number): MantisV2 FINE-TUNED end to end as a T2 member (the fair dose after
T2-D433 (b)'s frozen CLS + LR; the manager's ask 2026-11-10, the user's time-series seed).
  dump   (project env)  causally whitened windows (t2_d433_rocket.whiten = causal EA) + labels + the kit / seen1 fold masks -> DIR/ft_data.npz
  train  (eeg311: the `mantis` package)  one arm x seeds x protocols -> DIR/runs_<arm>/{proto}__mantisft__s{seed}.npz (bench format, last epoch)
Arms:
  sp   a learned linear spatial layer 41 -> 16 virtual channels (orthogonal init), each resized 480 -> 512 and encoded by ONE shared MantisV2
       (token generator re-implemented WITHOUT the package's no_grad on the patch statistics / difference, so the spatial layer gets gradients for
       amplitude too), CLS per virtual channel, concat 16 x 256 -> dropout 0.5 -> linear. Pretrained weights.
  sp0  the same, MantisV2 at RANDOM init (the twin).
  ch   the manager's shape: all 41 channels encoded by the shared pretrained MantisV2, attention pooling over channels (a learned query on
       CLS + a learned channel embedding), -> dropout 0.5 -> linear.
Training (one a-priori recipe, no selection on val / test): AdamW (backbone 1e-4, new layers 1e-3, wd 0.05), 40 epochs, cosine + 10 % warm-up, batch 32,
label smoothing 0.1, augmentation = circular time shift U{-12..12} + per-channel gain exp(U(-0.1, 0.1)), bf16 autocast; last-epoch probabilities.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def dump(a):
    sys.path.insert(0, str(HERE))
    import t2p_bench as B
    from t2_d433_rocket import whiten
    X, idx, meta = B.load_export(a.export)
    idx = idx.reset_index(drop=True)
    out = dict(X=whiten(X, idx), y=idx["y"].to_numpy().astype(np.int64))
    for proto in ("kit", "seen1"):
        (fname, trm, vam, tem), = B.folds(idx, proto)
        out.update({f"{proto}_tr": np.flatnonzero(trm), f"{proto}_va": np.flatnonzero(vam), f"{proto}_te": np.flatnonzero(tem)})
    np.savez(Path(a.dir) / "ft_data.npz", **out)
    print("[d434] dump done", out["X"].shape, flush=True)


def train(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mantis.architecture.version2 import MantisV2
    from safetensors.torch import load_file
    W = Path.home() / "workspace/datasets/pretrained_tsfm/mantis_v2"
    cfg = {k: v for k, v in json.loads((W / "config.json").read_text()).items() if k != "device"}
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"

    def tokgen(tg, x):                                   # TokenGeneratorUnit.forward with gradients through every input path
        xp = x.reshape(x.shape[0], tg.num_patches, -1)
        stats = [xp.mean(-1, keepdim=True), xp.std(-1, keepdim=True)]
        sc = [tg.scalar_encoders[i](stats[i]) for i in range(2)]
        emb = []
        for i, s in enumerate((F.pad(torch.diff(x, n=1, dim=2), (0, 1)), x)):
            e = tg.layer_norms[i](tg.convs[i](tg.ts_scaler(s)))
            emb.append(e.reshape(e.shape[0], tg.num_patches, -1, e.shape[2]).mean(2))
        return tg.linear_encoder(torch.cat([torch.cat(emb, -1), torch.cat(sc, -1)], -1))

    class Net(nn.Module):
        def __init__(self, arm, C):
            super().__init__()
            self.arm = arm
            self.enc = MantisV2(**cfg, device="cpu")
            if arm != "sp0":
                self.enc.load_state_dict(load_file(str(W / "model.safetensors")), strict=True)
            self.V = 16 if arm.startswith("sp") else C
            if arm.startswith("sp"):
                self.spat = nn.Linear(C, self.V, bias=False)
                nn.init.orthogonal_(self.spat.weight)
                self.head = nn.Linear(self.V * 256, 3)
            else:
                self.chemb = nn.Parameter(torch.zeros(C, 256))
                self.query = nn.Linear(256, 1)
                self.head = nn.Linear(256, 3)
            self.drop = nn.Dropout(0.5)

        def new_params(self):
            return [p for n, p in self.named_parameters() if not n.startswith("enc.")]

        def forward(self, x):                              # (B, C, T)
            if self.arm.startswith("sp"):
                x = torch.einsum("vc,bct->bvt", self.spat.weight, x)
            B, V, T = x.shape
            x = F.interpolate(x.reshape(B * V, 1, T), size=512, mode="linear", align_corners=False)
            cls = self.enc.transf_unit(tokgen(self.enc.tokgen_unit, x)).reshape(B, V, 256)
            if self.arm.startswith("sp"):
                return self.head(self.drop(cls.reshape(B, -1)))
            w = torch.softmax(self.query(cls + self.chemb).squeeze(-1), 1)
            return self.head(self.drop((w[..., None] * cls).sum(1)))

    D = Path(a.dir)
    Z = np.load(D / "ft_data.npz")
    X, y = torch.from_numpy(Z["X"]), torch.from_numpy(Z["y"])
    out = D / f"runs_{a.arm}"
    os.makedirs(out, exist_ok=True)
    epochs, bs = (1, 16) if a.smoke else (40, 32)
    for proto in a.protocols.split(","):
        tr, va, te = (Z[f"{proto}_{k}"] for k in ("tr", "va", "te"))
        if a.smoke:
            tr, va, te = tr[:48], va[:16], te[:16]
        for seed in [int(s) for s in a.seeds.split(",")]:
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            net = Net(a.arm, X.shape[1]).to(dev)
            enc_p = list(net.enc.parameters())
            opt = torch.optim.AdamW([{"params": enc_p, "lr": 1e-4}, {"params": net.new_params(), "lr": 1e-3}], weight_decay=0.05)
            steps = epochs * ((len(tr) + bs - 1) // bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1, anneal_strategy="cos",
                                                        div_factor=25, final_div_factor=1e3)
            for ep in range(epochs):
                net.train()
                perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=g)]
                tot = 0.0
                for s in range(0, len(perm), bs):
                    b = perm[s:s + bs]
                    xb = X[b].to(dev)
                    xb = torch.stack([torch.roll(v, int(k), -1) for v, k in zip(xb, torch.randint(-12, 13, (len(b),), generator=g))])
                    xb = xb * torch.exp((torch.rand(len(b), xb.shape[1], 1, generator=g) * 0.2 - 0.1).to(dev))
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        loss = F.cross_entropy(net(xb).float(), y[b].to(dev), label_smoothing=0.1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tot += loss.item() * len(b)
                if ep % 5 == 4 or ep == epochs - 1:
                    print(f"[d434] {a.arm} {proto} s{seed} ep {ep + 1}: train loss {tot / len(perm):.4f}", flush=True)
            net.eval()

            @torch.no_grad()
            def prob(ix):
                ps = []
                for s in range(0, len(ix), 64):
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        ps.append(torch.softmax(net(X[ix[s:s + 64]].to(dev)).float(), 1).cpu())
                return torch.cat(ps).numpy().astype(np.float32)
            pte, pva, ptr = prob(te), prob(va), prob(tr[:512])
            res = dict(protocol=proto, model="mantisft", arm=a.arm, seed=seed, epochs=epochs)
            np.savez_compressed(out / f"{proto}__mantisft__s{seed}.npz", test_idx=te, prob_last=pte, val_idx=va, prob_val_last=pva, prob_best=pte,
                                res=json.dumps(res))
            acc = lambda p, ix: float(np.mean(p.argmax(1) == Z["y"][ix]))  # noqa: E731
            print(f"[d434] {a.arm} {proto} s{seed}: window acc train {acc(ptr, tr[:512]):.3f} test {acc(pte, te):.3f} val {acc(pva, va):.3f}", flush=True)
    print(f"[d434] done {a.arm}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dump", "train"))
    ap.add_argument("--export")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", choices=("sp", "sp0", "ch"))
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--protocols", default="kit,seen1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dump(a) if a.step == "dump" else train(a)
