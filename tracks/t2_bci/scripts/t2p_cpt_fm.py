#!/usr/bin/env python
"""T2-D492: in-domain continued pretraining (CPT) of REVE-base / LaBraM-base on the unlabelled public windows ST-L's CPT saw.

Neither native objective can be run: LaBraM's masked-code target needs its VQ tokenizer (not held), and REVE's MAE decoder is not
public. So both get ONE shared objective, the nearest analogue of ST-L's MAE CPT: masked-patch reconstruction (SimMIM form).
  * input = the members' own front end: each pool's windows / a per-pool GLOBAL scale (the bench's 'none' constant: the median
    per-channel std of the pool's training windows), then the adapter's prep (FFT resample by duration to 200 Hz = 800 samples,
    x GAIN; REVE clamps), channels = the pool's names inside the FM's vocabulary (REVE: its position bank, exact; LaBraM: its
    128-name list, case-insensitive);
  * tokens = the FM's own (channel x 1-s patch) grid, P = 4 (REVE: 200-sample patches at stride 180; LaBraM: 4 x 200);
  * a random --mask-ratio of each window's tokens is masked by ZEROING the patch's samples at the input; the encoder's last-layer
    grid (B, C, P, D) feeds a fresh linear head (D -> 200) that reconstructs every masked patch, target = the patch normalised
    per token (MAE's norm_pix); loss = MSE over masked tokens only.
Pools = t2p_cpt's rules exactly (t2p_cpt.Pool / pool_keep: --proteus-pool kit_tv | seen1 | all + the --extra exports, draw p ∝
training windows, a fixed 5 % VAL per pool scored with fixed masks at step 0, every --eval-every steps and at the end).
Output <out>/: REVE -> model.safetensors (the full braindecode REVE state, the pinned file's layout); LaBraM -> checkpoint-cpt.pth
({'model': student.* keys} = the pinned labram-base.pth layout, the 16-row temporal table restored); head.pt; cpt_summary.json
(status ok, fm, base / ckpt sha256, args, pools, proteus_streams, curves) beside the checkpoint = t2p_fm.use_cpt's contract.
  python t2p_cpt_fm.py --fm reve --proteus <export_proteus_v1> --extra <dreyer2023 wide> <scherer2015> --out O --proteus-pool kit_tv
  ... --bench 20   # time 20 optimiser steps, write bench.json, no checkpoint
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2p_cpt as C  # noqa: E402  (pool rules, Pool, VAL_SEED)
import t2p_fm  # noqa: E402

t1 = t2p_fm.t1
GEOM = {"reve": (200, 180), "labram": (200, 200)}          # (patch length, stride) at 200 Hz


def vocab(fm, names):
    if fm == "reve":
        bank = json.loads(Path(t1.REVE_POS).read_text())
        return [i for i, c in enumerate(names) if c in bank]
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER
    canon = {c.upper() for c in LABRAM_CHANNEL_ORDER}
    return [i for i, c in enumerate(names) if str(c).upper() in canon]


class FMPool(C.Pool):
    """t2p_cpt.Pool (rows, 5 % val, 4-s crop) with the FM's channel vocabulary and a global scale."""

    def __init__(self, path, rows, label, fm):
        super().__init__(path, rows, label)
        names = list(json.loads((Path(path) / "meta.json").read_text())["ch_names"])
        self.keep = np.array(vocab(fm, names))
        self.ch = [names[i] for i in self.keep]
        self.dropped_ch = [c for c in names if c not in self.ch]
        assert len(self.ch) >= 8, f"{path}: only {len(self.ch)} channels in {fm}'s vocabulary"
        sel = np.sort(self.tr[np.linspace(0, len(self.tr) - 1, min(len(self.tr), 2000)).astype(int)])
        x = np.asarray(self.X[sel], dtype=np.float32)[:, self.keep]
        self.scale = float(np.median(x.std(-1)))
        assert self.scale > 0


def patches(fm, x):
    L, s = GEOM[fm]
    P = (x.shape[-1] - L) // s + 1
    return torch.stack([x[..., k * s:k * s + L] for k in range(P)], 2)              # (B, C, P, L)


def masked_loss(fm, enc, head, x, mask):
    """x (B, C, T) prepped; mask (B, C, P) bool -> MSE of the masked patches (norm_pix target)."""
    L, s = GEOM[fm]
    keep = torch.ones_like(x)
    for k in range(mask.shape[2]):
        keep[..., k * s:k * s + L] *= (~mask[:, :, k]).unsqueeze(-1).to(x.dtype)
    g = enc(x * keep, enc.n_layers)                                                  # (B, C, P, D)
    assert g.shape[:3] == mask.shape, (g.shape, mask.shape)
    tgt = patches(fm, x)
    tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.std(-1, keepdim=True) + 1e-5)
    pred = head(g.float())
    err = ((pred - tgt.float()) ** 2).mean(-1)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def lab_save_state(enc, pinned_path):
    """braindecode Labram keys -> the pinned labram-base.pth layout (student.*), inverting t1.labram_state; the temporal table gets
    its 16 rows back (the trained first P + 1 rows over the pinned ones). Round-trip asserted by the caller."""
    pin = torch.load(pinned_path, map_location="cpu", weights_only=False)["model"]
    inv = {}
    for k, v in enc.lab.state_dict().items():
        k2 = k.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
        if k2.startswith("patch_embed.temporal_conv."):
            k2 = "patch_embed." + k2[len("patch_embed.temporal_conv."):]
        k2 = {"position_embedding": "pos_embed", "temporal_embedding": "time_embed"}.get(k2, k2)
        if k2 == "time_embed":
            full = pin["student.time_embed"].clone()
            full[:, :v.shape[1]] = v.detach().cpu()
            v = full
        inv["student." + k2] = v.detach().cpu().clone()
    return {"model": inv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True, choices=("reve", "labram"))
    ap.add_argument("--proteus", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--proteus-pool", default="kit_tv", choices=sorted(C.PROTEUS_POOLS))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5, help="encoder peak lr (head x 20)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(f"[t2p_cpt_fm {time.time() - t0:7.0f}s] {s}", flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    idx = pd.read_csv(Path(a.proteus) / "index.csv", dtype={"subject": str, "session": str})
    sub, ses = idx["subject"].str.split("/").str[-1], idx["session"]
    if a.proteus_pool == "kit_tv":
        rows = np.flatnonzero(idx["split"].isin(["train", "val"]).to_numpy())
    else:
        rows = np.flatnonzero(C.pool_keep(sub, ses, a.proteus_pool))
    pools = [FMPool(a.proteus, rows, f"proteus_{a.proteus_pool}", a.fm)]
    pstreams = t2p_fm.proteus_streams(idx["subject"].to_numpy()[rows], ses.to_numpy()[rows])
    for e in a.extra:
        ie = pd.read_csv(Path(e) / "index.csv", dtype=str)
        assert not C.is_proteus_extra(ie), f"{e}: a PROTEUS-derived extra is not wired here"
        pools.append(FMPool(e, np.arange(len(ie)), "extra_unlabelled", a.fm))
    ntr = np.array([len(p.tr) for p in pools], float)
    p_draw = ntr / ntr.sum()
    encs = []
    for k, p in enumerate(pools):
        e = t2p_fm.make_encoder(a.fm, p.ch, log, 0.0, "gain")
        if k:                                                   # one backbone, a montage per pool
            if a.fm == "reve":
                e.reve = encs[0].reve
            else:
                e.lab = encs[0].lab
        encs.append(e.to(dev))
        log(f"pool {p.name} ({p.label}): {len(p.ch)} ch (dropped {p.dropped_ch}), {len(p.tr)} train / {len(p.va)} val, "
            f"scale {p.scale:.4g}, draw p {p_draw[k]:.3f}")
    enc0 = encs[0]
    head = nn.Linear(enc0.dim, GEOM[a.fm][0]).to(dev)
    bb = enc0.reve if a.fm == "reve" else enc0.lab
    params = [q for n_, q in bb.named_parameters() if q.requires_grad]
    opt = torch.optim.AdamW([dict(params=params, lr=a.lr, lr_scale=1.0), dict(params=list(head.parameters()), lr=a.lr * 20, lr_scale=20.0)],
                            lr=a.lr, weight_decay=a.wd)
    warm = max(1, int(a.warmup_frac * a.steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        q = (s - warm) / max(1, a.steps - warm)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * q))

    def batch(k, ids, g):
        p = pools[k]
        x = torch.as_tensor(p.draw(ids, g) / p.scale, device=dev)
        x = encs[k].prep(x)
        P = (x.shape[-1] - GEOM[a.fm][0]) // GEOM[a.fm][1] + 1
        m = torch.as_tensor(g.random((len(x), x.shape[1], P)) < a.mask_ratio, device=dev)
        return x, m

    vrng = {k: np.random.default_rng(C.VAL_SEED + k) for k in range(len(pools))}
    vfix = {k: [batch(k, p.va[j:j + a.batch], vrng[k]) for j in range(0, len(p.va), a.batch)] for k, p in enumerate(pools)}

    def val():
        bb.eval()
        r = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            for k, p in enumerate(pools):
                ls = [float(masked_loss(a.fm, encs[k], head, x, m)) for x, m in vfix[k]]
                r[p.name] = float(np.mean(ls))
        bb.train()
        return r

    curves = dict(train=[], val=[])
    curves["val"].append(dict(step=0, **val()))
    log(f"val @ 0: {curves['val'][-1]}")
    bb.train()
    n_steps = a.bench or a.steps
    tb, acc_l = time.time(), []
    for s in range(n_steps):
        f = lr_at(s)
        for gp in opt.param_groups:
            gp["lr"] = f * gp["lr_scale"]
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            k = int(rng.choice(len(pools), p=p_draw))
            ids = rng.choice(pools[k].tr, size=a.batch, replace=len(pools[k].tr) < a.batch)
            x, m = batch(k, ids, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = masked_loss(a.fm, encs[k], head, x, m) / a.accum
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
        opt.step()
        acc_l.append(tot)
        if (s + 1) % 50 == 0:
            curves["train"].append(dict(step=s + 1, loss=float(np.mean(acc_l[-50:]))))
        if not a.bench and ((s + 1) % a.eval_every == 0 or s + 1 == n_steps):
            curves["val"].append(dict(step=s + 1, **val()))
            log(f"step {s + 1}: train {np.mean(acc_l[-50:]):.4f} val {curves['val'][-1]}")
    dt = time.time() - tb
    if a.bench:
        (out / "bench.json").write_text(json.dumps(dict(fm=a.fm, steps=n_steps, s_per_step=dt / n_steps, train=acc_l), indent=1))
        log(f"bench: {dt / n_steps:.3f} s / step -> {a.steps} steps ≈ {dt / n_steps * a.steps / 3600:.2f} h")
        return
    base_path, base_sha = t1.checkpoint_sha(a.fm)
    if a.fm == "reve":
        from safetensors.torch import save_file
        ck = out / "model.safetensors"
        save_file({k_: v.detach().cpu().contiguous().clone() for k_, v in enc0.reve.state_dict().items()}, str(ck))
    else:
        ck = out / "checkpoint-cpt.pth"
        st = lab_save_state(enc0, base_path)
        torch.save(st, ck)
        chk = t1.labram_state(ck)                               # round trip through the adapter's own loader
        ref = enc0.lab.state_dict()
        for k_, v in chk.items():
            r_ = ref[k_].cpu()
            assert (v[:, :r_.shape[1]] if k_ == "temporal_embedding" else v).shape == r_.shape and torch.equal(
                (v[:, :r_.shape[1]] if k_ == "temporal_embedding" else v), r_), k_
        assert set(chk) == set(ref), sorted(set(chk) ^ set(ref))[:5]
    torch.save(head.state_dict(), out / "head.pt")
    h = hashlib.sha256(ck.read_bytes()).hexdigest()
    v0, v1 = curves["val"][0], curves["val"][-1]
    fell = {p.name: v1[p.name] < v0[p.name] for p in pools}
    summ = dict(status="ok", fm=a.fm, objective="masked-patch reconstruction (zeroed input patches, linear head, norm_pix MSE)",
                base_ckpt=str(base_path), base_sha256=base_sha, ckpt=str(ck), ckpt_sha256=h, args=vars(a), steps_done=n_steps,
                pools=[dict(name=p.name, label=p.label, n_train=len(p.tr), n_val=len(p.va), ch=p.ch, scale=p.scale, draw_p=float(q))
                       for p, q in zip(pools, p_draw)],
                proteus_streams=sorted(f"{u}|{v}" for u, v in pstreams), curves=curves, val_loss_fell=fell,
                elapsed_s=time.time() - t0, s_per_step=dt / n_steps)
    (out / "cpt_summary.json").write_text(json.dumps(summ, indent=1))
    log(f"done: {ck} sha256 {h[:12]}; val fell {fell}; {dt / 3600:.2f} h")


if __name__ == "__main__":
    main()
