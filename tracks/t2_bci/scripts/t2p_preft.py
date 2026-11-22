#!/usr/bin/env python
"""Track 2 (T2-D103): SUPERVISED multi-dataset pre-fine-tune of ST-EEGFormer on labelled public BCI corpora, at the PROTEUS
fine-tune's input contract, then the usual fine-tune from it (`t2p_bench --fm-ckpt <out>/checkpoint-cpt.pth`).

Model = the fine-tune's FMClf body (t2p_fm.make_encoder: ST adapter, calib gain, every layer) + a shared LayerNorm + one
linear head PER DATASET over that dataset's native classes (so MI-only corpora still teach, and Scherer2015's MI /
subtraction / word association and Shin2017B's subtraction vs rest keep their contrasts). Each micro-batch comes from one
dataset (its own channel set through the adapter's SourceChannels, D64); datasets are drawn with p ∝ √(train windows).
Input = what the fine-tune sees: each window demeaned and divided by the causal per-session per-channel RMS
(`ni26.stream` "sz", the session's windows in time order), then the adapter's gain prep (FFT resample to 128 Hz x 4 s).

Pools (`--pool DIR:LABEL_COLUMN`, repeatable): a t2p_moabb_export store (label column `event`) or a t2p_export store
(`trig_code`). Rows with an empty label are skipped; classes = the sorted unique labels. 5 % of each pool (fixed rng) is
held out for per-dataset accuracy (sanity only, never a selector). Fixed step budget.

Output <out>/: checkpoint-cpt.pth and cpt_summary.json in t2p_cpt's format (status, fm, base_sha256 = the pinned root
checkpoint's, ckpt_sha256, args.steps, pools), so `t2p_fm.use_cpt` accepts it; the summary also records the parent
checkpoint (the CPT one, or the pinned root) and the per-dataset curves. ST: {"model": encoder (vit) state}. MOR (T2-D106): the in-house checkpoint layout, loaded strictly.

Input contract = the fine-tune's (`--align` sz | wz, `--calib` gain | zwin; T2-D103 = sz + gain for ST, T2-D106 = wz +
zwin = the D88 MOR recipe). Channels = those the FM knows (ST: its channel table; MOR: its own channel specification).
`--init-ckpt` empty = start from the pinned root checkpoint of --fm (sha-checked by t1_fm.checkpoint_sha).

T2-D108 (all default OFF = the T2-D103 / D106 behaviour): `--holdout person` holds out ~10 % of each pool's SUBJECTS
(seeded) instead of 5 % of its windows; `--save-at 8000,12000,...` saves a checkpoint at each step and, at the end, copies
the one with the best mean held-out balanced accuracy over the pools (ties -> the earliest) to checkpoint-cpt.pth (the
selection reads the pools' held-out split only, never PROTEUS); `--mae-grad-ratio r` (MOR only) adds the MOR's own
masked loss on the same batch (fp32, `--mae-mask`, decoder frozen) with a PER-TERM clip: the CE and MAE gradients are each
clipped to norm 1, then summed as g_CE + w * g_MAE, w calibrated once at init (20 batches, no update) so that the MAE
term's encoder gradient norm is r x the CE term's.

T2-D123: `--mae-only` (MOR) drops the classification term: the loss is the MOR's own masked reconstruction only
(labels unused; evaluation = the pools' held-out MAE loss with fixed masks); `--exclude-subjects P04,P09` removes those
subjects' windows from every pool (the kit evaluation's CPT must not see the leaderboard users).

  python t2p_preft.py --fm st_large --init-ckpt <CPT checkpoint-cpt.pth> --names-from <export_proteus_v1> \
      --pool <scherer>:trig_code --pool <stieger2021_v1>:event ... --out O --steps 4000
"""

import argparse
import hashlib
import json
import math
import shutil
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
import t2p_fm  # noqa: E402

t1 = t2p_fm.t1
VAL_SEED = 20261014
T_WIN = 480


class LabelPool:
    """one labelled corpus: windows (memmap), labels, causal per-session sz scales, ST-known channels."""

    def __init__(self, spec, log, known, align="sz", holdout="window", exclude=()):
        path, col = spec.rsplit(":", 1)
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        assert abs(float(meta["sfreq"]) - 120.0) < 1e-6, f"{path}: sfreq {meta['sfreq']}"
        idx = pd.read_csv(path / "index.csv", dtype=str, keep_default_na=False)
        self.name = path.name
        self.X = np.load(path / "X.npy", mmap_mode="r")
        assert self.X.shape[-1] >= T_WIN
        names = list(meta["ch_names"])
        self.align = align
        self.keep = np.array([i for i, c in enumerate(names) if c in known])
        self.ch = [names[i] for i in self.keep]
        lab = idx[col].astype(str).str.strip()
        ok = (lab != "") & (lab != "nan") & (lab != "-1")
        if exclude:                         # T2-D123: e.g. the kit test users never enter the kit evaluation's CPT
            ok &= ~idx["subject"].astype(str).str.split("/").str[-1].isin(list(exclude))
        rows = np.flatnonzero(ok.to_numpy())
        # review T2 P2e: the PROTEUS streams this pool trains on (train + held-out rows), for the --for-protocol check
        self.streams = t2p_fm.proteus_streams(idx["subject"].to_numpy()[rows], idx["session"].to_numpy()[rows]) \
            if {"subject", "session"} <= set(idx.columns) else set()
        self.classes = sorted(lab[ok].unique().tolist())
        self.y = np.full(len(idx), -1, int)
        self.y[rows] = lab[ok].map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        # causal per-session RMS scales (ni26.stream "sz" = StreamAligner: demean, then / sqrt(mean of the per-channel
        # mean squares of the session's windows so far)); stream = subject|session, order = onset / kit order
        sess = (idx["subject"] + "|" + idx["session"]) if "stream" not in idx else idx["stream"]
        order = pd.to_numeric(idx["order"] if "order" in idx else idx["onset_s"], errors="coerce").to_numpy()
        run = pd.to_numeric(idx["run"], errors="coerce").fillna(0).to_numpy() if "run" in idx else np.zeros(len(idx))
        self.scale = np.ones((len(idx), len(self.keep)), np.float32)
        for _, g in pd.DataFrame(dict(s=sess, r=run, o=order, i=np.arange(len(idx)))).groupby("s", sort=False):
            ii = g.sort_values(["r", "o"])["i"].to_numpy()
            ms = np.empty((len(ii), len(self.keep)), np.float64)
            for a in range(0, len(ii), 4096):
                x = np.asarray(self.X[np.sort(ii[a:a + 4096])], np.float64)[:, self.keep, :T_WIN]
                srt = np.argsort(np.argsort(ii[a:a + 4096]))          # back to stream order
                x = x[srt]
                x = x - x.mean(-1, keepdims=True)
                ms[a:a + len(x)] = (x ** 2).mean(-1)
            self.scale[ii] = np.sqrt(np.cumsum(ms, 0) / np.arange(1, len(ii) + 1)[:, None] + 1e-12).astype(np.float32)
        rng = np.random.default_rng(VAL_SEED)
        isv = np.zeros(len(rows), bool)
        if holdout == "person":             # T2-D108: ~10 % of the SUBJECTS never trained on; <= 1000 of their windows scored
            subj = idx["subject"].astype(str).to_numpy()[rows]
            us = np.unique(subj)
            assert len(us) >= 2, f"{path}: --holdout person needs >= 2 subjects"
            hold = rng.choice(us, size=max(1, round(0.1 * len(us))), replace=False)
            out = np.isin(subj, hold)
            self.tr = rows[~out]
            self.va = np.sort(rng.choice(rows[out], size=min(int(out.sum()), 1000), replace=False))
            self.held = sorted(hold.tolist())
        else:
            isv[rng.choice(len(rows), size=min(max(1, len(rows) // 20), 1000), replace=False)] = True
            self.tr, self.va = rows[~isv], rows[isv]
            self.held = None
        log(f"pool {self.name}: {len(self.ch)} FM-known ch (dropped {[c for c in names if c not in known]}), "
            f"{len(self.tr)} train / {len(self.va)} val (holdout {holdout}{f', subjects {self.held}' if self.held else ''}), "
            f"classes {dict(zip(self.classes, np.bincount(self.y[rows])))}")

    def batch(self, ii, rng):
        ii = np.sort(ii)
        x = np.asarray(self.X[ii], np.float32)[:, self.keep]
        if x.shape[-1] > T_WIN:                                   # random 4 s crop of a longer window
            o = rng.integers(0, x.shape[-1] - T_WIN + 1, size=len(x))
            x = np.stack([xi[:, oi:oi + T_WIN] for xi, oi in zip(x, o)])
        x = x - x.mean(-1, keepdims=True)
        if self.align == "sz":                                    # causal per-session RMS (ni26.stream "sz")
            x = x / self.scale[ii][:, :, None]
        else:                                                     # "wz": per-window per-channel z (ni26.stream "wz")
            x = x / (x.std(-1, keepdims=True) + 1e-6)
        return x.astype(np.float32), self.y[ii]


class PreFT(nn.Module):
    def __init__(self, enc, pools):
        super().__init__()
        self.enc, self.layer = enc, enc.n_layers
        self.norm = nn.LayerNorm(enc.dim)
        self.heads = nn.ModuleDict({p.name.replace(".", "_"): nn.Linear(enc.dim, len(p.classes)) for p in pools})
        for h in self.heads.values():
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def forward(self, x, pool):
        g = self.enc(self.enc.prep(x), self.layer)
        return self.heads[pool.replace(".", "_")](self.norm(g).mean((1, 2)))

    def mae(self, x, ratio):
        """T2-D108: the in-house MOR model's own masked-reconstruction loss (not part of this public release)."""
        raise NotImplementedError("the in-house MOR model is not part of this public release")


def known_channels(fm):
    """channel names the FM can place: ST = its channel table (the in-house MOR branch is not part of this public release)."""
    if fm.startswith("st_"):
        return set(json.loads((t2p_fm.REPO / "tracks" / "t1_image" / "fm" / "steeg_channels.json").read_text()))
    assert fm in t1.MOR_NAMES, f"t2p_preft: unsupported fm {fm}"
    raise NotImplementedError("the in-house MOR model is not part of this public release")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default="st_large")
    ap.add_argument("--init-ckpt", default="", help="t2p_cpt checkpoint to start from (its cpt_summary.json is checked); "
                    "empty = the pinned root checkpoint of --fm")
    ap.add_argument("--align", default="sz", choices=("sz", "wz"), help="the fine-tune's stream alignment (sz: T2-D103 ST)")
    ap.add_argument("--calib", default="gain", choices=("gain", "zwin"), help="the fine-tune's FM input calib (gain: T2-D103 ST)")
    ap.add_argument("--decision", default="T2-D103", help="decision id recorded in the summary")
    ap.add_argument("--names-from", required=True, help="the PROTEUS export (the adapter's build montage)")
    ap.add_argument("--pool", action="append", required=True, metavar="DIR:LABEL_COLUMN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr-enc", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warm", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="window", choices=("window", "person"), help="T2-D108: person = ~10 %% of subjects")
    ap.add_argument("--save-at", default="", help="T2-D108: comma steps to checkpoint; the best mean held-out BA is kept")
    ap.add_argument("--mae-grad-ratio", type=float, default=0.0,
                    help="T2-D108 arm B: target |w g_MAE| / |g_CE| (encoder, after the per-term clip) at init; 0 = off")
    ap.add_argument("--mae-mask", type=float, default=0.75, help="mask ratio of the MOR's masked loss")
    ap.add_argument("--mae-only", action="store_true", help="T2-D123 (MOR): masked-reconstruction loss only, labels unused")
    ap.add_argument("--exclude-subjects", default="", help="T2-D123: comma subject ids removed from every pool (e.g. P04,P09)")
    ap.add_argument("--for-protocol", default="kit", choices=("kit", "seen1", "kitp10", "none"),
                    help="review T2 P2e: the protocol this checkpoint will be scored on; no pool (nor the --init-ckpt chain) may "
                    "hold one of its TEST streams (t2p_cpt's seen1 assertion). none = a sealed-phase checkpoint (every window)")
    args = ap.parse_args()
    save_at = sorted({int(s) for s in args.save_at.split(",") if s.strip()})
    assert all(0 < s <= args.steps and s % args.eval_every == 0 for s in save_at), "--save-at: steps must be evaluated"
    assert args.mae_grad_ratio == 0 or args.fm in t2p_fm.t1.MOR_NAMES, "--mae-grad-ratio: MOR only"
    assert not args.mae_only or (args.fm in t2p_fm.t1.MOR_NAMES and args.mae_grad_ratio == 0 and not args.save_at), \
        "--mae-only: MOR, no --mae-grad-ratio / --save-at"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[t2p_preft {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pinned = t1.checkpoint_sha(args.fm)[1]
    if args.init_ckpt:
        t2p_fm.use_cpt(args.fm, args.init_ckpt, log)
        parent = dict(t1.CPT_INFO[args.fm])
    else:
        parent = dict(ckpt=str(t1.CKPT[args.fm]), sha256=pinned, kind="pinned root")
        log(f"init = the pinned root checkpoint {t1.CKPT[args.fm]} (sha256 {pinned[:12]})")
    root_ckpt = str(t1.CKPT[args.fm])
    kit_names = json.loads((Path(args.names_from) / "meta.json").read_text())["ch_names"]
    enc = t2p_fm.make_encoder(args.fm, kit_names, log, 0.0, args.calib)
    known = known_channels(args.fm)
    excl = tuple(v for v in args.exclude_subjects.split(",") if v)
    pools = [LabelPool(s, log, known, args.align, args.holdout, excl) for s in args.pool]
    pstreams = set().union(*[p.streams for p in pools])
    if args.init_ckpt:
        chain = t2p_fm.cpt_pool_streams(args.init_ckpt)
        assert chain is not None or args.for_protocol == "none", f"--init-ckpt {args.init_ckpt}: no pool record (leak unverifiable)"
        pstreams |= chain or set()
    if args.for_protocol != "none":
        import t2p_bench
        _, eidx, _ = t2p_bench.load_export(args.names_from)
        test = t2p_bench.protocol_test_streams(eidx, args.for_protocol)
        leak = sorted(f"{u}|{v}" for u, v in test if (u, v) in pstreams or (u, "*") in pstreams)
        assert not leak, f"--for-protocol {args.for_protocol}: its test streams {leak} are in the pools (A10)"
    log(f"PROTEUS streams in the pools: {len(pstreams)}; clean for --for-protocol {args.for_protocol}")
    net = PreFT(enc, pools).to(dev)
    groups = t1.param_groups(net, args.lr_enc, args.lr_head, 1.0, args.wd, log)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    base = [g["lr"] for g in opt.param_groups]
    p_src = np.sqrt([len(p.tr) for p in pools])
    p_src = p_src / p_src.sum()
    log("sampling " + ", ".join(f"{p.name} {q:.3f}" for p, q in zip(pools, p_src)))
    wsteps = int(args.warm * args.steps)
    curves, t0 = [], time.time()

    @torch.no_grad()
    def evaluate_mae(step):                 # T2-D123: held-out masked-reconstruction loss per pool, fixed masks (seeded)
        net.eval()
        row, vr = dict(step=step), np.random.default_rng(0)
        torch.manual_seed(12345)
        for p in pools:
            ls_ = []
            eb = max(16, round(64 * 41 / len(p.ch)))
            for a in range(0, len(p.va), eb):
                x, _ = p.batch(p.va[a:a + eb], vr)
                with torch.autocast("cuda", enabled=False), t1.source_channels(net.enc, p.ch):
                    ls_.append(float(net.mae(torch.as_tensor(x, device=dev), args.mae_mask)) * len(x))
            row[p.name] = round(sum(ls_) / max(len(p.va), 1), 5)
        torch.manual_seed(args.seed + 7 + step)
        net.train()
        curves.append(row)
        log(f"eval step {step}: held-out MAE " + ", ".join(f"{k} {v}" for k, v in row.items() if k != "step"))
        return -float(np.mean([v for k, v in row.items() if k != "step"]))

    @torch.no_grad()
    def evaluate(step):
        if args.mae_only:
            return evaluate_mae(step)
        net.eval()
        row, ba = dict(step=step), {}
        vr = np.random.default_rng(0)
        for p in pools:
            prs, ys = [], []
            eb = max(16, round(128 * 41 / len(p.ch)))
            for a in range(0, len(p.va), eb):
                x, yb = p.batch(p.va[a:a + eb], vr)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"), t1.source_channels(net.enc, p.ch):
                    prs.append(net(torch.as_tensor(x, device=dev), p.name).float().argmax(1).cpu().numpy())
                ys.append(yb)
            pr, yy = np.concatenate(prs), np.concatenate(ys)
            row[p.name] = round(float((pr == yy).mean()), 4)
            ba[p.name] = round(float(np.mean([(pr[yy == c] == c).mean() for c in np.unique(yy)])), 4)
        row["ba"], row["mean_ba"] = ba, round(float(np.mean(list(ba.values()))), 4)
        net.train()
        curves.append(row)
        log(f"eval step {step}: " + ", ".join(f"{k} {v}" for k, v in row.items() if k not in ("step", "ba", "mean_ba")) +
            f" (chance ≈ " + ", ".join(f"{1 / len(p.classes):.2f}" for p in pools) + f"); held-out BA mean {row['mean_ba']}")
        return row["mean_ba"]

    def save(ck):
        if args.fm.startswith("st_"):
            torch.save({"model": {k_: v.detach().cpu() for k_, v in net.enc.vit.state_dict().items()}}, ck)
            return
        raise NotImplementedError("the in-house MOR model is not part of this public release")   # the MOR checkpoint layout
        del src

    train_p = [q for q in net.parameters() if q.requires_grad]
    w_mae, cal = 0.0, None
    if args.mae_grad_ratio > 0:             # T2-D108 arm B: calibrate w at init (own rng: the training batches stay = arm A's)
        enc_p = [q for q in net.enc.parameters() if q.requires_grad]
        enc_norm = lambda: float(torch.linalg.vector_norm(torch.stack([q.grad.norm() for q in enc_p if q.grad is not None])))  # noqa: E731
        crng, nce, nmae = np.random.default_rng(args.seed + 1), [], []
        for _ in range(20):
            p = pools[int(crng.choice(len(pools), p=p_src))]
            bs_k = max(8, min(args.bs, round(args.bs * 41 / len(p.ch))))
            x, yb = p.batch(crng.choice(p.tr, size=min(bs_k, len(p.tr)), replace=False), crng)
            xt = torch.as_tensor(x, device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"), t1.source_channels(net.enc, p.ch):
                F.cross_entropy(net(xt, p.name).float(), torch.as_tensor(yb, device=dev), label_smoothing=args.ls).backward()
            torch.nn.utils.clip_grad_norm_(train_p, 1.0)
            nce.append(enc_norm())
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=False), t1.source_channels(net.enc, p.ch):
                net.mae(xt, args.mae_mask).backward()
            torch.nn.utils.clip_grad_norm_(train_p, 1.0)
            nmae.append(enc_norm())
        opt.zero_grad(set_to_none=True)
        w_mae = float(args.mae_grad_ratio * np.mean(nce) / np.mean(nmae))
        cal = dict(ratio=args.mae_grad_ratio, w=w_mae, enc_gnorm_ce=float(np.mean(nce)), enc_gnorm_mae=float(np.mean(nmae)))
        log(f"MAE calibration (20 batches, after the per-term clip): encoder |g| CE {cal['enc_gnorm_ce']:.3f}, "
            f"MAE {cal['enc_gnorm_mae']:.3f} -> w = {w_mae:.3f} (ratio {args.mae_grad_ratio})")
    evaluate(0)
    net.train()
    tl, tn = 0.0, 0
    tm = gc = gm = 0.0
    saved = []
    for step in range(args.steps):
        k = int(rng.choice(len(pools), p=p_src))
        p = pools[k]
        # tokens per batch held at the PROTEUS fine-tune's (41 channels x bs): wider montages get smaller batches
        bs_k = max(8, min(args.bs, round(args.bs * 41 / len(p.ch))))
        x, yb = p.batch(rng.choice(p.tr, size=min(bs_k, len(p.tr)), replace=False), rng)
        f = (step + 1) / max(wsteps, 1) if step < wsteps else 0.5 * (1 + math.cos(math.pi * (step - wsteps) / max(args.steps - wsteps, 1)))
        for gr, lr0 in zip(opt.param_groups, base):
            gr["lr"] = lr0 * f
        xt = torch.as_tensor(x, device=dev)
        if args.mae_only:                   # T2-D123: the MOR's own masked loss only (fp32, as T2-D108 arm B's MAE term)
            with torch.autocast("cuda", enabled=False), t1.source_channels(net.enc, p.ch):
                loss = net.mae(xt, args.mae_mask)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"), t1.source_channels(net.enc, p.ch):
                loss = F.cross_entropy(net(xt, p.name).float(), torch.as_tensor(yb, device=dev), label_smoothing=args.ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gc += float(torch.nn.utils.clip_grad_norm_(train_p, 1.0))
        if w_mae > 0:                       # T2-D108 arm B: per-term clip, g = clip(g_CE) + w * clip(g_MAE)
            g_ce = [q.grad for q in train_p]
            for q in train_p:
                q.grad = None
            with torch.autocast("cuda", enabled=False), t1.source_channels(net.enc, p.ch):
                lm = net.mae(xt, args.mae_mask)
            lm.backward()
            gm += float(torch.nn.utils.clip_grad_norm_(train_p, 1.0))
            for q, g in zip(train_p, g_ce):
                q.grad = g if q.grad is None else (w_mae * q.grad if g is None else g + w_mae * q.grad)
            tm += float(lm.detach())
        opt.step()
        tl += float(loss.detach())
        tn += 1
        if (step + 1) % 100 == 0:
            extra = f" | mae {tm / tn:.4f}, pre-clip |g| ce {gc / tn:.2f} mae {gm / tn:.2f}" if w_mae > 0 else ""
            log(f"step {step + 1}/{args.steps} loss {tl / tn:.4f}{extra} ({(time.time() - t0) / (step + 1):.2f} s/step)")
            tl, tn, tm, gc, gm = 0.0, 0, 0.0, 0.0, 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            m = evaluate(step + 1)
            if step + 1 in save_at:
                cks = out / f"checkpoint-step{step + 1}.pth"
                save(cks)
                saved.append(dict(step=step + 1, mean_ba=m, path=str(cks), sha256=sha256_file(cks)))
                log(f"saved {cks.name} (held-out BA mean {m}, sha256 {saved[-1]['sha256'][:12]})")
    ck = out / "checkpoint-cpt.pth"
    selected = None
    if saved:                               # T2-D108: best mean held-out BA over the pools (ties -> the earliest step)
        selected = max(saved, key=lambda r: (r["mean_ba"], -r["step"]))
        shutil.copyfile(selected["path"], ck)
        log("selection (pools' held-out BA only): " + ", ".join(f"{r['step']} {r['mean_ba']}" for r in saved) +
            f" -> step {selected['step']}")
    else:
        save(ck)
    if args.mae_only:                       # sanity like t2p_cpt: the held-out MAE loss fell for every pool
        fell = {p.name: bool(curves[-1][p.name] < curves[0][p.name]) for p in pools}
        log(f"held-out MAE fell: {fell}")
    summ = dict(status="ok", fm=args.fm, kind=(f"MAE-only CPT ({args.decision})" if args.mae_only else f"supervised pre-fine-tune ({args.decision})"),
                val_loss_fell=fell if args.mae_only else None, base_sha256=pinned,
                ckpt_sha256=sha256_file(ck), parent=parent, args=vars(args),
                pools=[dict(name=p.name, ch=p.ch, classes=p.classes, n_train=int(len(p.tr)), n_val=int(len(p.va)),
                            held_subjects=p.held, p_sample=float(q)) for p, q in zip(pools, p_src)],
                saved=saved, selected=selected, mae_calibration=cal, curves=curves, seconds=round(time.time() - t0),
                proteus_streams=sorted(f"{u}|{v}" for u, v in pstreams), for_protocol=args.for_protocol)
    (out / "cpt_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    log(f"saved {ck} (sha256 {summ['ckpt_sha256'][:12]}); {summ['seconds']} s")


if __name__ == "__main__":
    main()
