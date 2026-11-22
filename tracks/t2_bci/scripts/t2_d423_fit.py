#!/usr/bin/env python
"""T2-D423 Stage A (pre-registered in docs/DECISIONS_T2.md): fit ReBrain's iEEG codec prior (configs/ReBrain_train_ieeg_gan.yaml) on the prepared
ds003029 runs (t2_d423_prep.py: 512 Hz, µV, good ECOG / SEEG channels), WITHOUT Lightning. models/rebrain.py's training_step, ported:
  * a batch = bs (16) windows of 4 s = 2048 samples, each from a run drawn in proportion to its duration, with nch (36) channels drawn without
    replacement (ReBrain's n_channels=36 subsampling; runs with fewer channels give all of them, so a batch has <= 576 single-channel segments);
  * discriminator step on hinge loss with the reconstruction DETACHED (Lightning's toggle_model freezes the generator in that block),
    grad clip 1, AdamW(lr d 1e-7, wd 0.01); generator step with the discriminator frozen: the Balancer over the five generator losses
    (weights r 1, l 1, s 1, f 2, a 1; rescale to total norm 1, EMA 0.999) + the RVQ commitment loss (weight 1000 inside the RVQ), grad clip 1,
    AdamW(lr g 1e-5, wd 0.01); deepspeed's FusedAdam (adam_w_mode, betas 0.9 / 0.999, eps 1e-8) -> torch.optim.AdamW with the same arguments;
  * cosine schedule with 4,000 warm-up steps over 96,000 (transformers' get_cosine_schedule_with_warmup, num_cycles 0.5, re-implemented);
  * fp32; seed 42 (the config's seed_everything); checkpoints every 10,000 steps + last; the train PRD (ReBrain's metric) logged.
    python tracks/t2_bci/scripts/t2_d423_fit.py --data <prep dir> --out <dir> [--steps 96000] [--smoke]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebrain as R  # noqa: E402
from rebrain.balancer import Balancer  # noqa: E402
from rebrain.discriminators import MultiResolutionDiscriminator  # noqa: E402
from rebrain.loss import discriminator_loss, generator_loss  # noqa: E402

L = R.SR * R.SEG_S
WEIGHTS = dict(reconstruction_loss=1.0, line_loss=1.0, spectral_loss=1.0, feature_loss=2.0, adversarial_loss=1.0)
DISC_RES = [(2048, 512, 2048), (1024, 256, 1024), (512, 128, 512), (256, 64, 256), (128, 32, 128)]


def cosine_warmup(opt, warmup, total):
    def f(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        prog = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def prd(x, y):
    return (torch.sqrt(((x - y) ** 2).sum(-1) / torch.clamp((x ** 2).sum(-1), min=1e-6)) * 100).mean()


class IEEGRuns:
    """ARM I: the prepared ds003029 runs (512 Hz, µV); a window = a run drawn in proportion to its duration, a 4 s offset, <= nch channels."""
    def __init__(self, data, smoke):
        meta = json.load(open(Path(data) / "manifest.json"))
        man = [m for m in meta["runs"] if m["n_ch"] > 0 and m["seconds"] > R.SEG_S + 1][:3 if smoke else None]
        if meta.get("packed"):                                                   # t2_d423_prep.py --pack (the sofia copy)
            flat = np.load(Path(data) / "flat.npy", mmap_mode="r" if smoke else None)
            self.runs = [flat[m["offset"]:m["offset"] + m["n_ch"] * m["T"]].reshape(m["n_ch"], m["T"]) for m in man]
        else:
            self.runs = [np.load(Path(data) / "runs" / f"{m['run']}.npy", mmap_mode="r" if smoke else None) for m in man]
        dur = np.array([r.shape[1] - L for r in self.runs], float); self.p = dur / dur.sum()

    def describe(self):
        return f"ARM I: {len(self.runs)} iEEG runs, {sum(r.shape[0] * r.shape[1] for r in self.runs) / R.SR / 3600:.1f} channel-hours"

    def batch(self, rng, bs, nch):
        segs = []
        for k in rng.choice(len(self.runs), size=bs, p=self.p):
            r = self.runs[k]; o = int(rng.integers(0, r.shape[1] - L))
            ch = rng.choice(r.shape[0], size=min(nch, r.shape[0]), replace=False)
            segs.append(np.asarray(r[np.sort(ch), o:o + L]))
        return segs


class ScalpPools:
    """ARM S (T2-D423 amendment, pre-registered): scalp EEG = the PROTEUS windows no fold scores (kit train minus seen1's val / test) + public MI exports
    (any pool named *proteus* refused). A window: a source drawn in proportion to sqrt(its windows), a window uniformly, <= nch channels, a random 4 s
    crop at the source rate, polyphase to 512 Hz (= 2048 samples). Sources with windows < 4 s, or a median window-channel std outside [1, 200] µV, are
    dropped and printed. µV as stored, no normalisation (as ReBrain)."""
    def __init__(self, proteus, pools, smoke):
        from fractions import Fraction
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import t2p_bench as B
        self.src, self.log = [], []
        X, idx, meta = B.load_export(proteus)
        (_, ktr, _, _), = B.folds(idx, "kit"); (_, _, sva, ste), = B.folds(idx, "seen1")
        rows = np.flatnonzero(ktr & ~sva & ~ste)
        sub, ses = idx["subject"].to_numpy()[rows], idx["session"].astype(str).to_numpy()[rows]
        assert set(sub) == {"P01", "P02", "P03", "P06", "P07", "P10"} and set(ses[sub == "P10"]) == {"1"}, (set(sub), set(ses[sub == "P10"]))
        cands = [("proteus_safe", X, rows, float(meta["sfreq"]))]
        for p in pools:
            p = Path(p); name = p.name
            if "proteus" in name.lower():
                self.log.append(f"{name}: REFUSED (a PROTEUS pool)"); continue
            if not (p / "X.npy").exists() or not (p / "meta.json").exists():
                self.log.append(f"{name}: dropped (no X.npy / meta.json)"); continue
            Xp = np.load(p / "X.npy", mmap_mode="r")
            cands.append((name, Xp, np.arange(len(Xp)), float(json.loads((p / "meta.json").read_text())["sfreq"])))
        for name, Xs, rr, fs in cands:
            if Xs.shape[-1] < int(round(R.SEG_S * fs)):
                self.log.append(f"{name}: dropped (windows {Xs.shape[-1] / fs:.2f} s < {R.SEG_S} s)"); continue
            pick = rr[np.linspace(0, len(rr) - 1, min(200, len(rr))).astype(int)]
            sd = float(np.median(np.asarray(Xs[pick]).std(-1)))
            if not 1.0 <= sd <= 200.0:
                self.log.append(f"{name}: dropped (median std {sd:.3g} µV outside [1, 200])"); continue
            fr = Fraction(R.SR / fs).limit_denominator(2000)
            self.src.append(dict(name=name, X=Xs, rows=rr, fs=fs, up=fr.numerator, down=fr.denominator, n0=int(round(R.SEG_S * fs)), sd=sd))
            self.log.append(f"{name}: {len(rr)} windows x {Xs.shape[1]} ch, {fs:g} Hz, {Xs.shape[-1] / fs:.2f} s, median std {sd:.1f} µV, {fr}")
        w = np.sqrt([len(d["rows"]) for d in self.src]); self.p = w / w.sum()

    def describe(self):
        return "ARM S: " + "; ".join(self.log) + " | draw p: " + ", ".join(f"{d['name']} {q:.3f}" for d, q in zip(self.src, self.p))

    def batch(self, rng, bs, nch):
        from scipy.signal import resample_poly
        segs = []
        for k in rng.choice(len(self.src), size=bs, p=self.p):
            d = self.src[k]; Xs = d["X"]
            r = int(d["rows"][rng.integers(len(d["rows"]))])
            ch = np.sort(rng.choice(Xs.shape[1], size=min(nch, Xs.shape[1]), replace=False))
            o = int(rng.integers(0, Xs.shape[-1] - d["n0"] + 1))
            seg = resample_poly(np.asarray(Xs[r, ch, o:o + d["n0"]], dtype=np.float64), d["up"], d["down"], axis=-1)
            seg = seg[:, :L] if seg.shape[-1] >= L else np.pad(seg, ((0, 0), (0, L - seg.shape[-1])), mode="edge")
            segs.append(seg.astype(np.float32))
        return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="ARM I: the prepared iEEG runs (t2_d423_prep.py)")
    ap.add_argument("--scalp", action="store_true", help="ARM S (T2-D423 amendment): fit on scalp EEG = --proteus safe windows + --pools")
    ap.add_argument("--proteus", default="", help="ARM S: the PROTEUS export (only windows no fold scores are used)")
    ap.add_argument("--pools", nargs="*", default=[], help="ARM S: public MI export dirs (X.npy + meta.json sfreq)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=96000)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--nch", type=int, default=36)
    ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true", help="20 steps on 3 runs, bs 2 x 4 channels (a code check only)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.steps, a.warmup, a.ckpt_every, a.bs, a.nch = 20, 5, 10, 2, 4
    src = ScalpPools(a.proteus, a.pools, a.smoke) if a.scalp else IEEGRuns(a.data, a.smoke)
    print(f"[fit] {src.describe()}; steps {a.steps}", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, quant = R.build_codec(a.nq)
    model, quant = model.to(dev).train(), quant.to(dev).train()
    disc = MultiResolutionDiscriminator(resolutions=DISC_RES).to(dev).train()
    opt_g = torch.optim.AdamW(list(model.parameters()) + list(quant.parameters()), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    sch_g, sch_d = cosine_warmup(opt_g, a.warmup, a.steps), cosine_warmup(opt_d, a.warmup, a.steps)
    bal = Balancer(weights=WEIGHTS)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time(); acc = {}
    for step in range(1, a.steps + 1):
        segs = src.batch(rng, a.bs, a.nch)
        x = torch.from_numpy(np.concatenate(segs)).float().to(dev)[:, None, :]          # (n, 1, L)
        q, _, commit = R.quantize(quant, model.encode(x)[0])
        y = model.decode(q)
        # discriminator step (generator frozen: the reconstruction is detached)
        for p in disc.parameters():
            p.requires_grad_(True)
        ls, _ = disc(x); lr_, _ = disc(y.detach())
        dl = discriminator_loss(ls, lr_)
        opt_d.zero_grad(set_to_none=True); dl.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0); opt_d.step(); sch_d.step()
        # generator step (discriminator frozen)
        for p in disc.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            ls, fs = disc(x)
        lr_, fr = disc(y)
        gl = generator_loss(y, x, fr, fs, lr_)
        opt_g.zero_grad(set_to_none=True)
        bal.backward(gl, y)
        commit = commit.mean(); commit.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(quant.parameters()), 1.0); opt_g.step(); sch_g.step()
        with torch.no_grad():
            cur = dict({k: float(v) for k, v in gl.items()}, commit=float(commit), disc=float(dl), prd=float(prd(x[:, 0], y[:, 0])))
        for k, v in cur.items():
            acc[k] = acc.get(k, 0.0) + v
        if step % (5 if a.smoke else 100) == 0 or step == a.steps:
            n = 5 if a.smoke else 100
            rec = dict(step=step, sec=round(time.time() - t0, 1), lr_g=sch_g.get_last_lr()[0], n_seg=int(x.shape[0]), **{k: v / n for k, v in acc.items()})
            log.write(json.dumps(rec) + "\n"); log.flush(); acc = {}
            print(f"[fit] step {step}: prd {rec['prd']:.1f} rec {rec['reconstruction_loss']:.3f} commit {rec['commit']:.3f} disc {rec['disc']:.3f} "
                  f"({rec['sec']:.0f} s, {rec['sec'] / step:.3f} s/step)", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            ck = dict(step=step, model=model.state_dict(), quantizer=quant.state_dict(), nq=a.nq, cfg=R.CFG, rvq=R.RVQ, args=vars(a))
            torch.save(ck, out / f"codec_step{step}.pt"); torch.save(ck, out / "codec_last.pt")
    print(f"[fit] DONE {a.steps} steps in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
