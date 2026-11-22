#!/usr/bin/env python
"""Track 2: continued MAE pretraining (CPT) of ST-EEGFormer on UNLABELLED public BCI EEG at the PROTEUS input contract,
then fine-tune from it with t2p_bench --fm-ckpt (the T1 D79 recipe, ported to 4 s windows).

Objective = the author's MAE (tracks/t1_image/fm/steeg_mae.py, vendored unmodified): masked-patch MSE on the z-scored
patches, mask ratio / lr / weight decay / betas read from the pinned checkpoint. Input = exactly the fine-tune input:
t2p_fm.STEEG4.prep with calib zwin (FFT resample of the 4 s window to 512 samples, per-window per-channel z), 1-based temporal
index, channel names through ST's table. Pools (one source per micro-batch, drawn proportional to its training windows):
  --proteus  the PROTEUS contract export, ONLY its kit train + val participants (P04 / P09 = the warm-up test never enter,
             so the leaderboard stays a clean gauge; final sealed-phase CPT may add them: public data);
  --extra    further exports (e.g. Dreyer2023 wide 0-8 s, Scherer2015): labels ignored; windows longer than 4 s give a random
             4 s crop per draw.
VAL = a fixed 5 % of each pool's windows (rng 20261013, seed-independent), never trained on, MAE loss with fixed masks at
step 0, every --eval-every steps and at the end; sanity = the val loss falls for every pool (summary val_loss_fell).
Fixed step budget (no selection on any number). T2-D109 (CPT v2; defaults OFF = the D89 behaviour): `--sampling sqrt` draws
a pool with p ∝ √(its training windows) instead of ∝ windows (the 126k-window Stieger2021 would otherwise take 63 % of the
draws); `--token-scale` holds the tokens per micro-batch at PROTEUS's (41 channels x --batch): a pool with C channels draws
min(batch, max(8, round(batch x 41 / C))) windows (Schirrmeister2017's 128 channels -> 20). Output <out>/: checkpoint-cpt.pth = {"model": the full MAE state},
cpt_summary.json (base ckpt + sha256, ckpt sha256, args, optimiser, pools, curves, throughput).

T2-D356 / T2-FM-D2 (the ST-base from-scratch insurance run, tag fm): `--store-split C's split --store-roots ... --store-src
<the in-house FM project snapshot>/src` replaces every PROTEUS / export pool with the in-house FM project's stores read through the in-house FM project's own loader
(t2p_store_pool.py: 4 s windows at 128 Hz, channel names -> ST's table, groups by montage, sqrt-of-windows source draws with the
sleep-PSG token cap); `--random-init` builds the MAE from the author's own init (seeded) with --lr / --wd / --mask-ratio / --betas
as given; `--state-every N` + `--resume` make a run resumable across walltime segments (state-cpt.pth: model + optimizer + step).

  python t2p_cpt.py --fm st_base --proteus <export_proteus_v1> --extra <dreyer wide> <scherer> --out O --steps 3000
  ... --bench 20     # time 20 optimizer steps, write bench.json, no checkpoint
"""

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tracks" / "t1_image"))
sys.path.insert(0, str(REPO / "src"))
import t2p_fm  # noqa: E402  (imports t1_fm too)
from fm import steeg_mae  # noqa: E402
from ni26.stream import car_windows  # noqa: E402

t1 = t2p_fm.t1
MAE_ARCH = {"st_small": "mae_vit_small_patch16", "st_base": "mae_vit_base_patch16", "st_large": "mae_vit_large_patch16"}
VAL_SEED = 20261013
T_WIN = 480                                                     # 4 s at 120 Hz: the PROTEUS contract window
# T2-D111 (iv): users whose every window enters (seen1 adds P04 / P10 session 1)
PROTEUS_POOLS = {"kit_tv": None, "seen1": ["P01", "P02", "P03", "P05", "P06", "P07", "P08"],
                 "seen1_out": ["P01", "P02", "P03", "P05", "P06", "P07", "P08"],
                 "all": [f"P{k:02d}" for k in range(1, 11)],      # the sealed twin (T2-D111 iv): every public window
                 "jcpt": ["P01", "P02", "P03", "P06", "P07"]}     # T2-D350 J-CPT: + P10 session 1; no typical-eval / P04 session
FS_ST, WIN_S = 128, 4.0                                         # T2-D350: ST's rate; the longest window (s) a draw keeps


def pool_keep(sub, ses, pool):
    """PROTEUS rows of a named pool (not kit_tv): its users' every window, + P04 / P10 session 1 for seen1 (T2-D111 iv); asserted free of
    seen1 test streams (P04 / P10 sessions 2-3, P09) unless the pool is 'all'. T2-D277: also the rule of --extra-proteus-pool.
    T2-D350 'jcpt': P01 P02 P03 P06 P07 + P10 session 1 (the local typical-eval users P05 / P08 / P09 / P10 s2-3 and every P04 session out)."""
    sub, ses = pd.Series(sub).reset_index(drop=True), pd.Series(ses).reset_index(drop=True)
    keep = (sub.isin(PROTEUS_POOLS[pool]) | (sub.isin(["P04", "P10"]) & (ses == "1") & (pool == "seen1"))
            | ((sub == "P10") & (ses == "1") & (pool == "jcpt"))).to_numpy()
    seen1_test = ((sub.isin(["P04", "P10"]) & ses.isin(["2", "3"])) | (sub == "P09")).to_numpy()
    assert pool == "all" or not seen1_test[keep].any(), "a seen1 test window in the CPT pool"
    assert pool != "jcpt" or not sub[keep].isin(["P04", "P05", "P08", "P09"]).any(), "a J-CPT holdout user in the pool"
    return keep


def is_proteus_extra(ie):
    """T2-D277: an --extra pool is PROTEUS-derived (the session crops) iff its subjects are Dreyer2026Proteus/<id>; other corpora with subject /
    session columns (Dreyer2023, Scherer2015) are not touched by --extra-proteus-pool; a mixed pool raises."""
    if not {"subject", "session"} <= set(ie.columns):
        return False
    m = ie["subject"].astype(str).str.startswith("Dreyer2026Proteus/")
    assert m.all() or not m.any(), "a mixed PROTEUS / other extra pool"
    return bool(m.all()) and len(m) > 0


class Pool:
    car = False                                                 # T2-D484: set per pool from --car (main)

    def __init__(self, path, rows, label, general=False):
        """general (T2-D350 J-CPT): any sampling rate and windows of 1-4 s or longer (a draw keeps min(window, 4 s), resampled by
        duration to ST's 128 Hz), >= 4 channels in ST's table (Muse); otherwise the v1 contract (120 Hz, >= 4 s, >= 8 channels)."""
        meta = json.loads((Path(path) / "meta.json").read_text())
        self.name, self.label = Path(path).name, label
        self.X = np.load(Path(path) / "X.npy", mmap_mode="r")
        self.rows = np.asarray(rows)
        names = list(meta["ch_names"])
        table = json.loads((REPO / "tracks" / "t1_image" / "fm" / "steeg_channels.json").read_text())
        self.keep = np.array([i for i, c in enumerate(names) if c in table])   # e.g. Stieger's CB1/CB2: not in ST's table
        self.ch = [names[i] for i in self.keep]
        self.dropped_ch = [c for c in names if c not in table]
        self.fs = float(meta["sfreq"])
        self.t_crop = T_WIN if not general else min(int(self.X.shape[-1]), int(round(WIN_S * self.fs)))
        self.n_out = int(round(FS_ST * self.t_crop / self.fs))          # 480 @ 120 Hz -> 512, the v1 prep's length
        if general:
            assert len(self.ch) >= 4, f"{path}: only {len(self.ch)} channels in ST's table"
            assert self.t_crop >= round(1.0 * self.fs) and self.n_out % 16 == 0, f"{path}: window {self.t_crop} @ {self.fs} Hz"
        else:
            assert len(self.ch) >= 8, f"{path}: only {len(self.ch)} channels in ST's table"
            assert abs(float(meta["sfreq"]) - 120.0) < 1e-6, f"{path}: sfreq {meta['sfreq']} != 120"
            assert self.X.shape[-1] >= T_WIN, f"{path}: windows shorter than 4 s"
        rng = np.random.default_rng(VAL_SEED)
        isv = np.zeros(len(self.rows), bool)
        # val = a fixed 5 % (capped at 2,000 windows: the large MOABB pools; every D89 pool is below the cap)
        isv[rng.choice(len(self.rows), size=min(max(1, len(self.rows) // 20), 2000), replace=False)] = True
        self.tr, self.va = self.rows[~isv], self.rows[isv]

    def draw(self, idx, rng):
        x = np.asarray(self.X[np.sort(idx)], dtype=np.float32)
        if len(self.keep) != x.shape[1]:
            x = x[:, self.keep]
        if self.car:                                            # T2-D484: the CAR view (as t2p_bench --car), before crop and prep
            x = car_windows(x)
        if x.shape[-1] > self.t_crop:                           # random 4 s crop of a longer (wide) window
            o = rng.integers(0, x.shape[-1] - self.t_crop + 1, size=len(x))
            x = np.stack([xi[:, oi:oi + self.t_crop] for xi, oi in zip(x, o)])
        return x


def spec_rows(ie, ent, log, name):
    """T2-D350: the rows of a --jcpt-spec pool: keep / drop {column: [values]} (a value matches the cell or its last '/' part), the PROTEUS
    holdout rule ('proteus_rule': a pool_keep name, for PROTEUS-derived pools), and T1's xuser4 holdout ('xuser4': {fold, seeds}: that fold's
    held-out + inner-val users of every seed, t1_train.xuser4_split). Returns (rows, a record of what was applied)."""
    rows = np.ones(len(ie), bool)
    last = lambda c: ie[c].astype(str).str.split("/").str[-1]  # noqa: E731
    for c, vals in (ent.get("keep") or {}).items():
        rows &= (ie[c].astype(str).isin(vals) | last(c).isin(vals)).to_numpy()
    for c, vals in (ent.get("drop") or {}).items():
        rows &= ~(ie[c].astype(str).isin(vals) | last(c).isin(vals)).to_numpy()
    rec = dict(keep=ent.get("keep"), drop=ent.get("drop"))
    if ent.get("proteus_rule"):
        assert is_proteus_extra(ie) or {"subject", "session"} <= set(ie.columns)
        rows &= pool_keep(last("subject").to_numpy(), ie["session"].astype(str).to_numpy(), ent["proteus_rule"])
        rec["proteus_rule"] = ent["proteus_rule"]
    if ent.get("xuser4"):
        sys.path.insert(0, str(REPO / "tracks" / "t1_image" / "scripts"))
        from t1_train import xuser4_split
        users = sorted(ie["subject"].astype(str).unique(), key=lambda s: int(s.split("/")[-1]))
        out_u = set()
        for sd in ent["xuser4"]["seeds"]:
            held, inner, _ = xuser4_split(users, int(ent["xuser4"]["fold"]), int(sd))
            out_u |= set(map(str, held)) | set(map(str, inner))
        rows &= ~ie["subject"].astype(str).isin(out_u).to_numpy()
        rec["xuser4_out"] = sorted(out_u, key=lambda s: int(s.split("/")[-1]))
    if "subject" in ie:
        rec["subjects_seen"] = sorted(ie["subject"].astype(str)[rows].unique().tolist())
    log(f"spec pool {name}: {int(rows.sum())} of {len(ie)} windows kept ({ {k: v for k, v in rec.items() if k != 'subjects_seen'} })")
    return np.flatnonzero(rows), rec


def encode(mae, x, chan, ratio, noise):
    t = mae.patch_embed(x)                                      # (B, S, C, D)
    B, S, C, D = t.shape
    t = t.reshape(B, S * C, D)
    seq = torch.arange(1, S + 1, device=x.device).view(1, S, 1).expand(B, S, C).reshape(B, S * C)
    t = t + mae.enc_temporal_emd(seq) + mae.enc_channel_emd(chan.view(1, 1, C).expand(B, S, C).reshape(B, S * C))
    keep = int(t.shape[1] * (1 - ratio))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    t = torch.gather(t, 1, ids_shuffle[:, :keep].unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([B, S * C], device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    cls = (mae.cls_token + mae.enc_temporal_emd.get_cls_token()).expand(B, -1, -1)
    t = torch.cat([cls, t], 1)
    for blk in mae.blocks:
        t = blk(t)
    return mae.norm(t), mask, ids_restore, (B, S, C, D)


def mae_loss_split(enc, dec, x, chan, ratio, noise):
    """T2-D125: the MAE loss with the ENCODER of one module (a fine-tuned ST classifier's vit: same module names) and the
    decoder of another (the CPT MAE, frozen)."""
    lat, mask, ids_restore, shp = encode(enc, x, chan, ratio, noise)
    pred = dec.forward_decoder(lat, chan.view(1, -1).expand(x.shape[0], -1), shp, ids_restore)
    return dec.forward_loss(x, pred.float(), mask)


def mae_loss(mae, x, chan, ratio, noise):
    lat, mask, ids_restore, shp = encode(mae, x, chan, ratio, noise)
    pred = mae.forward_decoder(lat, chan.view(1, -1).expand(x.shape[0], -1), shp, ids_restore)
    return mae.forward_loss(x, pred.float(), mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default="st_base", choices=sorted(MAE_ARCH))
    ap.add_argument("--proteus", default="", help="the PROTEUS export (required unless --store-split)")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=None, help="default = the checkpoint optimizer's lr")
    ap.add_argument("--wd", type=float, default=None, help="only for a weights-only checkpoint (no optimizer state)")
    ap.add_argument("--mask-ratio", type=float, default=None, help="only for a weights-only checkpoint (no args)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--sampling", default="prop", choices=("prop", "sqrt"), help="T2-D109: pool draw p ∝ windows | √windows")
    ap.add_argument("--token-scale", action="store_true", help="T2-D109: windows per micro-batch scaled by 41 / channels")
    ap.add_argument("--extra-exclude-subjects", default="", help="T2-D129: comma subject ids (e.g. P04,P09) dropped from every "
                    "--extra pool that has a subject column (the continuous-session PROTEUS pool of the kit evaluation)")
    ap.add_argument("--proteus-pool", default="kit_tv", choices=sorted(PROTEUS_POOLS),
                    help="PROTEUS windows in the pool (T2-D111 iv: seen1 / seen1_out = with / without the seen1 protocol's "
                    "seen users' session-1 windows; neither holds a seen1 test window)")
    ap.add_argument("--extra-proteus-pool", default="", choices=["", "seen1", "seen1_out", "all", "jcpt"],
                    help="T2-D277: the --proteus-pool rule (pool_keep) applied to every --extra pool with subject + session columns "
                    "(the continuous-session crops of the seen1 evaluation); '' = off")
    ap.add_argument("--jcpt-spec", default="", help="T2-D350 J-CPT: a JSON {pools: [{path, group, keep?, drop?, proteus_rule?, xuser4?}], "
                    "group_weights: {group: w}, proteus_group}: the cross-track pools (any rate, 1-4 s+ windows) with their holdouts; the "
                    "draw p of a pool = its group's weight x sqrt(its training windows) / the group's sum; replaces --extra / --sampling")
    ap.add_argument("--save-at", default="", help="T2-D350: comma steps; at each, <out>/ck_<save-tag>_s<step:05d>/{checkpoint-cpt.pth, "
                    "cpt_summary.json (status ok, args.steps = that step)}, a --fm-ckpt-ready checkpoint")
    ap.add_argument("--save-tag", default="j", help="T2-D350: the ck_ dir infix (t2p_bench's cpt_tag reads the dir's last 12 characters)")
    ap.add_argument("--schedule", default="cosine", choices=("cosine", "const"),
                    help="T2-D350: 'const' = warm-up then the peak lr held (intermediate --save-at checkpoints = shorter CPTs, WSD-style)")
    ap.add_argument("--random-init", action="store_true", help="T2-D356: the MAE from the author's own random init (seeded by --seed) "
                    "instead of the pinned checkpoint; --lr / --wd / --mask-ratio / --betas are then required / used as given; the "
                    "summary says random_init (t2p_fm.use_cpt skips only the pinned-base sha check for it)")
    ap.add_argument("--betas", default="0.9,0.999", help="--random-init only: AdamW betas (the pinned ST-base's optimizer: 0.9, 0.999)")
    ap.add_argument("--store-split", default="", help="T2-D356: the in-house FM project's split parquet (C's frozen split): the pool = its pool_paths "
                    "train / tier-1 val recordings, read through the in-house FM project's loader (t2p_store_pool.py); replaces --proteus / --extra")
    ap.add_argument("--store-roots", default="", help="comma store roots (the in-house FM project's --data), basenames distinct")
    ap.add_argument("--store-src", default="", help="the the in-house FM project snapshot's src/ (inhouse_fm on PYTHONPATH)")
    ap.add_argument("--store-masks", default="", help="default: <snapshot>/docs/data/pool_masks.json (must exist)")
    ap.add_argument("--store-hop", type=float, default=4.0, help="window hop (s) of the store index (windows are 4 s)")
    ap.add_argument("--store-min-ch", type=int, default=4, help="a recording needs >= this many channels in ST's table")
    ap.add_argument("--store-val-per-source", type=int, default=128, help="val windows per source (tier-1 recordings, cached once)")
    ap.add_argument("--store-sleep-share", type=float, default=0.25, help="cap on the sleep-PSG sources' delivered token share")
    ap.add_argument("--store-workers", type=int, default=16, help="DataLoader workers reading the store")
    ap.add_argument("--store-index-workers", type=int, default=0, help="the in-house FM project's metadata scan workers (0 = its default)")
    ap.add_argument("--state-every", type=int, default=0, help="T2-FM: write <out>/state-cpt.pth (model + optimizer + step) every N "
                    "steps and on USR1 / TERM (then exit 3 instead of a partial checkpoint); --resume continues from it")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state-cpt.pth (same recipe asserted; the data order "
                    "after the resume point differs, the lr schedule is exact)")
    ap.add_argument("--init-ckpt", default="", help="T2-FM-D6: start from a t2p_cpt checkpoint (the full MAE state; e.g. the shipped CPT v1) "
                    "instead of the pinned file; its summary must be status ok, the same --fm, base = the pinned sha; its PROTEUS "
                    "streams join this run's (the leak guard sees the lineage)")
    ap.add_argument("--store-sources", default="", help="T2-FM-D6: comma sources (the split's _source / dataset) kept in the store pool")
    ap.add_argument("--sup-dir", default="", help="T2-FM-D6: comma dirs of the in-house FM project's P1 source files (<name>.h5); enables the P1 term")
    ap.add_argument("--sup-src", default="", help="the the in-house FM project snapshot src/ for the P1 sampler + heads (default --store-src)")
    ap.add_argument("--sup-sources", default="", help="comma P1 source names (an EXPLICIT list; default t2p_p1.P1_SOURCES)")
    ap.add_argument("--sup-groups", default="", help="family-first group weights (default t2p_p1.P1_GROUPS, the in-house FM project's arm 2)")
    ap.add_argument("--sup-weight", type=float, default=0.02, help="lambda of the supervised term (P1 v2_shared: 0.02)")
    ap.add_argument("--sup-shift-weight", type=float, default=1.0)
    ap.add_argument("--sup-term-clip", type=float, default=2.0, help="the supervised encoder gradient <= c x the MAE one's norm")
    ap.add_argument("--sup-lr", type=float, default=1e-3, help="the heads' peak lr (the same schedule factor as the MAE)")
    ap.add_argument("--sup-token-budget", type=int, default=8192)
    ap.add_argument("--sup-max-channels", type=int, default=32)
    ap.add_argument("--sup-mix", type=int, default=3)
    ap.add_argument("--sup-holdout-every", type=int, default=2500)
    ap.add_argument("--sup-holdout-windows", type=int, default=256)
    ap.add_argument("--sup-no-hash", action="store_true", help="tests only: skip the P1 files' sha256 manifest")
    ap.add_argument("--car", action="store_true", help="T2-D484: common average reference (ni26.stream.car_windows, the window's good "
                    "channels) on every pool's draw, after the channel keep and before the crop / prep: a CPT that sees the CAR view")
    args = ap.parse_args()
    assert not (args.car and args.store_split), "--car: PROTEUS / export pools only"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def log(m):
        print(f"[t2p_cpt {time.time() - t0:6.0f}s] {m}", flush=True)

    store = None
    if args.store_split:                                        # T2-D356 / T2-FM-D2: the in-house FM project's stores through the in-house FM project's loader
        assert not (args.proteus or args.extra or args.jcpt_spec), \
            "--store-split replaces --proteus / --extra / --jcpt-spec (an fm-tag job reads fm data only)"
        from t2p_store_pool import StorePool
        store = StorePool(args.store_split, args.store_roots, args.store_src, log, masks=args.store_masks or None,
                          hop_s=args.store_hop, min_ch=args.store_min_ch, val_per_source=args.store_val_per_source,
                          sleep_share=args.store_sleep_share, batch=args.batch, token_scale=args.token_scale,
                          index_workers=args.store_index_workers or None,
                          sources=[v for v in args.store_sources.split(",") if v] or None)
        spec, pools, groups, recs = None, [], [], []
        pstreams = set(store.proteus_streams)
        chans = [torch.as_tensor(t1.STEEG.chan_spec(g["ch"])).to(dev) for g in store.groups]
        p_draw, mb = store.p_draw, store.mb
    else:
        assert args.proteus, "--proteus (or --store-split)"
        idx = pd.read_csv(Path(args.proteus) / "index.csv", dtype={"subject": str, "session": str})
        sub, ses = idx["subject"].str.split("/").str[-1], idx["session"]
        kit_tv = np.flatnonzero(idx["split"].isin(["train", "val"]).to_numpy())
        assert not idx.iloc[kit_tv]["split"].eq("test").any()
        if args.proteus_pool == "kit_tv":
            rows = kit_tv
        else:
            rows = np.flatnonzero(pool_keep(sub, ses, args.proteus_pool))
            assert args.proteus_pool != "all" or len(rows) == len(idx)
        log(f"PROTEUS pool {args.proteus_pool}: {len(rows)} windows, " +
            ", ".join(f"{k}: {v}" for k, v in pd.Series(sub[rows] + "/" + ses[rows]).value_counts().sort_index().items()))
        spec = json.loads(Path(args.jcpt_spec).read_text()) if args.jcpt_spec else None
        assert not (spec and (args.extra or args.sampling != "prop")), "--jcpt-spec replaces --extra / --sampling"
        pools = [Pool(args.proteus, rows, "proteus_kit_train_val" if args.proteus_pool == "kit_tv" else f"proteus_{args.proteus_pool}",
                      general=bool(spec))]
        excl = [v for v in args.extra_exclude_subjects.split(",") if v]
        pstreams = t2p_fm.proteus_streams(idx["subject"].to_numpy()[rows], ses.to_numpy()[rows])
        groups, recs = [spec["proteus_group"] if spec else None], [dict(proteus_pool=args.proteus_pool)]
        for ent in (spec or {}).get("pools", []):                  # T2-D350: the J-CPT pools with their holdouts
            ie = pd.read_csv(Path(ent["path"]) / "index.csv", dtype=str)
            rows_e, rec = spec_rows(ie, ent, log, Path(ent["path"]).name)
            pools.append(Pool(ent["path"], rows_e, f"jcpt_{ent['group']}", general=True))
            groups.append(ent["group"])
            recs.append(rec)
            if is_proteus_extra(ie):
                pstreams |= t2p_fm.proteus_streams(ie["subject"].to_numpy()[rows_e], ie["session"].to_numpy()[rows_e])
        for e in args.extra:
            ie = pd.read_csv(Path(e) / "index.csv", dtype=str)
            rows_e = np.arange(len(ie))
            if excl and "subject" in ie:
                rows_e = np.flatnonzero(~ie["subject"].str.split("/").str[-1].isin(excl).to_numpy())
                log(f"extra {Path(e).name}: {len(ie) - len(rows_e)} windows of {excl} excluded")
            if args.extra_proteus_pool and is_proteus_extra(ie):   # T2-D277: the seen1 rule on the session crops only
                k_ = pool_keep(ie["subject"].str.split("/").str[-1].to_numpy(), ie["session"].to_numpy(), args.extra_proteus_pool)
                n_ = len(rows_e)
                rows_e = rows_e[k_[rows_e]]
                log(f"extra {Path(e).name}: pool rule {args.extra_proteus_pool}: {len(rows_e)} of {n_} windows kept, " + ", ".join(
                    f"{k}: {v}" for k, v in pd.Series(ie["subject"].str.split("/").str[-1].to_numpy()[rows_e] + "/" + ie["session"].to_numpy()[rows_e])
                    .value_counts().sort_index().items()))
            pools.append(Pool(e, rows_e, "extra_unlabelled"))
            groups.append(None)                                     # T2-D484: keeps the summary's zip over every pool (since
            recs.append({})                                         # T2-D350 it listed only PROTEUS when --extra was given)
            if {"subject", "session"} <= set(ie.columns):          # review T2 P2e: PROTEUS-derived extras (continuous sessions)
                pstreams |= t2p_fm.proteus_streams(ie["subject"].to_numpy()[rows_e], ie["session"].to_numpy()[rows_e])
        for p in pools:
            p.car = bool(args.car)
        chans = [torch.as_tensor(t1.STEEG.chan_spec(p.ch)).to(dev) for p in pools]
        ntr = np.array([len(p.tr) for p in pools], float)
        p_draw = np.sqrt(ntr) if args.sampling == "sqrt" else ntr
        if spec:                                                    # T2-D350: group weight x sqrt(n) within the group
            gw = spec["group_weights"]
            assert set(groups) <= set(gw), (sorted(set(groups)), gw)
            sq = np.sqrt(ntr)
            p_draw = np.array([gw[g] * sq[k] / sum(sq[j] for j in range(len(pools)) if groups[j] == g) for k, g in enumerate(groups)])
        p_draw = p_draw / p_draw.sum()
        mb = [min(args.batch, max(8, round(args.batch * 41 / len(p.ch)))) if args.token_scale else args.batch for p in pools]
        for p, q, b in zip(pools, p_draw, mb):
            log(f"pool {p.name} ({p.label}): {len(p.ch)} ch, {len(p.tr)} train / {len(p.va)} val windows of {p.X.shape[-1]} samples; "
                f"draw p {q:.3f}, micro-batch {b}")

    if args.random_init:                                        # T2-D356: from scratch (the author's initialize_weights, seeded)
        if args.lr is None or args.wd is None or args.mask_ratio is None:
            raise SystemExit("--random-init: pass --lr / --wd / --mask-ratio")
        ck_path, base_sha = None, None
        mae = steeg_mae.__dict__[MAE_ARCH[args.fm]](norm_pix_loss=False)
        peak, wd, ratio = args.lr, args.wd, args.mask_ratio
        betas = tuple(float(v) for v in args.betas.split(","))
    else:
        ck_path, base_sha = t1.checkpoint_sha(args.fm)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        mae = steeg_mae.__dict__[MAE_ARCH[args.fm]](norm_pix_loss=False)
        mae.load_state_dict(sd, strict=True)                       # encoder AND decoder: CPT needs both
        ca = ck.get("args")
        ca = vars(ca) if ca is not None and not isinstance(ca, dict) else (ca or {})
        og = (ck.get("optimizer") or {}).get("param_groups", [{}])[0]
        peak = args.lr if args.lr is not None else og.get("lr")
        wd, ratio, betas = og.get("weight_decay"), ca.get("mask_ratio"), tuple(og.get("betas", (0.9, 0.999)))
        for name, got, flag, val in (("weight decay", wd, "--wd", args.wd), ("mask ratio", ratio, "--mask-ratio", args.mask_ratio)):
            if got is not None and val is not None and float(got) != val:
                raise SystemExit(f"{ck_path} has {name} {got}; {flag} {val} would silently override it")
        wd = wd if wd is not None else args.wd
        ratio = ratio if ratio is not None else args.mask_ratio
        if peak is None or wd is None or ratio is None:
            raise SystemExit(f"{ck_path}: no lr / wd / mask ratio in the checkpoint; pass --lr / --wd / --mask-ratio")
        del ck
    init_rec = None
    if args.init_ckpt:                                          # T2-FM-D6: continue from a t2p_cpt checkpoint (the shipped CPT v1)
        ip = Path(args.init_ckpt)
        isum = json.loads((ip.parent / "cpt_summary.json").read_text())
        ih = hashlib.sha256(ip.read_bytes()).hexdigest()
        assert isum.get("status") == "ok" and isum.get("fm") == args.fm, f"--init-ckpt summary {isum.get('status')} {isum.get('fm')}"
        assert isum.get("ckpt_sha256") == ih, f"--init-ckpt: file sha {ih[:12]} != its summary's {str(isum.get('ckpt_sha256'))[:12]}"
        assert not args.random_init and isum.get("base_sha256") == base_sha, "--init-ckpt: its base is not this run's pinned base"
        mae.load_state_dict(torch.load(ip, map_location="cpu", weights_only=False)["model"], strict=True)
        istreams = t2p_fm.cpt_pool_streams(ip)
        assert istreams is not None, "--init-ckpt: no PROTEUS stream record"
        pstreams |= set(istreams)
        init_rec = dict(path=str(ip), sha256=ih, steps=isum.get("steps_done"), pools=[q["name"] for q in isum.get("pools", [])],
                        proteus_streams=sorted(f"{u}|{v}" for u, v in istreams))
        log(f"--init-ckpt {ip} (sha256 {ih[:12]}; {init_rec['steps']} steps on {init_rec['pools']}); its PROTEUS streams joined")
    mae = mae.to(dev)
    sup = None
    if args.sup_dir:                                            # T2-FM-D6: the in-house FM project's P1 v2_shared term on ST features
        from t2p_p1 import P1_GROUPS, P1_SOURCES, STSup, encoder_params
        sup = STSup(args.sup_src or args.store_src, [d for d in args.sup_dir.split(",") if d],
                    [v for v in args.sup_sources.split(",") if v] or list(P1_SOURCES), args.sup_groups or P1_GROUPS, log,
                    D=mae.cls_token.shape[-1], seed=args.seed, token_budget=args.sup_token_budget, max_channels=args.sup_max_channels,
                    mix=args.sup_mix, shift_weight=args.sup_shift_weight, hash_files=not args.sup_no_hash)
        sup.heads.to(dev)
        enc_params = encoder_params(mae)
        sup_curve, sup_hold = [], []
    prep = t2p_fm.STEEG4.prep.__get__(type("P", (), dict(FS=t1.STEEG.FS, calib="zwin", CLIP_GAIN=t1.STEEG.CLIP_GAIN))())
    assert t1.STEEG.FS == FS_ST

    def pp(x, k):                   # T2-D350: by duration to 128 Hz for the J-CPT pools (= prep for a 4 s 120 Hz window)
        if store is not None:       # T2-D356: the store windows are already 4 s at 128 Hz (512 samples): per-window z only
            return t1.zwin(x.float(), 1e9)
        return t1.zwin(t1.fft_resample(x, pools[k].n_out), 1e9) if spec else prep(x)
    opt = torch.optim.AdamW([dict(params=list(mae.parameters()), lr_scale=1.0)], lr=peak, weight_decay=float(wd), betas=betas)
    if sup is not None:                                         # the heads: their own peak lr, the same schedule factor
        opt.add_param_group(dict(params=list(sup.heads.parameters()), lr=args.sup_lr, lr_scale=args.sup_lr / peak))
    warm = int(args.warmup_frac * args.steps)
    log(f"MAE {MAE_ARCH[args.fm]} from {ck_path or 'RANDOM INIT (seed %d)' % args.seed} (sha256 {(base_sha or '-')[:12]}): "
        f"lr {peak:.3g}, wd {wd}, betas {betas}, mask {ratio}, "
        f"batch {args.batch} x accum {args.accum} = {args.batch * args.accum}, {args.steps} steps, warm-up {warm}")

    vgen = torch.Generator(device="cpu").manual_seed(VAL_SEED)
    vnoise = {}

    @torch.no_grad()
    def evaluate():
        mae.eval()
        res = {}
        for s, chunks in (store.val.items() if store is not None else ()):    # T2-D356: tier-1 recordings, cached windows
            losses, w = [], []
            for ci, (names, W) in enumerate(chunks):
                ch = torch.as_tensor(t1.STEEG.chan_spec(names)).to(dev)
                b = min(args.batch, max(8, round(args.batch * 41 / len(names)))) if args.token_scale else args.batch
                for s0 in range(0, len(W), b):
                    x = pp(torch.as_tensor(W[s0:s0 + b]).to(dev), None)
                    key = (s, ci, s0)
                    if key not in vnoise:
                        vnoise[key] = torch.rand(len(x), (x.shape[-1] // 16) * x.shape[1], generator=vgen)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        losses.append(float(mae_loss(mae, x, ch, ratio, vnoise[key].to(dev))))
                    w.append(len(x))
            res[s] = float(np.average(losses, weights=w))
        for k, p in enumerate(pools):
            losses, w = [], []
            vr = np.random.default_rng(VAL_SEED + k)
            for s in range(0, len(p.va), mb[k]):
                x = torch.as_tensor(pp(torch.as_tensor(p.draw(p.va[s:s + mb[k]], vr)), k)).to(dev)
                key = (k, s)
                if key not in vnoise:
                    S = x.shape[-1] // 16
                    vnoise[key] = torch.rand(len(x), S * x.shape[1], generator=vgen)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    losses.append(float(mae_loss(mae, x, chans[k], ratio, vnoise[key].to(dev))))
                w.append(len(x))
            res[p.name] = float(np.average(losses, weights=w))
        mae.train()
        return res

    step0, resumed = 0, None
    state_path = out / "state-cpt.pth"
    fp = dict(fm=args.fm, random_init=bool(args.random_init), base_sha256=base_sha, steps=args.steps, lr=peak, wd=wd, mask=ratio,
              betas=list(betas), batch=args.batch, accum=args.accum, schedule=args.schedule, warmup_frac=args.warmup_frac,
              min_lr=args.min_lr, seed=args.seed, token_scale=args.token_scale, save_at=args.save_at, proteus=args.proteus,
              jcpt_spec=args.jcpt_spec, store_split_sha256=store.record["split_sha256"] if store is not None else None,
              init_ckpt_sha256=init_rec["sha256"] if init_rec else None, store_sources=args.store_sources,
              sup=None if sup is None else dict(sources=list(sup.paths), groups=args.sup_groups, weight=args.sup_weight,
                                                  clip=args.sup_term_clip, lr=args.sup_lr, mix=args.sup_mix, budget=args.sup_token_budget),
              **({"car": True} if args.car else {}))
    if args.resume:                                             # T2-FM: continue a run from its last state
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["fingerprint"] != fp:
            raise SystemExit(f"--resume: the recipe differs from {state_path}: "
                             f"{ {k: (st['fingerprint'].get(k), v) for k, v in fp.items() if st['fingerprint'].get(k) != v} }")
        mae.load_state_dict(st["model"])
        if sup is not None:
            sup.heads.load_state_dict(st["sup_heads"])
            sup_curve, sup_hold = st.get("sup_curve", []), st.get("sup_hold", [])
        opt.load_state_dict(st["optimizer"])
        step0, curve, train_curve = int(st["step"]), st["curve"], st["train_curve"]
        resumed = dict(step=step0, state=str(state_path), state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest())
        rng = np.random.default_rng((args.seed, step0))
        log(f"RESUMED at step {step0} from {state_path} (sha256 {resumed['state_sha256'][:12]})")
        del st
    else:
        curve = [dict(step=0, val=evaluate())]
        log(f"val @0 {curve[0]['val']}")
        train_curve = []
    tstep, t_load = time.time(), {"s": 0.0}
    mae.train()
    n_steps = args.bench or args.steps
    it = store.stream((args.seed, step0), args.store_workers) if store is not None else None
    if sup is not None:
        sup.start(step0)
        if not args.resume:
            sup_hold.append(dict(step=0, holdout=sup.holdout(mae, dev, args.sup_holdout_windows)))
            log(f"P1 holdout @0 { {k: round(v.get('balacc', float('nan')), 3) for k, v in sup_hold[-1]['holdout'].items()} }")

    def save_state(step_done):
        tmp = out / "state-cpt.pth.tmp"
        torch.save(dict(fingerprint=fp, step=step_done, model=mae.state_dict(), optimizer=opt.state_dict(), curve=curve,
                        train_curve=train_curve, **({} if sup is None else dict(sup_heads=sup.heads.state_dict(), sup_curve=sup_curve,
                                                                                  sup_hold=sup_hold))), tmp)
        tmp.replace(state_path)
        log(f"state saved at step {step_done} -> {state_path}")
    stop = {}                               # review T2 P3i: USR1 / TERM (t2p_cpt.slurm forwards USR1@600 as TERM) -> save, exit

    def _on_signal(signum, frame):
        stop["sig"] = signal.Signals(signum).name

    for sg in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sg, _on_signal)

    def summarise(steps_done):
        done = max(steps_done - step0, 1)
        sm = dict(fm=args.fm, base_ckpt=ck_path, base_sha256=base_sha, random_init=bool(args.random_init), args=vars(args),
                  lr_peak=peak, weight_decay=wd, betas=list(betas), mask_ratio=ratio, s_per_step=(time.time() - tstep) / done,
                  s_load_per_step=t_load["s"] / done,
                  peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None,
                  pools=[dict(name=p.name, label=p.label, ch=p.ch, dropped_ch=p.dropped_ch, n_train=len(p.tr), n_val=len(p.va), car=bool(p.car),
                              win=int(p.X.shape[-1]), p_draw=float(q), micro_batch=int(b),
                              **(dict(fs=p.fs, t_crop=p.t_crop, n_out=p.n_out, group=g, holdout=r) if spec else {}))
                         for p, q, b, g, r in zip(pools, p_draw, mb, groups, recs)],
                  proteus_streams=sorted(f"{u}|{v}" for u, v in pstreams), val_curve=curve, train_curve=train_curve[::10],
                  steps_done=steps_done, resumed_from=resumed, init_ckpt=init_rec)
        if sup is not None:
            sm["sup"] = dict(sup.record(), weight=args.sup_weight, term_clip=args.sup_term_clip, lr=args.sup_lr, curve=sup_curve[::10],
                             holdout=sup_hold)
        if store is not None:                                   # T2-D356: one pool entry per source (+ its montage groups)
            sm["pools"] = [dict(name=s, label="store", n_train=int(sum(g["n_windows"] for g in store.groups if g["source"] == s)),
                                n_val=v["n_val"], p_draw=v["p_draw"], split_train_h=v["split_train_h"], indexed_h=v["indexed_h"],
                                groups=[dict(ch=g["ch"], n_windows=g["n_windows"], p_draw=g["p_draw"], micro_batch=g["micro_batch"])
                                        for g in store.groups if g["source"] == s])
                           for s, v in store.record["sources"].items() if v["recordings_kept"]]
            sm["store"] = store.record
        return sm

    def save(sm, name):
        torch.save({"model": {k: v.detach().cpu() for k, v in mae.state_dict().items()}}, out / name)
        h = hashlib.sha256()
        with open(out / name, "rb") as fh:
            while b := fh.read(1 << 24):
                h.update(b)
        sm["ckpt_sha256"] = h.hexdigest()
        ((out / name).parent / "cpt_summary.json").write_text(json.dumps(sm, indent=1))    # beside its checkpoint (use_cpt reads it there)
        return h.hexdigest()

    save_at = {int(v) for v in args.save_at.split(",") if v}
    assert all(0 < v <= args.steps for v in save_at), args.save_at
    for step in range(step0, n_steps):
        f = (step + 1) / max(warm, 1) if step < warm else (
            1.0 if args.schedule == "const" else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(args.steps - warm, 1))))
        for g in opt.param_groups:
            g["lr"] = g.get("lr_scale", 1.0) * (args.min_lr + (peak - args.min_lr) * f if step >= warm else peak * f)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            if store is not None:                               # T2-D356: a micro-batch from the store workers
                tl = time.time()
                k, xb = next(it)
                t_load["s"] += time.time() - tl
                x, pname = pp(xb.to(dev, non_blocking=True), k), store.groups[k]["source"]
            else:
                k = int(rng.choice(len(pools), p=p_draw))
                p = pools[k]
                x, pname = torch.as_tensor(pp(torch.as_tensor(p.draw(rng.choice(p.tr, size=mb[k], replace=False), rng)), k)).to(dev), p.name
            noise = torch.rand(len(x), (x.shape[-1] // 16) * x.shape[1], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = mae_loss(mae, x, chans[k], ratio, noise) / args.accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} (pool {pname})")
            loss.backward()
            tot += float(loss.detach())
        if sup is not None:                                     # T2-FM-D6: the in-house FM project's P1 term + its per-term clip (encoder-only norms)
            g_mae = [None if q.grad is None else q.grad.detach().clone() for q in enc_params]
            sb = sup.stream.get(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                sl, sst = sup.loss(mae, sb, dev)
            if not torch.isfinite(sl):
                raise SystemExit(f"non-finite P1 loss at step {step}")
            (args.sup_weight * sl).backward()
            n_mae = math.sqrt(sum(float(g.double().pow(2).sum()) for g in g_mae if g is not None))
            gs = [None if q.grad is None else (q.grad - (0 if g is None else g)) for q, g in zip(enc_params, g_mae)]
            n_sup = math.sqrt(sum(float(g.double().pow(2).sum()) for g in gs if g is not None))
            sc = min(1.0, args.sup_term_clip * n_mae / n_sup) if n_sup > 0 else 1.0
            if sc < 1.0:
                for q, g, d in zip(enc_params, g_mae, gs):
                    if d is not None:
                        q.grad.copy_((0 if g is None else g) + sc * d)
            sup_curve.append(dict(step=step + 1, loss=float(sl.detach()), scale=sc, ratio=n_sup / max(n_mae, 1e-12),
                                  acc={k: round(v["acc"], 3) for k, v in sst.items()}))
        opt.step()
        if sup is not None and not args.bench and (step + 1) % args.sup_holdout_every == 0:
            sup_hold.append(dict(step=step + 1, holdout=sup.holdout(mae, dev, args.sup_holdout_windows)))
            log(f"P1 holdout @{step + 1} { {k: round(v.get('balacc', float('nan')), 3) for k, v in sup_hold[-1]['holdout'].items()} }")
        train_curve.append(dict(step=step + 1, loss=tot))
        if stop.get("sig") and not args.bench and args.state_every:      # T2-FM: resumable -> the state, exit 3
            save_state(step + 1)
            log(f"{stop['sig']}: state at step {step + 1}/{n_steps}; exit 3 (continue with --resume)")
            sys.exit(3)
        if stop.get("sig") and not args.bench:     # a PARTIAL checkpoint: use_cpt refuses status != ok until promoted by hand
            sm = summarise(step + 1)
            sm["status"] = "partial"                # the cosine schedule did not finish
            sm["stop"] = dict(signal=stop["sig"], step=step + 1, of=n_steps)
            sha = save(sm, "checkpoint-cpt-partial.pth")
            log(f"{stop['sig']}: saved checkpoint-cpt-partial.pth at step {step + 1}/{n_steps} (sha256 {sha[:12]}); exit 3")
            sys.exit(3)
        if (step + 1) % 50 == 0:
            log(f"step {step + 1}/{n_steps} loss {np.mean([c['loss'] for c in train_curve[-50:]]):.4f} "
                f"({(time.time() - tstep) / (step + 1 - step0):.2f} s/step, load {t_load['s'] / (step + 1 - step0):.2f})"
                + ("" if sup is None else f"; P1 loss {np.mean([c['loss'] for c in sup_curve[-50:]]):.4f} "
                   f"clip scale {np.mean([c['scale'] for c in sup_curve[-50:]]):.3f} sup/mae {np.mean([c['ratio'] for c in sup_curve[-50:]]):.3f}"))
        if not args.bench and ((step + 1) % args.eval_every == 0 or step + 1 == n_steps or step + 1 in save_at):
            curve.append(dict(step=step + 1, val=evaluate()))
            log(f"val @{step + 1} {curve[-1]['val']}")
        if not args.bench and step + 1 in save_at:            # T2-D350: a --fm-ckpt-ready intermediate checkpoint
            sm = summarise(step + 1)
            sm["args"] = dict(sm["args"], steps=step + 1, steps_planned=args.steps)
            sm["val_loss_fell"] = {k: curve[-1]["val"][k] < curve[0]["val"][k] for k in curve[0]["val"]}
            sm["status"] = "ok"
            sub = out / f"ck_{args.save_tag}_s{step + 1:05d}"
            sub.mkdir(exist_ok=True)
            sha = save(sm, sub / "checkpoint-cpt.pth")
            log(f"saved {sub.name}/checkpoint-cpt.pth at step {step + 1} (sha256 {sha[:12]})")
        if not args.bench and args.state_every and (step + 1) % args.state_every == 0 and step + 1 < n_steps:
            save_state(step + 1)
    summary = summarise(n_steps)
    sps = summary["s_per_step"]
    if args.bench:
        (out / "bench.json").write_text(json.dumps(summary, indent=1))
        log(f"BENCH {sps:.2f} s/step (load wait {summary['s_load_per_step']:.2f}), peak {summary['peak_mem_gb']} GB -> "
            f"{args.steps} steps ≈ {sps * args.steps / 3600:.2f} h")
        return
    summary["val_loss_fell"] = {k: curve[-1]["val"][k] < curve[0]["val"][k] for k in curve[0]["val"]}
    summary["status"] = "ok"
    sha = save(summary, "checkpoint-cpt.pth")
    log(f"DONE ckpt sha256 {sha[:12]}; val fell {summary['val_loss_fell']}")


if __name__ == "__main__":
    main()
