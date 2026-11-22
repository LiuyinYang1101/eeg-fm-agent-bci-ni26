"""T2-FM-D6: the in-house FM project's P1 v2_shared supervised stage, ported to ST-EEGFormer inside t2p_cpt.py (`--sup-*`).

What is REUSED unchanged from the in-house FM project's fork snapshot (on PYTHONPATH, read only): `SupSource` (the labelled source files,
their person holdout), `SupSamplerV2` (family-first mixed batches, post-cue motor crops, total-token budget, the per-step
deterministic draw), `SupStream` (prefetch) and `SupHeadsV2` with arm v2_shared (ONE masked-CE head per harmonised label
space: motor / erp / ssvep, per-source heads otherwise, a shift head per wide source; PooledTemporalMLP on the channel-mean
final grid, time kept; its own init generator).
What is ST-specific here (the in-house FM project's facts, 2026-11-06):
  * the crops (256 Hz, the in-house FM project's session front end, clipped at +-20) are decimated 2:1 by resample_poly AFTER the crop (the in-house
    FM project's trap c), then z-scored per window per channel (ST's input contract); ST's patch = 16 samples at 128 Hz = 125 ms =
    the MOR's 32 at 256 Hz, so crop patches (16 / 6 / 8) and the token budget map 1:1;
  * channel names = rec/ch_names[rec_idx] (aligned with X's channel axis) -> ST's table, case-insensitive; a name not in the
    table is DROPPED per window (never fed as zeros; the sampler's channel draw only touches present channels);
  * the feature = ST's encoder on the UNMASKED crop with per-window channel indices, final LayerNorm, the grid's channel mean.
The loss is `_sup_forward_v2`'s: the mean over the batch's datasets of each one's masked CE (+ shift_weight x the mean over
wide datasets of the on-time-vs-shifted BCE). t2p_cpt applies lambda and the in-house FM project's per-term clip (the supervised gradient on
the ENCODER <= c x the MAE gradient's norm, measured over the encoder; the heads get their full gradient).
"""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[3]
ST_TABLE = REPO / "tracks/t1_image/fm/steeg_channels.json"
# the in-house FM project's arm 2 (user approvals 2026-11-06 16:29 + 17:58): exactly these 11 sources, an explicit list (a dir glob would pick
# up the excluded Dreyer2023 / BNCI* / Sosulski / Nakanishi / ds005752 / liu2024 files that sit in the same dirs)
P1_SOURCES = ("lee2019_erp", "lee2019_mi", "stieger2021_mi", "cho2017", "alexmi", "schirrmeister2017", "lee2019_ssvep", "kalunga2016",
              "nieto2022_inner_speech", "ofner2017_me", "ofner2017_mi")
P1_GROUPS = "mi+me:0.504,erp:0.288,ssvep:0.108,speech:0.10"
P1_DENY = ("dreyer2023", "sosulski2019", "nakanishi2015", "liu2024")
ENCODER_PREFIX = ("patch_embed", "cls_token", "enc_channel_emd", "enc_temporal_emd", "blocks", "norm")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def encoder_params(mae):
    return [p for n, p in mae.named_parameters() if n.startswith(ENCODER_PREFIX) and p.requires_grad]


def st_pooled(mae, x, ch):
    """x (n, C, T128) float, ch (n, C) long ST table indices -> (n, T128 / 16, D): ST's encoder on the UNMASKED window,
    the final LayerNorm, the channel mean of the grid."""
    t = mae.patch_embed(x)                                       # (n, S, C, D)
    B, S, C, D = t.shape
    t = t.reshape(B, S * C, D)
    seq = torch.arange(1, S + 1, device=x.device).view(1, S, 1).expand(B, S, C).reshape(B, S * C)
    t = t + mae.enc_temporal_emd(seq) + mae.enc_channel_emd(ch.view(B, 1, C).expand(B, S, C).reshape(B, S * C))
    cls = (mae.cls_token + mae.enc_temporal_emd.get_cls_token()).expand(B, -1, -1)
    t = torch.cat([cls, t], 1)
    for blk in mae.blocks:
        t = blk(t)
    t = mae.norm(t)
    return t[:, 1:].reshape(B, S, C, D).mean(2)


class STSup:
    def __init__(self, src, dirs, names, groups, log, D, seed=0, head_seed=0x5D1, token_budget=8192, max_channels=32, batch_max=512,
                 mix=3, motor_crop=512, motor_start=(64, 256), shift_range=(26, 77), holdout_frac=0.10, holdout_seed=0,
                 shift_weight=1.0, dt=64, hidden=256, hash_files=True):
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import inhouse_fm
        from inhouse_fm.data.sup_stream import SupSamplerV2, SupSource, SupStream
        from inhouse_fm.train.sup_heads import SupHeadsV2
        assert Path(inhouse_fm.__file__).resolve().is_relative_to(Path(src).resolve()), f"inhouse_fm from {inhouse_fm.__file__}, not {src}"
        self._SupStream = SupStream
        bad = [n for n in names if n in P1_DENY or n.startswith(("bnci", "ds005752"))]
        assert not bad, f"excluded P1 sources asked: {bad}"
        self.paths = {}
        for n in names:
            hits = [Path(d) / f"{n}.h5" for d in dirs if (Path(d) / f"{n}.h5").exists()]
            assert len(hits) == 1, f"P1 source {n}: {len(hits)} files in {dirs}"
            self.paths[n] = hits[0]
        self.manifest = {n: dict(path=str(p), size=p.stat().st_size, mtime=p.stat().st_mtime,
                                 sha256=sha256(p) if hash_files else None) for n, p in self.paths.items()}
        self.sources = [SupSource(self.paths[n], n, holdout_frac, holdout_seed) for n in names]
        shift = [s.name for s in self.sources if s.shiftable]                    # v2: the shift head on every wide source
        self.sampler = SupSamplerV2(self.sources, groups, seed=int(seed), rank=0, world=1, token_budget=int(token_budget),
                                    max_channels=int(max_channels), batch_max=int(batch_max), mix=int(mix), motor_crop=int(motor_crop),
                                    motor_start=tuple(motor_start), shift_sources=shift, shift_range=tuple(shift_range))
        specs = {s.name: {"family": s.family, "T": self.sampler.crop_patches(i), "class_names": list(s.class_names),
                          "shift": s.name in self.sampler.shift_sources} for i, s in enumerate(self.sources)}
        self.heads = SupHeadsV2(int(D), specs, "v2_shared", dt=int(dt), hidden=int(hidden), seed=int(head_seed))
        self.shift_weight = float(shift_weight)
        table = json.loads(ST_TABLE.read_text())
        low = {k.lower(): k for k in table}
        self.st_idx, self.dropped = [], {}
        for s in self.sources:                                   # per source: (R, C_max) ST index, -1 = not in ST's table
            with h5py.File(s.path, "r") as f:
                raw = f["rec/ch_names"][...]
            nm = np.vectorize(lambda v: v.decode() if isinstance(v, bytes) else str(v))(raw)
            idx = np.full(nm.shape, -1, np.int64)
            for r in range(nm.shape[0]):
                for c in range(int(s.n_ch[r])):
                    k = low.get(nm[r, c].strip().lower())
                    if k is not None:
                        idx[r, c] = int(table[k])
                    else:
                        self.dropped.setdefault(s.name, set()).add(nm[r, c])
            self.st_idx.append(idx)
        self.dropped = {k: sorted(v) for k, v in self.dropped.items()}
        self.desc = self.sampler.describe()
        log(f"P1 sup: {len(self.sources)} sources, groups {groups}; heads {self.heads.spec()['ce']} + shift {self.heads.spec()['shift']}, "
            f"{self.heads.spec()['n_params']} params; channels not in ST's table (dropped): {self.dropped}")
        for n, v in self.desc["per_source"].items():
            log(f"P1 sup {n}: group {v['group']} family {v['family']} batch {v['batch_size']} tokens/window {v['tokens_per_window']} "
                f"crop {v['crop']} shift {v['shift']}")
        self.stream = None

    def start(self, step):
        self.stream = self._SupStream(self.sampler, int(step), prefetch=2)

    def _pooled_group(self, mae, batch, grp, dev):
        """the pooled features of one SupGroup's windows, in grp order: decimate after the crop, drop unmapped channels per
        window, re-z, encode sub-batches of equal kept-channel count."""
        x = resample_poly(np.asarray(grp.x, np.float32), 1, 2, axis=-1).astype(np.float32)   # (b, C, T/2) anti-aliased 2:1
        ch = np.stack([self.st_idx[int(batch.src_idx[b])][int(self.sources[int(batch.src_idx[b])].rec_idx[int(batch.rows[b])])]
                       [np.asarray(batch.chan[b], np.int64)] for b in grp.idx])           # (b, C) ST indices of the crop's channels
        out = [None] * len(grp.idx)
        keep = ch >= 0
        for k in sorted(set(keep.sum(1).tolist())):
            if k == 0:
                raise RuntimeError("a P1 window has no channel in ST's table")
            js = np.flatnonzero(keep.sum(1) == k)
            xs = np.stack([x[j][keep[j]] for j in js])
            cs = np.stack([ch[j][keep[j]] for j in js])
            xt = torch.as_tensor(xs, device=dev)
            xt = (xt - xt.mean(-1, keepdim=True)) / (xt.std(-1, keepdim=True, unbiased=False) + 1e-8)
            p = st_pooled(mae, xt, torch.as_tensor(cs, device=dev))
            for i, j in enumerate(js):
                out[j] = p[i]
        return out

    def loss(self, mae, batch, dev):
        """the in-house FM project's _sup_forward_v2 loss on ST features: (loss, per-source stats)."""
        heads, B = self.heads, batch.B
        on, sh = [None] * B, [None] * B
        for grp in batch.groups:
            for j, p in enumerate(self._pooled_group(mae, batch, grp, dev)):
                (sh if bool(grp.is_shift[j]) else on)[int(grp.idx[j])] = p
        assert all(v is not None for v in on), "every window needs an on-time crop"
        y = torch.as_tensor(batch.y, dtype=torch.long, device=dev)
        ce_terms, sh_terms, stats = [], [], {}
        for k in sorted(set(int(v) for v in batch.src_idx)):
            name = batch.src_names[k]
            idx = np.flatnonzero(batch.src_idx == k)
            h = torch.stack([on[i] for i in idx])
            key = heads.key_of[name]
            valid = torch.tensor(heads.valid(name), dtype=torch.bool, device=dev)
            logits = heads.ce[key](h).float().masked_fill(~valid, float("-inf"))
            tgt = torch.tensor(heads.class_map(name), dtype=torch.long, device=dev)[y[torch.as_tensor(idx, device=dev)]]
            ce_k = F.cross_entropy(logits, tgt)
            ce_terms.append(ce_k)
            rec = dict(ce=float(ce_k.detach()), acc=float((logits.argmax(-1) == tgt).float().mean()), n=int(len(idx)))
            if name in heads.skey_of:
                assert all(sh[i] is not None for i in idx), f"{name}: a wide source's windows came without their shifted crops"
                hs = torch.stack([sh[i] for i in idx])
                zl = heads.shift[heads.skey_of[name]](torch.cat([h, hs], 0)).squeeze(-1).float()
                tg = torch.cat([torch.zeros(len(idx), device=dev), torch.ones(len(idx), device=dev)])
                sh_k = F.binary_cross_entropy_with_logits(zl, tg)
                sh_terms.append(sh_k)
                rec["shift"] = float(sh_k.detach())
            stats[name] = rec
        loss = torch.stack(ce_terms).mean()
        if sh_terms:
            loss = loss + self.shift_weight * torch.stack(sh_terms).mean()
        return loss, stats

    @torch.no_grad()
    def holdout(self, mae, dev, n_windows=256, seed=0):
        """per source: balanced accuracy of the head over the source's own classes on its fixed holdout subset (the persons
        the sampler never draws), motor crops at the middle of their start range (the in-house FM project's evaluate_holdout_v2)."""
        heads = self.heads
        was = mae.training
        mae.eval()
        heads.eval()
        res = {}
        try:
            for i, s in enumerate(self.sources):
                ys, ps = [], []
                key = heads.key_of[s.name]
                valid = torch.tensor(heads.valid(s.name), dtype=torch.bool, device=dev)
                cmap = torch.tensor(heads.class_map(s.name), dtype=torch.long, device=dev)
                for b in self.sampler.holdout_batches(i, n_windows, seed):
                    for grp in b.groups:
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                            pool = self._pooled_group(mae, b, grp, dev)
                        lg = heads.ce[key](torch.stack(pool).float()).float().masked_fill(~valid, float("-inf"))
                        ps.append(torch.softmax(lg, -1)[:, cmap].argmax(-1).cpu().numpy())
                        ys.append(np.asarray(b.y)[grp.idx])
                if not ys:
                    res[s.name] = dict(n=0)
                    continue
                y, p = np.concatenate(ys), np.concatenate(ps)
                res[s.name] = dict(n=int(len(y)), balacc=float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])))
        finally:
            mae.train(was)
            heads.train(was)
        return res

    def record(self):
        return dict(sources=list(self.paths), manifest=self.manifest, sampler=self.desc, heads=self.heads.spec(),
                    channels_dropped=self.dropped, shift_weight=self.shift_weight)
