"""Track 2 PROTEUS: foundation-model classifiers on 4 s windows (41 ch, 120 Hz), reusing the T1 FM adapters.

The T1 adapters (``tracks/t1_image/scripts/t1_fm.py``: pinned checkpoints + sha checks, channel tables, LLRD param
groups) are 1 s-specific (8 patches / 1 patch per channel). The subclasses here only change the input length:
  st_*      ST-EEGFormer   128 Hz x 4 s = 512 samples, patch 16 -> 32 patches / channel (temporal PE table 500)
  mor*      the in-house MOR model (its adapter is not part of this public release)
  reve      REVE-base      200 Hz x 4 s = 800 samples, patch 200 / step 180 -> 4 patches (last 0.3 s unused)
  labram    LaBraM-base    200 Hz x 4 s = 800 samples, patch 200 -> 4 patches (temporal table rows 0..4)
Input contract for every FM: ``calib="zwin"`` (per-window, per-channel z-score after resampling), applied AFTER the
causal per-session alignment of ``ni26.stream`` (so ``align`` = none / sz / ea composes with it; ``wz`` is redundant).

``rand_<fm>`` = the same architecture at its seeded random init (the pretrained-vs-random control, MODELS §3.6).
MOR is an in-house model still in development; REVE is the organisers' baseline.
"""

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
T1S = REPO / "tracks" / "t1_image" / "scripts"
if str(T1S) not in sys.path:
    sys.path.insert(0, str(T1S))
import t1_fm as t1  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from ni26 import metrics  # noqa: E402

DUR = 4.0


def fm_mor_input(x, fs):
    """T2-D147: the in-house MOR model's own input front end (not part of this public release)."""
    raise NotImplementedError("the in-house MOR model is not part of this public release")


FP_MOR_FS = 256                          # the MOR front end's input rate


def _lora_depth(name):
    """review 2026-10-16 (T2 P2f): an input-dropout LoRA adapter, enc.lora_in['<adapted Linear's name, '.' -> '__'>'],
    sits at its block's depth (LLRD groups, layer clip, the readout-layer freeze), not at depth 0; None = not an adapter."""
    if name.startswith("lora_in."):
        b = t1.BLOCK.search(name.split(".")[1].replace("__", ".") + ".")
        if b:
            return int(b.group(3)) + 1
    return None


class STEEG4(t1.STEEG):
    def layer_of(self, name):
        d = _lora_depth(name)
        return super().layer_of(name) if d is None else d

    def prep(self, x):
        x = t1.fft_resample(x, int(round(self.FS * DUR)))
        return t1.zwin(x, 1e9) if self.calib == "zwin" else x * self.CLIP_GAIN

    def forward(self, x, layer, extras=False):
        v = self.vit
        t = v.patch_embed(x)                                              # (B, S, C, D)
        B, S, C, D = t.shape
        t = t.reshape(B, S * C, D)
        seq = torch.arange(1, S + 1, device=x.device).view(1, S, 1).expand(B, S, C).reshape(B, S * C)
        ch = self.active(self.chan_idx, C).view(1, 1, C).expand(B, S, C).reshape(B, S * C)
        t = t + v.enc_temporal_emd(seq) + v.enc_channel_emd(ch)
        cls = (v.cls_token + v.enc_temporal_emd.get_cls_token()).expand(B, -1, -1)
        t = v.pos_drop(torch.cat([cls, t], 1))
        for blk in v.blocks[:layer]:
            t = blk(t)
        if layer == self.n_layers:
            t = v.norm(t)
        g = t[:, 1:].reshape(B, S, C, D).permute(0, 2, 1, 3)              # (B, C, P, D)
        return (g, dict(cls=t[:, :1])) if extras else g

    def layer_grids(self, x, wanted):
        """T2-D131: {layer: (B, C, P, D)} in ONE pass for every layer in `wanted` (0 = the embedded tokens before the
        first block; raw block outputs; the final layer v.norm'd as in forward)."""
        v = self.vit
        t = v.patch_embed(x)
        B, S, C, D = t.shape
        t = t.reshape(B, S * C, D)
        seq = torch.arange(1, S + 1, device=x.device).view(1, S, 1).expand(B, S, C).reshape(B, S * C)
        ch = self.active(self.chan_idx, C).view(1, 1, C).expand(B, S, C).reshape(B, S * C)
        t = t + v.enc_temporal_emd(seq) + v.enc_channel_emd(ch)
        cls = (v.cls_token + v.enc_temporal_emd.get_cls_token()).expand(B, -1, -1)
        t = v.pos_drop(torch.cat([cls, t], 1))
        grid = lambda u: u[:, 1:].reshape(B, S, C, D).permute(0, 2, 1, 3)  # noqa: E731
        out = {0: grid(t)} if 0 in wanted else {}
        for i, blk in enumerate(v.blocks[:max(wanted)], 1):
            t = blk(t)
            if i in wanted:
                out[i] = grid(v.norm(t) if i == self.n_layers else t)
        return out


class MOR4(t1.MOR):
    """Adapter of the in-house MOR model for 4 s windows. Its code is unpublished and is not part of this public release."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("the in-house MOR model is not part of this public release")


class Reve4(t1.Reve):
    def prep(self, x):
        x = t1.fft_resample(x, int(round(self.FS * DUR)))
        return t1.zwin(x, self.CLIP) if self.calib == "zwin" else (x * self.GAIN).clamp(-self.CLIP, self.CLIP)

    def forward(self, x, layer, extras=False):
        B, C, T = x.shape
        P = (T - 200) // 180 + 1
        out = self.reve(x, pos=self.active(self.pos, C).expand(B, C, 3), return_output=True)
        return out[layer].reshape(B, C, P, self.dim)


class LaBraM4(t1.LaBraM):
    FS = int(200 * DUR)            # samples per window at 200 Hz: the parent sizes its temporal table as FS // PATCH patches

    def prep(self, x):
        x = t1.fft_resample(x, self.FS)
        return t1.zwin(x, 1e9) if self.calib == "zwin" else x * self.GAIN

    def forward(self, x, layer, extras=False):
        return super().forward(x, layer)


class NeuroRVQ4(t1.SourceChannels, nn.Module):
    """T2-D313 (RESEARCH ONLY: CC BY-NC weights, a private MI set in its pretraining, no template hosts it): NeuroRVQ-EEG FM v1's
    encoder from the shared pure-torch vendor src/ni26/fm/neurorvq.py (T3, exact parity vs the official repo @ 926e770d) on 4 s
    windows. Contract (raw uV, notch 50 / 60, band-pass 0.5-45 Hz, clip +-500, 200 Hz, 1-s patches): calib 'gain' only = the
    aligned unit-scale input x GAIN (15 uV per unit: LaBraM's 0.15 x 0.1 mV convention), then NrEnc.prep's FFT band-pass + notch
    and the resample BY DURATION to 200 Hz (4 s -> 800 samples -> 4 patches), clip +-500. Channels by lowercase 10-10 name (an
    unknown name raises). Grid (B, C, 4, 800) = the 4 inception branches' tokens concatenated (the reference's feature)."""
    FS, PATCH, GAIN = int(200 * DUR), 200, 15.0
    ALWAYS_USED = ()

    def __init__(self, fm, ch_names, calib, random_init, log):
        nn.Module.__init__(self)
        from ni26.fm.neurorvq import NR_CH, NrEnc, encoder_state
        assert calib == "gain", "NeuroRVQ: its contract is raw uV -> --fm-calib gain only"
        missing = [n for n in ch_names if str(n).lower() not in NR_CH]
        assert not missing, f"channels absent from NeuroRVQ's 104-name list: {missing}"
        self.calib, self.kit_names, self._nr_ch = calib, tuple(ch_names), NR_CH
        self.nr = NrEnc({"ch_names": list(ch_names)})
        if not random_init:
            path, _ = t1.checkpoint_sha(fm)
            self.nr.net.load_state_dict(encoder_state(torch.load(path, map_location="cpu", weights_only=True)), strict=True)
            log(f"NeuroRVQ: loaded {path} (strict, sha-checked); {len(ch_names)} channels -> rows {self.nr.spat.tolist()}")
        self.n_layers, self.dim = self.nr.n_layers, self.nr.dim

    def chan_spec(self, names):
        return torch.tensor([self._nr_ch.index(str(n).lower()) for n in names], dtype=torch.long)

    def prep(self, x):
        from ni26.fm.neurorvq import NrEnc
        fs_in = x.shape[-1] / DUR
        assert abs(fs_in - round(fs_in)) < 1e-6, f"window of {x.shape[-1]} samples is not an integer rate over {DUR} s"
        return NrEnc.prep(x * self.GAIN, fs_in=int(round(fs_in)), fs_out=int(self.FS / DUR))

    def forward(self, x, layer, extras=False):
        spat = self.active(self.nr.spat, x.shape[1])
        assert spat is self.nr.spat, "NeuroRVQ: kit montage only (no external-source channel sets in T2)"
        return self.nr(x, layer)

    def layer_of(self, name):
        d = _lora_depth(name)                             # T2-D313b: an input-dropout LoRA adapter sits at its block's depth
        if d is not None:
            return d
        if name.startswith("nr.net.blocks."):
            return int(name.split(".")[3]) + 1
        return 0                                          # patch_embed / pos_embed / time_embed / cls_token


def _register_neurorvq():
    """T2-D313: the shared vendor's pinned weights in T1's checkpoint tables (local path; sha256 prefix) without editing t1_fm."""
    from ni26.fm.neurorvq import SHA256, WEIGHTS, WEIGHTS_SOFIA
    t1.CKPT.setdefault("neurorvq", (Path(t1.ROOT) / WEIGHTS_SOFIA) if t1.ROOT else WEIGHTS)     # sofia: $S1_ROOT/models/...
    t1.SHA12.setdefault("neurorvq", SHA256[:12])


class CBraMod4(t1.SourceChannels, nn.Module):
    """T2-D341 (the goal T2-D324's distinct-member search): CBraMod (Wang et al., ICLR 2025; code MIT, weights Apache-2.0) as
    braindecode 1.8.1 builds it (`CBraMod`, T3 item 88's route = t3_stage.CBFM), strict-loaded from the pinned
    third_party/cbramod/pretrained_weights.pth but for the random final_layer (never reached: features only), sha-checked through
    T1's tables. Channel-agnostic (criss-cross attention + a conv positional encoding over the channel x patch grid; no channel
    ids), so the kit montage needs no vocabulary. Front end: FFT resample by duration to 200 Hz (4 s -> 800 samples = 4 one-second
    patches); calib 'gain' = the aligned unit-scale input x GAIN (0.15: LaBraM's D52 100-uV convention = CBraMod's pretraining unit),
    clipped at +-CLIP (T3's); 'zwin' = per-window z. Only the final layer is wired: forward(x, 12) -> (B, C, 4, 200)."""
    FS, PATCH, GAIN, CLIP = int(200 * DUR), 200, 0.15, 15.0
    ALWAYS_USED = ()

    def __init__(self, fm, ch_names, calib, random_init, log):
        nn.Module.__init__(self)
        from braindecode.models import CBraMod
        assert calib in ("gain", "zwin"), "CBraMod: --fm-calib gain | zwin"
        self.calib, self.kit_names = calib, tuple(ch_names)
        self.m = CBraMod(n_outputs=1, n_chans=len(ch_names), n_times=self.FS, sfreq=200)
        if not random_init:
            path, sha = t1.checkpoint_sha(fm)
            r = self.m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=False)
            assert all(k.startswith("final_layer.") for k in r.missing_keys) and not r.unexpected_keys, (r.missing_keys, r.unexpected_keys)
            log(f"CBraMod: loaded {path} (sha256 {sha[:12]}; every tensor but final_layer {r.missing_keys}); {len(ch_names)} channels")
        if fm.endswith("cbramodx"):     # T2-D363: the encoder output, as CBraMod's own fine-tunes (backbone.proj_out = Identity);
            self.m.proj_out = nn.Identity()  # plain 'cbramod' (T2-D341) read proj_out = the MAE reconstruction head (bitwise unchanged)
            log("CBraMod: proj_out -> Identity (features = the encoder output, the official downstream readout)")
        for p in self.m.final_layer.parameters():
            p.requires_grad_(False)
        self.register_buffer("spec", torch.arange(len(ch_names)), persistent=False)
        self.register_buffer("chan_idx", torch.arange(len(ch_names)), persistent=False)   # T2-D355: FMClf's grid head sizes itself by it
        self.n_layers, self.dim = 12, 200

    @staticmethod
    def chan_spec(names):
        return torch.arange(len(names))                   # no channel ids: only the count is checked (SourceChannels.active)

    def prep(self, x):
        x = t1.fft_resample(x, self.FS)
        assert x.shape[-1] == self.FS
        return t1.zwin(x, 1e9) if self.calib == "zwin" else (x * self.GAIN).clamp(-self.CLIP, self.CLIP)

    def forward(self, x, layer, extras=False):
        assert layer == self.n_layers, f"CBraMod: only the final layer ({self.n_layers}) is wired, got {layer}"
        B, C, T = x.shape
        self.active(self.spec, C)
        f = self.m(x, return_features=True)["features"]
        P = T // self.PATCH
        assert f.shape[0] == B and f.shape[1] == C and f.shape[-1] == self.dim and f.numel() == B * C * P * self.dim, f.shape
        return f.reshape(B, C, P, self.dim)

    def layer_of(self, name):
        if name.startswith("m.encoder.layers."):
            return int(name.split(".")[3]) + 1
        if name.startswith("m.final_layer."):
            return self.n_layers + 1                       # frozen by FMClf (> layer); never on the output path
        return self.n_layers if name.startswith("m.proj_out") else 0      # patch_embedding (+ its positional conv) = depth 0


def _register_cbramod():
    """T2-D341: the pinned CBraMod weights in T1's checkpoint tables (as T3's t3_feats does), without editing t1_fm."""
    for name in ("cbramod", "cbramodx"):    # T2-D363: cbramodx = the same pinned file, read at the encoder output
        t1.CKPT.setdefault(name, (Path(t1.ROOT) / "models/third_party/cbramod/pretrained_weights.pth") if t1.ROOT else
                           Path.home() / "workspace/datasets/pretrained_eeg_fms/cbramod/pretrained_weights.pth")
        t1.SHA12.setdefault(name, "0792cb808c14")


PROTEUS_USERS = [f"P{k:02d}" for k in range(1, 11)]
KIT_TV_USERS = ["P01", "P02", "P03", "P05", "P06", "P07", "P08", "P10"]          # the kit train + val participants
SEEN1_POOL_USERS = ["P01", "P02", "P03", "P05", "P06", "P07", "P08"]


def proteus_streams(subjects, sessions):
    """(subject, session) pairs of PROTEUS windows (subject 'Dreyer2026Proteus/P04'); other corpora's rows are ignored."""
    return {(str(u).split("/")[-1], str(s)) for u, s in zip(subjects, sessions) if str(u).startswith("Dreyer2026Proteus/")}


def cpt_pool_streams(path, _seen=()):
    """Review 2026-10-16 (T2 P2e): the PROTEUS (subject, session) streams whose windows (cue windows or continuous-session
    crops, labelled or not) trained the CPT / pre-FT checkpoint `path`, its --init-ckpt chain included; session '*' = every
    session of that subject. None = unknown (no cpt_summary.json next to it: a pinned root or a contract-only checkpoint).
    Summaries written after this review carry `proteus_streams`; older ones are read from their recorded arguments
    (t2p_cpt: --proteus-pool rule + a `proteus_sessions*` extra minus --extra-exclude-subjects; t2p_preft: every
    `export_proteus*` / `proteus_sessions*` pool minus --exclude-subjects)."""
    path = Path(path)
    sf = path.parent / "cpt_summary.json"
    if not sf.exists() or str(path) in _seen:
        return None
    summ = json.loads(sf.read_text())
    a = summ.get("args") or {}
    if "proteus_streams" in summ:
        out = {tuple(v.split("|", 1)) for v in summ["proteus_streams"]}
    else:
        out = set()
        if "proteus" in a:                                          # t2p_cpt.py
            rule = {"kit_tv": KIT_TV_USERS, "seen1": SEEN1_POOL_USERS, "seen1_out": SEEN1_POOL_USERS, "all": PROTEUS_USERS}
            pool = a.get("proteus_pool") or "kit_tv"
            out |= {(u, "*") for u in rule[pool]} | ({("P04", "1"), ("P10", "1")} if pool == "seen1" else set())
            excl = {v for v in (a.get("extra_exclude_subjects") or "").split(",") if v}
            extras = [Path(e).name for e in a.get("extra") or []]
        elif "pool" in a:                                           # t2p_preft.py
            excl = {v for v in (a.get("exclude_subjects") or "").split(",") if v}
            extras = [Path(str(p).rsplit(":", 1)[0]).name for p in a.get("pool") or []]
        elif summ.get("pools"):                                     # e.g. t2p_preft_finalize.py: pool names only, so no
            excl = set()                                            # exclusion is assumed (conservative)
            extras = [str(p.get("name", "")) for p in summ["pools"]]
        else:
            return None
        for nm in extras:
            if nm.startswith(("export_proteus", "proteus_sessions")):
                out |= {(u, "*") for u in PROTEUS_USERS if u not in excl}
            else:
                assert "proteus" not in nm.lower(), f"{sf}: pool {nm}: unknown PROTEUS-derived pool (no legacy rule)"
    init = a.get("init_ckpt") or ""
    if init:
        sub = cpt_pool_streams(init, tuple(_seen) + (str(path),))
        if sub is None:
            return None
        out |= sub
    return out


def cpt_leak(path, test_streams):
    """test streams of a protocol that the checkpoint's pools contain -> sorted list ([] = clean), or None (unknown)."""
    pool = cpt_pool_streams(path)
    if pool is None:
        return None
    return sorted(f"{u}|{s}" for u, s in test_streams if (u, s) in pool or (u, "*") in pool)


def _t3_cpt_lineage(path):
    """True for a checkpoint in the t3_cpt layout (a 'state' dict, no 'model'); t2p_cpt / t2p_preft checkpoints hold 'model'."""
    ck = t1.load_ck(path)
    out = isinstance(ck.get("state"), dict) and "model" not in ck
    del ck
    return out


_PINNED = {k: (t1.CKPT.get(k), t1.SHA12.get(k)) for k in set(t1.CKPT) | set(t1.SHA12)}   # T2-D392: the import-time pinned files


def reset_cpt(fm):
    """T2-D392: undo use_cpt for an ST-EEGFormer fm (the next seed of one build fine-tunes from ANOTHER checkpoint): the pinned
    CKPT / SHA12 back, the CPT lineage dropped, the sha cache cleared. No-op when fm is not redirected."""
    assert fm.startswith("st_"), "reset_cpt: ST-EEGFormer only (the MOR branch strict-loads its CPT state elsewhere)"
    if fm not in t1.CPT_INFO:
        return
    ck, sha = _PINNED[fm]
    t1.CKPT[fm], t1.SHA12[fm] = ck, sha
    t1.CPT_INFO.pop(fm)
    t1._SHA_CACHE.clear()


def use_cpt(fm, path, log):
    """Fine-tune from a t2p_cpt.py checkpoint instead of the pinned file: verify its cpt_summary.json (status ok, same fm,
    base sha = the pinned file's, ckpt sha = the file's), then point the T1 adapter's CKPT / SHA12 at it (ST, or the
    MOR from a t2p_preft checkpoint, T2-D106)."""
    import hashlib
    assert fm.startswith("st_") or fm in t1.MOR_NAMES or fm in ("reve", "labram"), "--fm-ckpt: ST-EEGFormer, MOR, or REVE / LaBraM (T2-D492)"
    path = Path(path)
    # T2-D492: a t2p_cpt_fm.py checkpoint: REVE = <dir>/model.safetensors (the adapter reads the DIR), LaBraM = <dir>/checkpoint-cpt.pth
    assert fm != "reve" or path.name == "model.safetensors", "--fm-ckpt reve: pass <cpt dir>/model.safetensors"
    if fm in t1.CPT_INFO:       # already redirected in this process (the next seed / arm): checkpoint_sha now hashes the CPT file
        got = t1.CPT_INFO[fm].get("ckpt") or t1.CPT_INFO[fm].get("path")
        assert got in (str(path), str(path.resolve())), f"--fm-ckpt: a second CPT checkpoint for {fm} in one process"
        return
    if fm in t1.MOR_NAMES and _t3_cpt_lineage(path):
        # T2-D141: a checkpoint of the in-house MOR lineage is loaded by T1's MOR branch (not part of this public release)
        t1.cpt_override_mor(fm, path, log)
        return
    if (path.parent / "cpt_summary.json").exists():
        summ = json.loads((path.parent / "cpt_summary.json").read_text())
    else:                               # FM program §0d item 9: a checkpoint with a scorecard contract (tracks/fm/fm_contract.py)
        con = json.loads(Path(str(path) + ".contract.json").read_text())
        assert con.get("status") == "ok" and con.get("family") == fm and \
            (con.get("pinned_keyset_match") or con.get("encoder_keyset_match")), f"--fm-ckpt contract {con}"
        summ = dict(status="ok", fm=fm, base_sha256=con["base_sha256"], ckpt_sha256=con["sha256"],
                    args=dict(steps=None), pools=[dict(name="(contract: see the checkpoint's own run record)")])
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    assert summ.get("status") == "ok" and summ.get("fm") == fm, f"--fm-ckpt: summary status/fm {summ.get('status')} {summ.get('fm')}"
    if summ.get("random_init") is True:     # T2-D356: a from-scratch t2p_cpt run has no pinned base; only the file sha is checked
        assert summ.get("base_sha256") is None and summ.get("base_ckpt") is None, "--fm-ckpt: random_init with a base checkpoint"
        pinned = None
    else:
        pinned = t1.checkpoint_sha(fm)[1]
        assert summ.get("base_sha256") == pinned, "--fm-ckpt: base sha mismatch"
    assert summ.get("ckpt_sha256") == h.hexdigest(), "--fm-ckpt: sha mismatch"
    t1.CKPT[fm], t1.SHA12[fm] = (path.parent if fm == "reve" else path), h.hexdigest()[:12]
    t1._SHA_CACHE.clear()
    ps = cpt_pool_streams(path)
    t1.CPT_INFO[fm] = dict(ckpt=str(path), sha256=h.hexdigest(), base_sha256=pinned, steps=summ["args"]["steps"],
                           pools=[p["name"] for p in summ["pools"]],
                           proteus_streams=None if ps is None else sorted(f"{u}|{s}" for u, s in ps))
    t1.CPT_INFO[fm]["random_init"] = bool(summ.get("random_init"))
    log(f"--fm-ckpt {path} (sha256 {h.hexdigest()[:12]}; {'FROM SCRATCH' if pinned is None else 'CPT of'} {fm}, {summ['args']['steps']} steps on "
        f"{[p['name'] for p in summ['pools']]})")


def make_encoder(fm, ch_names, log, drop_path=0.0, calib="zwin"):
    """calib: 'zwin' = per-window per-channel z after resampling (D82 default); 'gain' = the T1 adapters' fixed gain (+ clamp)
    on unit-scale input, i.e. relative channel amplitudes are KEPT (pair with align none = global scale, or sz = causal
    per-session per-channel scale)."""
    assert calib in ("zwin", "gain", "tp"), calib
    assert calib != "tp" or fm.removeprefix("rand_") in t1.MOR_NAMES, "calib tp = the MOR's own front end only"
    rand = fm.startswith("rand_")
    base = fm[5:] if rand else fm
    if rand:
        torch.manual_seed(int(os.environ.get("T1_FM_TWIN_SEED", "0")))
    if base == "neurorvq":
        _register_neurorvq()
    if base in ("cbramod", "cbramodx"):
        _register_cbramod()
    prov = t1.provenance(base, rand)
    if base.startswith("st_"):
        enc = STEEG4(base, ch_names, calib, rand, log, drop_path)
    elif base in t1.MOR_NAMES:
        enc = MOR4(base, ch_names, calib, rand, log, "referential")
    elif base == "reve":
        enc = Reve4(base, ch_names, calib, rand, log)
    elif base == "labram":
        enc = LaBraM4(base, ch_names, calib, rand, log)
    elif base == "neurorvq":                              # T2-D313 research screen
        enc = NeuroRVQ4(base, ch_names, calib, rand, log)
    elif base in ("cbramod", "cbramodx"):                 # T2-D341 / D363
        enc = CBraMod4(base, ch_names, calib, rand, log)
    else:
        raise SystemExit(f"unknown fm {fm}")
    enc.provenance = prov
    return enc


class FMClf(nn.Module):
    """encoder -> (B, C, P, D) grid at `layer` -> LayerNorm -> mean over tokens | attentive pooling -> linear."""

    def __init__(self, enc, layer, readout, n_out):
        super().__init__()
        self.enc, self.layer, self.readout = enc, layer, readout
        D = enc.dim
        self.norm = nn.LayerNorm(D)
        d_out = D
        if readout == "attn":
            self.pool_token = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.trunc_normal_(self.pool_token, std=0.02)
            self.pool = nn.MultiheadAttention(D, 8, batch_first=True)
            self.norm_pool = nn.LayerNorm(D)
        elif readout in ("early", "mlmean"):             # T2-D131: multi-layer readouts, one LayerNorm per layer
            L = enc.n_layers
            self.rl_layers = ([0, 1, 2, L] if not isinstance(enc, t1.MOR) else [1, 2, L]) if readout == "early" else \
                sorted({L // 2, (3 * L) // 4, L})
            self.rl_norms = nn.ModuleList([nn.LayerNorm(D) for _ in self.rl_layers])
            d_out = D * len(self.rl_layers) if readout == "early" else D
        elif readout == "attnres":                       # T2-D464 (the user's idea, arXiv 2603.15031 / 2604.03297): per token, a learned
            self.rl_layers = list(range(1, enc.n_layers + 1))   # softmax over ALL layer outputs replaces the last-layer read:
            self.ar_q = nn.Parameter(torch.zeros(D))            # alpha_i = softmax_i(q . RMSNorm(V_i) + b_i), q zero-initialised;
            b0 = torch.zeros(enc.n_layers)                      # b starts at (0, .., 0, 6): step 0 is 94.6 % the last layer for L = 24
            b0[-1] = 6.0
            self.ar_b = nn.Parameter(b0)
        elif readout == "tp":                            # T2-D131: a readout specific to the in-house MOR model
            assert isinstance(enc, t1.MOR), "--readout tp: MOR only"
            self.norm_tp = nn.LayerNorm(D)
        elif readout in ("grid", "grid64d"):             # T2-D131 'g': position-aware grid head (T1 D74 'grid'); T2-D520 'grid64d' = last year's
                                                         # winner's all_simple: per-token Linear(D, 64) -> flatten -> Dropout(0.1) -> Linear
            with torch.no_grad():
                dev_ = next(enc.parameters()).device
                n_tok = enc(enc.prep(torch.zeros(1, len(enc.chan_idx) if hasattr(enc, "chan_idx") else len(enc.xyz), 480,
                                                 device=dev_)), layer).shape
            gw = 64 if readout == "grid64d" else 32
            self.grid_tok = nn.Linear(D, gw)
            d_out = int(n_tok[1] * n_tok[2] * gw)
            if readout == "grid64d":
                self.grid_drop = nn.Dropout(0.1)
        else:
            assert readout == "mean", readout
        self.out = nn.Linear(d_out, n_out)
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
        for n_, p in enc.named_parameters():
            if enc.layer_of(n_) > layer and not n_.startswith(enc.ALWAYS_USED):
                p.requires_grad_(False)

    def features(self, x):
        if self.readout in ("early", "mlmean"):
            gs = self.enc.layer_grids(self.enc.prep(x), self.rl_layers)
            hs = [nrm(gs[li]).mean((1, 2)) for li, nrm in zip(self.rl_layers, self.rl_norms)]
            return torch.cat(hs, -1) if self.readout == "early" else torch.stack(hs).mean(0)
        if self.readout == "attnres":
            gs = self.enc.layer_grids(self.enc.prep(x), self.rl_layers)
            V = torch.stack([gs[li] for li in self.rl_layers])                                  # (L, B, C, P, D)
            Vn = V.float() * torch.rsqrt(V.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            a = torch.softmax(Vn @ self.ar_q.float() + self.ar_b.float().view(-1, 1, 1, 1), 0).to(V.dtype)
            return self.norm((a.unsqueeze(-1) * V).sum(0)).mean((1, 2))
        if self.readout == "tp":
            _, ex = self.enc(self.enc.prep(x), self.layer, extras=True)
            return self.norm_tp(ex["tp"]).mean(1)
        g = self.enc(self.enc.prep(x), self.layer)
        if self.readout in ("grid", "grid64d"):
            h = self.norm(g)
            td = getattr(self, "token_drop", 0.0)          # T2-D349b: whole-token dropout before the grid head (training only; 0 = unchanged)
            if td and self.training:
                h = h * ((torch.rand(h.shape[:3], device=h.device) >= td).to(h.dtype)[..., None] / (1.0 - td))
            f_ = self.grid_tok(h).flatten(1)
            return self.grid_drop(f_) if self.readout == "grid64d" else f_
        if self.readout == "mean":
            return self.norm(g).mean((1, 2))
        B, D = g.shape[0], g.shape[-1]
        kv = self.norm(g.reshape(B, -1, D))
        return self.norm_pool(self.pool(self.pool_token.expand(B, -1, -1).to(kv.dtype), kv, kv, need_weights=False)[0][:, 0])

    def forward(self, x):
        if hasattr(self, "amp_side"):                    # T2-X1: + the window's amplitude (AmpSide); absent = the FMClf of T2-D82..D215
            return self.out(self.features(x)) + self.amp_side(x)
        if hasattr(self, "dc_side"):                     # T2-D524: + a zero-init linear map of the window's causal-z window means (net._side, set per batch)
            assert getattr(self, "_side", None) is not None and len(self._side) == len(x), "T2-D524: set net._side for every forward"
            return self.out(self.features(x)) + self.dc_side(self._side.to(torch.float32)).to(torch.float32)
        return self.out(self.features(x))


class AmpSide(nn.Module):
    """T2-X1: the per-channel log-std of the incoming (aligned, not per-window-normalised) window, standardised by the training windows'
    mean / sd and clipped at +-5, into a zero-initialised linear layer added to the logits (at init the FMClf exactly). The FM's own zwin
    deletes this scale; the side input gives it back."""
    CLIP = 5.0

    def __init__(self, mu, sd, n_out):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.lin = nn.Linear(len(mu), n_out)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    @staticmethod
    def logstd(x):
        return torch.log(x.float().std(-1, unbiased=False) + 1e-6)

    def forward(self, x):
        return self.lin(((self.logstd(x) - self.mu) / self.sd).clamp(-self.CLIP, self.CLIP))


class FuseClf(FMClf):
    """FMClf + a CNN branch (a braindecode model without its final layer) on the SAME aligned window: the pooled FM
    feature and the LayerNormed CNN feature are concatenated and read out by one linear layer; trained jointly (the CNN
    branch is a head-group module at lr_head). The CNN runs in fp32 (FBCNet's log-variance is fragile in bf16)."""

    def __init__(self, enc, layer, readout, n_out, cnn, n_chans, n_times):
        super().__init__(enc, layer, readout, n_out)
        self.cnn = cnn
        self.cnn.final_layer = nn.Identity()
        with torch.no_grad():
            dc = int(self.cnn.eval()(torch.zeros(2, n_chans, n_times)).shape[-1])
        self.norm_cnn = nn.LayerNorm(dc)
        self.out = nn.Linear(enc.dim + dc, n_out)
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.features(x)
        with torch.autocast("cuda", enabled=False):
            hc = self.norm_cnn(self.cnn(x.float()))
        return self.out(torch.cat([h.float(), hc], 1))


@torch.no_grad()
def predict_proba(net, X, bs=128):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out.append(F.softmax(net(X[i:i + bs]).float(), 1))
    return torch.cat(out).cpu().numpy()


def lp_steps_guard(lp_epochs):
    return int(lp_epochs or 0) > 0


@torch.no_grad()
def predict_proba_side(net, X, S, bs=128):
    """T2-D524: predict_proba with the per-window side vectors S (net._side is set per chunk, then cleared)."""
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        net._side = S[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out.append(F.softmax(net(X[i:i + bs]).float(), 1))
    net._side = None
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def fm_side_dump(net, X, bs=128):
    """T2-D553 (NI26_SIDE_DUMP=1 only; unset = nothing computed): the member's PRE-HEAD logits on X (the same autocast as predict_proba_side) and
    its window-mean side head, so a reader can re-fit the head in-stream: softmax(L + S @ w.T + b) == predict_proba_side up to bf16 rounding
    of the head term."""
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out.append(net.out(net.features(X[i:i + bs])).float())
    return dict(logit_last=torch.cat(out).cpu().numpy(), w=net.dc_side.weight.detach().float().cpu().numpy(),
                b=net.dc_side.bias.detach().float().cpu().numpy())


@torch.no_grad()
def fm_feat_dump(net, X, bs=128):
    """T2-D555 (NI26_FEAT_DUMP=1 only; unset = nothing computed): the member's pooled pre-head FEATURES on X (net.features, the same autocast
    as predict_proba) and its main class head (net.out), so a reader can re-fit the main head in-stream: net.out(features) == the pre-side
    logits up to bf16 rounding."""
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out.append(net.features(X[i:i + bs]).float())
    return dict(feat=torch.cat(out).cpu().numpy(), w=net.out.weight.detach().float().cpu().numpy(), b=net.out.bias.detach().float().cpu().numpy())


VIEW_SHIFTS = (15, -15, 30)   # T2-D654 (Track3, pinned 2026-11-20 before any number): the circular shifts (samples at 120 Hz) of views v = 1, 2, 3
VIEW_GAIN = 0.2               # the per-channel gain range exp(U(-0.2, 0.2)) (= D557's training jitter)
VIEW_SEED0 = 65400            # view v's gains are drawn ONCE from torch.Generator(65400 + v)


def fm_view_spec(C, K):
    """T2-D654: the K - 1 FIXED test-time views beyond the identity (K in 2..4). View v (1..K-1) = a per-channel gain exp(U(-0.2, 0.2)) over the C
    channels drawn once from torch.Generator(65400 + v), then a circular time shift by VIEW_SHIFTS[v - 1] samples in TV.augment's order and roll
    convention (out[..., t] = x[..., (t - s) mod L]); NO channel dropout. Fixed (not per-window random): the lab and a template reproduce them bit for bit."""
    assert 2 <= int(K) <= 1 + len(VIEW_SHIFTS), K
    out = []
    for v in range(1, int(K)):
        g = torch.Generator().manual_seed(VIEW_SEED0 + v)
        out.append((torch.exp((torch.rand(int(C), generator=g) * 2 - 1) * VIEW_GAIN), int(VIEW_SHIFTS[v - 1])))
    return out


def fm_apply_view(X, gain, shift):
    """T2-D654: one view of the windows X (N, C, L): the per-channel gain, then the circular shift (TV.augment's gather convention)."""
    L = X.shape[-1]
    t = (torch.arange(L, device=X.device) - int(shift)) % L
    return (X * gain.to(device=X.device, dtype=X.dtype).view(1, -1, 1))[..., t]


@torch.no_grad()
def fm_view_dump(net, X, S, K, bs=128):
    """T2-D654 (NI26_VIEW_DUMP=K only; unset = nothing computed): for each fixed view v = 1..K-1 the member's pooled features (fm_feat_dump's
    forward) and its full probabilities (predict_proba, or predict_proba_side with the side vectors S of the IDENTITY window: the side path is not
    viewed). The identity itself stays prob_last / fm_feat_te. Global RNG untouched (the views use their own generators)."""
    spec = fm_view_spec(X.shape[1], K)
    feats, probs = [], []
    for gain, shift in spec:
        Xv = fm_apply_view(X, gain, shift)
        feats.append(fm_feat_dump(net, Xv, bs)["feat"])
        probs.append(predict_proba_side(net, Xv, S, bs) if S is not None else predict_proba(net, Xv, bs))
    return dict(feat=np.stack(feats), prob=np.stack(probs), gain=np.stack([g.numpy() for g, _ in spec]), shift=np.asarray([s for _, s in spec]))


def bandpass(X, band, fs=120.0):
    """Zero-phase Butterworth band-pass of each window on its own (order 4, sos; per-window, so stream-legal)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [float(band[0]), float(band[1])], btype="bandpass", fs=fs, output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, np.asarray(X, np.float32), axis=-1), dtype=np.float32)




def mor_drop_path(enc, p, log):
    """T2-D128 S2: block-level stochastic depth for the in-house MOR model (not part of this public release)."""
    raise NotImplementedError("the in-house MOR model is not part of this public release")


class _InLoRA(nn.Module):
    """one LoRA adapter: (alpha / r) * B A drop(x), B = 0 at init (the adapted model starts at the pretrained one)."""

    def __init__(self, in_f, out_f, r, alpha, dropout):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_f))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale, self.drop = alpha / r, nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.scale * F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype))




class _LoRAHook:
    """review 2026-10-16 (T2 P2f; = T1-D115's t1_fm._LoRAHook): the forward hook that adds an adapter's output to a frozen
    Linear's. A callable OBJECT, not a closure: copy.deepcopy (--fm-ema's averaged copy, mae_tta, a package) then maps
    `ads` to the COPY's own adapters (a closure kept calling the ORIGINAL adapters)."""

    def __init__(self, ads):
        self.ads = ads

    def __call__(self, mod, inp, out):
        return out + self.ads[None](inp[0])


def inject_lora_input(enc, r, alpha, dropout, log, roles=False):
    """T2-D130 (FM program §0d item 3): LoRA on EVERY Linear inside the encoder blocks with INPUT dropout (forward hooks),
    the base frozen, block norms + the final norm trainable (as t1.inject_lora). roles=True was a variant for the in-house
    MOR model, which is not part of this public release."""
    if roles:
        raise NotImplementedError("the in-house MOR model is not part of this public release")
    for p_ in enc.parameters():
        p_.requires_grad_(False)
    enc.lora_in = nn.ModuleDict()
    n = 0
    seen = set()
    for name, m in list(enc.named_modules()):
        b = t1.BLOCK.search(name)
        if not (isinstance(m, nn.Linear) and b and "dec_" not in name) or id(m) in seen:
            continue
        seen.add(id(m))
        key = name.replace(".", "__")
        enc.lora_in[key] = _InLoRA(m.in_features, m.out_features, r, alpha, dropout).to(m.weight.device)
        n += 1
        m.register_forward_hook(_LoRAHook({None: enc.lora_in[key]}))
    for name, m in enc.named_modules():
        b = t1.BLOCK.search(name)
        if ((b and "dec_" not in name) or name == "vit.norm") and "norm" in type(m).__name__.lower():
            for p_ in m.parameters(recurse=False):
                p_.requires_grad_(True)
    for p_ in enc.lora_in.parameters():
        p_.requires_grad_(True)
    log(f"LoRA (input dropout {dropout}) r={r} alpha={alpha}: {n} adapters; encoder trainable "
        f"{sum(p_.numel() for p_ in enc.parameters() if p_.requires_grad) / 1e6:.2f}M of {sum(p_.numel() for p_ in enc.parameters()) / 1e6:.1f}M")




def t2p_cpt_loss(enc, mae_dec, xz, noise, ratio):
    """T2-D132 (ST): the CPT MAE loss with the fine-tuned encoder (enc.vit) and the CPT decoder (frozen)."""
    import t2p_cpt
    return t2p_cpt.mae_loss_split(enc.vit, mae_dec, xz, enc.chan_idx.to(xz.device), ratio, noise)


def mor_mae_loss(enc, x, ratio):
    """T2-D132: the in-house MOR model's own masked-reconstruction loss (not part of this public release)."""
    raise NotImplementedError("the in-house MOR model is not part of this public release")


def supcon_loss(z, mask, tau):
    """SupCon over L2-normalised z (B, d) with an explicit positive mask (B, B; diagonal ignored); anchors without a
    positive are left out (Khosla et al. 2020, the 'out' form)."""
    z = F.normalize(z.float(), dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mask = mask & ~eye
    npos = mask.sum(1)
    ok = npos > 0
    if not ok.any():
        return z.sum() * 0.0
    return -((logp * mask).sum(1)[ok] / npos[ok]).mean()


class SupConPairs:
    """T2-D127 sampler (T1-D105 pattern): each step = n anchors + one positive each, drawn from the TRAINING windows.
    mode 'xuser' (arm 1): positive = same cue, another user; 'xsession' (arm 3): same user + same cue, another session.
    Anchors are drawn only among windows that have at least one positive."""

    def __init__(self, y, user, session, mode, seed):
        self.y, self.user, self.ses, self.mode = np.asarray(y), np.asarray(user), np.asarray(session), mode
        self.rng = np.random.default_rng(seed + 991)
        self.cand = {}
        for i in range(len(self.y)):
            if mode == "xuser":
                c = np.flatnonzero((self.y == self.y[i]) & (self.user != self.user[i]))
            else:
                c = np.flatnonzero((self.y == self.y[i]) & (self.user == self.user[i]) & (self.ses != self.ses[i]))
            if len(c):
                self.cand[i] = c
        self.anchors = np.array(sorted(self.cand))

    def draw(self, n):
        a = self.rng.choice(self.anchors, size=n, replace=len(self.anchors) < n)
        pos = np.array([self.rng.choice(self.cand[i]) for i in a])
        return np.concatenate([a, pos])

    def mask(self, rows):
        y, u, s_ = self.y[rows], self.user[rows], self.ses[rows]
        same_y = y[:, None] == y[None, :]
        if self.mode == "xuser":
            m = same_y & (u[:, None] != u[None, :])
        else:
            m = same_y & (u[:, None] == u[None, :]) & (s_[:, None] != s_[None, :])
        return torch.as_tensor(m)


def pkg_nll(lo, pc, y, w):
    """backlog 96 (G3, P3 v2): the NLL of the true class under the PACKAGE mix (1 - w) pc + w softmax(lo), pc = the fixed out-of-fold CNN
    probabilities of the same windows; the member's gradient is large where the CNN part is wrong (the boosting direction)."""
    comb = (1 - w) * pc + w * torch.softmax(lo.float(), 1)
    return -torch.log(comb.gather(1, y[:, None]).clamp_min(1e-8)).mean()


def wce(lo, y, ls, w=None, norm=False):
    """T2-D253: cross-entropy (label smoothing ls), per-window weighted when w is given (w = None: F.cross_entropy's mean).
    T2-D642 (W25): norm = train.py's 'wtr' semantics, sum w ce / sum w over the batch (self-normalised; a global rescale of w cancels);
    norm False = the D253 form below, unchanged (bitwise)."""
    if w is None:
        return F.cross_entropy(lo, y, label_smoothing=ls)
    if norm:
        return (F.cross_entropy(lo, y, label_smoothing=ls, reduction="none") * w).sum() / w.sum()
    return (F.cross_entropy(lo, y, label_smoothing=ls, reduction="none") * w).mean()


def d546_reinit(net, enc, seed, log):
    """T2-D546 arm (a): on a converged member, re-draw (i) the top ST encoder block and the encoder's final norm with D264's rule
    (xavier-uniform Linear weights, zero biases, LayerNorm 1 / 0), (ii) the head = FMClf.norm (LayerNorm 1 / 0) + FMClf.out (nn.Linear's own
    default init: kaiming-uniform a = sqrt(5), bias U(+-1 / sqrt(fan_in))), (iii) the dcmean side head (zero, its own init). One CPU generator
    seeded 7919 + seed, values copied onto the live (device) tensors."""
    g = torch.Generator().manual_seed(7919 + int(seed))
    n = 0
    with torch.no_grad():
        for m_ in list(enc.vit.blocks[-1].modules()) + [enc.vit.norm]:
            if isinstance(m_, nn.Linear):
                w_ = torch.empty(m_.weight.shape)
                nn.init.xavier_uniform_(w_, generator=g)
                m_.weight.copy_(w_)
                if m_.bias is not None:
                    m_.bias.zero_()
                n += 1
            elif isinstance(m_, nn.LayerNorm):
                m_.weight.fill_(1.0)
                m_.bias.zero_()
                n += 1
        net.norm.weight.fill_(1.0)
        net.norm.bias.zero_()
        w_ = torch.empty(net.out.weight.shape)
        nn.init.kaiming_uniform_(w_, a=math.sqrt(5), generator=g)
        net.out.weight.copy_(w_)
        bnd = 1.0 / math.sqrt(net.out.in_features)
        net.out.bias.copy_(torch.empty(net.out.bias.shape).uniform_(-bnd, bnd, generator=g))
        if hasattr(net, "dc_side"):
            net.dc_side.weight.zero_()
            net.dc_side.bias.zero_()
    log(f"T2-D546 (a): re-drew the top encoder block + final norm ({n} modules), the head (norm + out) and the side head")


def load_member_state(zip_path, file):
    """T2-D546 (b) build: a PACKAGED ST member as an init_state for fit_eval_fm. Returns (state, member, side_blob): the member's fp16
    weights upcast to fp32 (exact: every fp16 value is an fp32 value; the round trip back to fp16 is asserted bitwise), its side head
    (fm_<i>_side.pt, when the member has one) added as dc_side.weight / dc_side.bias, the member's config.json entry, and the side blob
    (kind, n0, mu, sd, weight, bias) for the caller's prior check."""
    import io
    import json as _json
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        cfg = _json.loads(z.read("config.json"))
        mem = [m for m in cfg["fm_members"] if m["file"] == file]
        assert len(mem) == 1, (zip_path, file, [m["file"] for m in cfg["fm_members"]])
        mem = mem[0]
        sd16 = torch.load(io.BytesIO(z.read(file)), map_location="cpu", weights_only=True)
        sb = None
        if mem.get("side"):
            sb = torch.load(io.BytesIO(z.read(mem["side"]["file"])), map_location="cpu", weights_only=True)
    assert not any(k.startswith("dc_side.") for k in sd16), "T2-D546: a side-head key inside the member file"
    st = {k: (v.float() if v.is_floating_point() else v) for k, v in sd16.items()}
    for k, v in sd16.items():
        if v.is_floating_point():
            assert v.dtype == torch.float16 and torch.equal(st[k].to(torch.float16), v), (k, v.dtype)
    if sb is not None:
        st["dc_side.weight"], st["dc_side.bias"] = sb["weight"].float(), sb["bias"].float()
    return st, mem, sb


def recomb_masks(L, S, fade, device=None, dtype=None):
    """T2-D583: S partition-of-unity masks over L samples (segment s = 1 inside its L / S block, raised-cosine cross-fades of `fade` samples
    centred on the internal joins); sum over s == 1 at every sample."""
    import torch
    seg = L // S
    t = torch.arange(L, dtype=torch.float64)
    M = torch.zeros(S, L, dtype=torch.float64)
    for s_ in range(S):
        lo, hi = s_ * seg, (s_ + 1) * seg
        m = ((t >= lo) & (t < hi)).double()
        if fade > 0:
            if s_ > 0:                                   # rising edge around the join at lo
                r = (t - (lo - fade / 2)) / fade
                m = torch.where((r > 0) & (r < 1), 0.5 - 0.5 * torch.cos(math.pi * r), m)
            if s_ < S - 1:                               # falling edge around the join at hi
                r = (t - (hi - fade / 2)) / fade
                m = torch.where((r > 0) & (r < 1), 0.5 + 0.5 * torch.cos(math.pi * r), m)
        M[s_] = m
    M = M / M.sum(0, keepdim=True)
    return M.to(device=device, dtype=dtype if dtype is not None else torch.float32)


def recomb_batch(xb, b, rb, side=None):
    """T2-D583: within-class, within-user segment recombination of a target batch (rb = recomb_setup's state): with probability p a window is
    rebuilt as sum_s mask_s * X[donor_s] with donor_s drawn uniformly from the window's own (user, class) group of the TRAINING windows.
    side (Track3 LOW on 9de94fd7): the fold's training side vectors (the dcmean causal z, which cannot be recomputed from an aligned window); when
    given, returns (xb, sb) with a recombined window's side vector = sum_s w_s * side[donor_s], w_s = mask s's mass (mean over samples, sum 1),
    and an untouched window's own side[b]. The RNG draws are the same with or without side, so the x path is unchanged."""
    import torch
    bc = b.detach().to("cpu").long()
    sb = side[b] if side is not None else None
    sel = torch.rand(len(bc), generator=rb["gen"]) < rb["p"]
    n_sel = int(sel.sum())
    rb["count"] += n_sel
    rb["seen"] += len(bc)
    if n_sel == 0:
        return (xb, sb) if side is not None else xb
    g = rb["gid"][bc[sel]]                               # (n_sel,) group ids
    size = rb["size"][g]                                 # (n_sel,)
    out = torch.zeros_like(xb[sel])
    osb = torch.zeros_like(sb[sel.to(sb.device)]) if side is not None else None
    for s_ in range(rb["S"]):
        j = (torch.rand(n_sel, generator=rb["gen"]) * size.double()).long().clamp(max=size - 1)
        donor = rb["members"][g, j]                      # training-window indices
        out = out + rb["masks"][s_].to(xb.dtype) * rb["X"][donor.to(rb["X"].device)]
        if side is not None:
            osb = osb + rb["mw"][s_].to(side.dtype) * side[donor.to(side.device)]
    xb = xb.clone()
    xb[sel.to(xb.device)] = out
    if side is None:
        return xb
    sb = sb.clone()
    sb[sel.to(sb.device)] = osb
    return xb, sb


def recomb_setup(Xtr, ytr, utr, S, p, fade, seed, log=print):
    """T2-D583: the (user, class) groups of the fold's TRAINING windows (the only pool) and the masks; its own CPU generator (seed + 5830)."""
    import torch
    assert S >= 2 and 0 < p <= 1 and Xtr.shape[-1] % S == 0, (S, p, Xtr.shape)
    u_ = np.asarray(utr).astype(np.int64)
    y_ = (ytr.detach().cpu().numpy() if hasattr(ytr, "detach") else np.asarray(ytr)).astype(np.int64)
    keys = sorted(set(zip(u_.tolist(), y_.tolist())))
    kid = {k: i for i, k in enumerate(keys)}
    gid = np.array([kid[(a, c)] for a, c in zip(u_.tolist(), y_.tolist())], np.int64)
    sizes = np.bincount(gid, minlength=len(keys))
    members = np.zeros((len(keys), int(sizes.max())), np.int64)
    for i in range(len(keys)):
        ix = np.flatnonzero(gid == i)
        members[i, :len(ix)] = ix
    rb = dict(S=int(S), p=float(p), X=Xtr, gid=torch.as_tensor(gid), size=torch.as_tensor(sizes), members=torch.as_tensor(members),
              masks=recomb_masks(Xtr.shape[-1], int(S), float(fade), device=Xtr.device), gen=torch.Generator(device="cpu").manual_seed(int(seed) + 5830),
              count=0, seen=0)
    rb["mw"] = rb["masks"].double().mean(1).to(rb["masks"].dtype)   # each slot's mass (sum 1): the side-vector weights (Track3 LOW)
    log(f"segment recombination: S {S} x {Xtr.shape[-1] // S} samples, p {p:g}, cross-fade {fade:g} samples; {len(keys)} (user, class) groups "
        f"of the training windows (sizes {int(sizes.min())}-{int(sizes.max())})")
    return rb


def mirror_setup(p, ch_names, seed, log=print):
    """T2-D618: the left-right mirror's state. The homologous map from the channel NAMES (10-20 rule: an odd number <-> the next even one,
    e.g. C3 <-> C4, PO7 <-> PO8; a 'z' name is midline and fixed); ASSERTED an involution whose fixed points are exactly the midline names;
    probability p per window; its own CPU generator (seed + 6180)."""
    import re
    import torch
    names = [str(c) for c in ch_names]
    pos = {c.upper(): i for i, c in enumerate(names)}
    assert 0 < float(p) <= 1 and len(pos) == len(names), (p, names)
    perm = []
    for c in names:
        m_ = re.fullmatch(r"([A-Za-z]+?)(\d+|[zZ])", c)
        assert m_, f"T2-D618: no 10-20 name: {c}"
        pre, num = m_.groups()
        if num in ("z", "Z"):
            perm.append(pos[c.upper()])
            continue
        h = f"{pre}{int(num) + 1 if int(num) % 2 else int(num) - 1}"
        assert h.upper() in pos, f"T2-D618: {c} has no homologous channel {h} in the montage"
        perm.append(pos[h.upper()])
    perm = np.asarray(perm, np.int64)
    assert np.array_equal(perm[perm], np.arange(len(names))), "T2-D618: the homologous map is not an involution"
    fixed = [names[i] for i in range(len(names)) if perm[i] == i]
    assert all(c[-1] in "zZ" for c in fixed), fixed
    pairs = [f"{names[i]}-{names[perm[i]]}" for i in range(len(names)) if perm[i] > i]
    log(f"left-right mirror: p {float(p):g} per target window; {len(pairs)} homologous pairs ({' '.join(pairs)}), {len(fixed)} midline fixed "
        f"({' '.join(fixed)}); the dcmean side z permuted with it")
    return dict(p=float(p), perm=torch.as_tensor(perm), C=len(names), gen=torch.Generator(device="cpu").manual_seed(int(seed) + 6180),
                count=0, seen=0, pairs=len(pairs))


def mirror_batch(xb, Sb, mr):
    """T2-D618: with probability p a window x (C, T) becomes x[perm] (left <-> right homologous channels, midline fixed); its dcmean side z
    (B, 2C) = [z(m), z(m - mean m)], both blocks in channel order, takes the same permutation per block (a mirrored stream's per-feature causal
    z = the permuted z when the training prior is mirrored with it). Sb None (no side head) passes through."""
    import torch
    sel = torch.rand(len(xb), generator=mr["gen"]) < mr["p"]
    n_sel = int(sel.sum())
    mr["count"] += n_sel
    mr["seen"] += len(xb)
    if n_sel == 0:
        return xb, Sb
    assert xb.shape[1] == mr["C"], ("T2-D618: the batch's channels != the montage", tuple(xb.shape), mr["C"])
    sd = sel.to(xb.device)
    perm = mr["perm"].to(xb.device)
    xb = torch.where(sd[:, None, None], xb[:, perm, :], xb)
    if Sb is not None:
        assert Sb.shape[1] == 2 * mr["C"], ("T2-D618: the side z is not the 2C dcmean layout", tuple(Sb.shape))
        p2 = torch.cat([perm, perm + mr["C"]])
        Sb = torch.where(sd[:, None], Sb[:, p2], Sb)
    return xb, Sb


def phase_setup(p, amax, seed, log=print):
    """T2-D586: the shared-phase Fourier surrogate's state (probability p per window, phase bound amax); its own CPU generator (seed + 5860)."""
    import torch
    assert 0 < float(p) <= 1 and 0 < float(amax) <= math.pi, (p, amax)
    log(f"shared-phase surrogate: p {float(p):g}, phi ~ U(-{float(amax):.4f}, +{float(amax):.4f}) per window and bin, shared by all channels; "
        "DC / Nyquist kept")
    return dict(p=float(p), amax=float(amax), gen=torch.Generator(device="cpu").manual_seed(int(seed) + 5860), count=0, seen=0)


def phase_batch(xb, ph):
    """T2-D586: with probability p a window x (C, T) becomes irfft(rfft(x) * exp(i phi)), phi (one value per frequency bin, the SAME for every
    channel, so every channel's amplitude spectrum and every cross-spectrum are kept) ~ U(-amax, +amax); DC (and Nyquist for even T) phi 0."""
    import torch
    sel = torch.rand(len(xb), generator=ph["gen"]) < ph["p"]
    n_sel = int(sel.sum())
    ph["count"] += n_sel
    ph["seen"] += len(xb)
    if n_sel == 0:
        return xb
    T = xb.shape[-1]
    nb = T // 2 + 1
    phi = (torch.rand(n_sel, nb, generator=ph["gen"], dtype=torch.float64) * 2 - 1) * ph["amax"]
    phi[:, 0] = 0.0
    if T % 2 == 0:
        phi[:, -1] = 0.0
    rot = torch.polar(torch.ones_like(phi), phi).to(torch.complex64).to(xb.device)
    sd = sel.to(xb.device)
    y = torch.fft.irfft(torch.fft.rfft(xb[sd].float(), dim=-1) * rot[:, None, :], n=T, dim=-1).to(xb.dtype)
    xb = xb.clone()
    xb[sd] = y
    return xb


def word_margin(lo, y, m=0.5):
    """T2-D596: mean over the confuser pairs of softplus(m - (z_y - z_c)); the pairs = (WORD, MI), (WORD, SUB) for WORD rows (y 2), (MI, WORD)
    for MI rows (y 0), (SUB, WORD) for SUB rows (y 1) (labels MI 0 / SUB 1 / WORD 2). lo (B, 3) float32 logits, y (B,) long."""
    import torch.nn.functional as F_
    zy = lo.gather(1, y[:, None])[:, 0]
    w = y == 2
    terms = [F_.softplus(m - (zy[w] - lo[w, 0])), F_.softplus(m - (zy[w] - lo[w, 1])), F_.softplus(m - (zy[~w] - lo[~w, 2]))]
    t = torch.cat(terms)
    return t.mean() if len(t) else lo.sum() * 0.0


def sym_kl(lo1, lo2):
    """T2-D588: 0.5 (KL(p1 || p2) + KL(p2 || p1)) from two logit tensors (float32), batch mean; gradients through both (R-Drop form)."""
    import torch.nn.functional as F_
    l1, l2 = F_.log_softmax(lo1.float(), -1), F_.log_softmax(lo2.float(), -1)
    return 0.5 * ((l1.exp() * (l1 - l2)).sum(-1).mean() + (l2.exp() * (l2 - l1)).sum(-1).mean())


def fit_eval_fm(fm, data, seed, ch_names, epochs=30, bs=64, lr_enc=1e-4, lr_head=1e-3, wd=0.05, llrd=1.0, ls=0.1,
                warm=0.1, readout="mean", layer=-1, drop_path=0.0, ema=0.0, device="cuda", log=print,
                lora=0, lora_alpha=0.0, lora_dropout=0.2, mixup=0.0, calib="zwin", fm_ckpt="", fuse="", band=None,
                return_state=False, supcon=0.0, supcon_mode="xuser", supcon_tau=0.1, qkv_lr_mult=1.0, layer_clip=False,
                lora_in=0, lora_in_roles=False, lora_in_dropout=0.1, mae_retain_every=0, mae_mask=0.75, return_model=False, vp=0,
                side="", rc=0.0, sw=None, aug=None, lp_epochs=0, reinit_top=0, wise=0.0, micro=1, head_freeze=False, head_wd=None,
                token_drop=0.0, init_state=None, tilt=0.0, pkg=None, train_eval=False, cont_reinit=False, recomb=None, phase=None, cons=0.0, wmargin=0.0, mirror=0.0,
                chemb_lr=None, dro_eta=0.0, grl=None, grl_lr=1e-3, grl_ramp=0.3, grl_hold=0.1):
    """Harnessing knobs (all off by default = the D82 recipe): layer (1-indexed exit, -1 = final), readout mean | attn,
    drop_path (ST only), ema (per-step decay of an evaluated EMA copy), llrd (depth-normalised layer decay), lora (rank;
    base frozen, norms + LoRA + head train; alpha default 4r), mixup (Beta(a, a) on the raw windows, both labels),
    fuse (a ni26.train CNN name, e.g. fbcnet: joint FM + CNN feature fusion, FuseClf; target corpus only),
    band ((lo, hi) Hz: per-window band-pass of every input window before the FM's own prep, e.g. FBCNet's 4-40 Hz),
    sw (T2-D253: per-window loss weights of the target training windows, aligned with Xtr; None = unweighted),
    aug (T2-D257: (gain_jit, tshift) = T2-D112's G1 / G2 on the target training windows, ni26.train.augment after rc and before vp;
    T2-D260: (gain_jit, tshift, chdrop) adds G4's whole-channel zeroing with probability chdrop per window and channel;
    per-window per-channel gain x exp(U(-g, g)) and a circular time shift U{-s..s} samples of the input rate; None = off),
    lp_epochs (T2-D263: LP-FT; the first lp_epochs epochs train the head only, the encoder frozen = no gradient, no AdamW state, no
    weight decay; then the encoder's groups re-warm over warm x the remaining steps and decay cosine to 0 with them; the head keeps the
    whole run's schedule; 0 = off),
    reinit_top (T2-D264: the top k ST encoder blocks + its final norm re-drawn before the fine-tune with the ST MAE's own init, a
    generator of their own so every other draw is the baseline's; 0 = off),
    wise (T2-D270: WiSE-FT on LP-FT, Wortsman et al. 2022; needs lp_epochs: the evaluated weights = (1 - wise) x the fine-tune's end
    + wise x the probe's end (the CPT'd encoder + the probed head), every floating tensor of the model; 0 = off)."""
    """data: Xtr,ytr,Xva,yva,cva,Xte,yte,cte (numpy, aligned windows). Same return keys as ni26.train.fit_eval."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    if band:
        data = dict(data, Xtr=bandpass(data["Xtr"], band), Xva=bandpass(data["Xva"], band) if len(data["Xva"]) else data["Xva"],
                    Xte=bandpass(data["Xte"], band), ext=[(bandpass(xe, band), ye, ne) for xe, ye, ne in data.get("ext", [])])
        log(f"input band-pass {band[0]:g}-{band[1]:g} Hz (per window, order 4, zero phase)")
    if fm_ckpt:
        use_cpt(fm, fm_ckpt, log)
    enc = make_encoder(fm, ch_names, log, drop_path, calib)
    if drop_path > 0 and fm in t1.MOR_NAMES:                  # make_encoder's drop path reaches ST only (T2-D128 S2)
        mor_drop_path(enc, drop_path, log)
    L = enc.n_layers if layer in (-1, None) else int(layer)
    if reinit_top:                      # T2-D264: re-initialise the top block(s) (Zhang et al. 2021) with steeg_mae._init_weights' rule:
        k_ = int(reinit_top)            # xavier-uniform Linear weights, zero biases, LayerNorm weight 1 / bias 0; + the final norm
        assert isinstance(enc, STEEG4) and 0 < k_ < enc.n_layers and L == enc.n_layers and not (lora or lora_in or mae_retain_every), reinit_top
        gr_ = torch.Generator().manual_seed(7919 + int(seed))
        with torch.no_grad():
            for m_ in [m_ for blk in enc.vit.blocks[-k_:] for m_ in blk.modules()] + [enc.vit.norm]:
                if isinstance(m_, nn.Linear):
                    assert m_.weight.device.type == "cpu", m_.weight.device
                    nn.init.xavier_uniform_(m_.weight, generator=gr_)
                    if m_.bias is not None:
                        nn.init.zeros_(m_.bias)
                elif isinstance(m_, nn.LayerNorm):
                    nn.init.ones_(m_.weight)
                    nn.init.zeros_(m_.bias)
        log(f"re-initialised the top {k_} of {enc.n_layers} encoder blocks + the final norm (the MAE's own init)")
    if lora:
        t1.inject_lora(enc, L, int(lora), float(lora_alpha or 4 * lora), float(lora_dropout), log)
    if lora_in:                         # T2-D130: input-dropout LoRA, alpha = 2r, roles = a variant for the in-house MOR model
        inject_lora_input(enc, int(lora_in), 2.0 * lora_in, float(lora_in_dropout), log, roles=lora_in_roles)
    if fuse:
        assert not data.get("ext"), "--fm-fuse: the CNN branch needs the target montage (no --extra)"
        from ni26 import train as T
        C_, L_ = data["Xtr"].shape[1], data["Xtr"].shape[2]
        torch.manual_seed(seed)
        net = FuseClf(enc, L, readout, 3, T.build_model(fuse, C_, 3, L_, 120.0), C_, L_).to(device)
        log(f"fusion: {fuse} branch {sum(p.numel() for p in net.cnn.parameters())} params, feature {net.norm_cnn.normalized_shape[0]}")
    else:
        net = FMClf(enc, L, readout, 3).to(device)
    if os.environ.get("NI26_FM_GRADCKPT") == "1":        # T2-D565: activation checkpointing of the ST encoder blocks (the SAME computation, re-run in
        from torch.utils.checkpoint import checkpoint as _ckpt   # backward; RNG state preserved) so the arms' batch fits a 32 GB GPU; off = bitwise as before
        assert hasattr(enc, "vit") and len(enc.vit.blocks), "NI26_FM_GRADCKPT: ST-EEGFormer encoders only"
        assert not ema, "NI26_FM_GRADCKPT: an EMA deepcopy would call the ORIGINAL blocks through these closures (the C' arms run ema 0)"
        for blk_ in enc.vit.blocks:
            def _fwd(*a_, _b=blk_, _f=blk_.forward, **k_):
                if _b.training and torch.is_grad_enabled():
                    return _ckpt(_f, *a_, use_reentrant=False, **k_)
                return _f(*a_, **k_)
            blk_.forward = _fwd
        log(f"NI26_FM_GRADCKPT: activation checkpointing on {len(enc.vit.blocks)} encoder blocks")
    if init_state is not None and side != "dcmean":       # T2-D397: start from a fine-tuned classifier's state (a per-user copy of the pooled member)
        assert not (fuse or side or lora or lora_in), "T2-D397: init_state on the plain FMClf only"
        net.load_state_dict(init_state, strict=True)
        log(f"init_state: {len(init_state)} tensors loaded (strict)")
    side_dc = side == "dcmean"
    if side_dc:                                           # T2-D524: the 82 causal-z window means at the logits (t2p_bench --side dcmean data)
        assert not fuse and not data.get("ext") and data.get("side_tr") is not None and len(data["side_tr"]) == len(data["Xtr"]), "fm dcmean side"
        assert not (lora or lora_in or mae_retain_every or wise or lp_steps_guard(lp_epochs) or train_eval or int(micro) != 1 or supcon > 0), \
            "T2-D524: the dcmean side head is wired for the plain / mixup CE path only"
        net.dc_side = nn.Linear(int(np.asarray(data["side_tr"]).shape[1]), 3).to(device)
        nn.init.zeros_(net.dc_side.weight)
        nn.init.zeros_(net.dc_side.bias)
        log(f"window-mean side head: {net.dc_side.in_features} -> 3 (zero init; plain / mixup CE)")
    elif side:                                            # T2-X1: the amplitude side input (a head-group module at lr_head)
        assert side == "amp" and not fuse and not data.get("ext"), side
        ls_ = AmpSide.logstd(torch.as_tensor(np.ascontiguousarray(data["Xtr"]))).numpy()
        mu_, sd_ = ls_.mean(0), ls_.std(0) + 1e-3
        net.amp_side = AmpSide(mu_, sd_, 3).to(device)
        log(f"amplitude side input: {len(mu_)} channel log-stds, training mean {float(mu_.mean()):+.3f}, sd {float(sd_.mean()):.3f} (zero-initialised)")
    if init_state is not None and side_dc:               # T2-D546: a converged SD member's state, its side head included (strict)
        assert not (fuse or lora or lora_in), "T2-D546: init_state with the dcmean side on the plain FMClf only"
        net.load_state_dict(init_state, strict=True)
        log(f"init_state (with the dcmean side head): {len(init_state)} tensors loaded (strict)")
    if cont_reinit:                                      # T2-D546 arm (a): re-draw the head + side head + the top ST block / final norm
        assert init_state is not None and isinstance(enc, STEEG4) and L == enc.n_layers and readout == "mean", "T2-D546: cont_reinit needs init_state, ST, mean readout"
        d546_reinit(net, enc, seed, log)
    if head_freeze or token_drop:                        # T2-D349b: the user's "random last-layer weights" / a token-level head regulariser
        assert readout in ("grid", "grid64d") and not fuse, "--fm-head-freeze / --fm-token-drop: the grid readouts only"
        if head_freeze:                                  # the random grid head stays as drawn; only its LayerNorm (net.norm) trains
            for p_ in list(net.grid_tok.parameters()) + list(net.out.parameters()):
                p_.requires_grad_(False)
        net.token_drop = float(token_drop)
        log(f"grid head: frozen random linear layers {bool(head_freeze)}, token dropout {token_drop:g}")
    groups = t1.param_groups(net, lr_enc, lr_head, llrd, wd, log)
    if head_wd is not None:                              # T2-D349b: the head's own weight decay (its decayed group only)
        for gr in groups:
            if gr.get("name") == "head":
                gr["weight_decay"] = float(head_wd)
        log(f"head weight decay {head_wd:g}")
    if qkv_lr_mult != 1.0:                                # T2-D128 S1: attention qkv projections at a lower lr
        qkv = {id(p_) for n_, p_ in net.enc.named_parameters() if "qkv" in n_ and p_.requires_grad}
        new = []
        for gr in groups:
            a_ = [p_ for p_ in gr["params"] if id(p_) in qkv]
            b_ = [p_ for p_ in gr["params"] if id(p_) not in qkv]
            if b_:
                new.append(dict(gr, params=b_))
            if a_:
                new.append(dict(gr, params=a_, lr=gr["lr"] * qkv_lr_mult, name=gr.get("name", "") + "_qkv"))
        groups = new
        log(f"qkv lr x {qkv_lr_mult} on {len(qkv)} tensors")
    if chemb_lr:                                          # T2-D644 (W27, astra): the ST channel-embedding table out of LLRD's depth-0 group
        assert isinstance(enc, STEEG4) and not (lora or lora_in), "T2-D644: --fm-chemb-lr is for a fully fine-tuned ST encoder"
        ce_ = enc.vit.enc_channel_emd.channel_transformation.weight   # nn.Embedding(145, D): rows never indexed get zero grad -> with weight
        assert ce_.requires_grad                                       # decay 0 AdamW leaves them EXACTLY unchanged (only the active rows move)
        src_ = [gr for gr in groups if any(q is ce_ for q in gr["params"])]
        assert len(src_) == 1 and sum(q is ce_ for q in src_[0]["params"]) == 1, "T2-D644: the table must sit in exactly one group"
        was_ = (src_[0].get("name"), src_[0]["lr"], src_[0]["weight_decay"])
        groups = [dict(gr, params=[q for q in gr["params"] if q is not ce_]) for gr in groups]
        groups = [gr for gr in groups if gr["params"]] + [dict(params=[ce_], lr=float(chemb_lr), weight_decay=0.0, name="enc_chemb")]
        log(f"T2-D644 channel-embedding table: its own group, peak lr {float(chemb_lr):g}, weight decay 0, the shared schedule "
            f"(was {was_[0]} lr {was_[1]:.3g} wd {was_[2]:g}); active rows {len(enc.chan_idx)} of {ce_.shape[0]}")
    layer_params = None
    if layer_clip:                                        # T2-D128 S3: each encoder layer's gradient clipped to norm 1
        layer_params = {}
        for n_, p_ in net.enc.named_parameters():
            if p_.requires_grad:
                layer_params.setdefault(net.enc.layer_of(n_), []).append(p_)
    mae_dec, prep_z = None, None
    if mae_retain_every:                # T2-D132: keep the CPT objective jointly (T3-D41 / LESSONS C24)
        assert fm_ckpt, "--fm-mae-retain needs --fm-ckpt (the CPT checkpoint holds the decoder)"
        if fm.startswith("st_"):
            import t2p_cpt
            mae_dec = t2p_cpt.steeg_mae.__dict__[t2p_cpt.MAE_ARCH[fm]](norm_pix_loss=False)
            mae_dec.load_state_dict(torch.load(fm_ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
            mae_dec = mae_dec.to(device).eval()
            for q in mae_dec.parameters():
                q.requires_grad_(False)
            S_ = t1.STEEG
            prep_z = STEEG4.prep.__get__(type("P", (), dict(FS=S_.FS, calib="zwin", CLIP_GAIN=S_.CLIP_GAIN))())
        else:                           # MOR: its own masked loss; the decoder frozen
            for n_, q in net.enc.named_parameters():
                if n_.startswith(("fm.dec_", "fm.mask_token")):
                    q.requires_grad_(False)
        groups = [dict(gr, params=[q for q in gr["params"] if q.requires_grad]) for gr in groups]
        groups = [gr for gr in groups if gr["params"]]
        log(f"MAE retained jointly every {mae_retain_every} steps (mask {mae_mask}; per-term clip 0.5 / 0.5, sum <= 1.0)")
    sc_pairs, proj = None, None
    if supcon > 0:                      # T2-D127: the projection head is NOT part of net (not in the saved state)
        assert not data.get("ext") and not fuse, "--fm-supcon: target corpus only"
        sc_pairs = SupConPairs(data["ytr"], data["utr"], data["str"], supcon_mode, seed)
        proj = nn.Sequential(nn.Linear(enc.dim, 256), nn.GELU(), nn.Linear(256, 128)).to(device)
        groups = groups + [dict(params=list(proj.parameters()), lr=lr_head, weight_decay=wd)]
        log(f"SupCon {supcon_mode}: lambda {supcon}, tau {supcon_tau}; {len(sc_pairs.anchors)} / {len(data['ytr'])} training "
            f"windows have a positive")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    base_lr = [gr["lr"] for gr in opt.param_groups]
    T_ = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
    Xtr, ytr, Xva, Xte = T_(data["Xtr"]), T_(data["ytr"], torch.long), T_(data["Xva"]), T_(data["Xte"])
    if side_dc:                                           # T2-D524
        Str_, Ste_ = T_(data["side_tr"]), T_(data["side_te"])
        Sva_ = T_(data["side_va"]) if len(data["yva"]) else None
    PP = (lambda m_, X_, S_: predict_proba_side(m_, X_, S_)) if side_dc else (lambda m_, X_, S_: predict_proba(m_, X_))  # noqa: E731
    # sources: the target corpus (its montage = the encoder's build montage) + external corpora in their NATIVE montages
    # (t1 SourceChannels: the FM sees each source's own channel names); one source per batch, p ~ its window count
    srcs = [(Xtr, ytr, None)] + [(T_(xe), T_(ye, torch.long), list(ne)) for xe, ye, ne in data.get("ext", [])]
    n_all = sum(len(s_[0]) for s_ in srcs)
    p_src = np.array([len(s_[0]) for s_ in srcs], float) / n_all
    rng = np.random.default_rng(seed)
    if float(cons) > 0:                 # T2-D588 (A; the manager after the H200 OOM of smoke 1638461): per-view batch = bs // 2, so the two views make
        bs = int(bs) // 2               # bs forwards per step = the base line's memory; the steps per epoch double (cons 0: untouched)
        log(f"T2-D588 (A): per-view batch {bs} ({2 * bs} forwards per step); steps per epoch {int(math.ceil(n_all / bs))}")
    spe = int(math.ceil(n_all / bs))
    total, wsteps = epochs * spe, int(warm * epochs * spe)
    ema_net = copy.deepcopy(net).eval() if ema > 0 else None
    step, best, curves, t0 = 0, (-1.0, -1, None), [], time.time()
    rvp = None
    if vp:                              # T2-D198: T2-D195's virtual participants on the target corpus's windows (the caller fills VP_BANK)
        from ni26 import train as TV
        assert TV.VP_BANK and TV.VP_BANK["mats"].shape[1] == Xtr.shape[1], "fm vp needs ni26.train.VP_BANK for the target montage"
        rvp = TV.Recipe("fm_vp", vp=int(vp))
        log(f"virtual participants on the target windows: {len(TV.VP_BANK['mats'])} draws")
    W_ = None
    wnorm_ = data.get("wtr") is not None          # T2-D642 (W25): data['wtr'] = train.py's user-balance weights, self-normalised per batch
    if wnorm_:
        assert sw is None, "fm: data['wtr'] (T2-D642) and sw (T2-D253) are exclusive"
        sw = data["wtr"]
        assert np.isfinite(np.asarray(sw, np.float64)).all() and (np.asarray(sw) > 0).all(), "fm wtr: finite and > 0"
        assert int(micro) == 1, "fm wtr (T2-D642): micro > 1 would self-normalise each chunk by its own sum w (not the batch's): micro 1 only"
        assert cons == 0 and wmargin == 0 and pkg is None and supcon == 0 and mae_retain_every == 0, \
            "fm wtr (T2-D642): the cons / word-margin / pkg / supcon / MAE-retain terms are not weighted (Track3): off with wtr"
    dro_ = float(dro_eta) > 0            # T2-D645 (FM-DRO, Track3's registration 2026-11-19 21:03:43): online group DRO over the training USERS
    if dro_:                              # (Sagawa et al. 2020) through D642's self-normalised path; 0 = this block and every hook below inert
        assert sw is None and data.get("wtr") is None, "FM-DRO (T2-D645): exclusive with sw / wtr"
        assert int(micro) == 1 and cons == 0 and wmargin == 0 and pkg is None and supcon == 0 and mae_retain_every == 0 and not fuse \
            and not data.get("ext"), "FM-DRO (T2-D645): the target corpus' plain / mixup CE path only (micro 1; no cons / margin / pkg / supcon / MAE / fuse / ext)"
        _, uinv_ = np.unique(np.asarray(data["utr"]), return_inverse=True)
        assert len(uinv_) == len(Xtr), "FM-DRO: data['utr'] must align with Xtr"
        UD_ = torch.as_tensor(uinv_, dtype=torch.long, device=device)
        nU_ = int(uinv_.max()) + 1
        q_dro = torch.full((nU_,), 1.0 / nU_, dtype=torch.float64, device=device)
        wnorm_ = True                     # the step's loss = sum_i w_i CE_i / sum_i w_i
        log(f"T2-D645 FM-DRO: eta {float(dro_eta):g}, q uniform over {nU_} training users; w_i = q_u(i) / n_u(i) (the user's batch count); "
            "after every step q_u <- q_u exp(eta L_u) (L_u = the batch-mean CE of u's rows; under mixup the anchor's user, the mixed CE), renormalised")
    grl_on = grl is not None              # T2-D649 (W29): a subject-adversarial head through a gradient-reversal layer (None = off, bitwise)
    if grl_on:
        assert int(micro) == 1 and cons == 0 and supcon == 0 and mae_retain_every == 0 and not fuse and not data.get("ext") and pkg is None, \
            "T2-D649: the target corpus' plain / mixup CE path only (micro 1; no cons / supcon / MAE / fuse / ext / pkg)"
        assert not dro_, "T2-D649: not with FM-DRO"
        from ni26.train import GradReverse as _GRL
        _, ug_inv_ = np.unique(np.asarray(data["utr"]), return_inverse=True)
        assert len(ug_inv_) == len(Xtr), "T2-D649: data['utr'] must align with Xtr"
        UG_ = torch.as_tensor(ug_inv_, dtype=torch.long, device=device)
        nUg_ = int(ug_inv_.max()) + 1
        rg_ = np.random.default_rng(int(seed) + 64900)    # (3) the engagement split: 10 % of each TRAINING user's windows held out from the adversary
        hold_ = np.zeros(len(ug_inv_), bool)
        for u_ in range(nUg_):
            iu_ = np.flatnonzero(ug_inv_ == u_)
            hold_[rg_.choice(iu_, max(1, int(round(float(grl_hold) * len(iu_)))), replace=False)] = True
        HOLD_ = torch.as_tensor(hold_, device=device)
        _st = torch.get_rng_state(); _cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(seed) + 64901)             # the adversary's own init; the global RNG restored after (the lambda = 0 twin == C's stream)
        adv_ = nn.Sequential(nn.Linear(enc.dim, 256), nn.GELU(), nn.Linear(256, nUg_)).to(device)
        torch.set_rng_state(_st)
        if _cst is not None:
            torch.cuda.set_rng_state_all(_cst)
        adv_opt_ = torch.optim.AdamW(adv_.parameters(), lr=float(grl_lr), weight_decay=0.0)
        grl_box_ = {}
        grl_hook_ = net.out.register_forward_pre_hook(lambda m_, inp_: grl_box_.__setitem__("f", inp_[0]))   # the pooled feature of THIS forward
        grl_curve_ = []
        log(f"T2-D649 GRL: lambda max {float(grl):g} (ramp over the first {float(grl_ramp):.0%} of steps), adversary lr {float(grl_lr):g}, {nUg_} users "
            f"(chance {1.0 / nUg_:.3f}); {int(hold_.sum())} / {len(hold_)} windows held out from the adversary (the engagement split)")
    if sw is not None:                  # T2-D253: per-window loss weights (target corpus only; plain or mixup batches)
        assert not data.get("ext") and not fuse and supcon == 0 and len(sw) == len(Xtr), "fm sw: target windows, no supcon / fuse"
        W_ = torch.as_tensor(np.asarray(sw, np.float32), device=device)
        log(f"per-window loss weights: {float(W_.min()):.2f}-{float(W_.max()):.2f}, mean {float(W_.mean()):.3f}")
    rrc = None
    if rc > 0:                          # T2-D235: T2-D230's re-colouring on the target corpus's windows (maps from its training users' inputs)
        from ni26 import train as TV
        assert int(data["utr"].max()) >= 1, "fm rc needs >= 2 training users"
        urc = torch.as_tensor(np.asarray(data["utr"]), dtype=torch.long, device=device)
        rcm = TV.rc_bank(Xtr, urc, int(data["utr"].max()) + 1, float(rc))
        rrc = TV.Recipe("fm_rc", rc=float(rc))
        log(f"re-colouring rc {rc:g} on the target windows: {len(rcm)} training users")
    rtl = TV.Recipe("fm_tilt", tilt=float(tilt)) if tilt > 0 else None   # backlog 94 (0 = off = bitwise)
    rrb = None
    if recomb:                          # T2-D583: within-user, within-class segment recombination (S, p); None = off = bitwise
        assert not data.get("ext") and not fuse and supcon == 0, "fm recomb: target windows only"
        rrb = recomb_setup(Xtr, ytr, data["utr"], int(recomb[0]), float(recomb[1]), 6.0, seed, log)
    rph = None
    if phase:                           # T2-D586: the shared-phase Fourier surrogate (p, amax); None = off = bitwise
        assert not data.get("ext") and not fuse and supcon == 0, "fm phase: target windows only"
        rph = phase_setup(float(phase[0]), float(phase[1]), seed, log)
    rmr = None
    if float(mirror) > 0:               # T2-D618: the left-right mirror (p); 0 = off = bitwise
        assert not data.get("ext") and not fuse and supcon == 0, "fm mirror: target windows only"
        rmr = mirror_setup(float(mirror), ch_names, seed, log)
    PK = None
    if pkg is not None:                 # backlog 96: + beta x the package NLL through the fixed out-of-fold CNN mix (None = off = bitwise)
        pk_p, pk_w, pk_beta = pkg
        assert len(pk_p) == len(Xtr) and not data.get("ext") and supcon == 0 and int(micro) == 1 and not fuse, "fm pkg: target windows, plain / mixup CE"
        assert 0 < float(pk_w) <= 1 and float(pk_beta) > 0, pkg
        PK = torch.as_tensor(np.asarray(pk_p, np.float32), device=device)
        pk_w, pk_beta = float(pk_w), float(pk_beta)
        log(f"package loss: + {pk_beta:g} x NLL((1 - {pk_w:g}) p_CNN_oof + {pk_w:g} p_member); OOF CNN p(y) mean "
            f"{float(PK[torch.arange(len(PK), device=device), torch.as_tensor(np.asarray(data['ytr']), device=device).long()].mean()):.3f}")
    raug = None
    if aug is not None:                 # T2-D257: T2-D112's gain jitter + time shift on the target corpus's windows (torch RNG, on only)
        from ni26 import train as TV
        cd_ = float(aug[2]) if len(aug) > 2 else 0.0   # T2-D260: channel dropout (a 2-tuple = T2-D257's path exactly)
        assert len(aug) in (2, 3) and (float(aug[0]) > 0 or int(aug[1]) > 0 or cd_ > 0) and 0 <= cd_ < 1, aug
        raug = TV.Recipe("fm_aug", gain_jit=float(aug[0]), tshift=int(aug[1]), chdrop=cd_)
        log(f"gain jitter {float(aug[0]):g} + time shift +-{int(aug[1])} samples + channel dropout {cd_:g} on the target windows")
    cons = float(cons)
    wmargin = float(wmargin)
    if wmargin > 0:                     # T2-D596: the WORD-margin loss (0 = off = bitwise)
        assert not data.get("ext") and not fuse and supcon == 0 and int(micro) == 1 and cons == 0 and not mae_retain_every, \
            "fm wmargin: target windows, the plain / mixup CE path only (not with consistency)"
        log(f"T2-D596: WORD-margin loss {wmargin:g} x mean softplus(0.5 - (z_y - z_c)) over (WORD, MI) / (WORD, SUB) / (MI, WORD) / (SUB, WORD), "
            f"ramped over the first 10 % of the steps")
    if cons > 0:                        # T2-D588: consistency between two independently augmented views (0 = off = bitwise)
        assert not data.get("ext") and not fuse and supcon == 0 and int(micro) == 1 and PK is None and not mae_retain_every, \
            "fm cons: target windows, the plain / mixup CE path only"
        assert any(r_ is not None for r_ in (rrb, rrc, raug, rph, rmr, rvp, rtl)), "fm cons: no augmentation is on, the two views would be identical"
        log(f"consistency: two independently augmented views per target batch (the same chain twice, the same mixup lambda / pairs), "
            f"loss 0.5 (CE1 + CE2) + {cons:g} x symmetric KL(p1, p2)")
    lp_steps = int(lp_epochs) * spe
    if lp_steps:                        # T2-D263: linear probe, then fine-tune (Kumar et al. 2022)
        assert 0 < int(lp_epochs) < epochs and not (lora or lora_in or mae_retain_every or fuse or supcon > 0 or data.get("ext")), lp_epochs
        enc_ix = [i_ for i_, gr in enumerate(opt.param_groups) if str(gr.get("name", "")).startswith("enc_")]
        enc_req = [q for q in net.enc.parameters() if q.requires_grad]
        assert enc_ix and enc_req and {id(q) for i_ in enc_ix for q in opt.param_groups[i_]["params"]} == {id(q) for q in enc_req}
        for q in enc_req:
            q.requires_grad_(False)
        t2_ = total - lp_steps
        w2_ = int(warm * t2_)
        log(f"linear probe: epochs 1-{int(lp_epochs)} of {epochs} train the head only ({len(enc_req)} encoder tensors frozen); then the "
            f"encoder re-warms over {w2_} of its {t2_} steps, cosine to 0")
    snap_lp = None
    if wise:                            # T2-D270: WiSE-FT (Wortsman et al. 2022) between the probe's end and the fine-tune's end
        assert lp_steps and 0 < float(wise) < 1 and ema_net is None, (wise, lp_steps, ema)
        log(f"WiSE-FT: the evaluated weights = {1 - float(wise):g} x the fine-tune's end + {float(wise):g} x the probe's end (step {lp_steps}: "
            f"the CPT'd encoder + the probed head)")
    for ep in range(epochs):
        net.train()
        perms = [torch.randperm(len(s_[0]), generator=g).to(device) for s_ in srcs]
        ptr = [0] * len(srcs)
        tl, tn = 0.0, 0
        tla_ = 0.0                          # T2-D649 (Track3 LOW): the adversary CE, logged apart from the task loss
        ckl, ckn = 0.0, 0                   # T2-D588: the epoch's consistency term
        wml, wmn, wms = 0.0, 0, 0.0         # T2-D596: the epoch's margin term; the same-class share of the mixup pairs (the full hard margin)
        for _ in range(spe):
            k = int(rng.choice(len(srcs), p=p_src))
            if ptr[k] + bs > len(srcs[k][0]):
                perms[k], ptr[k] = torch.randperm(len(srcs[k][0]), generator=g).to(device), 0
            b = perms[k][ptr[k]:ptr[k] + bs]
            ptr[k] += bs
            if sc_pairs is not None:        # T2-D127: anchors + positives instead of a plain batch
                rows_sc = sc_pairs.draw(bs // 2)
                b = torch.as_tensor(rows_sc, device=device)
            xb, yb, names = srcs[k][0][b], srcs[k][1][b], srcs[k][2]
            wb = W_[b] if W_ is not None else None
            if dro_:                        # T2-D645: the batch rows' DRO weights (q of the row's user / that user's count in the batch)
                ub_ = UD_[b]
                cnt_ = torch.bincount(ub_, minlength=nU_).to(torch.float64)
                wb = (q_dro[ub_] / cnt_[ub_]).float()
            Sb_ = Str_[b] if side_dc else None   # T2-D524's side vectors of the batch (the recombined ones under D583)
            xb_raw, Sb_raw = (xb, Sb_) if cons > 0 else (None, None)   # T2-D588: the un-augmented batch, for the second view
            if rrb is not None and k == 0:  # T2-D583: FIRST (on the aligned windows), then rc -> aug -> vp
                if side_dc:                 # Track3 LOW on 9de94fd7: the side z follows its donors (slot-mass-weighted mean)
                    xb, Sb_ = recomb_batch(xb, b, rrb, Str_)
                else:
                    xb = recomb_batch(xb, b, rrb)
            if rrc is not None and k == 0:  # T2-D235: before vp, as in the CNN recipes
                xb = TV.augment(xb, rrc, urc[b], rcm)
            if raug is not None and k == 0:  # T2-D257: after rc, before vp (ni26.train.augment's own order)
                xb = TV.augment(xb, raug)
            if rph is not None and k == 0:  # T2-D586: after rc -> aug, before vp
                xb = phase_batch(xb, rph)
            if rmr is not None and k == 0:  # T2-D618: after rc -> aug -> phase, before vp (the side z permuted with the window)
                xb, Sb_ = mirror_batch(xb, Sb_, rmr)
            if rvp is not None and k == 0:
                xb = TV.augment(xb, rvp)
            if rtl is not None and k == 0:  # backlog 94: the CNN's spectral tilt (last, as in ni26.train.augment)
                xb = TV.augment(xb, rtl)
            xb2 = Sb2_ = None
            if cons > 0 and k == 0:         # T2-D588: the SAME chain again on the un-augmented batch (independent draws)
                xb2, Sb2_ = xb_raw, Sb_raw
                if rrb is not None:         # its own donors -> its own side z (the D583 rule); without recomb Sb2_ = Sb_
                    if side_dc:
                        xb2, Sb2_ = recomb_batch(xb2, b, rrb, Str_)
                    else:
                        xb2 = recomb_batch(xb2, b, rrb)
                if rrc is not None:
                    xb2 = TV.augment(xb2, rrc, urc[b], rcm)
                if raug is not None:
                    xb2 = TV.augment(xb2, raug)
                if rph is not None:
                    xb2 = phase_batch(xb2, rph)
                if rmr is not None:         # T2-D618: its own draw, its own side z
                    xb2, Sb2_ = mirror_batch(xb2, Sb2_, rmr)
                if rvp is not None:
                    xb2 = TV.augment(xb2, rvp)
                if rtl is not None:
                    xb2 = TV.augment(xb2, rtl)
                if step == 0:
                    log(f"consistency views: mean |x1 - x2| {float((xb - xb2).abs().mean()):.4g} (mean |x1| {float(xb.abs().mean()):.4g})")
            f = (step + 1) / max(wsteps, 1) if step < wsteps else 0.5 * (1 + math.cos(math.pi * (step - wsteps) / max(total - wsteps, 1)))
            for gr, lr0 in zip(opt.param_groups, base_lr):
                gr["lr"] = lr0 * f
            if lp_steps:                    # T2-D263: encoder lr 0 for the probe, then its own warm-up + cosine over the remaining steps
                if step == lp_steps:
                    for q in enc_req:
                        q.requires_grad_(True)
                    if wise:                # T2-D270: the probe's end (the encoder untouched since it loaded)
                        snap_lp = {k_: v_.detach().clone() for k_, v_ in net.state_dict().items()}
                s2_ = step - lp_steps
                fe = 0.0 if s2_ < 0 else ((s2_ + 1) / max(w2_, 1) if s2_ < w2_ else 0.5 * (1 + math.cos(math.pi * (s2_ - w2_) / max(t2_ - w2_, 1))))
                for i_ in enc_ix:
                    opt.param_groups[i_]["lr"] = base_lr[i_] * fe
            if micro > 1:                   # TA568 screen: gradient accumulation over `micro` chunks of the batch (exact: chunk means
                # weighted by their share = the batch mean; one mixup lambda + permutation for the whole batch); plain / mixup CE only
                assert sc_pairs is None and not mae_retain_every, "fm micro-batching: the plain / mixup CE path only"
                opt.zero_grad(set_to_none=True)
                if mixup > 0:
                    lam = float(rng.beta(mixup, mixup))
                    j = torch.randperm(len(xb), device=xb.device)
                    xin, y2, w2 = lam * xb + (1 - lam) * xb[j], yb[j], (wb[j] if wb is not None else None)
                else:
                    xin, y2, w2 = xb, None, None
                tot = 0.0
                for ch in torch.arange(len(xb), device=xb.device).chunk(int(micro)):
                    fr = len(ch) / len(xb)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xtr.is_cuda), t1.source_channels(net.enc, names):
                        lo = net(xin[ch]).float()
                        l_ = wce(lo, yb[ch], ls, wb[ch] if wb is not None else None, norm=wnorm_)
                        if mixup > 0:
                            l_ = lam * l_ + (1 - lam) * wce(lo, y2[ch], ls, w2[ch] if w2 is not None else None, norm=wnorm_)
                    (l_ * fr).backward()
                    tot += float(l_.detach()) * fr
                loss = torch.tensor(tot, device=xb.device)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xtr.is_cuda), t1.source_channels(net.enc, names):
                    feat = None
                    if mixup > 0:
                        lam = float(rng.beta(mixup, mixup))
                        j = torch.randperm(len(xb), device=xb.device)
                        if side_dc:                 # T2-D524: the side vectors mixed with the same lambda / permutation
                            net._side = lam * Sb_ + (1 - lam) * Sb_[j]
                        lo = net(lam * xb + (1 - lam) * xb[j]).float()
                        loss = lam * wce(lo, yb, ls, wb, norm=wnorm_) + (1 - lam) * wce(lo, yb[j], ls, (wb if dro_ else wb[j]) if wb is not None else None, norm=wnorm_)
                        if xb2 is not None:         # T2-D588: view 2 with the same lambda / pairs (its side vectors mixed the same way)
                            if side_dc:
                                net._side = lam * Sb2_ + (1 - lam) * Sb2_[j]
                            lo2 = net(lam * xb2 + (1 - lam) * xb2[j]).float()
                            loss2 = lam * wce(lo2, yb, ls, wb, norm=wnorm_) + (1 - lam) * wce(lo2, yb[j], ls, wb[j] if wb is not None else None, norm=wnorm_)
                            kl_ = sym_kl(lo, lo2)
                            loss = 0.5 * (loss + loss2) + cons * kl_
                            ckl += float(kl_.detach()) * len(b)
                            ckn += len(b)
                        if wmargin > 0 and k == 0:  # T2-D596: the margin term, mixed with the same lambda / pairs
                            wm_ = lam * word_margin(lo, yb) + (1 - lam) * word_margin(lo, yb[j])
                            loss = loss + wmargin * min(1.0, (step + 1) / max(1, int(0.1 * total))) * wm_
                            wml += float(wm_.detach()) * len(b)
                            wmn += len(b)
                            wms += float((yb == yb[j]).float().sum())   # Track3 LOW 1: a mixed pair of two classes dilutes the margin
                            if step == 0:
                                log("T2-D596: the margin term reads the SAME logits tensor as the CE (net output incl. the side head W z + b)")
                        if PK is not None and k == 0:   # backlog 96: the mixed window's OOF CNN mix = the same lambda's mix
                            pcm = lam * PK[b] + (1 - lam) * PK[b][j]
                            loss = loss + pk_beta * (lam * pkg_nll(lo, pcm, yb, pk_w) + (1 - lam) * pkg_nll(lo, pcm, yb[j], pk_w))
                    elif sc_pairs is not None:
                        feat = net.features(xb)
                        loss = F.cross_entropy(net.out(feat).float(), yb, label_smoothing=ls)
                    else:
                        if side_dc:                 # T2-D524
                            net._side = Sb_
                        lo = net(xb).float()
                        loss = wce(lo, yb, ls, wb, norm=wnorm_)
                        if xb2 is not None:         # T2-D588: view 2
                            if side_dc:
                                net._side = Sb2_
                            lo2 = net(xb2).float()
                            kl_ = sym_kl(lo, lo2)
                            loss = 0.5 * (loss + wce(lo2, yb, ls, wb, norm=wnorm_)) + cons * kl_
                            ckl += float(kl_.detach()) * len(b)
                            ckn += len(b)
                        if wmargin > 0 and k == 0:  # T2-D596
                            wm_ = word_margin(lo, yb)
                            loss = loss + wmargin * min(1.0, (step + 1) / max(1, int(0.1 * total))) * wm_
                            wml += float(wm_.detach()) * len(b)
                            wmn += len(b)
                            wms += len(b)                   # un-mixed rows carry the full margin
                            if step == 0:
                                log("T2-D596: the margin term reads the SAME logits tensor as the CE (net output incl. the side head W z + b)")
                        if PK is not None and k == 0:   # backlog 96
                            loss = loss + pk_beta * pkg_nll(lo, PK[b], yb, pk_w)
                    if sc_pairs is not None and feat is not None:     # no mixup: CE and SupCon share one forward
                        loss = loss + supcon * supcon_loss(proj(feat.float()), sc_pairs.mask(rows_sc).to(device), supcon_tau)
                if grl_on:                      # T2-D649: the adversary on the MIXED pooled feature, through the GRL; user targets weighted as the task
                    lam_g_ = float(grl) * min(1.0, step / max(1.0, float(grl_ramp) * total))
                    alo_ = adv_(_GRL.apply(grl_box_["f"].float(), lam_g_))
                    ub_g_, kp_ = UG_[b], ~HOLD_[b]
                    if bool(kp_.any()):
                        wk_ = wb[kp_] if wb is not None else None
                        a_loss_ = wce(alo_[kp_], ub_g_[kp_], 0.0, wk_, norm=wnorm_)
                        if mixup > 0:
                            wj_ = (wb[j][kp_] if wb is not None else None)
                            a_loss_ = lam * a_loss_ + (1 - lam) * wce(alo_[kp_], ub_g_[j][kp_], 0.0, wj_, norm=wnorm_)
                        loss = loss + a_loss_
                        tla_ += float(a_loss_.detach()) * len(b)
                if dro_:                        # T2-D645: the q update from this step's per-row (mixed) CE, detached; absent users unchanged
                    with torch.no_grad():
                        ce_ = F.cross_entropy(lo.detach(), yb, label_smoothing=ls, reduction="none")
                        if mixup > 0:
                            ce_ = lam * ce_ + (1 - lam) * F.cross_entropy(lo.detach(), yb[j], label_smoothing=ls, reduction="none")
                        Lu_ = torch.zeros(nU_, dtype=torch.float64, device=device).index_add_(0, ub_, ce_.to(torch.float64)) / cnt_.clamp(min=1)
                        pr_ = cnt_ > 0
                        q_dro[pr_] = q_dro[pr_] * torch.exp(float(dro_eta) * Lu_[pr_])
                        q_dro /= q_dro.sum()
                opt.zero_grad(set_to_none=True)
                if grl_on:
                    adv_opt_.zero_grad(set_to_none=True)
                loss.backward()
                if sc_pairs is not None and feat is None:   # mixup: SupCon on the CLEAN batch after the CE backward freed its
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xtr.is_cuda), t1.source_channels(net.enc, names):
                        sc_ = supcon * supcon_loss(proj(net.features(xb).float()), sc_pairs.mask(rows_sc).to(device), supcon_tau)
                    sc_.backward()                          # activations (peak memory = one forward); gradients accumulate
                    loss = loss.detach() + sc_.detach()
            if layer_params is not None:
                for ps_ in layer_params.values():
                    torch.nn.utils.clip_grad_norm_(ps_, 1.0)
            if mae_retain_every and step % mae_retain_every == 0:   # T2-D132: per-term clip CE 0.5 + MAE 0.5, sum <= 1.0
                tp_ = [q for q in net.parameters() if q.requires_grad]
                torch.nn.utils.clip_grad_norm_(tp_, 0.5)
                g_ce = [q.grad for q in tp_]
                for q in tp_:
                    q.grad = None
                # ST: bf16 as its CPT; the MOR's masked loss in fp32 (as T2-D108 arm B)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xtr.is_cuda and mae_dec is not None), \
                        t1.source_channels(net.enc, names):
                    if mae_dec is not None:
                        xz = prep_z(xb)
                        noise = torch.rand(len(xz), (xz.shape[-1] // 16) * xz.shape[1], device=xz.device)
                        lm = t2p_cpt_loss(net.enc, mae_dec, xz, noise, mae_mask)
                    else:
                        lm = mor_mae_loss(net.enc, xb, mae_mask)
                lm.float().backward()
                torch.nn.utils.clip_grad_norm_(tp_, 0.5)
                for q, g_ in zip(tp_, g_ce):
                    q.grad = g_ if q.grad is None else (q.grad if g_ is None else q.grad + g_)
                loss = loss.detach() + lm.detach().float()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad] +
                                           (list(proj.parameters()) if proj is not None else []), 1.0)
            opt.step()
            if grl_on:
                adv_opt_.step()
            if ema_net is not None:
                with torch.no_grad():
                    for pe, pn in zip(ema_net.parameters(), net.parameters()):
                        pe.lerp_(pn, 1 - ema)
            step += 1
            tl += float(loss.detach()) * len(b)
            tn += len(b)
        ev = ema_net if ema_net is not None else net
        row = dict(epoch=ep + 1, train_loss=tl / max(tn, 1))
        if len(data["yva"]):
            pva = PP(ev, Xva, Sva_ if side_dc else None)
            row["val_cell_bacc"] = metrics.cell_balanced_accuracy(data["yva"], pva.argmax(1), data["cva"])[0]
            if row["val_cell_bacc"] > best[0]:
                best = (row["val_cell_bacc"], ep + 1, {k: v.detach().cpu().clone() for k, v in ev.state_dict().items()})
        if train_eval:                      # T2-D468: a CLEAN pass over the training windows (no mixup / augmentation / views), raw argmax
            row["train_cell_bacc_clean"] = metrics.cell_balanced_accuracy(data["ytr"], predict_proba(ev, Xtr).argmax(1), data["ctr"])[0]
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            row["test_cell_bacc"] = metrics.cell_balanced_accuracy(data["yte"], PP(ev, Xte, Ste_ if side_dc else None).argmax(1), data["cte"])[0]
        if grl_on:                          # T2-D649: the engagement record = the adversary's accuracy on the held-out training windows
            net.eval()
            with torch.no_grad():
                hi_ = torch.nonzero(HOLD_).flatten()
                hits_ = 0
                for k0_ in range(0, len(hi_), 64):
                    bi_ = hi_[k0_:k0_ + 64]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xtr.is_cuda), t1.source_channels(net.enc, srcs[0][2]):
                        fz_ = net.features(Xtr[bi_]).float()
                    hits_ += int((adv_(fz_).argmax(1) == UG_[bi_]).sum())
            net.train()
            lam_now_ = float(grl) * min(1.0, step / max(1.0, float(grl_ramp) * total))
            row.update(grl_lambda=lam_now_, adv_hold_acc=hits_ / max(len(hi_), 1),   # Track3 LOW: train_loss = task + adversary CE; the parts apart
                       train_adv_loss=tla_ / max(tn, 1), train_task_loss=(tl - tla_) / max(tn, 1))
            grl_curve_.append(dict(epoch=ep + 1, step=step, lam=lam_now_, adv_hold_acc=row["adv_hold_acc"]))
        curves.append(row)
        log(f"  ep {ep + 1} loss {row['train_loss']:.4f} val {row.get('val_cell_bacc', float('nan')):.4f} "
            f"test {row.get('test_cell_bacc', float('nan')):.4f} ({time.time() - t0:.0f}s)"
            + (f"; recombined {rrb['count']} / {rrb['seen']} target windows so far" if rrb is not None else "")
            + (f"; phase-rotated {rph['count']} / {rph['seen']} target windows so far" if rph is not None else "")
            + (f"; mirrored {rmr['count']} / {rmr['seen']} target windows so far" if rmr is not None else "")
            + (f"; cons KL {ckl / max(ckn, 1):.4g}" if cons > 0 else "")
            + (f"; wmargin {wml / max(wmn, 1):.4g} (same-class rows {wms / max(wmn, 1):.3f})" if wmargin > 0 else ""))
    if grl_on:                          # T2-D649: the adversary is training-only: the hook removed, the module never joins net / its state
        grl_hook_.remove()
        assert not any(k_.startswith("adv") for k_ in net.state_dict()), "T2-D649: the adversary leaked into the member state"
        log(f"T2-D649 GRL engagement (adversary held-out acc by epoch; chance {1.0 / nUg_:.3f}): "
            + ", ".join(f"{c_['epoch']}:{c_['adv_hold_acc']:.3f}" for c_ in grl_curve_[:: max(1, len(grl_curve_) // 10)]))
    if wise:                            # T2-D270: interpolate in place, then evaluate as before
        assert snap_lp is not None
        t_ft = metrics.cell_balanced_accuracy(data["yte"], predict_proba(net, Xte).argmax(1), data["cte"])[0]
        with torch.no_grad():
            for k_, v_ in net.state_dict().items():
                if v_.is_floating_point():
                    v_.lerp_(snap_lp[k_].to(v_.device), float(wise))
        t_ws = metrics.cell_balanced_accuracy(data["yte"], predict_proba(net, Xte).argmax(1), data["cte"])[0]
        log(f"WiSE-FT {float(wise):g}: test {t_ft:.4f} (the fine-tune's end) -> {t_ws:.4f} (interpolated)")
    if dro_:                                              # T2-D645: q's final distribution over the training users (descriptive, registered)
        qq_ = q_dro.cpu().numpy()
        log(f"T2-D645 FM-DRO final q: min {qq_.min():.4f} / max {qq_.max():.4f} (max / min {qq_.max() / qq_.min():.2f}); " + ", ".join(f"{v:.3f}" for v in qq_))
    ev = ema_net if ema_net is not None else net
    prob_last = PP(ev, Xte, Ste_ if side_dc else None)
    side_dump = fm_side_dump(ev, Xte) if (side_dc and os.environ.get("NI26_SIDE_DUMP") == "1") else None   # T2-D553; unset = None
    feat_dump = fm_feat_dump(ev, Xte) if os.environ.get("NI26_FEAT_DUMP") == "1" else None                 # T2-D555; unset = None
    view_dump = None                                     # T2-D654 (NI26_VIEW_DUMP=K, K in 2..4; unset = None, nothing computed)
    if os.environ.get("NI26_VIEW_DUMP"):
        assert feat_dump is not None, "T2-D654: NI26_VIEW_DUMP needs NI26_FEAT_DUMP=1 (the identity's features)"
        view_dump = fm_view_dump(ev, Xte, Ste_ if side_dc else None, int(os.environ["NI26_VIEW_DUMP"]))
    prob_val_last = PP(ev, Xva, Sva_ if side_dc else None) if len(data["yva"]) else None
    state_last = {k: v.detach().cpu().clone() for k, v in ev.state_dict().items()} if return_state else None
    net_last = copy.deepcopy(ev).cpu().eval() if return_model else None   # T2-D140: the model object at the last epoch
    prob_best = None
    if best[2] is not None:
        ev.load_state_dict(best[2])
        prob_best = PP(ev, Xte, Ste_ if side_dc else None)
    return dict(prob_val_last=prob_val_last, model=fm, seed=seed, n_params=int(sum(p.numel() for p in net.parameters())),
                n_train=int(len(Xtr)), best_epoch=int(best[1]), best_val=float(best[0]), seconds=round(time.time() - t0, 1),
                curves=curves, prob_last=prob_last, prob_best=prob_best, provenance=enc.provenance,
                state_last=state_last, net_last=net_last, chan_idx=enc.chan_idx.tolist() if hasattr(enc, "chan_idx") else None,
                **({"side_dump": side_dump} if side_dump is not None else {}), **({"feat_dump": feat_dump} if feat_dump is not None else {}),
                **({"view_dump": view_dump} if view_dump is not None else {}),   # T2-D654
                **({"dro_q": q_dro.cpu().numpy().tolist()} if dro_ else {}),
                **({"grl_curve": grl_curve_, "grl_chance": 1.0 / nUg_, "grl_ramp_end_step": float(grl_ramp) * total} if grl_on else {}))
