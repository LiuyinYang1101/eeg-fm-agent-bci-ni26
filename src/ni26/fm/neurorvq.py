"""ni26 shared (T3 builds it for all tracks; the user's NeuroRVQ research test, 2026-10-31): NeuroRVQ-EEG FM v1's ENCODER,
inference path only, in pure torch (the worker image has no einops). Self-contained (math / torch only), so a track can also
inline it into a submission.py. Vendored from github.com/KonstantinosBarmpas/NeuroRVQ @ 926e770d (NeuroRVQ_EEG/NeuroRVQ.py
`MultiDimentionalTemporalConv` + `NeuroRVQFM(use_as_encoder=True)`, NeuroRVQ_EEG/NeuroRVQ_modules.py `Block` / `Attention` /
`Mlp`; the model is CC BY-NC 4.0, the repo's licence file applies). Same functional ops, same order; einops rearranges written as
the reshape / permute they denote; dropout / drop-path omitted (identity in eval). Module names = the reference's, so
`NrEnc(cfg).net.load_state_dict(sd, strict=True)` takes the released state dict
(HF ntinosbarmpas/NeuroRVQ pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt) minus its pretraining-only
keys (head_pre_*, norm_pre.*, mask_token). tests/test_neurorvq.py checks it against the reference code (exact).
Input contract (the repo's preprocessing_eeg_example.py): uV, notch 50 / 60 Hz, band-pass 0.5-45 Hz, clip +-500, 200 Hz, 1-s
patches of 200 samples; spatial embedding row = the channel's index in the pretraining list (Muse TP9 / AF7 / AF8 / TP10 = 103 /
4 / 5 / 100); temporal embedding right-aligned at n_patches = 256 (fine_tuning: create_embedding_ix(n_time, 256, ...)).
Features: each of the 4 branches runs blocks[:layer] separately (as the reference), cls dropped, then concatenated: (B, C, S, 800).
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NR_CH = ("a1 a2 af3 af4 af7 af8 afz c1 c2 c3 c4 c5 c6 ccp1 ccp2 ccp3 ccp4 ccp5 ccp6 ccp7 ccp8 cfc1 cfc2 cfc3 cfc4 cfc5 cfc6 "
         "cfc7 cfc8 cp1 cp2 cp3 cp4 cp5 cp6 cpz cz eog f1 f10 f2 f3 f4 f5 f6 f7 f8 f9 fc1 fc2 fc3 fc4 fc5 fc6 fcz fp1 fp2 fpz "
         "ft7 ft8 fz iz loc o1 o2 oz p08 p1 p10 p2 p3 p4 p5 p6 p7 p8 p9 po1 po10 po2 po3 po4 po7 po8 po9 poz pz roc sp1 sp2 t1 "
         "t10 t2 t3 t4 t5 t6 t7 t8 t9 tp10 tp7 tp8 tp9").split()
NR_CFG = dict(embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0, out_chans=8, n_patches=256, patch_size=200,
              n_global=len(NR_CH), fs=200, hp=0.5, lp=45.0, clip=500.0)
NR_DROP = ("head_pre_", "norm_pre.", "mask_token")      # pretraining-only keys of the released checkpoint
_NR_MASK = {}                                            # prep's FFT keep-masks per (shape, band, device)
# the released checkpoint (HF ntinosbarmpas/NeuroRVQ, pretrained_models/foundation_models/, downloaded 2026-10-31; 234,352,952 B):
# local copy and sofia copy (challenge models/, s1 put); sha256 pinned
WEIGHTS = Path.home() / "workspace/datasets/pretrained_eeg_fms/neurorvq/weights/NeuroRVQ_EEG_foundation_model_v1.pt"
WEIGHTS_SOFIA = "models/neurorvq_eeg_fm_v1.pt"     # under $S1_ROOT (models/third_party is read-only); MANIFEST row 2026-10-31
SHA256 = "d14ef4eae2d4594d5f1f709226a3bb5f362afd84b68050d5da0fa0586aea886a"
REPO_COMMIT = "926e770d9d16b6aa308404280fa0cc0211a6f9fb"


def encoder_state(sd):
    """the released state dict -> the encoder's (the pretraining-only keys dropped), for NrEnc(cfg).net.load_state_dict(strict=True)"""
    return {k: v for k, v in sd.items() if not k.startswith(NR_DROP)}


def load_pretrained(path=None, cfg=None, check_sha=True):
    """NrEnc with the released weights, strict, eval mode (path None: $S1_ROOT/WEIGHTS_SOFIA on sofia, else WEIGHTS)."""
    import hashlib
    import os
    path = Path(path) if path else (Path(os.environ["S1_ROOT"]) / WEIGHTS_SOFIA if os.environ.get("S1_ROOT") else WEIGHTS)
    if check_sha:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == SHA256, f"{path}: sha256 {h[:12]} != pinned {SHA256[:12]}"
    enc = NrEnc(cfg or {})
    enc.net.load_state_dict(encoder_state(torch.load(path, map_location="cpu", weights_only=True)), strict=True)
    return enc.eval()


class _NrMDTC(nn.Module):
    """MultiDimentionalTemporalConv: 4 inception branches (kernels 21 / 15 / 9 / 5, then 9 / 7 / 5 / 3), GroupNorm(4), GELU, pools."""

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        for i, k in enumerate((21, 15, 9, 5), 1):
            setattr(self, f"conv1_{i}", nn.Conv2d(in_chans, out_chans, kernel_size=(1, k), padding=(0, k // 2)))
            setattr(self, f"norm1_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool1_{i}", nn.AvgPool2d(kernel_size=(1, 2)))
        self.gelu1 = nn.GELU()
        for i, k in enumerate((9, 7, 5, 3), 1):
            setattr(self, f"conv2_{i}", nn.Conv2d(out_chans, out_chans, kernel_size=(1, k), padding=(0, k // 2)))
            setattr(self, f"norm2_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool2_{i}", nn.AvgPool2d(kernel_size=(1, 4)))
        self.gelu2 = nn.GELU()

    def forward(self, x):                                  # (B, N, A, T) -> 4 x (B, N A, T' C)
        B, N, A, T = x.shape
        x = x.reshape(B, N * A, T).unsqueeze(1)            # 'B N A T -> B (N A) T', then (B, 1, NA, T)
        out = []
        for i in (1, 2, 3, 4):
            y = getattr(self, f"pool1_{i}")(self.gelu1(getattr(self, f"norm1_{i}")(getattr(self, f"conv1_{i}")(x))))
            y = getattr(self, f"pool2_{i}")(self.gelu2(getattr(self, f"norm2_{i}")(getattr(self, f"conv2_{i}")(y))))
            Bc, C, NA, Tp = y.shape
            out.append(y.permute(0, 2, 3, 1).reshape(Bc, NA, Tp * C))      # 'B C NA T -> B NA (T C)'
        return out


class _NrMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _NrAttention(nn.Module):
    """LaBraM attention: qkv without bias + q_bias / v_bias, qk LayerNorm (eps 1e-6) per head, explicit softmax(q k^T)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        hd = dim // num_heads
        self.scale = hd ** -0.5
        self.qkv = nn.Linear(dim, hd * num_heads * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(hd * num_heads))
        self.v_bias = nn.Parameter(torch.zeros(hd * num_heads))
        self.q_norm = nn.LayerNorm(hd, eps=1e-6)
        self.k_norm = nn.LayerNorm(hd, eps=1e-6)
        self.proj = nn.Linear(hd * num_heads, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q).type_as(v)
        k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(x)


class _NrBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _NrAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _NrMlp(dim, int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class _NrNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        D = c["embed_dim"]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.patch_embed = _NrMDTC(out_chans=c["out_chans"])
        self.pos_embed = nn.Parameter(torch.zeros(c["n_global"] + 1, D))
        self.time_embed = nn.Parameter(torch.zeros(c["n_patches"], D))
        self.blocks = nn.ModuleList(_NrBlock(D, c["num_heads"], c["mlp_ratio"]) for _ in range(c["depth"]))
        # the reference's heads / fc_norms (num_classes=0 -> Identity; LayerNorms unused on the encoder path) are not built


class NrEnc(nn.Module):
    """prep (B, C, 640) raw uV @ 128 Hz -> (B, C, 1000) @ 200 Hz (FFT band-pass 0.5-45 Hz + notch 50 / 60, clip +-500);
    forward(xp, layer) -> (B, C, S, 800) = the 4 branches' tokens after blocks[:layer], concatenated (S = 1-s patches)."""

    def __init__(self, cfg):
        super().__init__()
        self.c = c = dict(NR_CFG, **{k: v for k, v in cfg.items() if k in NR_CFG})
        self.net = _NrNet(c)
        idx = [NR_CH.index(str(n).lower()) for n in cfg.get("ch_names", ("TP9", "AF7", "AF8", "TP10"))]
        self.register_buffer("spat", torch.tensor(idx, dtype=torch.long), persistent=False)
        self._ix = {}
        self.n_layers, self.dim = c["depth"], 4 * c["embed_dim"]

    @staticmethod
    def prep(xb, fs_in=128, fs_out=200, hp=0.5, lp=45.0, notch=(50.0, 60.0), notch_hw=1.0, clip=500.0):
        n = xb.shape[-1]
        n_out = n * fs_out // fs_in
        Xf = torch.fft.rfft(xb.float(), dim=-1)
        key = (n, fs_in, hp, lp, tuple(notch), notch_hw, str(xb.device))
        if key not in _NR_MASK:       # built once per shape / device, outside any CUDA-graph capture (the warm-up call)
            f = torch.fft.rfftfreq(n, d=1.0 / fs_in)
            keep = (f >= hp) & (f <= lp)
            for f0 in notch:
                keep &= (f - f0).abs() > notch_hw
            _NR_MASK[key] = keep.to(xb.device)
        Xf = F.pad(Xf * _NR_MASK[key], (0, n_out // 2 + 1 - Xf.shape[-1]))
        return (torch.fft.irfft(Xf, n=n_out, dim=-1) * (n_out / n)).clamp(-clip, clip)

    def forward(self, xp, layer):
        n, c = self.net, self.c
        assert 0 <= layer <= self.n_layers, f"NeuroRVQ layer {layer} not in 0..{self.n_layers}"
        B, C, T = xp.shape
        S = T // c["patch_size"]
        x = xp[..., :S * c["patch_size"]].reshape(B, C, S, c["patch_size"])
        branches = n.patch_embed(x)
        key = (C, S, str(xp.device))
        if key not in self._ix:       # index tensors built once per shape / device (CUDA-graph safe: no host copy in capture)
            sp = F.pad(torch.repeat_interleave(self.spat, S), (1, 0), value=0)      # cls -> row 0, as the reference
            ti = torch.arange(c["n_patches"] - S, c["n_patches"], device=xp.device).repeat(C)
            self._ix[key] = (sp.to(xp.device), ti)
        sp, ti = self._ix[key]
        pos = n.pos_embed[sp].unsqueeze(0)                                             # (1, 1 + C S, D)
        tim = n.time_embed[ti].unsqueeze(0)
        cls = n.cls_token.expand(B, -1, -1)
        outs = []
        for xb in branches:
            h = torch.cat((cls, xb), dim=1) + pos
            h = torch.cat((h[:, :1], h[:, 1:] + tim), dim=1)
            for blk in n.blocks[:layer]:
                h = blk(h)
            outs.append(h[:, 1:])
        return torch.cat(outs, dim=-1).reshape(B, C, S, self.dim)
