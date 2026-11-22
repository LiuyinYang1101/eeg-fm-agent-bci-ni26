"""ST-EEGFormer in pure torch (inference path; no timm), shared by the T1 and T2 submissions (lead review L5).
Upstream model: ST-EEGFormer (a published model by the author of this repository); this file is a re-implementation written for this project (MIT).

The Codabench worker image has no timm (src/ni26/worker_guard/dists_442edcd.txt), so the vendored ST-EEGFormer
(tracks/t1_image/fm/steeg_vit.py, a timm VisionTransformer subclass) cannot ship. This file re-implements exactly the
encoder path the harness adapters run (t1_fm.STEEG.forward / t2p_fm.STEEG4.forward): EEG patch embedding (non-overlapping
patches of `patch` samples per channel, sequence-major tokens), + 1-based sinusoidal temporal code + learned channel
embedding (145-row table), CLS (+ the temporal row 0), timm pre-norm blocks (LayerNorm eps 1e-6, fused SDPA attention,
exact GELU MLP, no LayerScale / qk-norm), final norm only after the last block, CLS dropped from the grid.

Parameter and buffer NAMES match the timm model (`patch_embed.proj`, `enc_channel_emd.channel_transformation`,
`enc_temporal_emd.pe`, `cls_token`, `blocks.<i>.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}`, `norm`), so a
harness state dict loads with strict=True after dropping the `enc.vit.` prefix (or keeping it, via STClassifier).
Parity with the timm path: src/ni26/fm/steeg_pt_check.py.

Self-contained on purpose (imports torch only): the T2 packager inlines this source into submission.py. Keep every
top-level name prefixed (ST*, st_*) so it cannot clash with the template's names.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# size -> (embed dim, depth, heads); patch 16, mlp ratio 4, qkv bias (steeg_vit.vit_*_patch16)
ST_SIZES = {"st_small": (512, 8, 8), "st_base": (768, 12, 12), "st_large": (1024, 24, 16)}
ST_FS, ST_PATCH, ST_GAIN, ST_PE_ROWS, ST_N_CHAN_ROWS = 128, 16, 1.12, 512, 145


def st_fft_resample(x, n_new):
    """(B, C, T) -> (B, C, n_new); verbatim t1_fm.fft_resample (the resampler every ST adapter trains through)."""
    n = x.shape[-1]
    if n == n_new:
        return x
    Xf = torch.fft.rfft(x.float(), dim=-1)
    k = n_new // 2 + 1
    Xf = Xf[..., :k] if Xf.shape[-1] >= k else F.pad(Xf, (0, k - Xf.shape[-1]))
    return torch.fft.irfft(Xf, n=n_new, dim=-1) * (n_new / n)


def st_zwin(x, clip):
    """verbatim t1_fm.zwin: per-window per-channel z (population std), clamped."""
    return ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, unbiased=False) + 1e-8)).clamp(-clip, clip)


class STPatchEmbed(nn.Module):
    """steeg_vit.PatchEmbedEEG: (B, C, L) -> (B, S, C, D); x.unfold == its nn.Unfold((1, p), stride p) + reshape."""

    def __init__(self, patch, dim):
        super().__init__()
        self.p = patch
        self.proj = nn.Linear(patch, dim)

    def forward(self, x):
        return self.proj(x.unfold(-1, self.p, self.p).permute(0, 2, 1, 3))


class STChannelEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channel_transformation = nn.Embedding(ST_N_CHAN_ROWS, dim)

    def forward(self, idx):
        return self.channel_transformation(idx)


class STTemporalEmbed(nn.Module):
    """steeg_vit.TemporalPositionalEncoding(dim, 512): a fixed sinusoidal table (a persistent buffer in the checkpoint)."""

    def __init__(self, dim, rows=ST_PE_ROWS):
        super().__init__()
        pos = torch.arange(0, rows).unsqueeze(1)
        div = torch.exp((torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).float())
        pe = torch.zeros(1, rows, dim)
        pe[0, :, 0::2] = torch.sin(pos.float() * div)
        pe[0, :, 1::2] = torch.cos(pos.float() * div)
        self.register_buffer("pe", pe)

    def forward(self, seq):                    # (B, N) int -> (B, N, D)
        B, N = seq.shape
        return self.pe[0, seq.reshape(-1)].reshape(B, N, -1)


class STAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class STMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class STBlock(nn.Module):
    """timm vision_transformer.Block in eval mode with the ST config (no LayerScale, drop path / dropout inactive)."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = STAttention(dim, heads), STMlp(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class STViT(nn.Module):
    """The ST-EEGFormer encoder. forward(x, chan_idx, layer) on PREPARED input (128 Hz, gain or z; see STClassifier.prep)
    -> tokens (B, 1 + S*C, D) after `layer` blocks (layer = depth also applies the final norm), CLS first, sequence-major."""

    def __init__(self, size):
        super().__init__()
        dim, depth, heads = ST_SIZES[size]
        self.dim, self.depth = dim, depth
        self.patch_embed = STPatchEmbed(ST_PATCH, dim)
        self.enc_channel_emd = STChannelEmbed(dim)
        self.enc_temporal_emd = STTemporalEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([STBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, chan_idx, layer=None):
        layer = self.depth if layer in (None, -1) else int(layer)
        t = self.patch_embed(x)                                           # (B, S, C, D)
        B, S, C, D = t.shape
        t = t.reshape(B, S * C, D)
        seq = torch.arange(1, S + 1, device=x.device).view(1, S, 1).expand(B, S, C).reshape(B, S * C)
        ch = chan_idx.to(x.device).view(1, 1, C).expand(B, S, C).reshape(B, S * C)
        t = t + self.enc_temporal_emd(seq) + self.enc_channel_emd(ch)
        cls = (self.cls_token + self.enc_temporal_emd.pe[0, 0]).expand(B, -1, -1)
        t = torch.cat([cls, t], 1)
        for blk in self.blocks[:layer]:
            t = blk(t)
        if layer == self.depth:
            t = self.norm(t)
        return t

    @staticmethod
    def grid(t, C):
        """tokens (B, 1 + S*C, D) -> grid (B, C, S, D) without CLS (the adapters' `g`)."""
        B, N, D = t.shape
        return t[:, 1:].reshape(B, (N - 1) // C, C, D).permute(0, 2, 1, 3)


class _STHolder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit


class STClassifier(nn.Module):
    """t2p_fm.FMClf(STEEG4, readout='mean') in pure torch: prep -> encoder through `layer` -> LayerNorm (eps 1e-5) ->
    mean over (channels, patches) -> Linear. State-dict keys = FMClf's (`enc.vit.*`, `norm.*`, `out.*`).
    n_samples = the encoder's input length (128 Hz x window seconds, e.g. 512 for T2's 4 s); calib 'gain' | 'zwin'."""

    def __init__(self, size, chan_idx, n_samples, calib="gain", layer=-1, n_out=3):
        super().__init__()
        assert calib in ("gain", "zwin"), calib
        self.enc = _STHolder(STViT(size))
        self.calib, self.n_samples = calib, int(n_samples)
        self.layer = self.enc.vit.depth if layer in (None, -1) else int(layer)
        self.register_buffer("chan_idx", torch.as_tensor(chan_idx, dtype=torch.long), persistent=False)
        self.norm = nn.LayerNorm(self.enc.vit.dim)
        self.out = nn.Linear(self.enc.vit.dim, n_out)

    def prep(self, x):
        x = st_fft_resample(x, self.n_samples)
        return st_zwin(x, 1e9) if self.calib == "zwin" else x * ST_GAIN

    def features(self, x):
        g = STViT.grid(self.enc.vit(self.prep(x), self.chan_idx, self.layer), x.shape[1])
        return self.norm(g).mean((1, 2))

    def forward(self, x):
        return self.out(self.features(x))
