"""T2-D554 (the owner's 'speech recipe' bet, 2026-11-17; T2-D554b: + a learned spatial projection before the STFT, `specnet_sf`): a log-spectrogram 2-D CNN family with SpecAugment, trained from scratch.

Input: the aligned window (B, C, T) at 120 Hz. Per channel a log power spectrogram (Hann STFT, n_fft 64 = 0.53 s, hop 8 -> 61 frames;
bins 1..24 = 1.9-45 Hz), the EEG channels as the image's feature maps: (B, C, F 24, T' 61). A learned per-(channel, bin) affine input norm
(BatchNorm2d over C), then four conv blocks (3 x 3, BN, GELU; pooling 2 x 2 after blocks 2 and 3), global average pooling, dropout 0.3, a
linear head. SpecAugment in training only (torch RNG): 2 time masks <= 8 frames, 2 frequency masks <= 4 bins, each channel's spectrogram
dropped with p 0.1. Mixup / label smoothing / schedule come from the bench recipe (ni26.train.fit_eval). Eval mode is deterministic.
"""
import torch
import torch.nn as nn


class SpecNet(nn.Module):
    def __init__(self, n_chans, n_classes, n_times, sfreq, n_fft=64, hop=8, f_lo=1, f_hi=25, width=64, drop=0.3,
                 t_masks=2, t_w=8, f_masks=2, f_w=4, ch_p=0.1, spatial=0):
        super().__init__()
        # T2-D554b: spatial > 0 = a learned linear projection of the (aligned) channels to `spatial` virtual channels BEFORE the STFT
        # (the ShallowNet-style spatial front end); 0 = D554's specnet exactly
        self.spatial = nn.Conv1d(n_chans, spatial, 1, bias=False) if spatial else None
        n_chans = spatial or n_chans
        self.n_fft, self.hop, self.f_lo, self.f_hi = n_fft, hop, f_lo, f_hi
        self.t_masks, self.t_w, self.f_masks, self.f_w, self.ch_p = t_masks, t_w, f_masks, f_w, ch_p
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        self.inorm = nn.BatchNorm2d(n_chans)

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.GELU())
        self.body = nn.Sequential(blk(n_chans, width), blk(width, width), nn.MaxPool2d(2),
                                  blk(width, 2 * width), nn.MaxPool2d(2), blk(2 * width, 2 * width))
        self.drop = nn.Dropout(drop)
        self.final_layer = nn.Linear(2 * width, n_classes)

    def spec(self, x):
        B, C, T = x.shape
        z = torch.stft(x.reshape(B * C, T).float(), n_fft=self.n_fft, hop_length=self.hop, window=self.win, center=True, return_complex=True)
        p = (z.real ** 2 + z.imag ** 2)[:, self.f_lo:self.f_hi]          # (B C, F, T')
        return torch.log(p + 1e-6).reshape(B, C, p.shape[1], p.shape[2])

    def augment(self, s):
        B, C, Fq, Tt = s.shape
        dev = s.device
        m = torch.ones_like(s)
        for _ in range(self.t_masks):
            w = torch.randint(0, self.t_w + 1, (B,), device=dev)
            t0 = (torch.rand(B, device=dev) * (Tt - w).clamp(min=1)).long()
            ar = torch.arange(Tt, device=dev)[None]
            m = m * (~((ar >= t0[:, None]) & (ar < (t0 + w)[:, None])))[:, None, None, :].float()
        for _ in range(self.f_masks):
            w = torch.randint(0, self.f_w + 1, (B,), device=dev)
            f0 = (torch.rand(B, device=dev) * (Fq - w).clamp(min=1)).long()
            ar = torch.arange(Fq, device=dev)[None]
            m = m * (~((ar >= f0[:, None]) & (ar < (f0 + w)[:, None])))[:, None, :, None].float()
        if self.ch_p > 0:
            m = m * (torch.rand(B, C, 1, 1, device=dev) >= self.ch_p).float()
        mean = (s * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp(min=1.0)
        return s * m + mean * (1.0 - m)                                    # masked cells take the channel's mean log power

    def features(self, x):
        if self.spatial is not None:
            x = self.spatial(x.float())
        s = self.spec(x)
        if self.training:
            s = self.augment(s)
        h = self.body(self.inorm(s))
        return self.drop(h.mean((2, 3)))

    def forward(self, x):
        return self.final_layer(self.features(x))
