"""T2-D521: a RAW end-to-end net (the owner, relayed 2026-11-15 before 17:31: "test pure end to end with raw data approach, like learnable filter in cnn, or
even no preprocessing at all"). Input = the delivered window with ONE global gain (t2p_bench --aligns none --keep-mean: no per-window de-mean, no
per-stream alignment or scaling). Every step in front is learned:
  compression  x -> sign(x) log(1 + a_c |x|), a learned per-channel slope (log-parametrised; init 1 on the unit-scale input), in place of normalisation;
  temporal     a multi-scale learned bank: 8 filters at each of 0.1 / 0.25 / 0.5 / 1 s (12 / 30 / 60 / 120 taps at 120 Hz; < 2 Hz is resolvable);
  spatial      one learned spatial layer over all channels (32 maps), batch norm;
  readout      two pooled branches over 0.5 s windows (stride 0.1 s): power (square -> mean -> log, ShallowNet's) and LEVEL (signed mean: the slow
               / DC content the shipped members' de-mean removes), concatenated -> dropout -> linear.
Plain torch; the forward is (B, C, T) -> (B, K) logits like the braindecode members.
"""
import torch
import torch.nn as nn


class RawNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, sfreq=120.0, n_per=8, kernels_s=(0.1, 0.25, 0.5, 1.0), n_spat=32, pool_s=0.5,
                 stride_s=0.1, drop_prob=0.5):
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(n_chans))
        ks = [max(3, int(round(k * sfreq))) for k in kernels_s]
        self.temporal = nn.ModuleList([nn.Conv2d(1, n_per, (1, k), padding="same", bias=False) for k in ks])
        nt = n_per * len(ks)
        self.bn_t = nn.BatchNorm2d(nt)
        self.spatial = nn.Conv2d(nt, n_spat, (n_chans, 1), bias=False)
        self.bn_s = nn.BatchNorm2d(n_spat)
        p, s = int(round(pool_s * sfreq)), int(round(stride_s * sfreq))
        self.pool = nn.AvgPool2d((1, p), stride=(1, s))
        n_out_t = (n_times - p) // s + 1
        self.drop = nn.Dropout(drop_prob)
        self.final_layer = nn.Linear(2 * n_spat * n_out_t, n_outputs)

    def forward(self, x):
        a = torch.exp(self.log_a)[None, :, None]
        x = torch.sign(x) * torch.log1p(a * x.abs())
        x = x[:, None]                                            # (B, 1, C, T)
        h = torch.cat([conv(x) for conv in self.temporal], 1)     # (B, 32, C, T)
        h = self.bn_s(self.spatial(self.bn_t(h)))                 # (B, n_spat, 1, T)
        power = torch.log(torch.clamp(self.pool(h * h), min=1e-6))
        level = self.pool(h)
        f = torch.cat([power, level], 1).flatten(1)
        return self.final_layer(self.drop(f))
