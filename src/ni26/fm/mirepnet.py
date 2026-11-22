"""Row 77 / T2-D305: MIRepNet (Liu et al., KBS 2026, arXiv 2507.20254; code + weights MIT, github staraink/MIRepNet, HF starself/MIRepNet) as a
pure-torch member. The encoder is the published Conformer-style net (conv(1x25) -> conv(45x1) -> BN / ELU -> avgpool(75, 15) -> 1x1 conv to 256 ->
6 transformer blocks -> mean pool) with its key names, so MIRepNet.pth strict-loads (the decoder / mask token are dropped; the unused channel
embedding is kept for the strict load). Two quirks kept: attention scale 1/sqrt(256) (not the head dim), dropout 0.5.
MIRepMember wraps it for our windows: (B, C, T) at `sfreq` (already aligned by the caller, e.g. the bench's / package's causal EA) -> optional
zero-phase band-pass (`band`, MIRepNet's 8-30 Hz; None = broadband) -> FFT resample to 250 Hz -> the 45-channel template by inverse-distance
padding over all of our channels (template positions in mirepnet_template.json; present channels copied) -> encoder -> a new linear head.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPLATE = json.loads((Path(__file__).with_name("mirepnet_template.json")).read_text())
WEIGHTS = Path(os.environ.get("NI26_MIREP_PTH", str(Path.home() / "workspace/datasets/pretrained_eeg_fms/mirepnet/MIRepNet.pth")))
FS = 250.0


class PatchEmbedding(nn.Module):
    def __init__(self, emb=256, n_ch=45):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (1, 25))
        self.conv2 = nn.Conv2d(64, 128, (n_ch, 1))
        self.bn = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d((1, 75), (1, 15))
        self.projection = nn.Sequential(nn.Conv2d(128, emb, 1))
        self.chan_embed = nn.Embedding(45, emb)          # in the checkpoint, unused by the forward

    def forward(self, x):                                 # (B, 45, T) -> (B, tokens, emb)
        x = self.pool(F.elu(self.bn(self.conv2(self.conv1(x.unsqueeze(1))))))
        return self.projection(x).flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        self.emb, self.heads = emb, heads
        self.keys, self.queries, self.values, self.projection = (nn.Linear(emb, emb) for _ in range(4))

    def forward(self, x):
        B, N, E = x.shape
        sp = lambda t: t.view(B, N, self.heads, E // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.queries(x)), sp(self.keys(x)), sp(self.values(x))
        a = torch.softmax(q @ k.transpose(-1, -2) / self.emb ** 0.5, -1)        # the published scale: sqrt(emb)
        return self.projection((a @ v).transpose(1, 2).reshape(B, N, E))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def _block(emb):
    return nn.Sequential(Residual(nn.Sequential(nn.LayerNorm(emb), MultiHeadAttention(emb, 8), nn.Dropout(0.5))),
                         Residual(nn.Sequential(nn.LayerNorm(emb), nn.Sequential(nn.Linear(emb, 4 * emb), nn.GELU(), nn.Dropout(0.5),
                                                                                   nn.Linear(4 * emb, emb)), nn.Dropout(0.5))))


class MIRepNet(nn.Module):
    def __init__(self, emb=256, depth=6, n_classes=3):
        super().__init__()
        self.embedding = PatchEmbedding(emb)
        self.transformer = nn.Sequential(*[_block(emb) for _ in range(depth)])
        self.clshead = nn.Linear(emb, n_classes)

    def forward(self, x):
        return self.clshead(self.transformer(self.embedding(x)).mean(1))


def pad_map(ch_names):
    """(45, C): row = the template channel copied from ours, or its inverse-distance mix of ALL our channels (MIRepNet's padding)."""
    pos = TEMPLATE["positions"]
    ours = [c.upper() for c in ch_names]
    up = {k.upper(): v for k, v in pos.items()}
    missing = [c for c in ours if c not in up]
    assert not missing, f"no template position for {missing}"
    P = torch.tensor([up[c] for c in ours], dtype=torch.float64)
    W = torch.zeros(len(TEMPLATE["template"]), len(ours), dtype=torch.float64)
    for i, c in enumerate(TEMPLATE["template"]):
        if c.upper() in ours:
            W[i, ours.index(c.upper())] = 1.0
        else:
            w = 1.0 / ((P - torch.tensor(up[c.upper()], dtype=torch.float64)).norm(dim=1) + 1e-6)
            W[i] = w / w.sum()
    return W.float()


def load_pretrained(net, path=WEIGHTS):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    enc = {k: v for k, v in sd.items() if not (k.startswith("decoder") or k == "mask_token" or k.startswith("clshead"))}
    missing, unexpected = net.load_state_dict(enc, strict=False)
    assert not unexpected and set(missing) == {"clshead.weight", "clshead.bias"}, (missing, unexpected)
    return net


class MIRepMember(nn.Module):
    def __init__(self, ch_names, n_times, sfreq, n_classes=3, band=(8.0, 30.0), pretrained=True):
        super().__init__()
        self.n_times, self.sfreq, self.band = int(n_times), float(sfreq), band
        self.n_out = int(round(self.n_times * FS / self.sfreq))
        self.register_buffer("pad", pad_map(ch_names))
        f = torch.fft.rfftfreq(self.n_times, 1.0 / self.sfreq)
        self.register_buffer("bp", ((f >= band[0]) & (f <= band[1])).float() if band else torch.ones_like(f))
        self.net = MIRepNet(n_classes=3)
        if pretrained:
            load_pretrained(self.net)
        self.net.clshead = nn.Linear(256, n_classes)

    def prep(self, x):                                   # (B, C, T) at sfreq -> (B, 45, n_out) at 250 Hz: band-pass, resample, template padding
        X = torch.fft.rfft(x.float(), dim=-1) * self.bp
        n_new = self.n_out // 2 + 1
        Y = torch.zeros(X.shape[:-1] + (n_new,), dtype=X.dtype, device=X.device)
        m = min(n_new, X.shape[-1])
        Y[..., :m] = X[..., :m]
        x = torch.fft.irfft(Y, n=self.n_out, dim=-1) * (self.n_out / self.n_times)   # FFT resample to 250 Hz (amplitude kept)
        return torch.einsum("tc,bcl->btl", self.pad, x)

    def forward(self, x):
        return self.net(self.prep(x))
