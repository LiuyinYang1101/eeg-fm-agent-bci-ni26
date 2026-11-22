"""Track-2 submission: ensemble of braindecode CNNs trained by ni26 (github.com/LiuyinYang1101/eeg-fm-agent-bci-ni26).

Self-contained by contract (no extra modules): imports only torch, numpy, braindecode (all in the worker image).
Files read from meta["submission_dir"]: config.json (training contract) and weights.pt (list of state dicts).

The Adapter maps whatever the evaluation tensor looks like onto the training contract, because the sealed window
specification is not public yet (Q7):
  * channels by NAME (case-insensitive, 'EEG ' prefix / '-ref' suffix stripped); a missing channel is zero-filled
    and reported once (never a silent drop, LESSONS B10); same count without names -> positional;
  * sampling rate: FFT resampling of the window when meta["sfreq"] differs from training;
  * length: longer windows keep the FIRST n_times samples (windows are cue-aligned; information peaks just after
    the cue), shorter ones are zero-padded at the end;
  * amplitude: the kit pipeline delivers per-recording RobustScaled data (|x| ~ 1); if the batch median |x| is off by
    >100x (e.g. raw volts), each window is robust-scaled per channel instead.
predict(X) returns the argmax of the seed-averaged softmax, shape (B,), int64, on X's device.
"""

import inspect
import json

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver


def _norm(name):
    s = str(name).strip().upper()
    s = s[4:] if s.startswith("EEG ") else s
    return s.split("-")[0].strip()


def _resample(x, n_new):
    """FFT resampling along the last axis (window-local)."""
    n = x.shape[-1]
    if n == n_new:
        return x
    Xf = torch.fft.rfft(x, dim=-1)
    k = n_new // 2 + 1
    if Xf.shape[-1] >= k:
        Xf = Xf[..., :k]
    else:
        Xf = F.pad(Xf, (0, k - Xf.shape[-1]))
    return torch.fft.irfft(Xf, n=n_new, dim=-1) * (n_new / n)


class Adapter:
    def __init__(self, cfg, meta):
        self.ch_train = list(cfg["ch_names"])
        self.fs = float(cfg["sfreq"])
        self.n_times = int(cfg["n_times"])
        names = list(meta.get("ch_names") or [])
        if names:
            lookup = {}
            for i, n in enumerate(names):
                lookup.setdefault(_norm(n), i)
            self.idx = [lookup.get(_norm(c), -1) for c in self.ch_train]
        elif int(meta.get("n_chans") or -1) == len(self.ch_train):
            self.idx = list(range(len(self.ch_train)))
        else:
            raise ValueError("no channel names and a channel count different from training")
        self.missing = [c for c, i in zip(self.ch_train, self.idx) if i < 0]
        if self.missing:
            print(f"[ni26] {len(self.missing)} training channels absent at evaluation, zero-filled: {self.missing}")
        self.fs_in = float(meta.get("sfreq") or self.fs)

    def __call__(self, X):
        X = torch.as_tensor(X).float()
        sel = torch.tensor([max(i, 0) for i in self.idx], device=X.device)
        Z = X.index_select(1, sel)
        if self.missing:
            keep = torch.tensor([i >= 0 for i in self.idx], device=X.device, dtype=Z.dtype)
            Z = Z * keep[None, :, None]
        if abs(self.fs_in - self.fs) > 1e-6:
            Z = _resample(Z, int(round(Z.shape[-1] * self.fs / self.fs_in)))
        T = Z.shape[-1]
        if T > self.n_times:
            Z = Z[..., :self.n_times]
        elif T < self.n_times:
            Z = F.pad(Z, (0, self.n_times - T))
        med = Z.abs().median()
        if med < 1e-2 or med > 1e2:                          # not the kit's RobustScaled scale -> fall back
            c = Z.median(dim=-1, keepdim=True).values
            q = torch.quantile(Z, torch.tensor([0.25, 0.75], device=Z.device), dim=-1)
            Z = ((Z - c) / (q[1] - q[0]).clamp_min(1e-6)[..., None]).clamp(-20, 20)
        return Z


class Ensemble(torch.nn.Module):
    def __init__(self, nets, adapter):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.adapter = adapter

    @torch.inference_mode()
    def predict_proba(self, X):
        self.eval()
        Z = self.adapter(X)
        return torch.stack([F.softmax(n(Z).float(), dim=1) for n in self.nets]).mean(0)

    @torch.inference_mode()
    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_net(cfg):
    import braindecode.models as M
    cls = getattr(M, cfg["braindecode_class"])
    kw = dict(n_chans=len(cfg["ch_names"]), n_outputs=int(cfg["n_classes"]), n_times=int(cfg["n_times"]))
    if "sfreq" in inspect.signature(cls.__init__).parameters:
        kw["sfreq"] = float(cfg["sfreq"])
    return cls(**kw)


class Solver(CompetSolver):
    name = "ni26_cnn"

    def load_model(self, meta):
        torch.backends.cudnn.allow_tf32 = False        # exact fp32 inference on Ampere/Hopper (tiny models: no cost)
        torch.backends.cuda.matmul.allow_tf32 = False
        d = meta["submission_dir"]
        cfg = json.loads((d / "config.json").read_text())
        if int(meta.get("n_classes") or cfg["n_classes"]) != int(cfg["n_classes"]):
            raise ValueError(f"model has {cfg['n_classes']} classes, task wants {meta.get('n_classes')}")
        states = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
        nets = []
        for sd in states:
            net = build_net(cfg)
            net.load_state_dict(sd, strict=True)
            nets.append(net)
        return Ensemble(nets, Adapter(cfg, meta)).to(meta["device"]).eval()
