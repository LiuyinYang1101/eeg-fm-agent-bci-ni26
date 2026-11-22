"""Track-2 baselines that every candidate must beat (LESSONS C1: compact/classical controls through the SAME
pipeline). Inputs are windows exactly as predict() receives them (kit pipeline: 120 Hz, RobustScaled, ±20).

Classical (CPU): ``logvar_lda`` (8-30 Hz log-variance + shrinkage LDA), ``ts_lr`` (8-30 Hz OAS covariance ->
Riemannian tangent space -> logistic regression), ``fbts_lr`` (filter-bank tangent space: 4-8 ... 24-32 Hz).
Deep (GPU): ``eegnet`` (braindecode EEGNet, the kit's baseline architecture), AdamW + cosine, plain CE.

Band-pass filtering is window-local (zero-phase within the window): legal at inference, since predict() gets
the whole window. pyriemann is NOT in the Codabench image — a shipped tangent-space model must reimplement
the (few-line) matrix log/whitening in numpy/torch.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS_FB = [(4, 8), (8, 12), (12, 16), (16, 24), (24, 32)]


def bandpass(X, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1)


def logvar(X, fs, band=(8, 30)):
    return np.log(np.var(bandpass(X, *band, fs), axis=-1) + 1e-8)


def covs(X, fs, band):
    from pyriemann.estimation import Covariances
    return Covariances("oas").fit_transform(bandpass(X, *band, fs))


class TangentSpaceLR:
    """(Filter-bank) covariance -> tangent space at the training Riemannian mean -> scaled LR."""

    def __init__(self, fs, bands=((8, 30),), C=1.0):
        self.fs, self.bands, self.C = fs, list(bands), C

    def _feats(self, X, fit):
        from pyriemann.tangentspace import TangentSpace
        out = []
        for k, band in enumerate(self.bands):
            Cv = covs(X, self.fs, band)
            if fit:
                self.ts_[k] = TangentSpace(metric="riemann").fit(Cv)
            out.append(self.ts_[k].transform(Cv))
        return np.concatenate(out, axis=1)

    def fit(self, X, y):
        self.ts_ = {}
        F = self._feats(X, True)
        self.clf_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        self.clf_.fit(F, y)
        return self

    def predict(self, X):
        return self.clf_.predict(self._feats(X, False))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._feats(X, False))


class LogVarLDA:
    def __init__(self, fs, band=(8, 30)):
        self.fs, self.band = fs, band

    def fit(self, X, y):
        self.clf_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(logvar(X, self.fs, self.band), y)
        return self

    def predict(self, X):
        return self.clf_.predict(logvar(X, self.fs, self.band))

    def predict_proba(self, X):
        return self.clf_.predict_proba(logvar(X, self.fs, self.band))


BANDS_BP = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 55)]


class LogBandPowerLDA:
    """Per-channel log power in 5 bands incl. 30-55 Hz (EMG lives there; the pipeline keeps up to 60 Hz) + sLDA.
    Generic across channel types, so it is the feature set of the EEG/EOG/EMG channel-role study."""

    def __init__(self, fs, bands=BANDS_BP):
        self.fs, self.bands = fs, [b for b in bands if b[1] < fs / 2]

    def _f(self, X):
        return np.concatenate([logvar(X, self.fs, b) for b in self.bands], axis=1)

    def fit(self, X, y):
        self.clf_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._f(X), y)
        return self

    def predict(self, X):
        return self.clf_.predict(self._f(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._f(X))


class EEGNetClf:
    """braindecode EEGNet (kit baseline architecture). Selection-free: fixed epochs, no early stopping on test."""

    def __init__(self, n_chans, n_times, n_classes, epochs=40, lr=1e-3, wd=1e-2, bs=64, seed=0, device="cuda"):
        self.kw = dict(n_chans=n_chans, n_times=n_times, n_classes=n_classes)
        self.epochs, self.lr, self.wd, self.bs, self.seed, self.device = epochs, lr, wd, bs, seed, device

    def fit(self, X, y):
        import torch
        from braindecode.models import EEGNet
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net_ = EEGNet(n_chans=self.kw["n_chans"], n_outputs=self.kw["n_classes"],
                           n_times=self.kw["n_times"]).to(self.device)
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(Xt) / self.bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.1)
        lossf = torch.nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.bs):
                b = perm[i:i + self.bs]
                xb, yb = Xt[b].to(self.device, non_blocking=True), yt[b].to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net_(xb), yb)
                loss.backward()
                opt.step()
                sched.step()
        return self

    def predict(self, X):
        import torch
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.as_tensor(np.asarray(X[i:i + 256], dtype=np.float32)).to(self.device)
                out.append(self.net_(xb).argmax(1).cpu().numpy())
        return np.concatenate(out)


def make(name, fs, n_chans, n_times, n_classes, seed=0, device="cuda"):
    if name == "logvar_lda":
        return LogVarLDA(fs)
    if name == "logbp_lda":
        return LogBandPowerLDA(fs)
    if name == "ts_lr":
        return TangentSpaceLR(fs)
    if name == "fbts_lr":
        return TangentSpaceLR(fs, BANDS_FB)
    if name == "eegnet":
        return EEGNetClf(n_chans, n_times, n_classes, seed=seed, device=device)
    raise ValueError(name)
