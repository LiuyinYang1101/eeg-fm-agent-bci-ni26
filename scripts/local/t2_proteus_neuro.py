#!/usr/bin/env python
"""First-principles neurophysiology of the Track 2 task (PROTEUS public export; descriptive, zero GPU; the user's question 2026-11-13).
The export is the predict()-time window: 41 EEG ch, 120 Hz, 4 s from 1 s after the cue, kit filter 0.1-75 Hz + 50/60 Hz notch, no re-referencing,
no baseline. So every map is a CLASS CONTRAST (class minus the mean of the three classes), after removing each session's mean log power.
Figures -> outputs/t2/analysis/proteus/: contrast topomaps (grand average + t across users), contrast spectra at key channels, the
within-window time course, per-user maps, Fisher-ratio maps, cross-user consistency, and simple band-power decodability
(within-session CV vs leave-one-user-out) under three referencings (as delivered / common average / surface Laplacian). summary.json.
    mem-guard python scripts/local/t2_proteus_neuro.py
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import welch  # noqa: E402

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OUT = Path.home() / "workspace/outputs/neural_interface_challenge/t2/analysis/proteus"
OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(E / "meta.json"))
CH, FS = meta["ch_names"], float(meta["sfreq"])
X = np.load(E / "X.npy").astype(np.float64)
idx = pd.read_csv(E / "index.csv")
idx["u"] = idx.subject.str[-3:]
idx["sess"] = idx.u + "_s" + idx.session.astype(str)
Y = idx.y.to_numpy()
NAMES = ["MI", "SUB", "WORD"]
BANDS = {"delta 1-4": (1, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-20": (13, 20), "beta 20-30": (20, 30), "gamma 30-45": (30, 45)}
KEY = ["C3", "Cz", "C4", "FCz", "Fz", "CPz", "Pz", "P3", "P4", "TP7", "TP8", "F5"]
info = mne.create_info(CH, FS, "eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")


def reref(Xa, kind):
    if kind == "delivered":
        return Xa
    if kind == "car":
        return Xa - Xa.mean(1, keepdims=True)
    ep = mne.EpochsArray(Xa.astype(np.float64) / 1e6, info, verbose=False)
    return mne.preprocessing.compute_current_source_density(ep).get_data() * 1e6


def bandpow(Xa, nperseg=120):
    f, P = welch(Xa, fs=FS, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    return f, 10 * np.log10(P + 1e-20)                                   # dB, (N, C, F)


f, PdB = bandpow(X)
BP = np.stack([PdB[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS.values()], -1)   # (N, C, B) dB


def centre(A):
    out = A.copy()
    for s in idx.sess.unique():
        m = (idx.sess == s).to_numpy()
        out[m] -= A[m].mean(0, keepdims=True)
    return out


BPc, PdBc = centre(BP), centre(PdB)


def contrasts(A):
    """per user (sessions averaged): class mean - mean of the class means. -> (U, K, ...)"""
    res = []
    users = sorted(idx.u.unique())
    for u in users:
        per = []
        for s in idx[idx.u == u].sess.unique():
            m = (idx.sess == s).to_numpy()
            cm = np.stack([A[m & (Y == k)].mean(0) for k in range(3)])
            per.append(cm - cm.mean(0, keepdims=True))
        res.append(np.mean(per, 0))
    return users, np.stack(res)


users, CB = contrasts(BPc)                   # (U, K, C, B)
GA = CB.mean(0)
T = GA / (CB.std(0, ddof=1) / np.sqrt(len(users)) + 1e-12)
summ = {"users": users, "preprocessing_of_export": "kit filter 0.1-75 Hz, notch 50/60 Hz, 120 Hz, no re-reference, no baseline, 4 s from +1 s after cue"}


def topo_grid(A, title, fname, unit="dB", sym=True, cmap="RdBu_r", vlim=None, mask=None):
    nb = A.shape[-1]
    fig, ax = plt.subplots(3, nb, figsize=(2.3 * nb + 1, 7.2))
    v = vlim or np.percentile(np.abs(A), 98)
    for k in range(3):
        for b in range(nb):
            im, _ = mne.viz.plot_topomap(A[k, :, b], info, axes=ax[k, b], show=False, cmap=cmap, vlim=(-v, v) if sym else (0, v),
                                         contours=4, mask=None if mask is None else mask[k, :, b], mask_params=dict(markersize=5, markerfacecolor="k"))
            if k == 0:
                ax[k, b].set_title(list(BANDS)[b], fontsize=10)
        ax[k, 0].set_ylabel(NAMES[k], fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label=unit)
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


topo_grid(GA, "PROTEUS: class − mean of the three classes, 4 s task window (+1 to +5 s after cue), grand average over 10 users (dB)", "contrast_ga.png")
topo_grid(T, "PROTEUS: t across users of the class contrast (n = 10; dots: |t| > 2.26, p < 0.05 two-sided)", "contrast_t.png", unit="t",
          vlim=5, mask=np.abs(T) > 2.26)
summ["n_sig_channels"] = {NAMES[k]: {bn: int((np.abs(T[k, :, b]) > 2.26).sum()) for b, bn in enumerate(BANDS)} for k in range(3)}

# contrast spectra at key channels
_, CS = contrasts(PdBc)                      # (U, K, C, F)
fm = (f >= 1) & (f <= 45)
fig, ax = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
for j, ch in enumerate(KEY):
    a = ax.flat[j]
    ci = CH.index(ch)
    for k, c in zip(range(3), ("C0", "C3", "C2")):
        mu, se = CS[:, k, ci, fm].mean(0), CS[:, k, ci, fm].std(0, ddof=1) / np.sqrt(len(users))
        a.plot(f[fm], mu, c, label=NAMES[k])
        a.fill_between(f[fm], mu - se, mu + se, color=c, alpha=0.2)
    a.axhline(0, color="k", lw=0.5)
    a.set_title(ch)
    for lo in (8, 13, 30):
        a.axvline(lo, color="grey", lw=0.4, ls=":")
ax.flat[0].legend()
fig.supxlabel("Hz")
fig.supylabel("class − mean of classes (dB), mean ± SE over 10 users")
fig.suptitle("PROTEUS: which frequencies separate the three commands, per channel")
fig.savefig(OUT / "contrast_spectra.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# within-window time course: 1 s windows, 0.25 s hop
starts = np.arange(0, 480 - 120 + 1, 30)
tc = {}
for bn, (lo, hi) in (("alpha 8-13", (8, 13)), ("beta 13-20", (13, 20)), ("theta 4-8", (4, 8))):
    arr = []
    for s0 in starts:
        fw, Pw = welch(X[:, :, s0:s0 + 120], fs=FS, nperseg=120, axis=-1)
        arr.append(10 * np.log10(Pw[..., (fw >= lo) & (fw < hi)].mean(-1) + 1e-20))
    tc[bn] = np.stack(arr, -1)              # (N, C, Tt)
tcen = 1 + (starts + 60) / FS
fig, ax = plt.subplots(3, 5, figsize=(17, 8.5), sharex=True)
for r, bn in enumerate(tc):
    _, CT = contrasts(centre(tc[bn]))        # (U, K, C, Tt)
    for j, ch in enumerate(["C3", "C4", "Cz", "Fz", "Pz"]):
        a = ax[r, j]
        ci = CH.index(ch)
        for k, c in zip(range(3), ("C0", "C3", "C2")):
            mu, se = CT[:, k, ci].mean(0), CT[:, k, ci].std(0, ddof=1) / np.sqrt(len(users))
            a.plot(tcen, mu, c, label=NAMES[k])
            a.fill_between(tcen, mu - se, mu + se, color=c, alpha=0.2)
        a.axhline(0, color="k", lw=0.5)
        a.set_title(f"{ch}, {bn}", fontsize=9)
ax[0, 0].legend(fontsize=8)
fig.supxlabel("time after cue (s), centre of a 1 s window")
fig.supylabel("class − mean of classes (dB)")
fig.suptitle("PROTEUS: how the class differences evolve inside the 4 s window")
fig.savefig(OUT / "timecourse.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# per-user maps in the alpha and beta 13-20 bands
for b in (0, 2, 3):
    bn = list(BANDS)[b]
    fig, ax = plt.subplots(3, len(users), figsize=(1.9 * len(users), 6))
    v = np.percentile(np.abs(CB[..., b]), 97)
    for ui, u in enumerate(users):
        for k in range(3):
            mne.viz.plot_topomap(CB[ui, k, :, b], info, axes=ax[k, ui], show=False, cmap="RdBu_r", vlim=(-v, v), contours=0)
            if k == 0:
                ax[k, ui].set_title(u, fontsize=10)
        ax[0, 0].set_ylabel("MI")
        ax[1, 0].set_ylabel("SUB")
        ax[2, 0].set_ylabel("WORD")
    fig.suptitle(f"PROTEUS: the same contrast, user by user ({bn}; one shared scale ±{v:.2f} dB)", fontsize=11)
    fig.savefig(OUT / f"per_user_{bn.split()[0]}{bn.split()[1].replace('-', '_')}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

# consistency across users: leave-one-user-out correlation of the contrast topography
cons = {}
for b, bn in enumerate(BANDS):
    rs = []
    for ui in range(len(users)):
        others = np.delete(CB, ui, 0).mean(0)
        rs.append(np.mean([np.corrcoef(CB[ui, k, :, b], others[k, :, b])[0, 1] for k in range(3)]))
    cons[bn] = dict(mean_r=round(float(np.mean(rs)), 3), per_user={u: round(float(r), 3) for u, r in zip(users, rs)})
summ["loo_topography_r"] = cons

# Fisher ratio per channel x band (within session, then averaged)
FR = []
for s in idx.sess.unique():
    m = (idx.sess == s).to_numpy()
    A = BPc[m]
    cm = np.stack([A[Y[m] == k].mean(0) for k in range(3)])
    wv = np.mean([A[Y[m] == k].var(0) for k in range(3)], 0)
    FR.append(cm.var(0) / (wv + 1e-12))
FR = np.mean(FR, 0)
fig, ax = plt.subplots(1, len(BANDS), figsize=(2.6 * len(BANDS) + 1, 3))
v = np.percentile(FR, 99)
for b, bn in enumerate(BANDS):
    im, _ = mne.viz.plot_topomap(FR[:, b], info, axes=ax[b], show=False, cmap="viridis", vlim=(0, v), contours=0)
    ax[b].set_title(bn, fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.7, label="Fisher ratio")
fig.suptitle("PROTEUS: where the three commands separate within a session (between-class / within-class variance)", fontsize=10)
fig.savefig(OUT / "fisher.png", dpi=110, bbox_inches="tight")
plt.close(fig)
top = np.dstack(np.unravel_index(np.argsort(-FR.ravel())[:10], FR.shape))[0]
summ["fisher_top10"] = [(CH[c], list(BANDS)[b], round(float(FR[c, b]), 3)) for c, b in top]

# simple decodability of log band power under three referencings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402


def ba(yt, yp):
    return float(np.mean([np.mean(yp[yt == k] == k) for k in np.unique(yt)]))


dec = {}
for kind in ("delivered", "car", "laplacian"):
    Xr = reref(X, kind)
    fr, Pr = welch(Xr, fs=FS, nperseg=120, axis=-1)
    F = centre(np.stack([10 * np.log10(Pr[..., (fr >= lo) & (fr < hi)].mean(-1) + 1e-20) for lo, hi in BANDS.values()], -1)).reshape(len(X), -1)
    within, cross = [], []
    for s in idx.sess.unique():
        m = (idx.sess == s).to_numpy()
        p = cross_val_predict(LDA(solver="lsqr", shrinkage="auto"), F[m], Y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0))
        within.append(ba(Y[m], p))
    for u in users:
        te = (idx.u == u).to_numpy()
        p = LDA(solver="lsqr", shrinkage="auto").fit(F[~te], Y[~te]).predict(F[te])
        cross.append(ba(Y[te], p))
    dec[kind] = dict(within_session_cv=round(float(np.mean(within)), 3), cross_user_louo=round(float(np.mean(cross)), 3),
                     cross_user_per_user={u: round(c, 3) for u, c in zip(users, cross)})
    print(kind, dec[kind]["within_session_cv"], dec[kind]["cross_user_louo"], flush=True)
summ["bandpower_slda_ba"] = dec
# which region / band carries the decodable information? (CAR, the best referencing above)
Xr = reref(X, "car")
fr, Pr = welch(Xr, fs=FS, nperseg=120, axis=-1)
FB = centre(np.stack([10 * np.log10(Pr[..., (fr >= lo) & (fr < hi)].mean(-1) + 1e-20) for lo, hi in BANDS.values()], -1))
FRONT = [c for c in CH if c.startswith(("AF", "F")) and not c.startswith("FC")]
CENT = [c for c in CH if c.startswith(("C", "FC", "CP")) and c not in ("CPz",)] + ["CPz"]
PAR = [c for c in CH if c.startswith(("P", "PO", "TP"))]
bi = {b: i for i, b in enumerate(BANDS)}
subsets = {"all channels, all bands": (CH, list(BANDS)),
           "frontal AF/F, delta + theta": (FRONT, ["delta 1-4", "theta 4-8"]),
           "frontal AF/F, all bands": (FRONT, list(BANDS)),
           "central FC/C/CP, alpha + beta": (CENT, ["alpha 8-13", "beta 13-20", "beta 20-30"]),
           "central FC/C/CP, all bands": (CENT, list(BANDS)),
           "parietal P/PO/TP, all bands": (PAR, list(BANDS)),
           "all channels without delta + theta": (CH, ["alpha 8-13", "beta 13-20", "beta 20-30", "gamma 30-45"])}
reg = {}
for name, (chs, bns) in subsets.items():
    F = FB[:, [CH.index(c) for c in chs]][:, :, [bi[b] for b in bns]].reshape(len(X), -1)
    within, cross = [], []
    for s in idx.sess.unique():
        m = (idx.sess == s).to_numpy()
        p = cross_val_predict(LDA(solver="lsqr", shrinkage="auto"), F[m], Y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0))
        within.append(ba(Y[m], p))
    for u in users:
        te = (idx.u == u).to_numpy()
        cross.append(ba(Y[te], LDA(solver="lsqr", shrinkage="auto").fit(F[~te], Y[~te]).predict(F[te])))
    reg[name] = dict(n_features=int(F.shape[1]), within_session_cv=round(float(np.mean(within)), 3), cross_user_louo=round(float(np.mean(cross)), 3))
    print("region", name, reg[name], flush=True)
summ["region_band_slda_ba_car"] = reg
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps({k: summ[k] for k in ("n_sig_channels", "fisher_top10")}, indent=0)[:1500])
print({b: c["mean_r"] for b, c in cons.items()})
