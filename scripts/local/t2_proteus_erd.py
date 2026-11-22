#!/usr/bin/env python
"""Verification of T2-D473 (the user's doubt, 2026-11-13: "MI lacks clear ERD in motor regions […]"):
event-related (de)synchronisation of each command RELATIVE TO PRE-CUE REST, on the PROTEUS long export (-3 .. +8 s re cue, kit pipeline,
job 1613891; the contract window is exactly +1 .. +5 s of it). Spectrogram: 1 s Hann window, 0.1 s hop; baseline = -2.5 .. -0.5 s, per
trial-averaged session and class (dB re baseline). Grand average over the 10 users (sessions averaged within a user).
Figures -> outputs/t2/analysis/proteus_erd/: time-frequency maps at C3 / Cz / C4 per class; topomap grids (class x time) in mu 8-13 and
beta 13-30 Hz; ERD time courses at C3 / Cz / C4 (mean +- SE over users); per-user task-window mu / beta ERD at C3 / C4; summary.json.
    mem-guard python scripts/local/t2_proteus_erd.py [--car]
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
CAR = "--car" in sys.argv
L = Path.home() / "workspace/outputs/neural_interface_challenge/sofia_s1/ni26-t2p-export-long-1613891/export_proteus_long_m3p8"
OUT = Path.home() / f"workspace/outputs/neural_interface_challenge/t2/analysis/proteus_erd{'_car' if CAR else ''}"
OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(L / "meta.json"))
CH, FS, T0 = meta["ch_names"], float(meta["sfreq"]), float(meta["window_start"])
X = np.load(L / "X.npy", mmap_mode="r")
idx = pd.read_csv(L / "index.csv")
idx["u"] = idx.subject.str[-3:]
idx["sess"] = idx.u + "_s" + idx.session.astype(str)
Y = idx.y.to_numpy()
NAMES = ["MI", "SUB", "WORD"]
info = mne.create_info(CH, FS, "eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="raise")
# spectrogram of every trial, in chunks (memory)
f, t, _ = spectrogram(np.zeros(X.shape[-1]), fs=FS, window="hann", nperseg=int(FS), noverlap=int(FS) - int(0.1 * FS))
t = t + T0
fm = (f >= 2) & (f <= 45)
f = f[fm]
S = np.empty((len(X), len(CH), fm.sum(), len(t)), np.float32)
for a in range(0, len(X), 128):
    x = np.asarray(X[a:a + 128], np.float64)
    if CAR:
        x = x - x.mean(1, keepdims=True)
    _, _, P = spectrogram(x, fs=FS, window="hann", nperseg=int(FS), noverlap=int(FS) - int(0.1 * FS), axis=-1)
    S[a:a + 128] = P[:, :, fm].astype(np.float32)
bl = (t >= -2.5) & (t <= -0.5)
users = sorted(idx.u.unique())
ERD = []                                        # (U, K, C, F, T) dB re baseline
for u in users:
    per = []
    for s in idx[idx.u == u].sess.unique():
        m = (idx.sess == s).to_numpy()
        cl = []
        for k in range(3):
            P = S[m & (Y == k)].mean(0)                     # trial-averaged power
            cl.append(10 * np.log10(P / P[..., bl].mean(-1, keepdims=True)))
        per.append(np.stack(cl))
    ERD.append(np.mean(per, 0))
ERD = np.stack(ERD)
GA = ERD.mean(0)
summ = {"baseline_s": [-2.5, -0.5], "car": CAR, "users": users}
mu, beta = (f >= 8) & (f < 13), (f >= 13) & (f < 30)
task = (t >= 1.0) & (t <= 5.0)
# 1) time-frequency maps
fig, ax = plt.subplots(3, 3, figsize=(14, 9), sharex=True, sharey=True)
v = 3.0
for k in range(3):
    for j, ch in enumerate(["C3", "Cz", "C4"]):
        a = ax[k, j]
        im = a.pcolormesh(t, f, GA[k, CH.index(ch)], cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        a.axvline(0, color="k", lw=0.8)
        a.axvspan(1, 5, ymin=0, ymax=0.03, color="k")
        a.set_title(f"{NAMES[k]} at {ch}", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.6, label="dB re pre-cue rest (−2.5 … −0.5 s)")
fig.supxlabel("time re cue (s); black bar = the 4 s window the challenge delivers")
fig.supylabel("Hz")
fig.suptitle(f"PROTEUS: power relative to pre-cue rest, grand average of 10 users{' (common average reference)' if CAR else ''}")
fig.savefig(OUT / "tf_C3_Cz_C4.png", dpi=110, bbox_inches="tight")
plt.close(fig)
# 2) topomap grids class x time in mu and beta
spans = [(0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 7.5)]
for bname, bm in (("mu 8-13 Hz", mu), ("beta 13-30 Hz", beta)):
    fig, ax = plt.subplots(3, len(spans), figsize=(2.6 * len(spans) + 1, 7.2))
    vals = np.stack([[GA[k][:, bm][:, :, (t >= lo) & (t < hi)].mean((1, 2)) for lo, hi in spans] for k in range(3)])
    vv = max(1.0, np.percentile(np.abs(vals), 98))
    for k in range(3):
        for j, (lo, hi) in enumerate(spans):
            im, _ = mne.viz.plot_topomap(vals[k, j], info, axes=ax[k, j], show=False, cmap="RdBu_r", vlim=(-vv, vv), contours=4)
            if k == 0:
                ax[k, j].set_title(f"{lo:g}–{hi:g} s", fontsize=10)
        ax[k, 0].set_ylabel(NAMES[k], fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label="dB re pre-cue rest")
    fig.suptitle(f"PROTEUS: {bname} power relative to pre-cue rest (blue = desynchronisation), grand average", fontsize=11)
    fig.savefig(OUT / f"topo_{bname.split()[0]}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
# 3) time courses
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for r, (bname, bm) in enumerate((("mu 8-13", mu), ("beta 13-30", beta))):
    for j, ch in enumerate(["C3", "Cz", "C4"]):
        a = ax[r, j]
        for k, c in zip(range(3), ("C0", "C3", "C2")):
            tc = ERD[:, k, CH.index(ch)][:, bm].mean(1)            # (U, T)
            a.plot(t, tc.mean(0), c, label=NAMES[k])
            a.fill_between(t, tc.mean(0) - tc.std(0, ddof=1) / np.sqrt(len(users)), tc.mean(0) + tc.std(0, ddof=1) / np.sqrt(len(users)), color=c, alpha=0.2)
        a.axhline(0, color="k", lw=0.5)
        a.axvline(0, color="k", lw=0.8)
        a.axvspan(1, 5, color="grey", alpha=0.08)
        a.set_title(f"{ch}, {bname} Hz", fontsize=10)
ax[0, 0].legend()
fig.supxlabel("time re cue (s); shaded = the challenge's 4 s window")
fig.supylabel("dB re pre-cue rest, mean ± SE over users")
fig.savefig(OUT / "timecourse_C3_Cz_C4.png", dpi=110, bbox_inches="tight")
plt.close(fig)
# 4) numbers: task-window ERD per class at C3 / Cz / C4 (mean, t over users), and per user
st = {}
for bname, bm in (("mu", mu), ("beta", beta)):
    for ch in ("C3", "Cz", "C4", "Fz", "Pz"):
        vals = ERD[:, :, CH.index(ch)][:, :, bm][..., task].mean((2, 3))      # (U, K)
        st[f"{bname}@{ch}"] = {NAMES[k]: dict(mean_dB=round(float(vals[:, k].mean()), 3),
                                              t=round(float(vals[:, k].mean() / (vals[:, k].std(ddof=1) / np.sqrt(len(users)))), 2),
                                              n_users_below_0=int((vals[:, k] < 0).sum()),
                                              per_user={u: round(float(x), 2) for u, x in zip(users, vals[:, k])}) for k in range(3)}
summ["task_window_erd_dB"] = st
mi_vs = {}
for ch in ("C3", "C4", "Cz"):
    for bname, bm in (("mu", mu), ("beta", beta)):
        vals = ERD[:, :, CH.index(ch)][:, :, bm][..., task].mean((2, 3))
        d = vals[:, 0] - vals[:, 1:].mean(1)
        mi_vs[f"{bname}@{ch}"] = dict(mi_minus_others_dB=round(float(d.mean()), 3), t=round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(users)))), 2))
summ["mi_minus_others_task_window"] = mi_vs
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
for key in ("mu@C3", "mu@C4", "mu@Cz", "beta@C3", "beta@C4", "beta@Cz", "mu@Pz", "mu@Fz"):
    print(key, {k: (v["mean_dB"], v["t"], v["n_users_below_0"]) for k, v in st[key].items()})
print("MI minus mean(SUB, WORD), task window:", mi_vs)
