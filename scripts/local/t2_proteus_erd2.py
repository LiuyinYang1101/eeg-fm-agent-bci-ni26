#!/usr/bin/env python
"""Panel P11 (2026-11-14, T2 seat, descriptive, zero GPU): what else does the recording reference hide? Extends t2_proteus_erd.py (T2-D473's
verification) on the PROTEUS long export (-3 .. +8 s re cue; the contract window = +1 .. +5 s): power re pre-cue rest (-2.5 .. -0.5 s) per
user (sessions averaged), under the DELIVERED reference and under a common average reference (CAR), for
  (a) frontal-midline theta 4-8 Hz (Fz, FCz, F1, F2): mental arithmetic (SUB) vs the others;
  (b) left vs right fronto-temporal sites (F5 FC5 C5 TP7 vs F6 FC6 C6 TP8) in theta / alpha / beta / low gamma 30-45 Hz: WORD vs the others,
      and its laterality (left - right);
  (c) the frontal slow wave 2-4 Hz: its gradient front -> back (AF -> F -> FC -> C -> CP -> P rows; an ocular source falls off with distance
      from the eyes) and the per-user sign of SUB - MI at AF3 / AF4 / F5 / F6 (T2-D473: P10 reverses it);
  (d) the masked-signal count: per band and class, electrodes whose task-window (class - mean of the others) contrast has |t| > 2.26 over the
      10 users (t_0.975, 9 df), delivered vs CAR; a contrast present under CAR and absent as delivered = a second masked signal.
Writes outputs/t2/analysis/proteus_erd2/summary.json and prints the lines the seat file quotes.
    OMP_NUM_THREADS=4 mem-guard python scripts/local/t2_proteus_erd2.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

warnings.filterwarnings("ignore")
L = Path.home() / "workspace/outputs/neural_interface_challenge/sofia_s1/ni26-t2p-export-long-1613891/export_proteus_long_m3p8"
OUT = Path.home() / "workspace/outputs/neural_interface_challenge/t2/analysis/proteus_erd2"
OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(L / "meta.json"))
CH, FS, T0 = meta["ch_names"], float(meta["sfreq"]), float(meta["window_start"])
X = np.load(L / "X.npy", mmap_mode="r")
idx = pd.read_csv(L / "index.csv")
idx["u"] = idx.subject.str[-3:]
idx["sess"] = idx.u + "_s" + idx.session.astype(str)
Y = idx.y.to_numpy()
NAMES = ["MI", "SUB", "WORD"]
users = sorted(idx.u.unique())
BANDS = {"delta2-4": (2, 4), "theta4-8": (4, 8), "alpha8-13": (8, 13), "beta13-30": (13, 30), "gamma30-45": (30, 45)}
TCRIT = 2.262


def erd(car):
    f, t, _ = spectrogram(np.zeros(X.shape[-1]), fs=FS, window="hann", nperseg=int(FS), noverlap=int(FS) - int(0.1 * FS))
    t = t + T0
    fm = (f >= 2) & (f <= 45)
    f = f[fm]
    task, bl = (t >= 1.0) & (t <= 5.0), (t >= -2.5) & (t <= -0.5)
    B = {b: (f >= lo) & (f < hi) for b, (lo, hi) in BANDS.items()}
    # per trial: band power in the task window and in the baseline (C x bands), then dB per user / session / class
    Pt = np.empty((len(X), len(CH), len(B)), np.float64)
    Pb = np.empty_like(Pt)
    for a in range(0, len(X), 128):
        x = np.asarray(X[a:a + 128], np.float64)
        if car:
            x = x - x.mean(1, keepdims=True)
        _, _, P = spectrogram(x, fs=FS, window="hann", nperseg=int(FS), noverlap=int(FS) - int(0.1 * FS), axis=-1)
        P = P[:, :, fm]
        for j, bm in enumerate(B.values()):
            Pt[a:a + 128, :, j] = P[:, :, bm][..., task].mean((2, 3))
            Pb[a:a + 128, :, j] = P[:, :, bm][..., bl].mean((2, 3))
    E = []                                                       # (U, K, C, bands) dB re that session-class's own baseline
    for u in users:
        per = []
        for s in idx[idx.u == u].sess.unique():
            m = (idx.sess == s).to_numpy()
            per.append(np.stack([10 * np.log10(Pt[m & (Y == k)].mean(0) / Pb[m & (Y == k)].mean(0)) for k in range(3)]))
        E.append(np.mean(per, 0))
    return np.stack(E), list(B)


def tstat(v):
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))


out = {}
for ref in ("delivered", "car"):
    E, bn = erd(ref == "car")
    r = {}
    ci = {c: CH.index(c) for c in CH}
    th = bn.index("theta4-8")
    fmid = [ci[c] for c in ("Fz", "FCz", "F1", "F2")]
    v = E[:, :, fmid, th].mean(2)                                # (U, K)
    r["a_fm_theta"] = {NAMES[k]: round(float(v[:, k].mean()), 3) for k in range(3)} | {
        "SUB_minus_others": round(float((v[:, 1] - v[:, [0, 2]].mean(1)).mean()), 3), "t": round(tstat(v[:, 1] - v[:, [0, 2]].mean(1)), 2)}
    left, right = [ci[c] for c in ("F5", "FC5", "C5", "TP7")], [ci[c] for c in ("F6", "FC6", "C6", "TP8")]
    r["b_frontotemporal"] = {}
    for b in ("theta4-8", "alpha8-13", "beta13-30", "gamma30-45"):
        j = bn.index(b)
        lw = E[:, :, left, j].mean(2)
        rw = E[:, :, right, j].mean(2)
        dw = lw[:, 2] - lw[:, [0, 1]].mean(1)
        lat = (lw[:, 2] - rw[:, 2]) - ((lw[:, [0, 1]] - rw[:, [0, 1]]).mean(1))
        r["b_frontotemporal"][b] = dict(WORD_left=round(float(lw[:, 2].mean()), 3), WORD_minus_others_left=round(float(dw.mean()), 3),
                                        t=round(tstat(dw), 2), WORD_laterality_vs_others=round(float(lat.mean()), 3), t_lat=round(tstat(lat), 2))
    j = bn.index("delta2-4")
    rows = {"AF": ["AF3", "AF4"], "F": ["F5", "F3", "F1", "Fz", "F2", "F4", "F6"], "FC": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"],
            "C": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"], "CP": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"], "P": ["P5", "P3", "P1", "Pz", "P2", "P4", "P6"]}
    r["c_delta_gradient_SUB_minus_MI_dB"] = {rw_: round(float((E[:, 1, [ci[c] for c in cs], j] - E[:, 0, [ci[c] for c in cs], j]).mean()), 3)
                                             for rw_, cs in rows.items()}
    fr = [ci[c] for c in ("AF3", "AF4", "F5", "F6")]
    d = (E[:, 1, fr, j] - E[:, 0, fr, j]).mean(1)
    r["c_delta_per_user_SUB_minus_MI"] = {u: round(float(x), 2) for u, x in zip(users, d)}
    r["c_delta_class_vs_rest_frontal"] = {NAMES[k]: round(float(E[:, k, fr, j].mean()), 3) for k in range(3)}
    cnt = {}
    for b in bn:
        Eb = E[..., bn.index(b)]                                                         # (U, K, C)
        for k in range(3):
            con = Eb[:, k] - Eb[:, [q for q in range(3) if q != k]].mean(1)                 # (U, C)
            tt = con.mean(0) / (con.std(0, ddof=1) / np.sqrt(len(users)))
            sig = [CH[c] for c in np.flatnonzero(np.abs(tt) > TCRIT)]
            cnt[f"{NAMES[k]}@{b}"] = dict(n=len(sig), sites=sig[:8])
    r["d_significant_contrasts"] = cnt
    out[ref] = r
json.dump(out, open(OUT / "summary.json", "w"), indent=1)
for ref in out:
    r = out[ref]
    print(f"== {ref}")
    print(" (a) frontal-midline theta dB re rest:", r["a_fm_theta"])
    for b, v in r["b_frontotemporal"].items():
        print(f" (b) left fronto-temporal {b}:", v)
    print(" (c) delta 2-4 SUB - MI by row (front -> back):", r["c_delta_gradient_SUB_minus_MI_dB"])
    print(" (c) frontal delta class vs rest:", r["c_delta_class_vs_rest_frontal"], " per user SUB - MI:", r["c_delta_per_user_SUB_minus_MI"])
    print(" (d) significant contrasts (n sites):", {k: v["n"] for k, v in r["d_significant_contrasts"].items()})
d0, d1 = out["delivered"]["d_significant_contrasts"], out["car"]["d_significant_contrasts"]
print("(d) present under CAR, absent / weaker as delivered:", {k: (d0[k]["n"], d1[k]["n"], d1[k]["sites"]) for k in d1 if d1[k]["n"] >= d0[k]["n"] + 3})
