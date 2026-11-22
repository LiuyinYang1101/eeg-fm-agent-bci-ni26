#!/usr/bin/env python
"""T2-D515 features (registered 2026-11-15 before any number; CPU only): every PROTEUS window (export_proteus_v1, 2,655 x 41 x 480 at 120 Hz),
two references (as delivered, CAR = each window minus its channel mean), per window:
  * TSFEL 0.3.0 statistical + temporal + spectral (its default set) on all 41 channels;
  * catch22 (pycatch22) on all 41 channels;
  * an EEG block: Welch PSD (2-s segments, 50 % overlap) -> log band powers d 1-4 / t 4-8 / a 8-13 / b1 13-20 / b2 20-30 / g 30-40 / hi 40-58 Hz and
    relative powers (/ 1-58 Hz), 1/f slope and offset (log-log fit 2-40 Hz), spectral entropy (1-58 Hz), alpha peak (7-14 Hz); hemispheric
    log-asymmetries of the band powers for homologous pairs; coherence and phase-locking value (band-passed, Hilbert) between 12 fixed pairs in t / a / b.
Output: outputs/t2_d515/features.npz: F (N, D) float32 in the export's row order, names (D,), family (D,) = the block each column comes from.
    mem-guard python scripts/local/t2_d515_features.py
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OUT = Path.home() / "workspace/outputs/neural_interface_challenge/t2_d515"
FS = 120.0
BANDS = {"d": (1, 4), "t": (4, 8), "a": (8, 13), "b1": (13, 20), "b2": (20, 30), "g": (30, 40), "hi": (40, 58)}
PAIRS_ASYM = [("C3", "C4"), ("CP3", "CP4"), ("FC3", "FC4"), ("F3", "F4"), ("P3", "P4"), ("C1", "C2"), ("CP1", "CP2"), ("P1", "P2"), ("F1", "F2"),
              ("FC5", "FC6"), ("C5", "C6"), ("CP5", "CP6"), ("TP7", "TP8"), ("PO7", "PO8"), ("AF3", "AF4"), ("F5", "F6"), ("P5", "P6")]
PAIRS_SYNC = [("C3", "C4"), ("F3", "F4"), ("P3", "P4"), ("Fz", "Pz"), ("C3", "Pz"), ("C4", "Pz"), ("F3", "C3"), ("F4", "C4"), ("AF3", "AF4"),
              ("CP3", "CP4"), ("FC3", "FC4"), ("Cz", "Pz")]
SYNC_BANDS = {"t": (4, 8), "a": (8, 13), "b": (13, 30)}
META = json.loads((EXP / "meta.json").read_text())
DEMEAN = os.environ.get("T2_D515_DEMEAN") == "1"           # T2-D515d (an env var: the Pool workers re-import this module)
CH = list(META["ch_names"])


def eeg_block(x, tag):
    from scipy.signal import butter, coherence, hilbert, sosfiltfilt, welch
    f, P = welch(x, fs=FS, nperseg=240, noverlap=120, axis=-1)                     # (C, F)
    vals, names = [], []
    tot = P[:, (f >= 1) & (f <= 58)].sum(1) + 1e-12
    bp = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp[b] = P[:, m].sum(1) + 1e-12
        vals += [np.log(bp[b]), np.log(bp[b] / tot)]
        names += [[f"{tag}_lbp_{b}_{c}" for c in CH], [f"{tag}_rbp_{b}_{c}" for c in CH]]
    m = (f >= 2) & (f <= 40)
    lf, lp = np.log10(f[m]), np.log10(P[:, m] + 1e-12)
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef = np.linalg.lstsq(A, lp.T, rcond=None)[0]                                  # (2, C)
    vals += [coef[0], coef[1]]
    names += [[f"{tag}_slope_{c}" for c in CH], [f"{tag}_offset_{c}" for c in CH]]
    m = (f >= 1) & (f <= 58)
    q = P[:, m] / P[:, m].sum(1, keepdims=True)
    vals.append(-(q * np.log(q + 1e-12)).sum(1))
    names.append([f"{tag}_spent_{c}" for c in CH])
    m = (f >= 7) & (f <= 14)
    vals.append(f[m][P[:, m].argmax(1)])
    names.append([f"{tag}_apeak_{c}" for c in CH])
    ix = {c: i for i, c in enumerate(CH)}
    for b in BANDS:
        vals.append(np.array([np.log(bp[b][ix[l_]]) - np.log(bp[b][ix[r_]]) for l_, r_ in PAIRS_ASYM]))
        names.append([f"{tag}_asym_{b}_{l_}-{r_}" for l_, r_ in PAIRS_ASYM])
    for b, (lo, hi) in SYNC_BANDS.items():
        sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        ph = np.angle(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1))
        plv, coh = [], []
        for a_, b_ in PAIRS_SYNC:
            plv.append(np.abs(np.mean(np.exp(1j * (ph[ix[a_]] - ph[ix[b_]])))))
            fc, cxy = coherence(x[ix[a_]], x[ix[b_]], fs=FS, nperseg=240, noverlap=120)
            coh.append(cxy[(fc >= lo) & (fc < hi)].mean())
        vals += [np.array(plv), np.array(coh)]
        names += [[f"{tag}_plv_{b}_{a_}-{b_}" for a_, b_ in PAIRS_SYNC], [f"{tag}_coh_{b}_{a_}-{b_}" for a_, b_ in PAIRS_SYNC]]
    return np.concatenate([np.ravel(v) for v in vals]), [n for g in names for n in g]


def one(i):
    import pycatch22
    import tsfel
    X = np.load(EXP / "X.npy", mmap_mode="r")
    x0 = np.nan_to_num(np.asarray(X[i], np.float64))
    if DEMEAN:                                          # T2-D515d: the members' own front end (ni26.stream._demean) before every block
        x0 = x0 - x0.mean(-1, keepdims=True)
    out, names, fams = [], [], []
    for tag, x in (("raw", x0), ("car", x0 - x0.mean(0, keepdims=True))):
        cfg = {d: v for d, v in tsfel.get_features_by_domain().items() if d in ("statistical", "temporal", "spectral")}
        df = tsfel.time_series_features_extractor(cfg, x.T, fs=FS, verbose=0)
        out.append(df.to_numpy(np.float64).ravel())
        names += [f"{tag}_tsfel_{c}" for c in df.columns]
        fams += [f"tsfel_{tag}"] * df.shape[1]
        c22 = [pycatch22.catch22_all(list(x[c]))["values"] for c in range(len(CH))]
        out.append(np.asarray(c22, np.float64).ravel())
        names += [f"{tag}_c22_{j}_{c}" for c in CH for j in range(22)]
        fams += [f"catch22_{tag}"] * (22 * len(CH))
        v, n = eeg_block(x, tag)
        out.append(v)
        names += n
        fams += [f"{tag}_" + k.split("_")[1] if k.split("_")[1] in ("asym", "plv", "coh") else f"{tag}_spec" for k in n]
    return np.concatenate(out).astype(np.float32), names, fams


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    N = int(META.get("n_windows") or np.load(EXP / "X.npy", mmap_mode="r").shape[0])
    v0, names, fams = one(0)
    print(f"[d515] {len(names)} features per window", flush=True)
    F = np.empty((N, len(names)), np.float32)
    F[0] = v0
    with Pool(8) as pool:
        for i, (v, _, _) in enumerate(pool.imap(one, range(1, N), chunksize=8), start=1):
            F[i] = v
            if i % 300 == 0:
                print(f"[d515] {i} / {N}", flush=True)
    F[~np.isfinite(F)] = np.nan
    np.savez_compressed(OUT / ("features_dm.npz" if DEMEAN else "features.npz"), F=F, names=np.array(names), family=np.array(fams))
    fam, cnt = np.unique(fams, return_counts=True)
    print("[d515] families: " + ", ".join(f"{a} {b}" for a, b in zip(fam, cnt)), flush=True)


if __name__ == "__main__":
    main()
