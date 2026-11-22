#!/usr/bin/env python
"""T2-D487 (registered 2026-11-14 11:24 before any number): LAGGED phase coupling as a member. Per window of the contract export:
  18 nodes (T2-D473's maps: Fz FCz | Cz C3 C4 CP3 CP4 | F5 FC5 F6 FC6 TP7 TP8 | F3 F4 Pz P3 P4) -> 153 pairs x 3 bands (theta 4-8, mu 8-13,
  beta 13-30 Hz); from 1-s Hann segments with 50 % overlap inside the window (7 segments, 1-Hz bins) the cross-spectra S_xy, averaged over
  segments and the band's bins:  imcoh = Im(S_xy) / sqrt(S_xx S_yy)  (signed; the member)   wpli = |sum Im| / sum |Im| (unsigned; descriptive twin).
  Per stream (kit order) a causal re-centring: minus the mean of the stream's windows 1 .. t-1 (none at t = 1); z-scored with each protocol's
  training windows; L2 logistic, C 0.05 fixed, fitted on the protocol's training windows; probabilities for its val and test windows, written in
  the bench's npz form (test_idx / prob_last / val_idx / prob_val_last), so the reader treats it as one more (seed-free) member.
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mem-guard python scripts/local/t2_d487_wpli.py
  -> outputs/t2_d487/{kit,seen1}__{imcoh,wpli}.npz, features.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OUT = Path.home() / "workspace/outputs/neural_interface_challenge/t2_d487"
OUT.mkdir(parents=True, exist_ok=True)
NODES = ["Fz", "FCz", "Cz", "C3", "C4", "CP3", "CP4", "F5", "FC5", "F6", "FC6", "TP7", "TP8", "F3", "F4", "Pz", "P3", "P4"]
BANDS = {"theta": (4, 8), "mu": (8, 13), "beta": (13, 30)}
C_LR = 0.05


def features(X, ch, fs=120.0, seg=120, hop=60):
    """(N, C, T) -> imcoh, wpli: (N, n_pairs * n_bands) each; pairs i < j over NODES, band-major."""
    ci = [ch.index(n) for n in NODES]
    x = np.asarray(X[:, ci], np.float64)
    N, C, T = x.shape
    starts = range(0, T - seg + 1, hop)
    w = np.hanning(seg)
    F = np.stack([np.fft.rfft((x[:, :, s:s + seg] - x[:, :, s:s + seg].mean(-1, keepdims=True)) * w, axis=-1) for s in starts], 1)  # (N, S, C, f)
    freqs = np.fft.rfftfreq(seg, 1 / fs)
    iu = np.triu_indices(C, 1)
    im_all, wp_all = [], []
    for lo, hi in BANDS.values():
        b = (freqs >= lo) & (freqs < hi)
        Fb = F[..., b]                                                     # (N, S, C, nb)
        S = np.einsum("nsif,nsjf->nsfij", Fb, Fb.conj())                   # (N, S, nb, C, C)
        Sm = S.mean((1, 2))                                                # (N, C, C)
        p = np.real(np.diagonal(Sm, axis1=1, axis2=2))
        imc = np.imag(Sm) / np.sqrt(p[:, :, None] * p[:, None, :] + 1e-30)
        ims = np.imag(S)
        wpli = np.abs(ims.sum((1, 2))) / (np.abs(ims).sum((1, 2)) + 1e-30)
        im_all.append(imc[:, iu[0], iu[1]])
        wp_all.append(wpli[:, iu[0], iu[1]])
    return np.concatenate(im_all, 1).astype(np.float32), np.concatenate(wp_all, 1).astype(np.float32)


def recentre(Fe, idx):
    """per stream in kit order: minus the mean of the stream's windows 1 .. t-1 (none at t = 1)."""
    out = Fe.copy()
    for _, g in idx.groupby("stream", sort=False):
        rows = g.sort_values("order")["i"].to_numpy()
        cs = np.cumsum(Fe[rows], 0)
        for k in range(1, len(rows)):
            out[rows[k]] = Fe[rows[k]] - cs[k - 1] / k
    return out


def main():
    X, idx, meta = B.load_export(EXP)
    ch = list(meta["ch_names"])
    assert all(n in ch for n in NODES), [n for n in NODES if n not in ch]
    imc, wpl = features(X, ch, float(meta["sfreq"]))
    print(f"features: {imc.shape[1]} per kind on {len(imc)} windows ({len(NODES)} nodes, {len(BANDS)} bands)", flush=True)
    np.savez_compressed(OUT / "features.npz", imcoh=imc, wpli=wpl, nodes=np.array(NODES), bands=np.array(list(BANDS)))
    from sklearn.linear_model import LogisticRegression
    y = idx["y"].to_numpy().astype(int)
    summary = {}
    for kind, Fe in (("imcoh", imc), ("wpli", wpl)):
        Fr = recentre(Fe.astype(np.float64), idx)
        for proto in ("kit", "seen1"):
            ((fname, trm, vam, tem),) = B.folds(idx, proto)
            tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
            mu, sd = Fr[tr].mean(0), Fr[tr].std(0) + 1e-9
            Z = (Fr - mu) / sd
            clf = LogisticRegression(C=C_LR, max_iter=3000).fit(Z[tr], y[tr])
            assert list(clf.classes_) == [0, 1, 2]
            pte, pva = clf.predict_proba(Z[te]), clf.predict_proba(Z[va])
            np.savez_compressed(OUT / f"{proto}__{kind}.npz", test_idx=te, prob_last=pte.astype(np.float32), val_idx=va,
                                prob_val_last=pva.astype(np.float32), prob_best=pte.astype(np.float32),
                                res=json.dumps(dict(kind=kind, C=C_LR, nodes=NODES, bands=BANDS, recentre="causal per stream, windows 1..t-1")))
            acc_te = float((pte.argmax(1) == y[te]).mean())
            summary[f"{proto}_{kind}"] = dict(train=len(tr), val=len(va), test=len(te), test_window_acc=round(acc_te, 4))
            print(f"{proto} {kind}: {len(tr)} train / {len(va)} val / {len(te)} test windows; test window accuracy {acc_te:.3f} (descriptive)", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
