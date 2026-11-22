#!/usr/bin/env python
"""T2-D467 (written before any number): a BLOCK of classical (non-neural) members, every one with causal per-stream re-centring.
Members (bench-format files, one deterministic draw written at s0..s11, per protocol kit / seen1, trained on that protocol's training
windows only):
  fbcsp  : per band (D328's 7 bands), multi-class CSP (pyriemann, 6 filters, log-variance) on re-centred band covariances -> shrinkage LDA
  mdm    : Riemannian minimum distance to the class means on re-centred broadband (4-40 Hz) covariances (no hyperparameter)
  xdts   : xDAWN covariances (4 filters per class, Ledoit-Wolf) on causally EA-aligned broadband windows -> tangent space -> L2 LR
  fbts   : T2-D328's filter-bank tangent space + LR (re-used from outputs/t2_d328/runs, not refit)
Hyperparameters fixed a priori (nfilter 6 / 4, LDA shrinkage 'auto') except the LR C of xdts, chosen in {0.01, 0.1, 1} by
leave-one-user-out on the protocol's TRAINING users (as D328). Re-centring: W_t = (mean of the stream's covariances 1..t)^(-1/2), shrink 0.05.
    mem-guard python tracks/t2_bci/scripts/t2_d467_classical.py --export E --out DIR
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_d328_fbts import BANDS, SHRINK, inv_sqrt  # noqa: E402

CS = (0.01, 0.1, 1.0)


def band_covs(X, lo, hi, fs):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    Xb = sosfiltfilt(sos, X.astype(np.float64), axis=-1)
    Xb = Xb - Xb.mean(-1, keepdims=True)
    C = X.shape[1]
    covs = np.matmul(Xb, Xb.transpose(0, 2, 1)) / Xb.shape[-1]
    tr = np.trace(covs, axis1=1, axis2=2) / C
    return Xb, (1 - SHRINK) * covs + SHRINK * tr[:, None, None] * np.eye(C)


def recentre(covs, idx, Xb=None):
    """causal per stream: W_t from the running mean of covs 1..t; returns W C W (and W X if Xb is given)."""
    order = idx["order"].to_numpy()
    out = np.empty_like(covs)
    xo = None if Xb is None else np.empty_like(Xb)
    for _, g in idx.groupby("stream", sort=False):
        rows = g.index.to_numpy()[np.argsort(order[g.index.to_numpy()], kind="stable")]
        run = np.zeros(covs.shape[1:])
        for k, r in enumerate(rows, 1):
            run += covs[r]
            W = inv_sqrt(run / k)
            out[r] = W @ covs[r] @ W
            if xo is not None:
                xo[r] = W @ Xb[r]
    return out, xo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import t2p_bench as B
    from pyriemann.classification import MDM
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.spatialfilters import CSP
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, idx, meta = B.load_export(a.export)
    X = np.asarray(X)
    idx = idx.reset_index(drop=True)
    fs = float(meta["sfreq"])
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].str[-3:].to_numpy()
    bandc = []
    for lo, hi in BANDS:
        _, c = band_covs(X, lo, hi, fs)
        bandc.append(recentre(c, idx)[0])
        print(f"[d467] band {lo}-{hi} re-centred", flush=True)
    Xbb, cbb = band_covs(X, 4, 40, fs)
    cbb_r, Xbb_r = recentre(cbb, idx, Xbb)
    print("[d467] broadband re-centred", flush=True)
    os.makedirs(a.out, exist_ok=True)

    def fit_fbcsp(tr):
        fs_ = [CSP(nfilter=6, log=True).fit(bc[tr], y[tr]) for bc in bandc]
        F = lambda r: np.hstack([f.transform(bc[r]) for f, bc in zip(fs_, bandc)])  # noqa: E731
        lda = LDA(solver="lsqr", shrinkage="auto").fit(F(tr), y[tr])
        return lambda r: lda.predict_proba(F(r))

    def fit_mdm(tr):
        m = MDM().fit(cbb_r[tr], y[tr])
        return lambda r: m.predict_proba(cbb_r[r])

    def fit_xdts(tr, c):
        xd = XdawnCovariances(nfilter=4, estimator="lwf").fit(Xbb_r[tr], y[tr])
        ts = TangentSpace(metric="riemann").fit(xd.transform(Xbb_r[tr]))
        F = lambda r: ts.transform(xd.transform(Xbb_r[r]))  # noqa: E731
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=4000)).fit(F(tr), y[tr])
        return lambda r: lr.predict_proba(F(r))

    rep = {}
    for proto in ("kit", "seen1"):
        (fname, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        users = sorted(set(subj[tr]))
        score = {}
        for c in CS:
            accs = []
            for u in users:
                fit, hold = tr[subj[tr] != u], tr[subj[tr] == u]
                p = fit_xdts(fit, c)(hold).argmax(1)
                accs.append(np.mean([np.mean(p[y[hold] == k] == k) for k in np.unique(y[hold])]))
            score[c] = float(np.mean(accs))
        cbest = max(score, key=score.get)
        models = {"fbcsp": fit_fbcsp(tr), "mdm": fit_mdm(tr), "xdts": fit_xdts(tr, cbest)}
        rep[proto] = dict(xdts_C=cbest, xdts_louo=score)
        for name, f in models.items():
            pte, pva = f(te).astype(np.float32), f(va).astype(np.float32)
            res = dict(protocol=proto, fold=fname, model=name, bands=BANDS if name == "fbcsp" else [4, 40], shrink=SHRINK,
                       C=cbest if name == "xdts" else None)
            for s in range(12):
                np.savez_compressed(f"{a.out}/{proto}__{name}__s{s}.npz", test_idx=te, prob_last=pte, val_idx=va, prob_val_last=pva,
                                    prob_best=pte, res=json.dumps(res))
        print(f"[d467] {proto}: written; xdts C {cbest} (LOUO {score})", flush=True)
    json.dump(rep, open(f"{a.out}/fit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
