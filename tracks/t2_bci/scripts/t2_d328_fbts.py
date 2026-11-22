#!/usr/bin/env python
"""T2-D328: a classical member, filter-bank tangent space + L2 logistic regression with causal per-stream re-centering (FBTS),
as pre-registered (zero GPU). Per band: the window covariance (shrink 0.05), re-centred by the running mean covariance of the
stream's windows 1..t (the template's EA recursion: _inv_sqrt(mean, 0.05)), log map at identity, upper triangle with the
off-diagonal x sqrt(2). C in {0.01, 0.1, 1} by leave-one-user-out on the TRAINING users only. Bench-format result files
(test_idx, prob_last, val_idx, prob_val_last), one deterministic draw written at s0..s11.
  python t2_d328_fbts.py --export E --out DIR
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
BANDS = ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30), (30, 40))
SHRINK = 0.05
CS = (0.01, 0.1, 1.0)


def inv_sqrt(R, shrink=SHRINK):
    C = R.shape[-1]
    R = (1 - shrink) * R + shrink * np.trace(R) / C * np.eye(C)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, w.max() * 1e-8)
    return (V * w ** -0.5) @ V.T


def logm_spd(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.log(np.maximum(w, 1e-12))) @ V.T


def features(X, idx, fs):
    n, C, _ = X.shape
    iu = np.triu_indices(C)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    F = np.empty((n, len(BANDS), len(iu[0])), np.float32)
    order = idx["order"].to_numpy()
    for b, (lo, hi) in enumerate(BANDS):
        sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        Xb = sosfiltfilt(sos, X.astype(np.float64), axis=-1)            # inside each 4 s window (deployable per window)
        Xb = Xb - Xb.mean(-1, keepdims=True)
        covs = np.matmul(Xb, Xb.transpose(0, 2, 1)) / Xb.shape[-1]
        tr = np.trace(covs, axis1=1, axis2=2) / C
        covs = (1 - SHRINK) * covs + SHRINK * tr[:, None, None] * np.eye(C)
        for _, g in idx.groupby("stream", sort=False):
            rows = g.index.to_numpy()[np.argsort(order[g.index.to_numpy()], kind="stable")]
            run = np.zeros((C, C))
            for k, r in enumerate(rows, 1):
                run += covs[r]
                W = inv_sqrt(run / k)
                F[r, b] = (logm_spd(W @ covs[r] @ W)[iu] * wts).astype(np.float32)
        print(f"[d328] band {lo}-{hi} Hz done", flush=True)
    return F.reshape(n, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import t2p_bench as B
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, idx, meta = B.load_export(a.export)
    idx = idx.reset_index(drop=True)
    y = idx["y"].to_numpy().astype(int)
    F = features(np.asarray(X), idx, float(meta["sfreq"]))
    os.makedirs(a.out, exist_ok=True)
    subj = idx["subject"].str[-3:].to_numpy()
    rep = {}
    for proto in ("kit", "seen1"):
        (fname, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        users = sorted(set(subj[tr]))
        score = {}
        for c in CS:                                                     # LOUO on the training users only
            accs = []
            for u in users:
                fit, hold = tr[subj[tr] != u], tr[subj[tr] == u]
                if len(set(y[fit])) < 3 or len(hold) == 0:
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(F[fit], y[fit])
                p = m.predict(F[hold])
                accs.append(np.mean([np.mean(p[y[hold] == k] == k) for k in np.unique(y[hold])]))
            score[c] = float(np.mean(accs))
        cbest = max(score, key=score.get)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=cbest, max_iter=4000)).fit(F[tr], y[tr])
        pte, pva = m.predict_proba(F[te]).astype(np.float32), m.predict_proba(F[va]).astype(np.float32)
        res = dict(protocol=proto, fold=fname, model="fbts", C=cbest, louo=score, bands=BANDS, shrink=SHRINK)
        for s in range(12):
            np.savez_compressed(f"{a.out}/{proto}__fbts__s{s}.npz", test_idx=te, prob_last=pte, val_idx=va, prob_val_last=pva,
                                prob_best=pte, res=json.dumps(res))
        rep[proto] = dict(C=cbest, louo=score)
        print(f"[d328] {proto}: C {cbest} (LOUO {score})", flush=True)
    json.dump(rep, open(f"{a.out}/fit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
