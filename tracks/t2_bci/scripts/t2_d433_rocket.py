#!/usr/bin/env python
"""T2-D433 (pre-registered in docs/DECISIONS_T2.md before any D433 number): a TIME-SERIES-CLASSIFIER member, MiniRocket-style multivariate random
convolutions + L2 logistic regression, on causally whitened windows (the user's seed 2026-11-10: time-series models "from a different angle").
  whitening  per stream, window t is multiplied by inv_sqrt of the running mean covariance of windows 1..t (T2-D328's recursion; causal, label-free);
  kernels    K = 1000 per draw: one of the 84 MiniRocket length-9 kernels (three taps 2, six taps -1), dilation 2^U(0, log2((T-1)/8)), applied to the
             SUM of a random channel subset (size 2^U(0, log2 10), 1..9 channels), 'same' zero padding;
  features   per kernel: PPV = mean(out > bias) with the bias a U(0.1, 0.9) quantile of the kernel's output on one random TRAINING window of the
             protocol, and log variance of the output (2 K features);
  classifier StandardScaler + LogisticRegression, C in {1e-4, 1e-3, 1e-2} by leave-one-user-out on the TRAINING users (draw 0, reused for 1..11).
Bench-format result files (test_idx, prob_last, val_idx, prob_val_last), 12 draws = 12 kernel seeds. CPU only.
  mem-guard python t2_d433_rocket.py --export E --out DIR
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
K = 1000
CS = (1e-4, 1e-3, 1e-2)
KERN = np.full((84, 9), -1.0, np.float32)
for _i, _c in enumerate(itertools.combinations(range(9), 3)):
    KERN[_i, list(_c)] = 2.0


def whiten(X, idx):
    from t2_d328_fbts import inv_sqrt
    X = np.asarray(X, np.float64)
    X = X - X.mean(-1, keepdims=True)
    covs = np.matmul(X, X.transpose(0, 2, 1)) / X.shape[-1]
    out = np.empty(X.shape, np.float32)
    order = idx["order"].to_numpy()
    for _, g in idx.groupby("stream", sort=False):
        rows = g.index.to_numpy()[np.argsort(order[g.index.to_numpy()], kind="stable")]
        run = np.zeros(covs.shape[1:])
        for k, r in enumerate(rows, 1):
            run += covs[r]
            out[r] = inv_sqrt(run / k) @ X[r]
    return out


def draw(seed, C, T):
    g = np.random.default_rng(seed)
    dil = np.floor(2 ** g.uniform(0, np.log2((T - 1) / 8), K)).astype(int)
    nch = np.clip(np.floor(2 ** g.uniform(0, np.log2(10), K)).astype(int), 1, 9)
    A = np.zeros((K, C), np.float32)
    for k in range(K):
        A[k, g.choice(C, nch[k], replace=False)] = 1.0
    return dict(w=KERN[g.integers(0, 84, K)], dil=dil, A=A, q=g.uniform(0.1, 0.9, K), g=g)


def conv_out(Xt, P, d):
    m = np.flatnonzero(P["dil"] == d)
    Z = torch.einsum("kc,nct->nkt", torch.from_numpy(P["A"][m]), Xt)
    return m, Fn.conv1d(Z, torch.from_numpy(P["w"][m])[:, None, :], groups=len(m), dilation=int(d), padding=4 * int(d))


@torch.no_grad()
def features(Xw, P, train_rows):
    n = len(Xw)
    ref = P["g"].choice(train_rows, 64, replace=False)
    pick = P["g"].integers(0, 64, K)
    Xr = torch.from_numpy(Xw[ref])
    bias = np.empty(K, np.float32)
    for d in np.unique(P["dil"]):
        m, o = conv_out(Xr, P, d)
        for j, k in enumerate(m):
            bias[k] = np.quantile(o[pick[k], j].numpy(), P["q"][k])
    F = np.empty((n, 2 * K), np.float32)
    for s in range(0, n, 256):
        Xt = torch.from_numpy(Xw[s:s + 256])
        for d in np.unique(P["dil"]):
            m, o = conv_out(Xt, P, d)
            F[s:s + 256, m] = (o > torch.from_numpy(bias[m])[None, :, None]).float().mean(-1).numpy()
            F[s:s + 256, K + m] = torch.log(o.var(-1) + 1e-6).numpy()
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=12)
    a = ap.parse_args()
    import t2p_bench as B
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    X, idx, meta = B.load_export(a.export)
    idx = idx.reset_index(drop=True)
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].str[-3:].to_numpy()
    Xw = whiten(X, idx)
    print(f"[d433] whitened {Xw.shape}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rep = {}
    for proto in ("kit", "seen1"):
        (fname, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        cbest, score = None, {}
        for s in range(a.draws):
            F = features(Xw, draw(1000 * (proto == "seen1") + s, Xw.shape[1], Xw.shape[2]), tr)
            if cbest is None:
                for c in CS:
                    accs = []
                    for u in sorted(set(subj[tr])):
                        fit, hold = tr[subj[tr] != u], tr[subj[tr] == u]
                        if len(set(y[fit])) < 3 or len(hold) == 0:
                            continue
                        p = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(F[fit], y[fit]).predict(F[hold])
                        accs.append(np.mean([np.mean(p[y[hold] == k] == k) for k in np.unique(y[hold])]))
                    score[c] = float(np.mean(accs))
                cbest = max(score, key=score.get)
                print(f"[d433] {proto}: C {cbest} (LOUO {score})", flush=True)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=cbest, max_iter=4000)).fit(F[tr], y[tr])
            pte, pva = m.predict_proba(F[te]).astype(np.float32), m.predict_proba(F[va]).astype(np.float32)
            res = dict(protocol=proto, fold=fname, model="rocket", C=cbest, louo=score, K=K, seed=s)
            np.savez_compressed(f"{a.out}/{proto}__rocket__s{s}.npz", test_idx=te, prob_last=pte, val_idx=va, prob_val_last=pva, prob_best=pte,
                                res=json.dumps(res))
            print(f"[d433] {proto} s{s}: window acc test {np.mean(pte.argmax(1) == y[te]):.3f} val {np.mean(pva.argmax(1) == y[va]):.3f}", flush=True)
        rep[proto] = dict(C=cbest, louo=score)
    json.dump(rep, open(f"{a.out}/fit.json", "w"), indent=1)
    print("[d433] done", flush=True)


if __name__ == "__main__":
    main()
