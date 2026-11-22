#!/usr/bin/env python
"""T2-D539b refit COST (the manager's condition 1, before any template code): wall time of the in-stream side-head re-fit = t2_d539_read.py's
fit_head EXACTLY (scipy L-BFGS-B, float64, gtol 1e-8, maxiter 500, from the trained head, lam 1), per fit and per stream, on this CPU,
single-threaded (OMP 1, as the worker's predict path) and with the default threads. Inputs: the REAL dumped side nets of draw 819 (logits,
side z, trained heads) and the real 819 teacher's confR labels, so the iteration counts are the read's own; plus the export's stream lengths.
Per stream: 3 re-fits (60 / 100 / 150) x the side heads present (5 or 10 per family x 2 families). Descriptive.
    mem-guard env OMP_NUM_THREADS=1 python scripts/local/t2_d539b_refit_cost.py
"""
import glob
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
idx = pd.read_csv(EXP / "index.csv")
STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()
CKS = (60, 100, 150)


def fit_head(Ls, Ss, ys, W0, b0, lam):          # = scripts/local/t2_d539_read.py fit_head, verbatim
    K, D = W0.shape
    n = len(ys)
    th0 = np.concatenate([W0.ravel(), b0])
    Y1 = np.eye(K)[ys]

    def fg(th):
        W, b = th[:K * D].reshape(K, D), th[K * D:]
        z = Ls + Ss @ W.T + b
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        G = (np.exp(z - lse[:, None]) - Y1) / n
        dth = th - th0
        f = float((lse - z[np.arange(n), ys]).mean() + 0.5 * lam * (dth @ dth))
        return f, np.concatenate([(G.T @ Ss).ravel(), G.sum(0)]) + lam * dth

    r = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(gtol=1e-8, maxiter=500))
    return r.x[:K * D].reshape(K, D), r.x[K * D:], bool(r.success), int(r.nit)


def conf_half(pp):
    ya, cf = pp.argmax(1), pp.max(1)
    keep = np.zeros(len(ya), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return ya, keep


lens = pd.Series(STREAM).value_counts()
print(f"export streams: {len(lens)}; windows per stream: median {lens.median():.0f}, mean {lens.mean():.1f}, max {lens.max()}; "
      f"> 60: {(lens > 60).sum()}, > 100: {(lens > 100).sum()}, > 150: {(lens > 150).sum()}", flush=True)
times, nits = {c: [] for c in CKS}, {c: [] for c in CKS}
files = sorted(glob.glob(str(O / "t2_d540/side_sh/runs/lc9_*__s819.npz")) + glob.glob(str(O / "t2_d540/side_ifn/runs/lc9_*__s819.npz")))
assert files, "no 819 side dumps"
for path in files:
    z = np.load(path)
    te = z["test_idx"].astype(np.int64)
    L, Wh, bh = (z[k].astype(np.float64) for k in ("side_logit_last", "side_w", "side_b"))
    S0 = z["side_te"].astype(np.float64)
    f = int(re.search(r"lc9_(\d+)__", path).group(1))
    tz = np.load(O / f"t2_d540/teacher_p1/lc9_{f}__s819.npz")
    pos = {int(r): k for k, r in enumerate(tz["rows"].astype(np.int64))}
    post = {c: tz[f"post_r{c}"].astype(np.float64) for c in CKS}
    st_te, od_te = STREAM[te], ORD[te]
    for s in np.unique(st_te):
        j = np.flatnonzero(st_te == s)
        j = j[np.argsort(od_te[j], kind="stable")]
        for c in CKS:
            if len(j) <= c:
                break
            a = j[:c]
            pp = np.stack([post[c][pos[int(r)]] for r in te[a]])
            fin = np.isfinite(pp).all(1)
            ya, keep = conf_half(pp[fin])
            sel, ys = a[fin][keep], ya[keep].astype(np.int64)
            t0 = time.perf_counter()
            _, _, ok, nit = fit_head(L[sel], S0[sel], ys, Wh, bh, 1.0)
            times[c].append(time.perf_counter() - t0)
            nits[c].append(nit)
for c in CKS:
    t = np.asarray(times[c]) * 1e3
    print(f"re-fit at {c}: {len(t)} fits; ms per fit median {np.median(t):.2f}, mean {t.mean():.2f}, p95 {np.percentile(t, 95):.2f}, max {t.max():.2f}; "
          f"iterations median {np.median(nits[c]):.0f}, max {max(nits[c])}", flush=True)
per_head_stream = sum(np.mean(times[c]) for c in CKS)          # a stream long enough for all three re-fits
print(f"per side head and long stream (3 re-fits): {per_head_stream * 1e3:.1f} ms; x 10 heads (S5, 2 families) {10 * per_head_stream:.2f} s; "
      f"x 20 heads (S10) {20 * per_head_stream:.2f} s", flush=True)
nfit = {c: int((lens > c).sum()) for c in CKS}
tot5 = 10 * sum(nfit[c] * np.mean(times[c]) for c in CKS)
tot10 = 20 * sum(nfit[c] * np.mean(times[c]) for c in CKS)
print(f"the export's {len(lens)} streams ({len(idx)} windows): total re-fit {tot5:.1f} s (S5) / {tot10:.1f} s (S10) = "
      f"{1e3 * tot5 / len(idx):.2f} / {1e3 * tot10 / len(idx):.2f} ms per window", flush=True)
