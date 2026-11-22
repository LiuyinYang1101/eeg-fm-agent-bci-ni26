#!/usr/bin/env python
"""T2-D515b runtime price (CPU): the features a FINAL GBM (all public windows, causal per-stream z with the all-window prior) selects (top 300 by gain,
union over seeds 1-3), and the per-window cost of computing only those: TSFEL restricted to the selected functions on the selected channels per
reference (an upper bound: TSFEL's own extractor per function), catch22 on the selected channels, the EEG block whole (cheap). Also: how many selected
features need statsmodels (TSFEL's autocorr only; not in the worker image).
    mem-guard python scripts/local/t2_d515_price.py
"""
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import t2p_bench as B  # noqa: E402
import t2_d515_features as FX  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
_, idx, _ = B.load_export(str(EXP))
STREAM, ORD, Y = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy().astype(int)
z = np.load(O / "t2_d515/features.npz")
F, NAMES, FAM = z["F"].astype(np.float32), z["names"], z["family"]
PRM = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.3, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=8)
tr = np.arange(len(F))
mu0, sd0 = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-6
Z = np.empty_like(F)
for st in np.unique(STREAM):
    r = np.flatnonzero(STREAM == st)
    r = r[np.argsort(ORD[r], kind="stable")]
    x = np.where(np.isfinite(F[r]), F[r], mu0)
    n = 10.0 + np.arange(1, len(r) + 1)[:, None]
    m = (10.0 * mu0 + np.cumsum(x, 0)) / n
    v = (10.0 * (sd0 ** 2 + mu0 ** 2) + np.cumsum(x * x, 0)) / n - m ** 2
    Z[r] = (x - m) / np.sqrt(np.maximum(v, 1e-12))
sel = set()
for sd in (1, 2, 3):
    m1 = lgb.LGBMClassifier(n_estimators=150, seed=sd, **PRM).fit(Z, Y)
    sel |= set(np.argsort(-m1.booster_.feature_importance("gain"))[:300].tolist())
sel = sorted(sel)
names = [str(NAMES[j]) for j in sel]
fams = [str(FAM[j]) for j in sel]
print(f"[price] union of the three seeds' top 300: {len(sel)} features; by family: " + ", ".join(f"{f} {fams.count(f)}" for f in sorted(set(fams))), flush=True)
auto = [n for n in names if "Autocorrelation" in n]
print(f"[price] selected features that need statsmodels (TSFEL autocorr): {len(auto)}", flush=True)
# which TSFEL functions, on which channels, per reference
import tsfel  # noqa: E402
cfg_all = {d: v for d, v in tsfel.get_features_by_domain().items() if d in ("statistical", "temporal", "spectral")}
need = {}
for n in names:
    if "_tsfel_" in n:
        ref, rest = n.split("_tsfel_", 1)
        ch, feat = rest.split("_", 1)
        need.setdefault(ref, {}).setdefault(feat.split("_")[0], set()).add(ch)
print("[price] TSFEL functions needed: " + "; ".join(f"{r}: {sorted(v)}" for r, v in need.items())[:900], flush=True)
X = np.load(EXP / "X.npy", mmap_mode="r")
CH = FX.CH
ts = []
for i in range(0, 60):
    x0 = np.nan_to_num(np.asarray(X[i * 40], np.float64))
    t0 = time.time()
    for ref, fns in need.items():
        x = x0 if ref == "raw" else x0 - x0.mean(0, keepdims=True)
        for fn, chs in fns.items():
            sub = {d: {k: v for k, v in feats.items() if k == fn} for d, feats in cfg_all.items()}
            sub = {d: f for d, f in sub.items() if f}
            if not sub:
                continue
            cols = sorted(int(c) for c in chs)          # TSFEL names its columns by the input column index
            tsfel.time_series_features_extractor(sub, x[cols].T, fs=120.0, verbose=0)
    t_ts = time.time() - t0
    t0 = time.time()
    import pycatch22  # noqa: E402
    for ref in ("raw", "car"):
        x = x0 if ref == "raw" else x0 - x0.mean(0, keepdims=True)
        chs = {n.split("_")[-1] for n in names if n.startswith(f"{ref}_c22_")}
        for c in chs:
            pycatch22.catch22_all(list(x[CH.index(c)]))
    t_c22 = time.time() - t0
    t0 = time.time()
    FX.eeg_block(x0, "raw")
    FX.eeg_block(x0 - x0.mean(0, keepdims=True), "car")
    t_eeg = time.time() - t0
    ts.append((t_ts, t_c22, t_eeg))
t = np.median(np.array(ts), 0) * 1e3
print(f"[price] per window (median of 60, this CPU, one thread): TSFEL-restricted {t[0]:.1f} ms + catch22 {t[1]:.1f} ms + EEG block {t[2]:.1f} ms = {t.sum():.1f} ms", flush=True)
np.save(O / "t2_d515/selected_final.npy", np.array(sel))
