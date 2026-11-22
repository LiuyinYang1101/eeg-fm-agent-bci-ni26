#!/usr/bin/env python
"""T2-D525 (registered 2026-11-16 before any number; CPU): D487's imaginary-coherency member on the K = 9 panel. D487's features
(outputs/t2_d487/features.npz, imcoh 2,655 x 459), D487's form exactly (causal per-stream re-centring, z-score on the fold's training
windows, L2 logistic C 0.05), one fit per lc9 fold on that fold's training rows -> probabilities for EVERY window
(outputs/t2_d525/pred_imcoh.npz: te_<f> = all rows, bag_<f> = (N, 3); the reader realigns by row, training rows are never scored).
    mem-guard python scripts/local/t2_d525_imcoh.py
"""
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
from t2_d487_wpli import C_LR, recentre  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
X, idx, meta = B.load_export(str(EXP))
del X
y = idx["y"].to_numpy().astype(int)
Fe = np.load(O / "t2_d487/features.npz")["imcoh"].astype(np.float64)
assert Fe.shape == (len(idx), 459), Fe.shape
Fr = recentre(Fe, idx)
N = len(idx)
out = {}
for f in range(10):
    ((name, trm, _, tem),) = B.folds(idx, f"lc9_{f}")
    tr = np.flatnonzero(trm)
    mu, sd = Fr[tr].mean(0), Fr[tr].std(0) + 1e-9
    Z = (Fr - mu) / sd
    clf = LogisticRegression(C=C_LR, max_iter=3000).fit(Z[tr], y[tr])
    assert list(clf.classes_) == [0, 1, 2]
    p = clf.predict_proba(Z).astype(np.float32)
    te = np.flatnonzero(tem)
    out[f"te_{f}"], out[f"bag_{f}"] = np.arange(N), p
    print(f"[d525] {name}: {len(tr)} train rows, {len(te)} test rows; test window acc {float((p[te].argmax(1) == y[te]).mean()):.3f} (descriptive)", flush=True)
(O / "t2_d525").mkdir(exist_ok=True)
np.savez_compressed(O / "t2_d525/pred_imcoh.npz", **out)
print("[d525] wrote outputs/t2_d525/pred_imcoh.npz")
