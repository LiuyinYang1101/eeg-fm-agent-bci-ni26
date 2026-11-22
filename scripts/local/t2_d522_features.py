#!/usr/bin/env python
"""T2-D522 (registered 2026-11-15 before any D522 number; CPU): the window-mean tree family with the TIME COURSE of the mean as inputs: per channel the
8 half-second means of the delivered window (0.5 s = 60 samples at 120 Hz) and of its CAR view = 2 x 41 x 8 = 656 features (their average over the 8 bins
is D516's 82 window means, so the tree can rebuild those). Same format as outputs/t2_d515/features_dc.npz (F, names, family); read by
scripts/local/t2_d515_read.py --features features_dc8.npz --seeds 1,2,3 (-> pred_s1-2-3_dc8.npz; the in-fold top-300 selection applies).
    mem-guard python scripts/local/t2_d522_features.py
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
import t2p_bench as B  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
X, idx, meta = B.load_export(str(EXP))
CH = list(meta["ch_names"])
Xd = np.asarray(X, dtype=np.float64)
N, C, T = Xd.shape
assert T == 480 and C == len(CH), (T, C)
raw = Xd.reshape(N, C, 8, T // 8).mean(-1)                       # (N, C, 8): the 0.5-s bin means
car = raw - raw.mean(1, keepdims=True)                           # the CAR view's bin means (CAR commutes with the time mean)
F = np.concatenate([raw.reshape(N, C * 8), car.reshape(N, C * 8)], 1).astype(np.float32)
names = np.array([f"dc8_{v}_{c}_b{b}" for v in ("raw", "car") for c in CH for b in range(8)])
family = np.array(["dc8_raw"] * (C * 8) + ["dc8_car"] * (C * 8))
assert F.shape[1] == len(names) == 656
# the bins' average must equal D516's window means exactly (float32 tolerance)
dc = np.load(O / "t2_d515/features_dc.npz")["F"]
assert np.allclose(F[:, :C * 8].reshape(N, C, 8).mean(-1), dc[:, :C], atol=1e-3), "the bin means do not average to D516's window means"
np.savez_compressed(O / "t2_d515/features_dc8.npz", F=F, names=names, family=family)
print(f"[d522] features_dc8.npz: {F.shape}, finite {bool(np.isfinite(F).all())}")
