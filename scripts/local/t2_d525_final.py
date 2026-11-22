#!/usr/bin/env python
"""T2-D525 / W18 build input (zero GPU; PREPARATION, used only if the imcoh switch keeps in the D532 fresh-draw read and the owner types the W18
line): the imcoh family's FINAL model for a package. D487's 459 imaginary-coherency features of EVERY public window
(outputs/t2_d487/features.npz), causally re-centred per stream (D487's recentre = the template's ICFamily state), z-scored on every labelled
public window, one L2 logistic regression (C 0.05) on all of them; plus the family's start vector (T2-D459's rule, as W16's dcfam: the mean
posterior of the K = 9 folds' strangers, from D525's fold fits outputs/t2_d525/pred_imcoh.npz).
Output: outputs/t2_d525_final/imcoh.npz (mu, sd, coef, intercept, start) + imcoh_meta.json.
Checks: the template's arithmetic (softmax(coef @ (r - mu) / sd + intercept)) == sklearn predict_proba on every public window (<= 1e-9), and the
W18 template's ICFamily on raw windows == the same probabilities on the first streams (the feature path, end to end).
    mem-guard python scripts/local/t2_d525_final.py
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
from t2_d487_wpli import C_LR, recentre  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
OUT = O / "t2_d525_final"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
X, idx, meta = B.load_export(str(EXP))
Y, U = idx["y"].to_numpy().astype(int), idx["subject"].to_numpy()
Fe = np.load(O / "t2_d487/features.npz")["imcoh"].astype(np.float64)
assert Fe.shape == (len(idx), 459) and np.isfinite(Fe).all(), Fe.shape
Fr = recentre(Fe, idx)
mu, sd = Fr.mean(0), Fr.std(0) + 1e-9                      # D525's form (every labelled public window is training here)
Z = (Fr - mu) / sd
clf = LogisticRegression(C=C_LR, max_iter=3000).fit(Z, Y)
assert list(clf.classes_) == [0, 1, 2]
coef, icpt = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64)
sc = Z @ coef.T + icpt
p_arr = np.exp(sc - sc.max(1, keepdims=True))
p_arr /= p_arr.sum(1, keepdims=True)
d = float(np.abs(p_arr - clf.predict_proba(Z)).max())
print(f"[d525 final] LR on {len(Y)} windows; template arithmetic vs sklearn max |dp| {d:.2e}; in-sample window acc {float((p_arr.argmax(1) == Y).mean()):.3f}", flush=True)
assert d <= 1e-9, d
zp = np.load(O / "t2_d525/pred_imcoh.npz")
per = []
for f in range(10):
    ((_, _, _, tem),) = B.folds(idx, f"lc9_{f}")
    te = np.flatnonzero(tem)
    held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
    rows = np.flatnonzero(U == held)
    pos = {int(r): k for k, r in enumerate(zp[f"te_{f}"])}
    per.append(zp[f"bag_{f}"][np.array([pos[int(r)] for r in rows])].mean(0))
start = np.mean(per, 0)
start = start / start.sum()
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "imcoh.npz", mu=mu, sd=sd, coef=coef, intercept=icpt, start=start)

# end to end: the W18 template's ICFamily on the raw windows of the first streams == p_arr (the feature path + the causal state)
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")                                # the template imports the kit (as t2p_w18_selftest.py loads it)
spec = importlib.util.spec_from_file_location("t2p_submission_w18", REPO / "tracks/t2_bci/submission/t2p_submission_w18.py")
W18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W18)
arrs = dict(np.load(OUT / "imcoh.npz"))
ch = list(meta["ch_names"])
fam = W18.ICFamily(arrs, [ch.index(n) for n in W18.IC_NODES], float(meta["sfreq"]))
dmax, nwin = 0.0, 0
for _, g in list(idx.groupby("stream", sort=False))[:3]:
    fam.reset()
    for r in g.sort_values("order")["i"].to_numpy():
        q = fam(np.asarray(X[r], np.float32))            # the dtype predict() hands the family
        dmax, nwin = max(dmax, float(np.abs(q - p_arr[r]).max())), nwin + 1
print(f"[d525 final] W18 ICFamily on raw windows vs the fit, {nwin} windows of 3 streams: max |dp| {dmax:.2e}", flush=True)
assert dmax <= 1e-6, dmax
sha = hashlib.sha256((OUT / "imcoh.npz").read_bytes()).hexdigest()
json.dump(dict(n=int(len(Y)), C=C_LR, start=[round(float(v), 4) for v in start], sha256=sha, check_sklearn=d, check_template=dmax,
               template_windows=nwin, rule="T2-D525 / W18 (preparation)"), open(OUT / "imcoh_meta.json", "w"), indent=1)
print(f"[d525 final] imcoh.npz written (sha256 {sha[:16]}); start vector {np.round(start, 4).tolist()}", flush=True)
