#!/usr/bin/env python
"""T2-D546 CPU smoke of the warm-start continuation code path (rand_st_small on synthetic windows; no numbers of any read):
  (1) the base fit (side dcmean, mixup) with return_state;
  (2) arm (b) with epochs 0 from the base state = the base's own predictions EXACTLY (the strict load incl. the side head);
  (3) arm (a) with epochs 0: the head / side head / top block re-drawn (predictions differ; dc_side zero);
  (4) both arms with 1 epoch at lr x 0.1, warm 0: run end to end, finite outputs, curves present.
    mem-guard env OMP_NUM_THREADS=4 python scripts/local/t2_d546_smoke.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tracks/t2_bci/scripts").resolve()))
sys.path.insert(0, str(Path("src").resolve()))
import t2p_fm  # noqa: E402

meta = json.load(open(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json"))
ch = list(meta["ch_names"])
rng = np.random.default_rng(0)
n_tr, n_te = 32, 16
X = rng.standard_normal((n_tr + n_te, len(ch), 480)).astype(np.float32)
y = rng.integers(0, 3, n_tr + n_te)
S = rng.standard_normal((n_tr + n_te, 82)).astype(np.float32)
cells = np.array([f"c{i % 2}" for i in range(n_tr + n_te)])
data = dict(Xtr=X[:n_tr], ytr=y[:n_tr], Xva=X[:0], yva=y[:0], cva=cells[:0], Xte=X[n_tr:], yte=y[n_tr:], cte=cells[n_tr:],
            side_tr=S[:n_tr], side_va=S[:0], side_te=S[n_tr:])
kw = dict(epochs=1, bs=16, lr_enc=1e-4, lr_head=1e-3, wd=0.05, llrd=0.75, readout="mean", device="cpu", mixup=0.4, calib="gain",
          side="dcmean", log=lambda m: print(f"  {m}", flush=True))
fm = "rand_st_small"
print("(1) base", flush=True)
r = t2p_fm.fit_eval_fm(fm, data, 828, ch, **dict(kw, return_state=True))
assert r["state_last"] is not None and any(k.startswith("dc_side") for k in r["state_last"]), "the state must carry the side head"
print("(2) arm b, 0 epochs", flush=True)
rb0 = t2p_fm.fit_eval_fm(fm, data, 50828, ch, **dict(kw, epochs=0, warm=0.0, init_state=r["state_last"]))
d_b = float(np.abs(rb0["prob_last"] - r["prob_last"]).max())
print(f"    max |p(b, 0 ep) - p(base)| = {d_b:.3g}", flush=True)
assert d_b <= 1e-6, d_b
print("(3) arm a, 0 epochs", flush=True)
ra0 = t2p_fm.fit_eval_fm(fm, data, 50828, ch, **dict(kw, epochs=0, warm=0.0, init_state=r["state_last"], cont_reinit=True))
d_a = float(np.abs(ra0["prob_last"] - r["prob_last"]).max())
print(f"    max |p(a, 0 ep) - p(base)| = {d_a:.3g} (must be > 0)", flush=True)
assert d_a > 1e-4, d_a
print("(4) both arms, 1 epoch at lr x 0.1", flush=True)
for arm in ("a", "b"):
    rc = t2p_fm.fit_eval_fm(fm, data, 50828, ch, **dict(kw, lr_enc=1e-5, lr_head=1e-4, warm=0.0, init_state=r["state_last"], cont_reinit=(arm == "a")))
    assert np.isfinite(rc["prob_last"]).all() and rc["prob_last"].shape == (n_te, 3) and len(rc["curves"]) == 1, arm
    print(f"    arm {arm}: ok, curves {rc['curves']}", flush=True)
print("D546 smoke: PASS", flush=True)
