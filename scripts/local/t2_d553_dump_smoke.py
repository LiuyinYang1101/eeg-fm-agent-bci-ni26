#!/usr/bin/env python
"""T2-D553 CPU smoke of the FM side-head DUMP (no numbers of any read): a REAL W18 b2 member with its side head (st_large, dcmean) on synthetic
windows, 1 epoch at lr x 0.1 from its state:
  (1) NI26_SIDE_DUMP unset -> fit_eval_fm returns no side_dump (the default path is unchanged);
  (2) NI26_SIDE_DUMP=1 -> side_dump = {logit_last (n_te, 3), w (3, 82), b (3,)}; softmax(L + S_te @ w.T + b) == prob_last (CPU: no autocast,
      so equal to float32 rounding), and the dump's head == the state's dc_side;
  (3) the same seed with and without the dump gives the same prob_last (the dump is read-only).
    mem-guard env OMP_NUM_THREADS=8 python scripts/local/t2_d553_dump_smoke.py <W18 b2 zip> [file]   (default fm_2.pt)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("tracks/t2_bci/scripts").resolve()))
sys.path.insert(0, str(Path("src").resolve()))
import t2p_fm  # noqa: E402

ZIP = sys.argv[1]
FILE = sys.argv[2] if len(sys.argv) > 2 else "fm_2.pt"
meta = json.load(open(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json"))
ch = list(meta["ch_names"])
rng = np.random.default_rng(0)
n_tr, n_te = 32, 24
X = rng.standard_normal((n_tr + n_te, len(ch), int(meta.get("n_times", 480)))).astype(np.float32)
y = rng.integers(0, 3, n_tr + n_te)
S = rng.standard_normal((n_tr + n_te, 82)).astype(np.float32)
cells = np.array([f"c{i % 2}" for i in range(n_tr + n_te)])
data = dict(Xtr=X[:n_tr], ytr=y[:n_tr], Xva=X[:0], yva=y[:0], cva=cells[:0], Xte=X[n_tr:], yte=y[n_tr:], cte=cells[n_tr:],
            side_tr=S[:n_tr], side_va=S[:0], side_te=S[n_tr:], utr=np.arange(n_tr) % 4)
kw = dict(bs=16, wd=0.05, llrd=0.75, rc=0.75, tilt=0.15, readout="mean", device="cpu", mixup=0.4, calib="gain", side="dcmean",
          lr_enc=1e-5, lr_head=1e-4, warm=0.0, epochs=1, log=lambda m: print(f"  {m}", flush=True))
st0, m0, sb0 = t2p_fm.load_member_state(ZIP, FILE)
os.environ.pop("NI26_SIDE_DUMP", None)
r0 = t2p_fm.fit_eval_fm("st_large", data, int(m0["seed"]) + 50000, ch, **dict(kw, init_state=st0, return_state=False))
assert "side_dump" not in r0, "the default path returned a dump"
print("(1) NI26_SIDE_DUMP unset: no side_dump", flush=True)
os.environ["NI26_SIDE_DUMP"] = "1"
r1 = t2p_fm.fit_eval_fm("st_large", data, int(m0["seed"]) + 50000, ch, **dict(kw, init_state=st0, return_state=True))
sd = r1["side_dump"]
L, w, b = (np.asarray(sd[k], np.float64) for k in ("logit_last", "w", "b"))
assert L.shape == (n_te, 3) and w.shape == (3, 82) and b.shape == (3,), (L.shape, w.shape, b.shape)
z = L + S[n_tr:].astype(np.float64) @ w.T + b
p = np.exp(z - z.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
dp = float(np.abs(p - r1["prob_last"]).max())
dw = float(np.abs(w - r1["state_last"]["dc_side.weight"].float().numpy()).max())
db = float(np.abs(b - r1["state_last"]["dc_side.bias"].float().numpy()).max())
print(f"(2) NI26_SIDE_DUMP=1: softmax(L + S w^T + b) vs prob_last max |dp| {dp:.2e}; head vs state |dw| {dw:.1e} |db| {db:.1e}", flush=True)
assert dp <= 1e-5 and dw == 0.0 and db == 0.0, (dp, dw, db)
d01 = float(np.abs(r1["prob_last"] - r0["prob_last"]).max())
print(f"(3) prob_last with vs without the dump: max |dp| {d01:.2e}", flush=True)
assert d01 == 0.0, d01
print("D553 dump smoke: PASS", flush=True)
