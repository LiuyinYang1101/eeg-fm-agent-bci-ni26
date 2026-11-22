#!/usr/bin/env python
"""T2-D546 (b) BUILD-path CPU smoke (the manager's conditions ≈ 22:5x): the REAL W18 b2 members fm_2 (seed 622, plain) / fm_4 (seed 624,
CAR) from the package, st_large with the dcmean side head, on synthetic windows (no numbers of any read):
  (1) t2p_fm.load_member_state: fp16 -> fp32 upcast, the round trip back to fp16 bitwise (asserted inside), the side head as dc_side.*;
  (2) 0-epoch continuation from the fp32 upcast state with the build's knobs (lr x 0.1, warm 0, llrd 0.75, rc 0.75, tilt 0.15, mixup 0.4,
      calib gain, side dcmean, data seed s + 50,000): every tensor of state_last == the upcast state, and max |dp| vs the STORED fp16
      values loaded the same way (the stored member) -- the upcast changes nothing beyond fp16 rounding iff this is 0;
  (3) 1 epoch at lr x 0.1 from the same state: runs end to end, finite, moves the predictions, curves present.
    mem-guard env OMP_NUM_THREADS=8 python scripts/local/t2_d546b_smoke.py <W18 bN zip> [file:seed ...]   (default fm_2.pt:622 fm_4.pt:624 = b2;
    b1 = fm_2.pt:612 fm_6.pt:614, T2-D549)
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tracks/t2_bci/scripts").resolve()))
sys.path.insert(0, str(Path("src").resolve()))
import t2p_fm  # noqa: E402

ZIP = sys.argv[1]
PAIRS = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2:]] or [("fm_2.pt", 622), ("fm_4.pt", 624)]
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
          lr_enc=1e-5, lr_head=1e-4, warm=0.0, log=lambda m: print(f"  {m}", flush=True))
for file, seed in PAIRS:
    print(f"(1) {file}: load_member_state", flush=True)
    st0, m0, sb0 = t2p_fm.load_member_state(ZIP, file)
    assert m0["seed"] == seed and "dc_side.weight" in st0 and sb0 is not None, (m0["seed"], seed)
    with zipfile.ZipFile(ZIP) as z:
        sd16 = torch.load(io.BytesIO(z.read(file)), map_location="cpu", weights_only=True)
    st16 = dict(sd16, **{"dc_side.weight": sb0["weight"], "dc_side.bias": sb0["bias"]})
    print(f"    {len(st0)} tensors (side head included), fp16 round trip bitwise; car {bool(m0.get('car'))}, tilt {m0.get('tilt')}", flush=True)
    print(f"(2) {file}: 0-epoch continuation (fp32 upcast) vs the stored fp16 values", flush=True)
    r_ref = t2p_fm.fit_eval_fm("st_large", data, seed + 50000, ch, **dict(kw, epochs=0, init_state=st16, return_state=False))
    r0 = t2p_fm.fit_eval_fm("st_large", data, seed + 50000, ch, **dict(kw, epochs=0, init_state=st0, return_state=True))
    same = all(torch.equal(r0["state_last"][k].cpu().to(st0[k].dtype), st0[k]) for k in st0)
    dp = float(np.abs(r0["prob_last"] - r_ref["prob_last"]).max())
    print(f"    state_last == upcast state: {same}; max |p(0-epoch, fp32 upcast) - p(stored fp16)| = {dp:.3g}", flush=True)
    assert same and dp == 0.0, (same, dp)
    print(f"(3) {file}: 1 epoch at lr x 0.1", flush=True)
    r1 = t2p_fm.fit_eval_fm("st_large", data, seed + 50000, ch, **dict(kw, epochs=1, init_state=st0, return_state=False))
    d1 = float(np.abs(r1["prob_last"] - r0["prob_last"]).max())
    assert np.isfinite(r1["prob_last"]).all() and r1["prob_last"].shape == (n_te, 3) and len(r1["curves"]) == 1 and d1 > 0, d1
    print(f"    ok: max |p(1 epoch) - p(0 epochs)| = {d1:.3g}; curves {r1['curves']}", flush=True)
print("D546b build-path smoke: PASS", flush=True)
