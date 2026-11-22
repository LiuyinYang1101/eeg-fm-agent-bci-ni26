#!/usr/bin/env python
"""Track1's T2 experiment C19 (the manager's ask 2026-11-15 01:xx; SEAT_T1_on_T2.md, section C19): the 19-REAL-shared-channel export =
every PROTEUS window cut to the 19 channel names PROTEUS and Scherer2015 share (PROTEUS order) + Scherer's three-command windows on the
same 19 channels at their NATIVE values (no spline), one scale constant (median window x channel std = PROTEUS's on these channels).
Index as t1seat_t2_xday_export.py (PROTEUS rows first, their kit split kept; Scherer S1..S9, sessions '1' / '2', split 'ext').
    mem-guard python scripts/local/t1seat_t2_c19_export.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path.home() / "workspace/datasets/neural_interface_challenge"
O = Path.home() / "workspace/outputs/neural_interface_challenge/t1seat_t2"
OUT = O / "export_c19_v1"
CODE = {4: 0, 2: 1, 1: 2}
mp = json.loads((D / "t2_proteus/export_proteus_v1/meta.json").read_text())
ms = json.loads((D / "t2_bci/windows/scherer2015_cue3s4/meta.json").read_text())
names = [c for c in mp["ch_names"] if c in ms["ch_names"]]
assert len(names) == 19, names
jp, js = [mp["ch_names"].index(c) for c in names], [ms["ch_names"].index(c) for c in names]
idx = pd.read_csv(O / "export_combo_v1/index.csv", dtype={"session": str})      # the same rows, in the same order
XP = np.load(D / "t2_proteus/export_proteus_v1/X.npy", mmap_mode="r")
XS = np.load(D / "t2_bci/windows/scherer2015_cue3s4/X.npy", mmap_mode="r")
isx = pd.read_csv(D / "t2_bci/windows/scherer2015_cue3s4/index.csv")
keep = np.flatnonzero(isx.code.isin(list(CODE)).to_numpy())
assert len(idx) == len(XP) + len(keep) and (idx.y.to_numpy()[len(XP):] == isx.code.map(CODE).to_numpy()[keep]).all()
xp = np.asarray(XP, np.float32)[:, jp]
xs = np.asarray(XS[keep], np.float64)[:, js]
k = float(np.median(xp[::5].astype(np.float64).std(-1))) / float(np.median(xs[::5].std(-1)))
OUT.mkdir(parents=True, exist_ok=True)
X = np.concatenate([xp, (xs * k).astype(np.float32)])
np.save(OUT / "X.npy", X)
idx.to_csv(OUT / "index.csv", index=False)
sha = hashlib.sha256((OUT / "X.npy").read_bytes()).hexdigest()
(OUT / "meta.json").write_text(json.dumps(dict(study="c19: Dreyer2026Proteus + Scherer2015Individually on their 19 shared channels", sfreq=120.0,
                                              n_chans=19, n_times=480, ch_names=names, n_windows=int(len(idx)), n_proteus=int(len(XP)),
                                              scherer_scale=k, x_sha256=sha), indent=1))
print(f"c19 export: {X.shape}, channels {names}, Scherer scale x{k:.4f}, sha {sha[:16]} -> {OUT}")
