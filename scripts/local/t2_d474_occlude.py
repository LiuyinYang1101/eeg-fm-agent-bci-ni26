#!/usr/bin/env python
"""T2-D474 (descriptive; written before any number): do our members USE the frontal slow-wave (eye-movement-like) cue on held-out users?
The kit-trained twin package (t2p_d304_skl_twin.zip: trained on the kit's training users only, its own decoder) is run stream by stream on
the kit val (P05, P08) and test (P04, P09) windows, as delivered and with the < 4 Hz content removed (4th-order Butterworth high-pass, zero
phase, inside each window) at (a) AF3 / AF4 / F5 / F6, (b) every AF / F channel, (c) every channel. Cell-averaged balanced accuracy per user.
    gpu-submit mem-guard python scripts/local/t2_d474_occlude.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(REPO / "src"))
from t2p_repack import load, run_stream  # noqa: E402
from ni26 import kit, metrics  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
ZIP = O / "sofia_s1/ni26-t2p-repack-d304-skl-twin-1548789/t2p_d304_skl_twin.zip"
PKG = O / "t2_d474/pkg"
if not (PKG / "submission.py").exists():
    PKG.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(ZIP).extractall(PKG)
meta_e = json.loads((E / "meta.json").read_text())
CH = meta_e["ch_names"]
idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
X = np.load(E / "X.npy", mmap_mode="r")
kit.use_track("bci_decoding")
dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = dict(device=dev, ch_names=CH, sfreq=meta_e["sfreq"], n_chans=len(CH), n_times=int(X.shape[-1]), n_classes=3)
m = load(PKG, "twin", meta)
sos = butter(4, 4.0, btype="highpass", fs=float(meta_e["sfreq"]), output="sos")
ARMS = {"delivered": [], "AF3/AF4/F5/F6 < 4 Hz removed": ["AF3", "AF4", "F5", "F6"],
        "all AF/F < 4 Hz removed": [c for c in CH if c.startswith(("AF", "F")) and not c.startswith("FC")],
        "every channel < 4 Hz removed": list(CH)}
idx["u"] = idx.subject.str[-3:]
sel = idx[idx.u.isin(["P04", "P05", "P08", "P09"]) & idx.split.isin(["val", "test"])]
rows = []
for arm, chs in ARMS.items():
    ci = [CH.index(c) for c in chs]
    for _, g in sel.groupby("stream", sort=False):
        g = g.sort_values("order")
        xs = np.asarray(X[g["i"].to_numpy()], np.float32)
        if ci:
            xs = xs.copy()
            xs[:, ci] = sosfiltfilt(sos, xs[:, ci].astype(np.float64), axis=-1).astype(np.float32)
        pr, _ = run_stream(m, xs, dev)
        for c in g.cell.unique():
            mk = (g.cell == c).to_numpy()
            yy = g.y.to_numpy()[mk]
            rows.append((arm, g.u.iloc[0], c, float(np.mean([np.mean(pr[mk][yy == k] == k) for k in np.unique(yy)]))))
    print(arm, "done", flush=True)
D = pd.DataFrame(rows, columns=["arm", "u", "cell", "ba"])
tab = D.groupby(["arm", "u"]).ba.mean().unstack()
tab["typical (P05 P08 P09)"] = D[D.u != "P04"].groupby("arm").ba.mean()
tab = tab.loc[list(ARMS)]
print(tab.round(4).to_string())
(O / "t2_d474").mkdir(exist_ok=True)
tab.round(4).to_csv(O / "t2_d474/occlusion.csv")
