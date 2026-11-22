#!/usr/bin/env python
"""T2-D480: ni26.stream.lap_windows (nearest-neighbour surface Laplacian per window). 1) a field common to all channels (a reference /
far-field term) is removed exactly; 2) the result equals x_c - mean of its 4 nearest neighbours, by brute force; 3) a flat channel and a NaN
channel are left out of their neighbours' means, and the NaN channel stays NaN; 4) single windows (C, T) and batches agree; 5) the
neighbour sets are symmetric enough to be local (every neighbour within 7 cm on the standard_1005 head; edge channels reach 6.5 cm).
    python tests/test_lap_windows.py -> OK"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26.stream import lap_neighbours, lap_windows  # noqa: E402

CH = ['C2', 'CP2', 'P2', 'Pz', 'CPz', 'Cz', 'C1', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'TP7', 'CP5', 'CP3', 'C3', 'PO8', 'P6', 'P4', 'TP8', 'CP6',
      'CP4', 'C6', 'C4', 'FC6', 'FC4', 'FC2', 'F6', 'F4', 'F2', 'AF4', 'AF3', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1', 'FCz', 'C5']
rng = np.random.default_rng(0)
X = rng.standard_normal((5, len(CH), 64)).astype(np.float32)
common = rng.standard_normal((5, 1, 64)).astype(np.float32) * 10
Y0, Y1 = lap_windows(X, CH), lap_windows(X + common, CH)
assert np.allclose(Y0, Y1, atol=1e-4), "a field common to every channel must vanish"
nb = lap_neighbours(CH, 4)
ref = X - X[:, nb].mean(2)
assert np.allclose(Y0, ref, atol=1e-5), "x_c - mean of its 4 nearest neighbours"
Xb = X.copy()
Xb[1, CH.index("Cz")] = 0.0
Xb[1, CH.index("C1"), 7] = np.nan
Yb = lap_windows(Xb, CH)
assert np.isnan(Yb[1, CH.index("C1")]).all() or np.isnan(Yb[1, CH.index("C1"), 7]), "the NaN channel stays NaN"
for c in range(len(CH)):
    gn = [j for j in nb[c] if CH[j] not in ("Cz", "C1")]
    if c in (CH.index("Cz"), CH.index("C1")) or not gn:
        continue
    assert np.allclose(Yb[1, c], Xb[1, c] - Xb[1, gn].mean(0), atol=1e-5), CH[c]
assert np.allclose(lap_windows(X[2], CH), Y0[2], atol=1e-6), "a single window = the batch row"
import mne  # noqa: E402
pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
P = np.array([pos[c] for c in CH])
dmax = max(np.linalg.norm(P[c] - P[j]) for c in range(len(CH)) for j in nb[c])
assert dmax < 0.07, f"a neighbour {dmax:.3f} m away is not local"   # edge channels of this sparse montage reach 6.5 cm
json.dumps(nb.tolist())
print("OK")
