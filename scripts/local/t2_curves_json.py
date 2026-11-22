#!/usr/bin/env python
"""T2 learning curves beside every run (the owner's directive 2026-11-16 ≈ 20:2x): for each run dir given, read every runs/*.npz (or *.npz)
and write <dir>/curves.json = {file: {"curves": res['curves'] (train loss / epoch, test cell BA every 5 epochs), "cont": the D546 twins'
continuation curves when present, "epochs", "seed", "fold"}}. Descriptive data only; a run dir that already has curves.json is rewritten.
    OMP_NUM_THREADS=1 python scripts/local/t2_curves_json.py <run dir> [<run dir> ...]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

for d in map(Path, sys.argv[1:]):
    files = sorted(glob.glob(str(d / "runs/*.npz"))) or sorted(glob.glob(str(d / "*.npz")))
    out = {}
    for f in files:
        z = np.load(f)
        r = json.loads(str(z["res"])) if "res" in z.files else {}
        e = dict(curves=r.get("curves"), epochs=r.get("epochs"), seed=r.get("seed"), fold=r.get("fold"), twin_suffix=r.get("twin_suffix"))
        if "cont_curves" in z.files:
            e["cont"] = json.loads(str(z["cont_curves"]))
        out[Path(f).name] = e
    (d / "curves.json").write_text(json.dumps(out))
    print(f"{d}: curves.json for {len(out)} run files", flush=True)
