#!/usr/bin/env python
"""T2-D524 / W18 rehearsal input (PREPARATION; a FAKE donor, never a candidate, never uploaded): W17 build 1's zip with every FM member given
a small RANDOM window-mean side head (fm_<i>_side.pt in t2p_final --fm-side's format: kind dcmean, n0 10, side.pt's prior mu / sd, weight
(3, 82) and bias (3) ~ N(0, 0.05), seed 0) and its config entry side = {kind, n0, file}. Purpose: rehearse t2p_repack --fm-from with a
side-head donor (the members are W17's own, so the donor check is exact) before the real SD donors (1620636 / 37) land.
    mem-guard python scripts/local/t2_w18_fakedonor.py [--base W17.zip --out FAKE.zip]      (sofia G2: both paths given)
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

O = Path.home() / "workspace/outputs/neural_interface_challenge"
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(O / "sofia_s1/ni26-t2p-repack-t2p_fin1_w17-1619616/t2p_fin1_w17.zip"))
ap.add_argument("--out", default=str(O / "t2_w18_rehearsal/fakedonor.zip"))
A = ap.parse_args()
BASE = Path(A.base)
R = Path(A.out).parent
D = R / (Path(A.out).stem + "_tree")
if D.exists():
    shutil.rmtree(D)
D.mkdir(parents=True)
with zipfile.ZipFile(BASE) as z:
    names = z.namelist()
    z.extractall(D)
cfg = json.loads((D / "config.json").read_text())
sb = torch.load(D / cfg["side"]["file"], map_location="cpu", weights_only=True)
rng = np.random.default_rng(0)
side = []
for i, m in enumerate(cfg["fm_members"]):
    f = f"fm_{i}_side.pt"
    torch.save(dict(kind="dcmean", n0=10.0, mu=torch.as_tensor(np.asarray(sb["mu"], np.float64)), sd=torch.as_tensor(np.asarray(sb["sd"], np.float64)),
                    weight=torch.as_tensor(rng.normal(0, 0.05, (3, len(sb["mu"]))), dtype=torch.float32),
                    bias=torch.as_tensor(rng.normal(0, 0.05, 3), dtype=torch.float32)), D / f)
    m["side"] = dict(kind="dcmean", n0=10.0, file=f)
    side.append(f)
(D / "config.json").write_text(json.dumps(cfg, indent=1))
zp = Path(A.out)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in names + side:
        z.write(D / f, f)
shutil.rmtree(D)
print(f"[fakedonor] {zp} ({zp.stat().st_size / 1e6:.0f} MB): {len(side)} FM members with random side heads {side}")
