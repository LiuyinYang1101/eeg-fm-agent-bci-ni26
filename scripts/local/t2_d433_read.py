#!/usr/bin/env python
"""T2-D433 read (pre-registered before any D433 number; zero GPU): the random-convolution member (outputs/t2_d433/runs) in the classical-complement
form of T2-D408. Reports, under SKλ on t2_cluster_ci's cells: the member ALONE beside FBTS alone (all cells, non-P04); the error phi with M7 (decoded
window errors, held-out windows, 12 draws pooled); M7RK05 = M7 + the member at 0.05 vs M7 (typical, W50, per user via t2_cluster_ci).
KILL: member alone below FBTS alone on all cells, OR (phi > 0.5 AND the add's typical <= 0).
BAR (the standard bar + the user's R2 guard): M7RK05 vs M7 typical >= +0.003 with >= 9 / 12 AND W50 >= 0 AND no user below -0.020 -> a runtime / ZIP
pricing row and a fresh-draw replicate; otherwise NOT ADOPTED.
    mem-guard python scripts/local/t2_d433_read.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
R = O / "t2_d433"
NAME = sys.argv[1] if len(sys.argv) > 1 else "rocket"          # "mantis" = part (b)
TAG = {"rocket": "ROCKET", "mantis": "MANTIS"}[NAME]
J = json.load(open(O / "t2_d408/spec_m7.json"))
J["members"][NAME] = {"offset": 0, **{p: str(R / f"runs/{p}__{NAME}__s{{s}}.npz") for p in ("kit", "seen1")}}
M7 = J["arms"]["M7"]
J["arms"] = {"M7": M7, "M7RK05": {**{k: 0.95 * v for k, v in M7.items()}, NAME: 0.05}, TAG: {NAME: 1.0}, "FBTS": {"fbts": 1.0}}
json.dump(J, open(R / f"spec_{NAME}.json", "w"), indent=1)
idx = pd.read_csv(EXP / "index.csv")
lev = {}
for arm in (TAG, "FBTS", "M7", "M7RK05"):
    d = cell_rows(J, arm, 12, idx)
    lev[arm] = (d.ba.mean(), d[d.subj != "P04"].ba.mean())
    print(f"{arm:8s} alone / package level: all cells {lev[arm][0]:.4f}, non-P04 {lev[arm][1]:.4f}")


def errors(arm):
    out = []
    for proto in ("kit", "seen1"):
        for i in range(12):
            for split in ("test", "val"):
                mix, ix = 0.0, None
                for m, w in J["arms"][arm].items():
                    z = np.load(J["members"][m][proto].format(s=J["members"][m].get("offset", 0) + i))
                    ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    mix = mix + w * p.astype(np.float64)
                P = causal_sinkhorn(mix, (idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]), 10, 3, 0.95).argmax(1)
                out.append(P != idx.y.to_numpy()[ix])
    return np.concatenate(out).astype(float)


phi = float(np.corrcoef(errors(TAG), errors("M7"))[0, 1])
phi_f = float(np.corrcoef(errors("FBTS"), errors("M7"))[0, 1])
print(f"error phi with M7: {TAG} {phi:.3f} (FBTS {phi_f:.3f})")
out = subprocess.run([sys.executable, "-I", str(HERE / "t2_cluster_ci.py"), str(R / f"spec_{NAME}.json"), "M7", "M7RK05"], capture_output=True, text=True, check=True).stdout
(R / f"cluster_M7RK05_{NAME}.txt").write_text(out)
print(out)
users = {u: float(v) for u, v in re.findall(r"^\s+(P\d\d): ([+-]\d\.\d+)", out, re.M)}
typ, up = re.search(r"typical: Δ ([+-]\d\.\d+) \((\d+) / 12", out).groups()
typ, up = float(typ), int(up)
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", out).group(1))
kill = lev[TAG][0] < lev["FBTS"][0] or (phi > 0.5 and typ <= 0)
ok = typ >= 0.003 and up >= 9 and w50 >= 0 and min(users.values()) >= -0.020
print(f"== VERDICT: {'KILLED' if kill else ('PASS -> pricing row + replicate' if ok else 'NOT ADOPTED')}  [member {lev[TAG][0]:.4f} vs FBTS "
      f"{lev['FBTS'][0]:.4f}; phi {phi:.3f}; add typical {typ:+.4f} ({up} / 12), W50 {w50:+.4f}, worst user {min(users.values()):+.4f}]")
