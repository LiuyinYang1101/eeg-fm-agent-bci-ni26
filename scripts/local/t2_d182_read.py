#!/usr/bin/env python
"""T2-D182 (zero compute, pre-registered in docs/DECISIONS_T2.md): per-member temperature scaling before the package's linear pool.
For each member and index, one scalar T > 0 is fitted on the KIT protocol's val windows (P05 / P08; not part of the mix objective) by
minimising the NLL of softmax(log p / T); the same T is applied to that member's kit AND seen1 files (val and test). Members = the adopted
package on fresh seeds (T2-D164's cA: FBCNet r1_pulxx blend seeds 48-59, ShallowNet r1_pulx r 1/2 blend seeds 24-35, ST-L, MOR). Then
t2p_pkgshape: c_temp (tempered members) vs cA (untempered).  python scripts/local/t2_d182_read.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"; D = O / "t2_d182"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def temper(p, T):
    lp = np.log(np.clip(p.astype(np.float64), 1e-12, 1.0)) / T
    lp -= lp.max(1, keepdims=True); q = np.exp(lp)
    return (q / q.sum(1, keepdims=True)).astype(np.float32)


def main():
    import pandas as pd
    y = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})["y"].to_numpy().astype(int)   # as t2p_pkgshape
    spec = json.load(open(O / "t2_d164/spec.json"))
    keep = ("fbcb_pulxxh", "shb_pulx", "stl", "mor")
    m_new, temps = {}, {}
    for k in keep:
        mem = spec["members"][k]; off = int(mem["offset"]); (D / k).mkdir(parents=True, exist_ok=True)
        temps[k] = []
        for i in range(12):
            zk = np.load(mem["kit"].format(s=off + i))
            pv, yv = zk["prob_val_last"], y[zk["val_idx"]]
            nll = lambda T: -np.mean(np.log(np.clip(temper(pv, T)[np.arange(len(yv)), yv], 1e-12, 1)))
            T = float(minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x)
            temps[k].append(round(T, 4))
            for p in ("kit", "seen1"):
                z = np.load(mem[p].format(s=off + i))
                np.savez_compressed(D / k / f"{p}__t__s{i}.npz", test_idx=z["test_idx"], val_idx=z["val_idx"],
                                    prob_last=temper(z["prob_last"], T), prob_val_last=temper(z["prob_val_last"], T),
                                    prob_best=temper(z["prob_last"], T), res=json.dumps(dict(temperature=T, src=mem[p].format(s=off + i))))
        m_new[k + "_t"] = {"offset": 0, **{p: str(D / k / f"{p}__t__s{{s}}.npz") for p in ("kit", "seen1")}}
    members = {k: spec["members"][k] for k in keep}; members.update(m_new)
    arms = {"cA": {"fbcb_pulxxh": 0.5, "shb_pulx": 0.2, "stl": 0.2, "mor": 0.1},
            "c_temp": {"fbcb_pulxxh_t": 0.5, "shb_pulx_t": 0.2, "stl_t": 0.2, "mor_t": 0.1}}
    (D / "spec.json").write_text(json.dumps({"n": 12, "anchor": "cA", "bar": "d141", "members": members, "arms": arms}, indent=2) + "\n")
    (D / "temperatures.json").write_text(json.dumps(temps, indent=1) + "\n")
    print({k: (min(v), float(np.median(v)), max(v)) for k, v in temps.items()}, flush=True)
    with open(D / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(D / "spec.json"),
                             "--out", str(D / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    print("rc", rc)


if __name__ == "__main__":
    main()
