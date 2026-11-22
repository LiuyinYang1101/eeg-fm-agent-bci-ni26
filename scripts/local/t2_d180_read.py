#!/usr/bin/env python
"""T2-D180 (zero compute, pre-registered in docs/DECISIONS_T2.md): the per-user copy gain when the CNN members are 3-SEED averages per index
instead of single seeds. Per index i = 0..11, member files are the mean of three fresh-seed files (prob_last / prob_val_last; test / val
indices asserted equal):
  anchor c3_0: FBCNet r1 blend (PERUSER copies, r 1/3) over seeds {24, 36, 48} + i; ShallowNet pooled r1 over {12, 24, 36} + i
  arm    c3_1: FBCNet r1_pulxx blend (r 1/3) over seeds {36, 48, 60} + i; ShallowNet r1_pulx blend (r 1/2) over {12, 24, 36} + i
  ST-L / MOR: the single shipped members (seeds 0-11) in both. Then t2p_pkgshape (c3_1 vs c3_0). The 1-seed reference = T2-D160 (+0.0141).
  python scripts/local/t2_d180_read.py
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"; S = O / "sofia_s1"; D = O / "t2_d180"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
one = lambda pat: (lambda g: (g[0] if len(g) == 1 else sys.exit(f"need one dir for {pat}: {g}")))(sorted(glob.glob(str(pat))))
SRC = {
    "fbc_r1": [(O / "t2_d155/route_fbc_r1", 24, "fbcnet_bboth__none__n00__r1fb3"), (O / "t2_d156/route_fbc_r1_r3", 36, "fbcnet_bboth__none__n00__r1fb3"),
               (O / "t2_d159/route_fbc_r1_r3", 48, "fbcnet_bboth__none__n00__r1fb3")],
    "fbc_pulxx": [(O / "t2_d156/route_fbc_r1_pulxx_r3", 36, "fbcnet_bboth__none__n00__r1_pulxxfb3"), (O / "t2_d159/route_fbc_r1_pulxx_r3", 48, "fbcnet_bboth__none__n00__r1_pulxxfb3"),
                  (O / "t2_d168/route_fbc_r1_pulxx", 60, "fbcnet_bboth__none__n00__r1_pulxxfb3")],
    "sh_pool": [(Path(one(S / "ni26-t2p-d159-sh-r1_pulx-*")) / "runs", 12, "shallow__ea__n00__r1_pulx"),
                (Path(one(S / "ni26-t2p-d164-sh-r1_pulx-[0-9]*")) / "runs", 24, "shallow__ea__n00__r1_pulx"),
                (Path(one(S / "ni26-t2p-d169-sh-r1_pulx-[0-9]*")) / "runs", 36, "shallow__ea__n00__r1_pulx")],
    "sh_pulx": [(O / "t2_d159/route_sh_pulx_r2", 12, "shallow__ea__n00__r1_pulxfb2"), (O / "t2_d164/route_sh_r1_pulx_r2", 24, "shallow__ea__n00__r1_pulxfb2"),
                (O / "t2_d169/route_sh_r1_pulx_r2", 36, "shallow__ea__n00__r1_pulxfb2")],
}


def main():
    for name, srcs in SRC.items():
        (D / name).mkdir(parents=True, exist_ok=True)
        for p in ("kit", "seen1"):
            for i in range(12):
                zs = [np.load(d / f"{p}__{stem}__s{base + i}.npz") for d, base, stem in srcs]
                for k in ("test_idx", "val_idx"):
                    assert all(np.array_equal(z[k], zs[0][k]) for z in zs), (name, p, i, k)
                pt = np.mean([z["prob_last"].astype(np.float64) for z in zs], 0)
                pv = np.mean([z["prob_val_last"].astype(np.float64) for z in zs], 0)
                np.savez_compressed(D / name / f"{p}__avg__s{i}.npz", test_idx=zs[0]["test_idx"], prob_last=pt.astype(np.float32), val_idx=zs[0]["val_idx"],
                                    prob_val_last=pv.astype(np.float32), prob_best=pt.astype(np.float32),
                                    res=json.dumps(dict(avg_of=[f"{d.name}:{stem}:s{base + i}" for d, base, stem in srcs])))
    base = json.load(open(O / "t2_d153/spec.json"))["members"]
    m = {k: base[k] for k in ("stl", "mor")}
    for name in SRC:
        m[name] = {"offset": 0, **{p: str(D / name / f"{p}__avg__s{{s}}.npz") for p in ("kit", "seen1")}}
    arms = {"c3_0": {"fbc_r1": 0.5, "sh_pool": 0.2, "stl": 0.2, "mor": 0.1}, "c3_1": {"fbc_pulxx": 0.5, "sh_pulx": 0.2, "stl": 0.2, "mor": 0.1}}
    (D / "spec.json").write_text(json.dumps({"n": 12, "anchor": "c3_0", "bar": "d141", "members": m, "arms": arms}, indent=2) + "\n")
    with open(D / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[2] / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E),
                             "--spec", str(D / "spec.json"), "--out", str(D / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    print("rc", rc)


if __name__ == "__main__":
    main()
