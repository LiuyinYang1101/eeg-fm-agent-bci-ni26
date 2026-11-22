#!/usr/bin/env python
"""T2-D476 ship form (template W13): ONE start vector PER FAMILY, by T2-D459's rule applied to each family's own mix: the mean raw family-mix
row of the control shape (M7's weights; CNN draws of D419 / D429, FM draws of D442) over ALL held-out cells (kit + seen1, test + val, 12 draws),
every held-out user weighted equally (mean of the per-user means; P04 included). Families = the package's: cnn (FBCNet), cnn2 (ShallowNet),
cnn3 (IFNet), fm (the five ST-L, shipped weights inside). Written to outputs/t2_d476/start_vectors.json; prints its sha256.
    mem-guard python scripts/local/t2_d476_vector.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = S[src]
C = J["arms"]["M7f"]
FAM = {"cnn": ["fbc_tv8c"], "cnn2": ["sh_tv8c"], "cnn3": ["ifn"], "fm": [k for k in C if k.startswith("n_")]}
assert sorted(sum(FAM.values(), [])) == sorted(C)
U = pd.read_csv(EXP / "index.csv").subject.str[-3:].to_numpy()
acc = {f: {} for f in FAM}
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            for f, ms in FAM.items():
                mix, ix = 0.0, None
                for m in ms:
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or (ix == ix_).all()
                    ix, mix = ix_, mix + C[m] * p.astype(np.float64)
                mix = mix / mix.sum(1, keepdims=True)
                for u in np.unique(U[ix]):
                    acc[f].setdefault(u, []).append(mix[U[ix] == u])
out_v, per_u = {}, {}
for f in FAM:
    per = {u: np.concatenate(v).mean(0) for u, v in acc[f].items()}
    vec = np.mean(list(per.values()), 0)
    vec = np.round(vec / vec.sum(), 4)
    vec[-1] = round(1.0 - vec[:-1].sum(), 4)
    out_v[f], per_u[f] = vec.tolist(), {u: np.round(v, 4).tolist() for u, v in per.items()}
out = O / "t2_d476"
out.mkdir(exist_ok=True)
txt = json.dumps({"per_family": out_v, "classes": ["mi", "sub", "word"], "per_user": per_u,
                  "rule": "T2-D459's rule per family: mean of per-user mean raw family-mix rows, control shape, all held-out cells"}, indent=1)
(out / "start_vectors.json").write_text(txt)
print(json.dumps(out_v))
print("sha256", hashlib.sha256(txt.encode()).hexdigest())
