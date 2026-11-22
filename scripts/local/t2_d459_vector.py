#!/usr/bin/env python
"""T2-D459 ship form: the ONE decoder start vector = the mean raw mix row of the control shape (M7's weights; fresh CNN draws of D419 / D429,
FM draws of D442) over ALL held-out cells (kit + seen1, test + val, 12 draws), every held-out user weighted equally (mean of the
per-user means; P04 included: no selection). Written to outputs/t2_d459/start_vector.json; prints its sha256.
    mem-guard python scripts/local/t2_d459_vector.py
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
U = pd.read_csv(EXP / "index.csv").subject.str[-3:].to_numpy()
acc = {}
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            mix, ix = 0.0, None
            for m, w in J["arms"]["M7f"].items():
                z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                mix = mix + w * p.astype(np.float64)
            mix = mix / mix.sum(1, keepdims=True)
            for u in np.unique(U[ix]):
                acc.setdefault(u, []).append(mix[U[ix] == u])
per = {u: np.concatenate(v).mean(0) for u, v in acc.items()}
vec = np.mean(list(per.values()), 0)
vec = np.round(vec / vec.sum(), 4)
vec[-1] = round(1.0 - vec[:-1].sum(), 4)
out = O / "t2_d459"
out.mkdir(exist_ok=True)
txt = json.dumps({"start": vec.tolist(), "classes": ["mi", "sub", "word"], "per_user": {u: np.round(v, 4).tolist() for u, v in per.items()},
                  "n_windows": {u: int(sum(len(a) for a in v)) for u, v in acc.items()}, "rule": "mean of per-user mean raw mix rows, control shape, all held-out cells"}, indent=1)
(out / "start_vector.json").write_text(txt)
print(txt)
print("sha256", hashlib.sha256(txt.encode()).hexdigest())
