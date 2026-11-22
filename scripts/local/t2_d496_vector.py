#!/usr/bin/env python
"""T2-D496 (W14-lite): the fm family's ship start vector for the 3 + 3 member mix (the first three plain + three CAR ST-L, equal weights), by T2-D459's
rule (mean of the per-user mean raw family-mix rows over ALL held-out cells: kit + seen1, test + val; every held-out user weighted equally; P04
included), on A13's draws (D455 plain at seed index i + k, the D475 pair's CAR nets at i + 5 .. i + 7; 10 draws). The plain-five vector on the
same draws is printed beside it (how far the CAR members move the fm start), with T2-D476's shipped fm vector (D442's draws) for reference.
Writes outputs/t2_d496/fm_start_vector.json; prints its sha256 (copied to tracks/t2_bci/submission/configs/t2_d496_fm_start_vector.json).
    mem-guard python scripts/local/t2_d496_vector.py
"""
import glob
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
S = O / "sofia_s1"


def stl(pat):
    out = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{S}/{pat.format(pr=pr)}-[0-9]*")
        assert len(ds) == 1, (pat, pr, ds)
        f = glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s472*.npz")
        assert len(f) == 1, f
        out[pr] = f[0].replace("__s472.npz", "__s{s}.npz")
    return out


pp, pc = stl("ni26-t2p-d455-ctl-{pr}"), stl("ni26-t2p-d475-stlcar-{pr}")
U = pd.read_csv(EXP / "index.csv").subject.str[-3:].to_numpy()
MIXES = {"fm_5plain": lambda pr, i: [pp[pr].format(s=472 + (i + k) % 14) for k in range(5)],
         "fm_5plain_3car": lambda pr, i: [pp[pr].format(s=472 + (i + k) % 14) for k in range(5)] + [pc[pr].format(s=472 + (i + k) % 14) for k in (5, 6, 7)],
         "fm_3plain_3car": lambda pr, i: [pp[pr].format(s=472 + (i + k) % 14) for k in range(3)] + [pc[pr].format(s=472 + (i + k) % 14) for k in (5, 6, 7)]}
acc = {f: {} for f in MIXES}
for proto in ("kit", "seen1"):
    for i in range(10):
        for split in ("test", "val"):
            for f, files in MIXES.items():
                mix, ix = 0.0, None
                fs = files(proto, i)
                for fn in fs:
                    z = np.load(fn)
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or (ix == ix_).all()
                    ix, mix = ix_, mix + p.astype(np.float64) / len(fs)
                mix = mix / mix.sum(1, keepdims=True)
                for u in np.unique(U[ix]):
                    acc[f].setdefault(u, []).append(mix[U[ix] == u])
vecs = {}
for f in MIXES:
    per = {u: np.concatenate(v).mean(0) for u, v in acc[f].items()}
    vec = np.mean(list(per.values()), 0)
    vec = np.round(vec / vec.sum(), 4)
    vec[-1] = round(1.0 - vec[:-1].sum(), 4)
    vecs[f] = vec.tolist()
shipped = json.load(open(O / "t2_d476/start_vectors.json"))["per_family"]["fm"]
out = O / "t2_d496"
out.mkdir(exist_ok=True)
txt = json.dumps({"fm": vecs["fm_3plain_3car"], "fm_5plain_3car_same_draws": vecs["fm_5plain_3car"], "fm_5plain_same_draws": vecs["fm_5plain"], "fm_shipped_W13_D442_draws": shipped,
                  "classes": ["mi", "sub", "word"], "rule": "T2-D459's rule on the fm family's own mix (three plain + three CAR, equal weights), A13's draws (T2-D496)"}, indent=1)
(out / "fm_start_vector.json").write_text(txt)
print("fm (3 plain + 3 CAR):", vecs["fm_3plain_3car"], "| 5 + 3:", vecs["fm_5plain_3car"], "| 5 plain, same draws:", vecs["fm_5plain"], "| W13's shipped fm (D442 draws):", shipped)
print("sha256", hashlib.sha256(txt.encode()).hexdigest())
