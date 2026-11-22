#!/usr/bin/env python
"""Recipe explainer (2026-11-12, descriptive, zero GPU): how M7 builds up from a stock baseline, on ONE metric: cell-averaged balanced
accuracy on typical held-out users (all but P04), kit + seen1, test + val cells, 12 draws; each step read twice: the raw argmax of the
mix, and after the causal Sinkhorn decode (10 iterations, n0 3, lambda 0.95). Also each M7 member alone. Writes JSON to argv[1].
    mem-guard python scripts/local/t2_explainer_ladder.py <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
J = json.load(open(O / "t2_d408/spec_m7.json"))
M = dict(J["members"])
M["eegnet_stock"] = {"offset": 360, **{p: f"{S}/ni26-t2p-d432-eegnet-stock-5090/runs/{p}__eegnet__ea__n00__r1__s{{s}}.npz" for p in ("kit", "seen1")}}
M["fbc_stock"] = {"offset": 12, **{p: f"{S}/ni26-t2p-d146-fbc-r1-{p}-{j}/runs/{p}__fbcnet_bboth__none__n00__r1__s{{s}}.npz" for p, j in (("kit", 1516836), ("seen1", 1516837))}}
M["fbc_recipe"] = {"offset": 192, **{p: f"{S}/ni26-t2p-d296r-fbc-{p}-5090/runs/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75__s{{s}}.npz" for p in ("kit", "seen1")}}
M["fbc_copies"] = {"offset": 192, **{p: f"{O}/t2_d296r/route_fbc_r1_pulxx_vp2_e180_vr_rc75/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75fb3__s{{s}}.npz" for p in ("kit", "seen1")}}
M7 = J["arms"]["M7"]
cnn = {k: M7[k] for k in ("fbc_tv8c", "sh_tv8c", "ifn")}
norm = lambda d: {k: v / sum(d.values()) for k, v in d.items()}  # noqa: E731
LADDER = [
    ("Stock EEGNet (braindecode defaults + causal EA)", {"eegnet_stock": 1.0}, "T2-D432"),
    ("Stock FBCNet (filter-bank CNN, 60 epochs, no augmentation)", {"fbc_stock": 1.0}, "T2-D146"),
    ("FBCNet with our training recipe (virtual participants + references, re-colouring, 180 epochs)", {"fbc_recipe": 1.0}, "T2-D193..D230, D296r"),
    ("+ per-user fine-tuned copies, picked by a session fingerprint", {"fbc_copies": 1.0}, "T2-D159 / D160, D296r"),
    ("+ 8 test-time views (= M7's FBCNet member)", {"fbc_tv8c": 1.0}, "T2-D296 / D302"),
    ("+ ShallowNet and IFNet members (the CNN stack)", norm(cnn), "T2-D345r, D384"),
    ("+ five ST-EEGFormer-large members (= M7)", M7, "T2-D384 / D408"),
]
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
cache = {}


def probs(m, proto, i, split):
    k = (m, proto, i, split)
    if k not in cache:
        z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
        cache[k] = (z["test_idx"], z["prob_last"].astype(np.float64)) if split == "test" else (z["val_idx"], z["prob_val_last"].astype(np.float64))
    return cache[k]


def score(arm):
    out = {"raw": [], "decoded": []}
    per_user = {}
    for i in range(12):
        cells = {"raw": [], "decoded": []}
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                mix, ix = 0.0, None
                for m, w in arm.items():
                    ix_, p = probs(m, proto, i, split)
                    assert ix is None or (ix == ix_).all()
                    ix, mix = ix_, mix + w * p
                for kind, P in (("raw", mix.argmax(1)), ("decoded", causal_sinkhorn(mix, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1))):
                    for c in np.unique(CELL[ix]):
                        mk = CELL[ix] == c
                        ba = np.mean([np.mean(P[mk][Y[ix][mk] == v] == v) for v in np.unique(Y[ix][mk])])
                        u = U[ix][mk][0]
                        if u != "P04":
                            cells[kind].append(ba)
                        if kind == "decoded":
                            per_user.setdefault(u, []).append(ba)
        for kind in out:
            out[kind].append(np.mean(cells[kind]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}, {u: float(np.mean(v)) for u, v in per_user.items()}


res = {"metric": "cell-averaged balanced accuracy, typical held-out users (all but P04), kit + seen1 protocols, test + val cells, mean of 12 seed draws",
       "ladder": [], "members": []}
prev = None
for name, arm, row in LADDER:
    s, pu = score(arm)
    res["ladder"].append(dict(step=name, raw=round(s["raw"][0], 4), decoded=round(s["decoded"][0], 4), decoded_sd=round(s["decoded"][1], 4),
                              delta=None if prev is None else round(s["decoded"][0] - prev, 4), row=row, per_user_decoded={u: round(v, 4) for u, v in sorted(pu.items())}))
    prev = s["decoded"][0]
    print(res["ladder"][-1], flush=True)
NAMES = {"fbc_tv8c": "FBCNet", "sh_tv8c": "ShallowNet (EA)", "ifn": "IFNet (EA)", "lp10": "ST-EEGFormer-large, LP-FT", "e100f0": "ST-EEGFormer-large",
         "e100f1": "ST-EEGFormer-large", "e100f2": "ST-EEGFormer-large", "e100f3": "ST-EEGFormer-large"}
for m, w in M7.items():
    s, _ = score({m: 1.0})
    res["members"].append(dict(member=m, family=NAMES[m], share=round(w, 4), alone_raw=round(s["raw"][0], 4), alone_decoded=round(s["decoded"][0], 4)))
    print(res["members"][-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
