#!/usr/bin/env python
"""Track1's T2 experiment XDAY, read 6 (registered in SEAT_T1_on_T2.md before its number; descriptive): the own-copy gain on day-2 windows
1-60 vs 61+ (decode over the whole stream; BA per half), from the xd files.   mem-guard python scripts/local/t1seat_t2_xday_read6.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
from t2_cluster_ci import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge/t1seat_t2"
idx = pd.read_csv(O / "export_combo_v1/index.csv", dtype={"session": str})
U, STREAM, ORD, Y = idx.subject.str.split("/").str[-1].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy()
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", 0.5714, 1 / 3), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vr", 0.4286, 1 / 2)}
rows = []
for s in range(780, 786):
    Z = {f: np.load(O / f"runs/xd__{v[0]}tv8cpu__s{s}.npz") for f, v in FAM.items()}
    ix = Z["fbc"]["test_idx"]
    users = list(Z["fbc"]["users"])
    own, n = np.array([users.index(u) for u in U[ix]]), np.arange(len(ix))
    for arm, m in (("m0", 0), ("m1", 1), ("m2", 2), ("copy", 99)):
        mix = 0
        for f, (st, w, r0) in FAM.items():
            r = min(1.0, m * r0)
            a, b = Z[f]["prob_last"].astype(float), Z[f]["prob_users_test"].astype(float)[own, n]
            mix = mix + w * causal_sinkhorn((1 - r) * a / a.sum(1, keepdims=True) + r * b / b.sum(1, keepdims=True), (STREAM[ix], ORD[ix]), 10, 3, 0.95)
        pr = mix.argmax(1)
        for u in np.unique(U[ix]):
            for half, hm in (("early", ORD[ix] < 60), ("late", ORD[ix] >= 60)):
                k = (U[ix] == u) & hm
                rows.append(dict(seed=s, arm=arm, user=u, half=half, ba=np.mean([np.mean(pr[k & (Y[ix] == q)] == q) for q in range(3)])))
T = pd.DataFrame(rows).groupby(["half", "user", "arm"]).ba.mean().unstack()
for name, a, b in (("m2 - m1", "m2", "m1"), ("copy-only - m1", "copy", "m1"), ("m1 - m0", "m1", "m0")):
    e, l = T.loc["early", a] - T.loc["early", b], T.loc["late", a] - T.loc["late", b]
    print(f"XDAY read 6 {name}: early half {e.mean():+.4f}, late half {l.mean():+.4f}; users late < early {int((l < e).sum())} / 9")
print("   levels: " + " ".join(f"{h} {a} {T.loc[h, a].mean():.4f}" for h in ("early", "late") for a in ("m0", "m1", "m2", "copy")))
