#!/usr/bin/env python
"""Track1's T2 experiment C19, the read (rules fixed in SEAT_T1_on_T2.md, section C19, before any number): R41 / A / B / B' on the kit's
strangers (test + val users), pooled nets + tv8, the shipped causal decode per family, CNN mix 0.5714 / 0.4286, cell-averaged BA.
Kill: max(B, B') - A < +0.010 on the strangers.       mem-guard python scripts/local/t1seat_t2_c19_read.py"""
import json
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
idx = pd.read_csv(O / "export_c19_v1/index.csv", dtype={"session": str})          # PROTEUS rows first = export_proteus_v1's rows
U, STREAM, ORD, Y, CELL = idx.subject.str.split("/").str[-1].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
FAM = {"fbc": ("fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8", 0.5714), "sh": ("shallow__ea__n00__r1_pulx_vp2_e180_vrtv8", 0.4286)}
ARMS = {"R41": ("r41", "kit"), "A": ("c19", "kit"), "B": ("c19", "kx"), "B'": ("c19", "kxh")}
SEEDS = range(790, 796)
rows = []
for s in SEEDS:
    for arm, (d, proto) in ARMS.items():
        D = {}
        for f, (st, w) in FAM.items():
            z = np.load(O / f"c19/{d}/{proto}__{st}__s{s}.npz")
            ix = np.concatenate([z["test_idx"], z["val_idx"]])
            p = np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64)
            D[f] = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[ix], ORD[ix]), 10, 3, 0.95)
        for name, pr in (("mix", sum(FAM[f][1] * D[f] for f in FAM).argmax(1)), ("fbc", D["fbc"].argmax(1)), ("sh", D["sh"].argmax(1))):
            for c in np.unique(CELL[ix]):
                m = CELL[ix] == c
                rows.append(dict(seed=s, arm=arm, what=name, cell=c, user=U[ix][m][0],
                                 ba=float(np.mean([np.mean(pr[m][Y[ix][m] == q] == q) for q in range(3)]))))
C = pd.DataFrame(rows)
C.to_csv(O / "c19/cells.csv", index=False)
res = {}
for what in ("mix", "fbc", "sh"):
    S = C[C.what == what].groupby(["arm", "seed"]).ba.mean().unstack("arm")            # cell mean per seed, all four strangers
    T = C[(C.what == what) & (C.user != "P04")].groupby(["arm", "seed"]).ba.mean().unstack("arm")
    Uu = C[C.what == what].groupby(["arm", "user"]).ba.mean().unstack("arm")
    print(f"C19 {what}: strangers " + " ".join(f"{a} {S[a].mean():.4f}" for a in ARMS) + " | typical (non-P04) " + " ".join(f"{a} {T[a].mean():.4f}" for a in ARMS), flush=True)
    for a, b in (("A", "R41"), ("B", "A"), ("B'", "A"), ("B", "R41")):
        d, dt = S[a] - S[b], T[a] - T[b]
        print(f"   {a} - {b}: {d.mean():+.4f} ({int((d > 0).sum())} / 6 seeds up), typical {dt.mean():+.4f} ({int((dt > 0).sum())} / 6); per user "
              + " ".join(f"{u} {Uu.loc[u, a] - Uu.loc[u, b]:+.3f}" for u in Uu.index), flush=True)
        res[f"{what}:{a}-{b}"] = dict(d=float(d.mean()), up=int((d > 0).sum()), typical=float(dt.mean()))
best = max(res["mix:B-A"]["d"], res["mix:B'-A"]["d"])
print(f"C19 KILL RULE: max(B, B') - A = {best:+.4f} on the strangers (CNN mix) -> {'SURVIVES (>= +0.010): the stack read follows' if best >= 0.010 else 'KILLED (< +0.010)'}", flush=True)
res["verdict"] = "survives" if best >= 0.010 else "killed"
json.dump(res, open(O / "c19/read.json", "w"), indent=1)
