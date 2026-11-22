#!/usr/bin/env python
"""T2-D462 (written before any number): the per-user copy blend weight r scaled by m in M7's shape, by returning / new. See the row.
    mem-guard python scripts/local/t2_d462_r.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
W = J["arms"]["M7f"]
FAM = {"fbc_tv8c": ("ni26-t2p-d419-fbc_e180_s324-5090", "t2_d419/route_fbc_e180", 324), "sh_tv8c": ("ni26-t2p-d429-sh-ctl-5090", "t2_d429/route_sh_ctl", 348),
       "ifn": ("ni26-t2p-d429-ifn-ctl-5090", "t2_d429/route_ifn_ctl", 348)}
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL, SES = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy(), idx.session.to_numpy().astype(int)
MS = (0.0, 0.5, 1.0, 1.33, 1.67, 2.0)
rows = []
for proto in ("kit", "seen1"):
    for i in range(12):
        for split, kp, ki, ku in (("test", "prob_last", "test_idx", "prob_users_test"), ("val", "prob_val_last", "val_idx", "prob_users_val")):
            fmix, ix = 0.0, None
            for m, w in W.items():
                if m in FAM:
                    continue
                z = np.load(J["members"][m][proto].format(s=J["members"][m].get("offset", 0) + i))
                ix, fmix = z[ki], fmix + w * z[kp].astype(np.float64)
            parts = {}
            for m, (src, rt, off) in FAM.items():
                a = np.load([f for f in glob.glob(f"{O}/sofia_s1/{src}/runs/{proto}__*tv8cpu__s{off + i}.npz")][0], allow_pickle=True)
                b = np.load(glob.glob(f"{O}/{rt}/{proto}__*__s{off + i}.npz")[0], allow_pickle=True)
                r0 = float(re.search(r"r=([0-9.]+)", json.loads(str(b["res"]))["routed"]).group(1))
                assert (a[ki] == ix).all() and (b[ki] == ix).all()
                sh, Pu, routed = a[kp].astype(np.float64), a[ku].astype(np.float64), b[kp].astype(np.float64)
                err = np.abs((1 - r0) * sh[None] + r0 * Pu - routed[None]).max(2)
                k = err.argmin(0)
                k[np.abs(routed - sh).max(1) < 1e-7] = -1
                assert (np.where(k >= 0, err.min(0), 0) < 1e-5).all(), (m, "route not recoverable")
                parts[m] = (sh, Pu[np.maximum(k, 0), np.arange(len(ix))], k >= 0, r0)
            u = U[ix]
            grp = np.where(u == "P04", np.where((proto == "seen1") & (SES[ix] > 1), "ret_P04", "new_P04"), np.where((proto == "seen1") & (SES[ix] > 1), "ret_typ", "new_typ"))
            for mm in MS:
                mix = fmix.copy()
                for m, (sh, pk, has, r0) in parts.items():
                    r = min(1.0, mm * r0)
                    mix += W[m] * np.where(has[:, None], (1 - r) * sh + r * pk, sh)
                pred = causal_sinkhorn(mix, (ST[ix], OD[ix]), 10, 3, 0.95).argmax(1)
                rows.append(pd.DataFrame(dict(m=mm, proto=proto, i=i, split=split, u=u, grp=grp, cell=CELL[ix], y=Y[ix], ok=pred == Y[ix])))
D = pd.concat(rows, ignore_index=True)
key = ["proto", "i", "split", "u", "grp", "cell"]
ba = D.groupby(["m"] + key + ["y"]).ok.mean().groupby(["m"] + key).mean()
ref = ba[1.0]
print("level at shipped r by group:", ref.groupby("grp").mean().round(4).to_dict())
for mm in MS:
    if mm == 1.0:
        continue
    d = (ba[mm] - ref).rename("d").reset_index()
    t = d[d.u != "P04"]
    dd = t.groupby("i").d.mean()
    g = d.groupby("grp").d.mean()
    print(f"m {mm:4.2f} vs shipped: typical {dd.mean():+.4f} ({int((dd > 0).sum()):2d}/12)  W50 {(g['ret_typ'] + g['new_typ']) / 2:+.4f}  " + " ".join(f"{k} {v:+.4f}" for k, v in g.items())
          + "  | " + " ".join(f"{k} {v:+.4f}" for k, v in d.groupby("u").d.mean().items()), flush=True)
