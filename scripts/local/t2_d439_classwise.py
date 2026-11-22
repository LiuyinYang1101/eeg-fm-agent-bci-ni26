#!/usr/bin/env python
"""T2-D439 (pre-registered in docs/DECISIONS_T2.md before any number; zero GPU): CLASS-WISE stacking of M7's member groups.
Step 0 (gate): per family (FBCNet / ShallowNet / IFNet / the five ST-L) and class, recall of the family ALONE under the causal SKλ decode on
  non-P04 cells (kit + seen1, test + val, 12 draws); interaction I[g, c] = R[g, c] - R[g, .] - R[., c] + R[., .]. Fit only iff max |I| >= 0.03.
Step 1 (fit): mix[c] = sum_g w_g a[g, c] p_g[c], w = M7's shipped shares, a = 1 is M7 exactly; log a in [-log 1.5, log 1.5] (a priori),
  fitted by L-BFGS-B on the cell-balanced log loss of the normalised mix, LEAVE-ONE-USER-OUT: user u's rows are mixed with the a fitted on
  the other four users' rows (both protocols, test + val) of the same draw; then SKλ, cell BA, vs M7 through the same code.
BAR: typical (non-P04) >= +0.003 with >= 9 / 12 draws up and no non-P04 user < -0.01.
    mem-guard python scripts/local/t2_d439_classwise.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d408/spec_m7.json"))
M7 = J["arms"]["M7"]
GRP = {"FBC": ["fbc_tv8c"], "SH": ["sh_tv8c"], "IFN": ["ifn"], "STL": [k for k in M7 if k not in ("fbc_tv8c", "sh_tv8c", "ifn")]}
W = {g: sum(M7[m] for m in ms) for g, ms in GRP.items()}
assert abs(sum(W.values()) - 1) < 1e-9 and len(GRP["STL"]) == 5, W
idx = pd.read_csv(EXP / "index.csv")
N = 12


def load(proto, split, i):
    P, ix = {}, None
    for g, ms in GRP.items():
        acc = 0.0
        for m in ms:
            Mm = J["members"][m]
            z = np.load(Mm[proto].format(s=Mm.get("offset", 0) + i))
            ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
            acc = acc + M7[m] / W[g] * p.astype(np.float64)
        P[g] = acc
    d = idx.iloc[ix]
    return dict(P=P, y=d.y.to_numpy(), cell=d.cell.to_numpy(), u=d.subject.str[-3:].to_numpy(), key=(d.stream.to_numpy(), d.order.to_numpy()))


def decode(mix, key):
    return causal_sinkhorn(mix, key, 10, 3, 0.95).argmax(1)


def cell_ba(pred, B):
    return {(c, B["u"][B["cell"] == c][0]): np.mean([np.mean(pred[(B["cell"] == c) & (B["y"] == k)] == k) for k in np.unique(B["y"][B["cell"] == c])])
            for c in np.unique(B["cell"])}


BL = {(p, s, i): load(p, s, i) for p in ("kit", "seen1") for s in ("test", "val") for i in range(N)}
G = list(GRP)
# ---- step 0
R = np.zeros((len(G), 3))
for gi, g in enumerate(G):
    rec = []
    for B in BL.values():
        pr = decode(B["P"][g], B["key"])
        nt = B["u"] != "P04"
        rec.append([np.mean(pr[nt & (B["y"] == k)] == k) for k in range(3)])
    R[gi] = np.mean(rec, 0)
I = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
print("step 0: recall per class of each family alone (SKλ, non-P04 rows, mean over blocks); classes 0 / 1 / 2 as index.y")
for gi, g in enumerate(G):
    print(f"  {g:4s} share {W[g]:.3f}  recall {np.round(R[gi], 3).tolist()}  mean {R[gi].mean():.3f}  interaction {np.round(I[gi], 3).tolist()}")
gate = float(np.abs(I).max())
print(f"  max |interaction| = {gate:.4f} (gate 0.03)")
if gate < 0.03:
    print("== VERDICT (pre-registered): CLOSED BY THE TABLE (no family x class structure >= 0.03); no fit")
    sys.exit(0)
# ---- step 1
LIM = np.log(1.5)


def mixf(B, la, rows=slice(None)):
    a = np.exp(la.reshape(len(G), 3))
    return sum(W[g] * a[gi] * B["P"][g][rows] for gi, g in enumerate(G))


def fit(blocks, held):
    Ps, ys, ws = [], [], []
    for B in blocks:
        r = B["u"] != held
        _, inv, cnt = np.unique(B["cell"][r].astype(str) + "|" + B["y"][r].astype(str), return_inverse=True, return_counts=True)
        Ps.append({g: B["P"][g][r] for g in G}); ys.append(B["y"][r]); ws.append(1.0 / cnt[inv])
    y, w = np.concatenate(ys), np.concatenate(ws)
    PG = np.stack([W[g] * np.concatenate([p[g] for p in Ps]) for g in G])        # (G, n, 3)
    oh = np.eye(3)[y]

    def f(la):
        a = np.exp(la.reshape(len(G), 3))
        m = np.einsum("gc,gnc->nc", a, PG)
        s = m.sum(1)
        py = (m * oh).sum(1)
        loss = -(w * np.log(py / s)).sum() / w.sum()
        dm = (w / w.sum())[:, None] * (-(oh / py[:, None]) + 1.0 / s[:, None])   # dloss / dm
        return loss, (np.einsum("nc,gnc->gc", dm, PG) * a).ravel()

    r = minimize(f, np.zeros(len(G) * 3), jac=True, method="L-BFGS-B", bounds=[(-LIM, LIM)] * (len(G) * 3))
    return r.x


rows, A = [], []
for i in range(N):
    blocks = [BL[(p, s, i)] for p in ("kit", "seen1") for s in ("test", "val")]
    la = {u: fit(blocks, u) for u in ("P04", "P05", "P08", "P09", "P10")}
    A.append(np.mean([np.exp(la[u]).reshape(len(G), 3) for u in la], 0))
    for (p, s, ii), B in BL.items():
        if ii != i:
            continue
        base = mixf(B, np.zeros(len(G) * 3))
        new = base.copy()
        for u in la:
            r = B["u"] == u
            if r.any():
                new[r] = mixf(B, la[u], r)
        new = new / new.sum(1, keepdims=True) * base.sum(1, keepdims=True)
        b0, b1 = cell_ba(decode(base, B["key"]), B), cell_ba(decode(new, B["key"]), B)
        rows += [dict(i=i, proto=p, subj=k[1], d=b1[k] - b0[k], ba=b0[k]) for k in b0]
D = pd.DataFrame(rows)
print("\nstep 1: fitted multipliers a[g, c] (mean over draws and held-out users; 1 = M7; bounds 0.667 … 1.5)")
for gi, g in enumerate(G):
    print(f"  {g:4s} {np.round(np.mean(A, 0)[gi], 3).tolist()}   sd over draws {np.round(np.std(A, 0)[gi], 3).tolist()}")
print(f"M7 level through this code: typical {D[D.subj != 'P04'].ba.mean():.4f}")
pu = D.groupby(["subj", "i"]).d.mean().groupby("subj")
print("per user Δ (draws up): " + ", ".join(f"{u} {s.mean():+.4f} ({int((s > 0).sum())})" for u, s in pu))
td = D[D.subj != "P04"].groupby("i").d.mean()
ret = D[(D.subj == "P10") | ((D.subj == "P04") & (D.proto == "seen1"))].groupby("i").d.mean()
new = D[D.subj.isin(["P05", "P08", "P09"]) | ((D.subj == "P04") & (D.proto == "kit"))].groupby("i").d.mean()
w50 = 0.5 * ret + 0.5 * new
print(f"typical Δ {td.mean():+.4f} ({int((td > 0).sum())} / {N}); W50 {w50.mean():+.4f} ({int((w50 > 0).sum())} / {N}); P04 {D[D.subj == 'P04'].d.mean():+.4f}")
worst = min(s.mean() for u, s in pu if u != "P04")
ok = td.mean() >= 0.003 and (td > 0).sum() >= 9 and worst >= -0.01
print("== VERDICT (pre-registered): " + ("BAR MET -> fresh-seed replicate + template pricing" if ok else "NOT MET"))
