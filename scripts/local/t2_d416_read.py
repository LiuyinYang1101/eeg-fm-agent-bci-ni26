#!/usr/bin/env python
"""T2-D416 (pre-registered in docs/DECISIONS_T2.md before any D416 number; upgrade-all step 1, zero GPU, under the user's R2 typed 2026-11-09):
CROSS-FITTED RE-WEIGHTING of a stack's OWN members (no member change, so the ZIP's members and runtime are unchanged; a config repack).
  - members + shipped weights: M7 = outputs/t2_d408/spec_m7.json arm "M7" (decision arm M7rw); M6 = outputs/t2_d378/spec.json arm "S3f" (M6rw, reported);
  - fit: member weights on the simplex (softmax parametrisation, L-BFGS) minimising the CELL-BALANCED log loss of the linear pool
    sum_m w_m p_m on ONE protocol (every cell, test + val splits, all 12 draws pooled; each cell's windows weighted 1 / n_cell);
  - cross-fit: kit cells are scored with the weights fitted on seen1 and seen1 cells with the weights fitted on kit; causal SKλ (10, n0 3, λ 0.95);
  - vs the shipped weights on the same cells: typical (P05 P08 P09 P10; cell-weighted per draw) Δ and draws up, P04, P10, newT, per user,
    returning / new / W50 as t2_cluster_ci;
  - BAR (decision arm M7rw): typical >= +0.003 with >= 9 / 12 AND W50 >= 0 AND no single user below -0.020 (R2);
  - the full-fit weights (both protocols) are reported = the weights a pass would repack.
    mem-guard python scripts/local/t2_d416_read.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
R = O / "t2_d416"
R.mkdir(exist_ok=True)
idx = pd.read_csv(E / "index.csv")
STRM, ORDR, Y, CELL, SUBJ = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy()
N = 12


def load(spec, arm):
    J = json.load(open(spec))
    W = J["arms"][arm]
    names = list(W)
    D = {}
    for proto in ("kit", "seen1"):
        for i in range(N):
            for split in ("test", "val"):
                ps, ix = [], None
                for m in names:
                    M = J["members"][m]
                    assert M.get("temp", 1) == 1, m
                    z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                    ix = ix_
                    ps.append(p.astype(np.float64))
                D[(proto, i, split)] = (ix, np.stack(ps))                      # (n_members, n_windows, 3)
    return names, np.array([W[m] for m in names]), D


def fit(D, protos, n_m):
    P_true, wts = [], []
    for (proto, i, split), (ix, ps) in D.items():
        if proto not in protos:
            continue
        P_true.append(ps[:, np.arange(len(ix)), Y[ix]].T)                     # (n_windows, n_members): p_m(y)
        cells = CELL[ix]
        _, inv, cnt = np.unique(cells, return_inverse=True, return_counts=True)
        wts.append(1.0 / cnt[inv])
    Pt, wt = np.concatenate(P_true), np.concatenate(wts)
    wt = wt / wt.sum()

    def f(th):
        w = np.exp(th - th.max()); w /= w.sum()
        q = Pt @ w
        loss = -(wt * np.log(np.clip(q, 1e-12, None))).sum()
        g_w = -(wt[:, None] * Pt / np.clip(q, 1e-12, None)[:, None]).sum(0)
        return loss, w * (g_w - (w * g_w).sum())
    r = minimize(f, np.zeros(n_m), jac=True, method="L-BFGS-B")
    w = np.exp(r.x - r.x.max())
    return w / w.sum()


def cells_df(D, w_by_proto):
    rows = []
    for (proto, i, split), (ix, ps) in D.items():
        mix = np.tensordot(w_by_proto[proto], ps, 1)
        P = causal_sinkhorn(mix, (STRM[ix], ORDR[ix]), 10, 3, 0.95).argmax(1)
        for c in np.unique(CELL[ix]):
            mk = CELL[ix] == c
            yy = Y[ix][mk]
            rows.append(dict(proto=proto, split=split, i=i, cell=c, subj=SUBJ[ix][mk][0], ba=np.mean([np.mean(P[mk][yy == k] == k) for k in np.unique(yy)])))
    return pd.DataFrame(rows)


def compare(A, B):
    key = ["proto", "split", "i", "cell", "subj"]
    Dd = A.merge(B, on=key, suffixes=("_a", "_b"))
    Dd["d"] = Dd.ba_b - Dd.ba_a
    out = {}
    typ = Dd[Dd.subj.isin(["P05", "P08", "P09", "P10"])].groupby("i").d.mean()
    out["typical"], out["typical_up"] = float(typ.mean()), int((typ > 0).sum())
    for g, mk in (("P04", Dd.subj == "P04"), ("P10", Dd.subj == "P10"), ("newT", Dd.subj.isin(["P05", "P08", "P09"]))):
        s = Dd[mk].groupby("i").d.mean()
        out[g], out[g + "_up"] = float(s.mean()), int((s > 0).sum())
    out["users"] = {u: float(v) for u, v in Dd.groupby(["subj", "i"]).d.mean().groupby("subj").mean().items()}
    ret = Dd[(Dd.subj == "P10") | ((Dd.subj == "P04") & (Dd.proto == "seen1"))]
    new = Dd[Dd.subj.isin(["P05", "P08", "P09"]) | ((Dd.subj == "P04") & (Dd.proto == "kit"))]
    w50 = 0.5 * ret.groupby("i").d.mean() + 0.5 * new.groupby("i").d.mean()
    out.update(returning=float(ret.d.mean()), new=float(new.d.mean()), W50=float(w50.mean()), W50_up=int((w50 > 0).sum()),
               level_a=float(A[A.subj.isin(["P05", "P08", "P09", "P10"])].groupby("i").ba.mean().mean()))
    return out


res = {}
for tag, spec, arm in (("M7rw", O / "t2_d408/spec_m7.json", "M7"), ("M6rw", O / "t2_d378/spec.json", "S3f")):
    names, w0, D = load(spec, arm)
    w_kit, w_seen1, w_all = fit(D, {"kit"}, len(names)), fit(D, {"seen1"}, len(names)), fit(D, {"kit", "seen1"}, len(names))
    base = cells_df(D, {"kit": w0, "seen1": w0})
    xfit = cells_df(D, {"kit": w_seen1, "seen1": w_kit})
    r = compare(base, xfit)
    r["weights"] = {"shipped": dict(zip(names, w0.round(4).tolist())), "fit_kit": dict(zip(names, w_kit.round(4).tolist())),
                    "fit_seen1": dict(zip(names, w_seen1.round(4).tolist())), "fit_all": dict(zip(names, w_all.round(4).tolist()))}
    r["insample_all"] = {k: v for k, v in compare(base, cells_df(D, {"kit": w_all, "seen1": w_all})).items() if k in ("typical", "typical_up", "W50", "P04")}
    res[tag] = r
    print(f"== {tag} (cross-fitted vs shipped weights): typical {r['typical']:+.4f} ({r['typical_up']} / 12), P04 {r['P04']:+.4f}, P10 {r['P10']:+.4f}, "
          f"newT {r['newT']:+.4f}, returning {r['returning']:+.4f}, new {r['new']:+.4f}, W50 {r['W50']:+.4f} ({r['W50_up']} / 12); "
          f"users {', '.join(f'{u} {v:+.4f}' for u, v in sorted(r['users'].items()))}; level {r['level_a']:.4f}")
    for k, v in r["weights"].items():
        print(f"   {k}: {v}")
    print(f"   in-sample (full fit, descriptive): {r['insample_all']}")
json.dump(res, open(R / "read.json", "w"), indent=1)
r = res["M7rw"]
ok = r["typical"] >= 0.003 and r["typical_up"] >= 9 and r["W50"] >= 0 and min(r["users"].values()) >= -0.020
print(f"== VERDICT (M7rw; typical >= +0.003 on >= 9 / 12, W50 >= 0, no user < -0.020): {'PASS -> repack M7 with fit_all weights + gates' if ok else 'NOT MET'}")
