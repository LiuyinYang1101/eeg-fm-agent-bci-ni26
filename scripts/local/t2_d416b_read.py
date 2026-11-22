#!/usr/bin/env python
"""T2-D416b (pre-registered in docs/DECISIONS_T2.md before any D416b number; zero GPU; the user's R2): re-weighting of M7's own member GROUPS fitted
ON THE METRIC ITSELF (T2-D416's log-loss objective lost even in-sample: an objective mismatch, the manager's flag).
  - groups (M7 = outputs/t2_d408/spec_m7.json arm M7): FBCNet (fbc_tv8c), ShallowNet (sh_tv8c), IFNet (ifn), ST-L (lp10 + e100f0-f3, the group share
    split EQUALLY over its 5 members, as shipped); shipped shares 0.3143 / 0.2357 / 0.1000 / 0.3500;
  - grid: every (f, s, i, t) >= 0 summing to 1 in 0.05 steps (1,771 points) + the shipped point;
  - objective on a protocol: the mean over the 12 draws of the cell-mean balanced accuracy over ALL that protocol's cells (test + val), decoded by the
    shipped causal SKλ (10 it, n0 3, λ 0.95); ties -> the point nearest the shipped shares (L1);
  - cross-fit: kit cells scored with the seen1-argmax shares, seen1 cells with the kit-argmax shares; vs the shipped shares on the same cells;
  - SANITY (pre-registered): the in-sample argmax over BOTH protocols must beat the shipped shares on that objective, else the fit is broken (reported, no verdict);
  - BAR: typical >= +0.003 with >= 9 / 12 AND W50 >= 0 AND no single user below -0.020 (R2); a pass -> repack M7 with the all-fit shares (config only).
    mem-guard python scripts/local/t2_d416b_read.py
"""
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
R = O / "t2_d416b"
R.mkdir(exist_ok=True)
idx = pd.read_csv(E / "index.csv")
STRM, ORDR, Y, CELL, SUBJ = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy()
N = 12
GROUPS = {"FB": ["fbc_tv8c"], "SH": ["sh_tv8c"], "IF": ["ifn"], "ST": ["lp10", "e100f0", "e100f1", "e100f2", "e100f3"]}
J = json.load(open(O / "t2_d408/spec_m7.json"))
W7 = J["arms"]["M7"]
assert sorted(W7) == sorted(sum(GROUPS.values(), [])) and all(abs(W7[m] - 0.07) < 1e-9 for m in GROUPS["ST"])
SHIP = np.array([sum(W7[m] for m in GROUPS[g]) for g in GROUPS])
D = {}


def load():
    for proto in ("kit", "seen1"):
        for i in range(N):
            for split in ("test", "val"):
                gp, ix = [], None
                for g, ms in GROUPS.items():
                    acc = 0.0
                    for m in ms:
                        M = J["members"][m]
                        z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                        ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                        assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                        ix, acc = ix_, acc + p.astype(np.float64) / len(ms)
                    gp.append(acc)
                D[(proto, i, split)] = (ix, np.stack(gp))


def cell_ba(proto_i_split, w):
    ix, gp = D[proto_i_split]
    P = causal_sinkhorn(np.tensordot(w, gp, 1), (STRM[ix], ORDR[ix]), 10, 3, 0.95).argmax(1)
    out = []
    for c in np.unique(CELL[ix]):
        mk = CELL[ix] == c
        yy = Y[ix][mk]
        out.append((c, SUBJ[ix][mk][0], float(np.mean([np.mean(P[mk][yy == k] == k) for k in np.unique(yy)]))))
    return out


def objective(w):
    w = np.asarray(w, dtype=np.float64)
    res = {}
    for proto in ("kit", "seen1"):
        per_i = [np.mean([b for (_, _, b) in sum((cell_ba((proto, i, s), w) for s in ("test", "val")), [])]) for i in range(N)]
        res[proto] = float(np.mean(per_i))
    return res


def init():
    load()


def cells_df(w_by_proto):
    rows = []
    for (proto, i, split) in D:
        for c, u, b in cell_ba((proto, i, split), w_by_proto[proto]):
            rows.append(dict(proto=proto, split=split, i=i, cell=c, subj=u, ba=b))
    return pd.DataFrame(rows)


def compare(A, B):
    Dd = A.merge(B, on=["proto", "split", "i", "cell", "subj"], suffixes=("_a", "_b"))
    Dd["d"] = Dd.ba_b - Dd.ba_a
    out = {}
    typ = Dd[Dd.subj.isin(["P05", "P08", "P09", "P10"])].groupby("i").d.mean()
    out["typical"], out["typical_up"] = float(typ.mean()), int((typ > 0).sum())
    for g, mk in (("P04", Dd.subj == "P04"), ("P10", Dd.subj == "P10"), ("newT", Dd.subj.isin(["P05", "P08", "P09"]))):
        out[g] = float(Dd[mk].groupby("i").d.mean().mean())
    out["users"] = {u: float(v) for u, v in Dd.groupby(["subj", "i"]).d.mean().groupby("subj").mean().items()}
    ret = Dd[(Dd.subj == "P10") | ((Dd.subj == "P04") & (Dd.proto == "seen1"))]
    new = Dd[Dd.subj.isin(["P05", "P08", "P09"]) | ((Dd.subj == "P04") & (Dd.proto == "kit"))]
    w50 = 0.5 * ret.groupby("i").d.mean() + 0.5 * new.groupby("i").d.mean()
    out.update(W50=float(w50.mean()), W50_up=int((w50 > 0).sum()))
    return out


if __name__ == "__main__":
    grid = [np.array(p) / 20 for p in itertools.product(range(21), repeat=4) if sum(p) == 20]
    pts = [SHIP] + grid
    with Pool(12, initializer=init) as P:
        objs = P.map(objective, pts, chunksize=8)
    load()
    K = np.array([o["kit"] for o in objs]); S = np.array([o["seen1"] for o in objs]); A = (K + S) / 2
    dist = np.array([np.abs(p - SHIP).sum() for p in pts])

    def argmax(v):
        best = v.max()
        cand = np.flatnonzero(v >= best - 1e-12)
        return int(cand[np.argmin(dist[cand])])
    ik, is_, ia = argmax(K), argmax(S), argmax(A)
    names = list(GROUPS)
    show = lambda j: dict(zip(names, np.round(pts[j], 4).tolist()))  # noqa: E731
    print(f"grid {len(pts)} points; shipped objective kit {K[0]:.4f} seen1 {S[0]:.4f} both {A[0]:.4f}")
    print(f"fit kit {show(ik)} (kit obj {K[ik]:.4f}); fit seen1 {show(is_)} (seen1 obj {S[is_]:.4f}); fit both {show(ia)} (both obj {A[ia]:.4f})")
    sane = A[ia] > A[0] + 1e-12
    print(f"== SANITY (in-sample both-fit beats the shipped shares): {'ok' if sane else 'BROKEN'} ({A[ia] - A[0]:+.4f})")
    base = cells_df({"kit": SHIP, "seen1": SHIP})
    r = compare(base, cells_df({"kit": pts[is_], "seen1": pts[ik]}))
    print(f"== CROSS-FIT vs shipped: typical {r['typical']:+.4f} ({r['typical_up']} / 12), P04 {r['P04']:+.4f}, P10 {r['P10']:+.4f}, newT {r['newT']:+.4f}, "
          f"W50 {r['W50']:+.4f} ({r['W50_up']} / 12); users {', '.join(f'{u} {v:+.4f}' for u, v in sorted(r['users'].items()))}")
    ok = sane and r["typical"] >= 0.003 and r["typical_up"] >= 9 and r["W50"] >= 0 and min(r["users"].values()) >= -0.020
    json.dump(dict(shipped=dict(zip(names, SHIP.round(4).tolist())), fit_kit=show(ik), fit_seen1=show(is_), fit_both=show(ia), sane=bool(sane), xfit=r,
                   obj_shipped=dict(kit=K[0], seen1=S[0]), passed=bool(ok)), open(R / "read.json", "w"), indent=1)
    print(f"== VERDICT (pre-registered): {'PASS -> repack M7 with the both-fit shares + gates' if ok else ('NO VERDICT (fit broken)' if not sane else 'NOT MET')}")
