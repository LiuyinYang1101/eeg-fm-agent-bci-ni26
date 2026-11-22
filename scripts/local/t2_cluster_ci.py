#!/usr/bin/env python
"""Participant-clustered uncertainty for a package-shape read (the external review G1 item 2, 2026-11-05): an arm vs its anchor under SKλ
(10 iters, n0 3, λ 0.95), kit + seen1, test + val splits, the same cells as t2_d320_rescreen.py. Reports:
  - per user: mean Δ cell BA over draws (and draws up);
  - typical (P05 P08 P09 P10): the rescreen's cell-weighted Δ, a user-cluster bootstrap 90 % interval (2,000 resamples of the 4 users), and LOPO
    (typical Δ with each user left out);
  - groups: returning (P10 + P04's later sessions in seen1), new (P05 P08 P09 + P04 in kit), P04 alone.
  python scripts/local/t2_cluster_ci.py <spec.json> <anchor> <arm> [--n 12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tracks" / "t2_bci" / "scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def cell_rows(J, arm, n, idx, protos=("kit", "seen1")):
    rows = []
    for proto in protos:
        for i in range(n):
            for split in ("test", "val"):
                mix, ix = 0.0, None
                for m, w in J["arms"][arm].items():
                    M = J["members"][m]
                    z = np.load(M[proto].format(s=M.get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                    ix, mix = ix_, mix + w * p.astype(np.float64)
                P = causal_sinkhorn(mix, (idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]), 10, 3, 0.95).argmax(1)
                y, cells = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix]
                for c in np.unique(cells):
                    mk = cells == c
                    rows.append(dict(proto=proto, split=split, i=i, cell=c, subj=idx.subject.str[-3:].to_numpy()[ix][mk][0],
                                     ses=str(idx.session.to_numpy()[ix][mk][0]),
                                     ba=np.mean([np.mean(P[mk][y[mk] == k] == k) for k in np.unique(y[mk])])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("anchor")
    ap.add_argument("arm")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    J = json.loads(Path(a.spec).read_text())
    idx = pd.read_csv(EXP / "index.csv")
    A, B = cell_rows(J, a.anchor, a.n, idx), cell_rows(J, a.arm, a.n, idx)
    key = ["proto", "split", "i", "cell", "subj", "ses"]
    D = A.merge(B, on=key, suffixes=("_a", "_b"))
    D["d"] = D.ba_b - D.ba_a
    per_user = D.groupby(["subj", "i"]).d.mean().groupby("subj")
    print("per user: mean Δ (draws up / 12)")
    for u, s in per_user:
        print(f"  {u}: {s.mean():+.4f} ({int((s > 0).sum())} / {a.n})")
    typ = ["P05", "P08", "P09", "P10"]
    T = D[D.subj.isin(typ)]
    td = T.groupby("i").d.mean()                                   # the rescreen's typical: cell-weighted, per draw
    cells_u = {u: T[T.subj == u] for u in typ}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        pick = rng.choice(typ, size=len(typ), replace=True)
        boot.append(pd.concat([cells_u[u] for u in pick]).d.mean())
    lo, hi = np.percentile(boot, [5, 95])
    print(f"typical: Δ {td.mean():+.4f} ({int((td > 0).sum())} / {a.n} draws up); user-cluster bootstrap 90 % [{lo:+.4f}, {hi:+.4f}]")
    print("  LOPO (typical without one user): " + ", ".join(f"-{u} {T[T.subj != u].d.mean():+.4f}" for u in typ))
    ret = D[(D.subj == "P10") | ((D.subj == "P04") & (D.proto == "seen1"))]
    new = D[D.subj.isin(["P05", "P08", "P09"]) | ((D.subj == "P04") & (D.proto == "kit"))]
    print(f"returning (P10 + seen1 P04): {ret.d.mean():+.4f}; new (P05 P08 P09 + kit P04): {new.d.mean():+.4f}; P04 alone: {D[D.subj == 'P04'].d.mean():+.4f}")
    w50 = 0.5 * ret.groupby("i").d.mean() + 0.5 * new.groupby("i").d.mean()   # the manager 2026-11-07: W50 co-primary (descriptive) next to W25
    print(f"W50 (½ returning + ½ new, ≈ the sealed composition; co-primary, descriptive): Δ {w50.mean():+.4f} ({int((w50 > 0).sum())} / {a.n} draws up)")


if __name__ == "__main__":
    main()
