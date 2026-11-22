#!/usr/bin/env python
"""End-phase pair choice (the manager's 20:2x ask; descriptive, zero GPU): for the candidate pairs, the expected sealed BEST-OF-2 under two
weightings of the internal paired read (outputs/t2_endphase/spec.json, SKλ, 12 draws):
  W25 = the pre-registered 'typical' (every non-P04 cell: P05 / P08 / P09 new + P10 s2-3 returning; 25 % returning cells);
  W50 = ½ returning (P10 + seen1 P04 cells) + ½ new (P05 / P08 / P09 + kit P04 cells)  (≈ the sealed composition).
Per arm: Δ vs S1 (mean over draws). Correlations between two candidates: rho_draw = corr over the 12 draws of their metric LEVELS (shared-seed
member noise; S1 not subtracted, so a shared anchor cannot inflate it) and rho_cell = corr over cells of their per-cell Δ vs S1 (shared per-user
failure profile). Sealed model: X ~ N(0.808 + tau * Δ_X, sigma²), tau 0.35, sigma = sigma_draw 0.005, corr(X, Y) = rho; E[max] closed form.
  python t2_pair_bestof2.py --export E --spec outputs/t2_endphase/spec.json [--arms S1,M6,M6pv,M7]
"""
import argparse
import json
import sys
from itertools import combinations
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

SK = dict(iters=10, n0=3, lam=0.95)
S1_SEALED, TAU, SIGMA = 0.808, 0.35, 0.005


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def per_cell(spec, arm, d):
    """-> DataFrame (draw, proto, cell, user, ba) for one arm."""
    M, w = spec["members"], spec["arms"][arm]
    strm, ordr, y, cells, subj = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy(), d.cell.to_numpy(), d.subject.str[-3:].to_numpy()
    rows = []
    for i in range(spec["n"]):
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                ix, mix = None, 0.0
                for m, wm in w.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or np.array_equal(ix, ix_), (arm, m, proto, i, split)
                    ix, mix = ix_, mix + wm * p.astype(np.float64)
                pr = causal_sinkhorn(mix, (strm[ix], ordr[ix]), SK["iters"], SK["n0"], SK["lam"]).argmax(1)
                for c in np.unique(cells[ix]):
                    k = cells[ix] == c
                    rows.append((i, proto, c, subj[ix][k][0], bal(y[ix][k], pr[k])))
    return pd.DataFrame(rows, columns=["draw", "proto", "cell", "user", "ba"])


def metrics(df):
    """per draw: W25, W50 (and the returning / new parts)."""
    out = []
    for i, g in df.groupby("draw"):
        typ = g[g.user != "P04"].ba.mean()
        ret = g[(g.user == "P10") | ((g.user == "P04") & (g.proto == "seen1"))].ba.mean()
        new = g[g.user.isin(["P05", "P08", "P09"]) | ((g.user == "P04") & (g.proto == "kit"))].ba.mean()
        out.append(dict(draw=i, W25=typ, W50=0.5 * ret + 0.5 * new, ret=ret, new=new))
    return pd.DataFrame(out).set_index("draw")


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def emax(mu1, mu2, s, rho):
    th = sqrt(max(2 * s * s * (1 - rho), 1e-12))
    dl = mu1 - mu2
    return mu1 * Phi(dl / th) + mu2 * Phi(-dl / th) + th * exp(-0.5 * (dl / th) ** 2) / sqrt(2 * pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--arms", default="S1,M6,M6pv,M7")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    d = pd.read_csv(f"{a.export}/index.csv")
    arms = a.arms.split(",")
    C = {x: per_cell(spec, x, d) for x in arms}
    Mx = {x: metrics(C[x]) for x in arms}
    res = {"model": dict(S1_sealed=S1_SEALED, tau=TAU, sigma=SIGMA)}
    for wk in ("W25", "W50"):
        dl = {x: float((Mx[x][wk] - Mx["S1"][wk]).mean()) for x in arms if x != "S1"}
        mu = {x: S1_SEALED + TAU * v for x, v in dl.items()}
        print(f"\n== {wk}: Δ vs S1 (internal) and expected sealed (single)")
        for x in dl:
            print(f"  {x:6s} Δ {dl[x]:+.4f}  -> {mu[x]:.4f}")
        res[wk] = dict(delta=dl, single=mu, pairs={})
        for x, y in combinations([v for v in arms if v != "S1"], 2):
            r_draw = float(np.corrcoef(Mx[x][wk], Mx[y][wk])[0, 1])
            cx = C[x].groupby(["proto", "cell"]).ba.mean() - C["S1"].groupby(["proto", "cell"]).ba.mean()
            cy = C[y].groupby(["proto", "cell"]).ba.mean() - C["S1"].groupby(["proto", "cell"]).ba.mean()
            r_cell = float(np.corrcoef(cx, cy)[0, 1])
            e = {f"rho={r:.2f}": round(emax(mu[x], mu[y], SIGMA, r), 4) for r in sorted({round(r_draw, 2), round(r_cell, 2), 0.5, 0.9})}
            res[wk]["pairs"][f"{x}+{y}"] = dict(rho_draw=round(r_draw, 3), rho_cell=round(r_cell, 3), E_best=e)
            print(f"  pair {x}+{y}: rho_draw {r_draw:.2f}, rho_cell {r_cell:.2f}; E[best of 2] " + ", ".join(f"{k} {v:.4f}" for k, v in e.items()))
    Path(a.spec).with_name("bestof2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
