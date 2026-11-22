#!/usr/bin/env python
"""Track 2: seed-paired comparison of two K-member probability mixes (weights + causal prior correction) on the kit split,
scored like the leaderboard (cell-BA test P04 + P09 / val P05 + P08, per participant), with the D88 bar (2).

  python t2p_mixpair.py --export E --runs R1,R2 --members A,B,C --arm "0.4,0.2,0.4" --anchor "0.6,0.2,0.2" \
      [--arm-prior 1:3 --anchor-prior 1:3] --seeds 6-11

Each member stem must resolve to exactly one file per seed across --runs (no duplicates).
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import metrics  # noqa: E402
from t2p_paired import seeds_arg  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--arm-prior", default="1:3")
    ap.add_argument("--anchor-prior", default="1:3")
    ap.add_argument("--seeds", required=True)
    args = ap.parse_args()
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    mem = args.members.split(",")
    cfg = {"arm": (np.array([float(v) for v in args.arm.split(",")]), [float(v) for v in args.arm_prior.split(":")]),
           "anchor": (np.array([float(v) for v in args.anchor.split(",")]), [float(v) for v in args.anchor_prior.split(":")])}

    def one(m, s):
        g = [f for r in args.runs.split(",") for f in glob.glob(f"{r}/{m}__s{s}.npz")]
        assert len(g) == 1, (m, s, g)
        return np.load(g[0])

    rows = []
    for s in seeds_arg(args.seeds):
        Z = [one(m, s) for m in mem]
        for split, k, ik in (("test", "prob_last", "test_idx"), ("val", "prob_val_last", "val_idx")):
            ix = Z[0][ik]
            assert all(np.array_equal(z[ik], ix) for z in Z)
            P = [z[k].astype(np.float64) for z in Z]
            for who, (w, (a, n0)) in cfg.items():
                pr = causal_prior_correction(sum(wi * Pi for wi, Pi in zip(w / w.sum(), P)), (strm[ix], ordr[ix]), a, n0).argmax(1)
                row = dict(seed=s, split=split, who=who, ba=metrics.cell_balanced_accuracy(y[ix], pr, cells[ix])[0],
                           pooled=metrics.pooled_balanced_accuracy(y[ix], pr))
                for p_ in np.unique(subj[ix]):
                    m_ = subj[ix] == p_
                    row[p_] = metrics.cell_balanced_accuracy(y[ix][m_], pr[m_], cells[ix][m_])[0]
                rows.append(row)
    df = pd.DataFrame(rows)
    pt = df.pivot_table(index="seed", columns=["split", "who"], values="ba")
    pd.set_option("display.width", 200)
    print(f"members {mem}\narm {args.arm} prior {args.arm_prior} | anchor {args.anchor} prior {args.anchor_prior}")
    print(pt.to_string(float_format=lambda v: f"{v:.4f}"))
    res = {}
    for sp in ("test", "val"):
        d = pt[(sp, "arm")] - pt[(sp, "anchor")]
        res[sp] = d.mean()
        print(f"{sp}: arm {pt[(sp, 'arm')].mean():.4f} vs anchor {pt[(sp, 'anchor')].mean():.4f} = {d.mean():+.4f} "
              f"({int((d > 0).sum())}/{len(d)} up)")
    pp = df.groupby("who")[[c for c in df.columns if c.startswith("P")]].mean()
    print("per participant:", (pp.loc["arm"] - pp.loc["anchor"]).round(4).to_dict())
    pl = df[df.split == "test"].groupby("who")["pooled"].mean()
    print(f"pooled test: arm {pl['arm']:.4f} vs anchor {pl['anchor']:.4f}")
    print(f"D88 bar (2): {'MET' if res['test'] >= 0.01 and res['val'] >= 0 else 'not met'}")


if __name__ == "__main__":
    main()
