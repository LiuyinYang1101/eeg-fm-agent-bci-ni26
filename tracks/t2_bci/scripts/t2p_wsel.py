#!/usr/bin/env python
"""Track 2: VAL-ONLY weight selection for a K-member probability package (T2-D99 / T2-D116 rule). Prints the mean val
cell-BA (P05 + P08, prior correction after the mix) over --seeds for every weight vector; never reads test probabilities.

  python t2p_wsel.py --export E --runs R1,R2 --members A,B,C --weights "0.8,0.2,0;0.7,0.2,0.1" --seeds 0-4
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
    ap.add_argument("--weights", required=True, help="';'-separated weight vectors, one weight per member")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--priors", default="1:3", help="';'-separated alpha:n0 pairs of the causal prior correction (T2-D120)")
    args = ap.parse_args()
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    mem = args.members.split(",")
    W = [np.array([float(v) for v in w.split(",")]) for w in args.weights.split(";")]
    assert all(len(w) == len(mem) for w in W)
    files = {}
    for m in mem:
        d = {}
        for r in args.runs.split(","):
            for f in glob.glob(f"{r}/{m}__s*.npz"):
                if not f.endswith(".tmp.npz"):
                    s = int(Path(f).stem.split("__s")[-1])
                    assert s not in d, (m, s)
                    d[s] = f
        files[m] = d
    rows = []
    for s in seeds_arg(args.seeds):
        Z = [np.load(files[m][s]) for m in mem]
        va = Z[0]["val_idx"]
        assert all(np.array_equal(z["val_idx"], va) for z in Z)
        P = [z["prob_val_last"].astype(np.float64) for z in Z]
        for w in W:
            mix = sum(wi * Pi for wi, Pi in zip(w / w.sum(), P))
            for pa in args.priors.split(";"):
                a, n0 = (float(v) for v in pa.split(":"))
                pr = causal_prior_correction(mix, (strm[va], ordr[va]), a, n0).argmax(1)
                rows.append(dict(seed=s, w=",".join(f"{v:g}" for v in w) + f" prior {pa}",
                                 val=metrics.cell_balanced_accuracy(y[va], pr, cells[va])[0]))
    df = pd.DataFrame(rows)
    out = df.groupby("w", sort=False)["val"].agg(["mean", "std", "count"])
    print(f"members {mem}; seeds {seeds_arg(args.seeds)}; VAL only")
    print(out.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
