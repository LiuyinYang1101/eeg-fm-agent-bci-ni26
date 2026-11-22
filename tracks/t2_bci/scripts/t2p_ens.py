#!/usr/bin/env python
"""Seed-paired probability ensembles of t2p_bench runs on the competition (kit) split, scored like the leaderboard.

A member is a bench result-file stem without its seed suffix, e.g. 'kit__fbcnet__none__n00__r1' or
'kit__fm-mor_p1v2-mean__wz__n00__fmE30b64le0.0001lh0.001wd0.05ll1ema0dp0'. For every seed present in ALL members,
the members' test (and saved val) probabilities are averaged with --weights, optionally re-balanced by the causal output
prior correction (--tta prior:1:3, the D88 default), and scored: test = P04 + P09 cell-BA / pooled BA / acc (the
leaderboard), val = P05 + P08 cell-BA. Prints per-seed means and the all-seed ensemble, members alone beside.

Usage: python t2p_ens.py --export <dir> --runs <dirs> --members A,B[,C] [--weights 1,1] [--tta prior:1:3]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import metrics  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--members", required=True, help="comma list of result-file stems without __s<seed>.npz")
    ap.add_argument("--weights", default="")
    ap.add_argument("--tta", default="prior:1:3")
    ap.add_argument("--seeds", default="", help="comma list: score only these seeds (e.g. a pre-registered fresh-seed set)")
    args = ap.parse_args()
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    tta = None
    if args.tta.startswith("prior"):
        p_ = args.tta.split(":")
        tta = (float(p_[1]), float(p_[2]))
    mem = args.members.split(",")
    w = np.array([float(v) for v in args.weights.split(",")]) if args.weights else np.ones(len(mem))
    w = w / w.sum()
    files = {}
    for m in mem:
        files[m] = {}
        for d in args.runs.split(","):
            for f in glob.glob(str(Path(d) / f"{m}__s*.npz")):
                if f.endswith(".tmp.npz"):
                    continue
                files[m][int(Path(f).stem.split("__s")[-1])] = f
        assert files[m], f"no result files for member {m}"
    seeds = sorted(set.intersection(*[set(v) for v in files.values()]))
    if args.seeds:
        want = {int(v) for v in args.seeds.split(",")}
        assert want <= set(seeds), f"seeds {sorted(want - set(seeds))} missing for some member"
        seeds = sorted(want)
    assert seeds, "no seed shared by all members"

    def score(P, ix):
        Pc = causal_prior_correction(P, (strm[ix], ordr[ix]), *tta) if tta else P
        pr = Pc.argmax(1)
        return (metrics.cell_balanced_accuracy(y[ix], pr, cells[ix])[0], metrics.pooled_balanced_accuracy(y[ix], pr),
                float(np.mean(y[ix] == pr)))

    rows = []
    sums = {k: None for k in ["ens"] + mem}
    vsums = {k: None for k in ["ens"] + mem}
    for s in seeds:
        Z = {m: np.load(files[m][s]) for m in mem}
        te, va = Z[mem[0]]["test_idx"], Z[mem[0]]["val_idx"]
        for m in mem:
            assert np.array_equal(Z[m]["test_idx"], te), f"{m}: test windows differ"
        P = {m: Z[m]["prob_last"].astype(np.float64) for m in mem}
        P["ens"] = sum(wi * P[m] for wi, m in zip(w, mem))
        has_val = all(len(Z[m]["prob_val_last"]) for m in mem) and all(np.array_equal(Z[m]["val_idx"], va) for m in mem)
        V = {m: Z[m]["prob_val_last"].astype(np.float64) for m in mem} if has_val else {}
        if has_val:
            V["ens"] = sum(wi * V[m] for wi, m in zip(w, mem))
        for k in ["ens"] + mem:
            t = score(P[k], te)
            v = score(V[k], va)[0] if has_val else np.nan
            rows.append(dict(seed=s, who=k, test=t[0], pooled=t[1], acc=t[2], val=v))
            sums[k] = P[k] if sums[k] is None else sums[k] + P[k]
            if has_val:
                vsums[k] = V[k] if vsums[k] is None else vsums[k] + V[k]
    df = pd.DataFrame(rows)
    out = df.groupby("who", sort=False)[["test", "pooled", "acc", "val"]].mean()
    out["test_sd"] = df.groupby("who", sort=False)["test"].std()
    out["allseed_test"] = [score(sums[k], te)[0] for k in out.index]
    out["allseed_val"] = [score(vsums[k], va)[0] if vsums[k] is not None else np.nan for k in out.index]
    d = df.pivot(index="seed", columns="who", values="test")
    out["seeds_ens_beats_member"] = [f"{int((d['ens'] > d[k]).sum())}/{len(d)}" if k != "ens" else "" for k in out.index]
    pd.set_option("display.width", 250)
    print(f"seeds {seeds}; weights {dict(zip(mem, np.round(w, 3)))}; tta {args.tta or 'none'}")
    print(out.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
