#!/usr/bin/env python
"""Track 2: seed-paired arm vs anchor on the leaderboard metric (cell-BA, last epoch, causal prior correction a 1, n0 3),
for the kit split and the T2-D111 seen-user protocol (seen1). Supersedes the scratchpad paired.py (validated on D97).

  python t2p_paired.py --export E --runs R1,R2 --arm STEM --anchor STEM --seeds 0-11 \
      [--pkg STEM --role fbc|st] [--bar d88|seen|cpt]

STEM = a bench result-file stem without __s<seed>.npz (its first token names the protocol). Groups:
  kit    test = P04 + P09 (the leaderboard), val = P05 + P08
  seen1  seen = P04 + P10 sessions 2-3 (8 cells), p09 = P09 (2 cells), val = P05 + P08, unseen = P05 P08 P09 (6 cells)
--pkg: also score the package shape 0.8 FBCNet + 0.2 ST-L, the arm / anchor taking --role and STEM the other member
(same seed). Prior correction runs per stream on the mixed probabilities, as in the package.
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
from t2p_report import causal_prior_correction  # noqa: E402

GROUPS = {"kit": ("test", "val"), "kitp10": ("test", "val"), "seen1": ("seen", "p09", "val", "unseen")}
BARS = {  # (group, min delta) pairs that must all hold
    "d88": [("test", 0.01), ("val", 0.0)],
    "seen": [("seen", 0.01)],
    "cpt": [("seen", 0.01), ("unseen", -0.005)],
}


def seeds_arg(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--pkg", default="")
    ap.add_argument("--role", default="fbc", choices=("fbc", "st"))
    ap.add_argument("--w-st", type=float, default=0.2)
    ap.add_argument("--bar", default="", choices=("",) + tuple(BARS))
    ap.add_argument("--quiet", action="store_true", help="no per-seed table")
    args = ap.parse_args()
    seeds = seeds_arg(args.seeds)
    proto = args.arm.split("__")[0]
    assert proto == args.anchor.split("__")[0] and (not args.pkg or args.pkg.split("__")[0] == proto), "protocols differ"
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj, ses = idx["subject"].to_numpy(), idx["session"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()

    def files(stem):
        d = {}
        for r in args.runs.split(","):
            for f in glob.glob(f"{r}/{stem}__s*.npz"):
                if f.endswith(".tmp.npz"):
                    continue
                s = int(Path(f).stem.split("__s")[-1])
                assert s not in d, (stem, s, f, d.get(s))
                d[s] = f
        return d

    F = {k: files(v) for k, v in (("arm", args.arm), ("anchor", args.anchor), ("pkg", args.pkg)) if v}
    miss = {k: [s for s in seeds if s not in v] for k, v in F.items()}
    assert not any(miss.values()), f"missing seeds {miss}"

    def full(z):
        P = np.full((len(idx), 3), np.nan)
        P[z["test_idx"]] = z["prob_last"]
        if len(z["val_idx"]):
            P[z["val_idx"]] = z["prob_val_last"]
        return P

    masks = {}
    z0 = np.load(F["arm"][seeds[0]])
    te, va = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
    te[z0["test_idx"]], va[z0["val_idx"]] = True, True
    if proto == "seen1":
        masks = dict(seen=te & np.isin(subj, ["P04", "P10"]), p09=te & (subj == "P09"), val=va, unseen=va | (te & (subj == "P09")))
        assert set(ses[masks["seen"]]) <= {"2", "3"}
    else:
        masks = dict(test=te, val=va)

    def score(P):
        out = {}
        Pc = np.full_like(P, np.nan)
        ok = ~np.isnan(P[:, 0])
        Pc[ok] = causal_prior_correction(P[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
        pr = np.nan_to_num(Pc).argmax(1)
        for g, m in masks.items():
            assert ok[m].all(), g
            out[g] = metrics.cell_balanced_accuracy(y[m], pr[m], cells[m])[0]
        for p_ in np.unique(subj[ok]):
            m = ok & (subj == p_)
            out[p_] = metrics.cell_balanced_accuracy(y[m], pr[m], cells[m])[0]
        return out

    rows = []
    for s in seeds:
        Z = {k: np.load(v[s]) for k, v in F.items()}
        for k in ("anchor",) + (("pkg",) if args.pkg else ()):
            assert np.array_equal(Z[k]["test_idx"], Z["arm"]["test_idx"]) and np.array_equal(Z[k]["val_idx"], Z["arm"]["val_idx"]), (k, s)
        P = {k: full(z) for k, z in Z.items()}
        who = {"arm": P["arm"], "anchor": P["anchor"]}
        if args.pkg:
            w = args.w_st
            for k in ("arm", "anchor"):
                who[f"pkg_{k}"] = ((1 - w) * P[k] + w * P["pkg"]) if args.role == "fbc" else ((1 - w) * P["pkg"] + w * P[k])
        for k, Pk in who.items():
            rows.append(dict(seed=s, who=k, **score(Pk)))
    df = pd.DataFrame(rows)
    groups = list(GROUPS[proto])
    parts = sorted(c for c in df.columns if c.startswith("P"))
    pd.set_option("display.width", 250)
    if not args.quiet:
        print(df.pivot_table(index="seed", columns="who", values=groups[0]).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n{proto}: arm {args.arm}\n       anchor {args.anchor}" + (f"\n       pkg {args.pkg} (role {args.role}, w_st {args.w_st})" if args.pkg else "")
          + f"\n       seeds {seeds}")
    res = {}
    for pre in [""] + (["pkg_"] if args.pkg else []):
        a, b = pre + "arm", pre + "anchor"
        A, B = df[df.who == a].set_index("seed"), df[df.who == b].set_index("seed")
        line = []
        for g in groups + parts:
            d = A[g] - B[g]
            res[(pre, g)] = float(d.mean())
            if g in groups:
                line.append(f"{g} {A[g].mean():.4f} vs {B[g].mean():.4f} = {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)} up)")
        print(f"  {pre or 'member '}{' | '.join(line)}")
        print(f"  {pre or 'member '}per participant: " + ", ".join(f"{p_} {res[(pre, p_)]:+.4f}" for p_ in parts))
    if args.bar:
        ok = all(res[("", g)] >= t for g, t in BARS[args.bar])
        print(f"  bar {args.bar} {BARS[args.bar]} on the member: {'MET' if ok else 'not met'}")


if __name__ == "__main__":
    main()
