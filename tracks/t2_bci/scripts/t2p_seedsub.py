#!/usr/bin/env python
"""Track 2 T2-D121 step 1 as a repo tool (review 2026-10-16 T2 P2g; the original lived in a scratchpad that vanished):
does averaging MORE seeds per member help in package shape? Zero compute, from per-seed bench result files on disk.

A member = a t2p_bench result-file stem without its seed suffix and a package weight. One draw: the ARM takes a random
`--arm` count of each member's seeds (without replacement), the ANCHOR a random `--anchor` count drawn as a SUBSET of
the arm's seeds; each = sum_m w_m * mean over its seeds of p_m (the package's output; prior correction --tta after the
mix, as the template), scored on the kit test (P04 + P09, cell-BA: the leaderboard) and the kit val (P05 + P08).
Δ = arm − anchor, paired within the draw; --draws draws from np.random.default_rng(--rng).

  python t2p_seedsub.py --export <export_proteus_v1> \
      --member <dir>:kit__fbcnet__none__n00__r1:0.6 --member <dir>:kit__shallow__ea__n00__r1:0.2 \
      --member <dir>:kit__fm-st_large-...:0.2 --arm 10,10,3 --anchor 5,5,1 [--draws 200 --rng 20261015 --seeds 0-11]

T2-D121 step 1 (2026-10-15): fbc_kit = outputs/t2_d111/fbc_kit, sh_kit = outputs/t2_d116/sh_kit, ST-L mixup =
sofia_s1/ni26-t2p-d112-stlmix-1458293/runs; reported test 0.5608 vs 0.5535 (E[Δ] +0.0072, P(Δ > 0) 0.78), val 0.7554 vs
0.7534 (+0.0020, 0.59). The draw order of the lost script is not known, so a re-run matches in distribution, not draw
for draw.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import metrics  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402


def seed_list(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--member", action="append", required=True, metavar="DIR:STEM:WEIGHT")
    ap.add_argument("--arm", required=True, help="seeds per member in the arm, e.g. 10,10,3")
    ap.add_argument("--anchor", required=True, help="seeds per member in the anchor (a subset of the arm's), e.g. 5,5,1")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--rng", type=int, default=20261015)
    ap.add_argument("--seeds", default="0-11", help="the seed pool every member must hold")
    ap.add_argument("--tta", default="prior:1:3")
    ap.add_argument("--out", default="", help="optional JSON with the per-draw scores")
    args = ap.parse_args()
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    tta = tuple(float(v) for v in args.tta.split(":")[1:]) if args.tta.startswith("prior") else None
    pool = seed_list(args.seeds)
    mems = []
    for spec in args.member:
        d, stem, w = spec.rsplit(":", 2)
        P, V = [], []
        te = va = None
        for s in pool:
            z = np.load(Path(d) / f"{stem}__s{s}.npz")
            te = z["test_idx"] if te is None else te
            va = z["val_idx"] if va is None else va
            assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), f"{stem} s{s}: windows differ"
            P.append(z["prob_last"].astype(np.float64))
            V.append(z["prob_val_last"].astype(np.float64))
        mems.append(dict(stem=stem, w=float(w), P=np.stack(P), V=np.stack(V), te=te, va=va))
    te, va = mems[0]["te"], mems[0]["va"]
    for m in mems:
        assert np.array_equal(m["te"], te) and np.array_equal(m["va"], va), f"{m['stem']}: windows differ across members"
    W = np.array([m["w"] for m in mems])
    assert abs(W.sum() - 1) < 1e-9, f"weights sum to {W.sum()}"
    n_arm, n_anc = seed_list(args.arm), seed_list(args.anchor)
    assert len(n_arm) == len(n_anc) == len(mems) and all(b <= a <= len(pool) for a, b in zip(n_arm, n_anc))

    def score(sel, key, ix):
        p = sum(m["w"] * m[key][s].mean(0) for m, s in zip(mems, sel))
        if tta:
            p = causal_prior_correction(p, (strm[ix], ordr[ix]), *tta)
        return metrics.cell_balanced_accuracy(y[ix], p.argmax(1), cells[ix])[0]

    rng = np.random.default_rng(args.rng)
    rows = []
    for _ in range(args.draws):
        arm = [rng.choice(len(pool), size=a, replace=False) for a in n_arm]
        anc = [rng.choice(s_, size=b, replace=False) for s_, b in zip(arm, n_anc)]
        rows.append(dict(arm_test=score(arm, "P", te), anc_test=score(anc, "P", te),
                         arm_val=score(arm, "V", va), anc_val=score(anc, "V", va)))
    df = pd.DataFrame(rows)
    res = {}
    for split in ("test", "val"):
        d = df[f"arm_{split}"] - df[f"anc_{split}"]
        res[split] = dict(arm=float(df[f"arm_{split}"].mean()), anchor=float(df[f"anc_{split}"].mean()), d=float(d.mean()),
                          p_up=float((d > 0).mean()), sd_arm=float(df[f"arm_{split}"].std()), sd_anchor=float(df[f"anc_{split}"].std()))
        r = res[split]
        print(f"{split}: arm {r['arm']:.4f} vs anchor {r['anchor']:.4f}, E[Δ] {r['d']:+.4f} (P(Δ > 0) {r['p_up']:.2f}); "
              f"sd {r['sd_anchor']:.4f} -> {r['sd_arm']:.4f}", flush=True)
    print(f"{args.draws} draws, rng {args.rng}, arm {args.arm} vs anchor {args.anchor}, seeds {args.seeds}, "
          f"weights {W.round(3).tolist()}, tta {args.tta}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(args=vars(args), summary=res, draws=rows), indent=1))


if __name__ == "__main__":
    main()
