#!/usr/bin/env python
"""T2-D405 read (pre-registered 2026-11-07 06:19; this script written before any D405 number): IM test-time adaptation of the pooled FBCNet,
member level, every member under the shipped causal SKλ (10 it, n0 3, λ 0.95), seeds 300-311.
  selection: IM1 vs IM2 by kit VAL cell BA (P05 P08), only if the pick beats no-TTA on kit val (else NOT MET, as T2-D113)
  judged on kit TEST + seen1 TEST cells (val users never judged): typical = non-P04 cells; new = P09 + kit P04; returning = seen1 P04 + P10;
  W50 = ½ returning + ½ new (co-primary, descriptive). Bar: picked − no-TTA typical >= +0.010 with >= 9 / 12, new >= 0, P04 >= -0.010.
  python t2_d405_read.py --export E --runs DIR [--out JSON]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

STEM = "{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15{v}__s{s}.npz"
SEEDS = range(300, 312)


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def cells(d, ix, P):
    pr = causal_sinkhorn(P, (d.stream.to_numpy()[ix], d.order.to_numpy()[ix]), 10, 3, 0.95).argmax(1)
    y, c, u = d.y.to_numpy()[ix], d.cell.to_numpy()[ix], d.subject.str[-3:].to_numpy()[ix]
    return [(u[c == k][0], k, bal(y[c == k], pr[c == k])) for k in np.unique(c)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", required=True, help="comma list of run dirs (kit and seen1 may sit in one)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    dirs = [Path(x) for x in a.runs.split(",")]

    def load(p, v, s):
        f = [dd / STEM.format(p=p, v=v, s=s) for dd in dirs if (dd / STEM.format(p=p, v=v, s=s)).exists()]
        assert len(f) == 1, (p, v, s, f)
        return np.load(f[0])
    rows = []
    for s in SEEDS:
        for v in ("", "plIM1", "plIM2"):
            for p in ("kit", "seen1"):
                z = load(p, v, s)
                if p == "kit":
                    rows += [dict(seed=s, v=v or "base", p=p, split="val", user=u, cell=c, ba=b) for u, c, b in cells(d, z["val_idx"], z["prob_val_last"])]
                rows += [dict(seed=s, v=v or "base", p=p, split="test", user=u, cell=c, ba=b) for u, c, b in cells(d, z["test_idx"], z["prob_last"])]
    D = pd.DataFrame(rows)
    val = D[D.split == "val"].groupby("v").ba.mean()
    print("[D405] kit VAL cell BA (selection): " + " · ".join(f"{k} {x:.4f}" for k, x in val.items()))
    pick = max(("plIM1", "plIM2"), key=lambda k: val[k])
    out = dict(val=val.to_dict(), pick=pick)
    if val[pick] <= val["base"]:
        out["verdict"] = "NOT MET (no IM variant beats no-TTA on kit val)"
    te = D[D.split == "test"]

    def grp(g):
        typ = g[g.user != "P04"].ba.mean()
        new = g[(g.user == "P09") | ((g.user == "P04") & (g.p == "kit"))].ba.mean()
        ret = g[(g.user == "P10") | ((g.user == "P04") & (g.p == "seen1"))].ba.mean()
        return pd.Series(dict(typical=typ, new=new, returning=ret, P04=g[g.user == "P04"].ba.mean(), W50=0.5 * ret + 0.5 * new))
    G = te.groupby(["v", "seed"]).apply(grp)
    for v in ("plIM1", "plIM2"):
        Dv = G.loc[v] - G.loc["base"]
        out[v] = dict(Dv.mean().to_dict(), up=int((Dv.typical > 0).sum()))
        print(f"[D405] {v} − no-TTA (judged test cells): typical {Dv.typical.mean():+.4f} ({int((Dv.typical > 0).sum())} / 12), new {Dv.new.mean():+.4f}, "
              f"returning {Dv.returning.mean():+.4f}, P04 {Dv.P04.mean():+.4f}, W50 {Dv.W50.mean():+.4f}")
    print("[D405] no-TTA level: " + " · ".join(f"{k} {x:.4f}" for k, x in G.loc["base"].mean().items()))
    if "verdict" not in out:
        o = out[pick]
        met = o["typical"] >= 0.010 and o["up"] >= 9 and o["new"] >= 0 and o["P04"] >= -0.010
        out["verdict"] = f"{'MET' if met else 'NOT MET'} (pick {pick})"
    print(f"[D405] VERDICT: {out['verdict']}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
