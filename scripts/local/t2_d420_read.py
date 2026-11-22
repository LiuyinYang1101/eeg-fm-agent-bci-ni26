#!/usr/bin/env python
"""T2-D420 read (pre-registered 2026-11-10 in docs/DECISIONS_T2.md before any D420 number; adapted from t2_d405_read.py): causal test-time
adaptation of M7's pooled FBCNet (r1_pulxx_vp2_e180_vr_rc75, align none), member level, every member under the shipped causal SKλ
(10 it, n0 3, λ 0.95), fresh seeds 312-323. Variants: plEA1 / plEA2 = EATA-filtered IM (lr 1e-5 / 3e-5), plAT1 / plAT2 = anchored mean-teacher TTT
(lr 1e-5 / 3e-5); constants fixed a priori in t2p_bench (EATA_E0, EATA_MIN, ANC).
  selection: the four variants by kit VAL cell BA (P05 P08), only if the pick beats no-TTA on kit val (else NOT MET, as T2-D113 / D405)
  judged on kit TEST + seen1 TEST cells (val users never judged): typical = non-P04 cells; new = P09 + kit P04; returning = seen1 P04 + P10;
  W50 = ½ returning + ½ new. Per-user Δ = by user over both protocols' test cells.
  BAR (R3 + the R2 user guard): pick − no-TTA typical >= +0.015 with W50 > 0 and no user below −0.020 -> the package row T2-D420p
  (the adapted pooled FBCNet inside M7's shape, runtime priced first); typical <= 0 (or the selection clause fails) -> KILLED;
  in between -> one more dose on fresh seeds.
  python t2_d420_read.py --export E --runs DIR[,DIR] [--out JSON]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

STEM = "{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75{v}__s{s}.npz"
SEEDS = range(312, 324)
VARS = ("plEA1", "plEA2", "plAT1", "plAT2")


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
        for v in ("",) + VARS:
            for p in ("kit", "seen1"):
                z = load(p, v, s)
                if p == "kit":
                    rows += [dict(seed=s, v=v or "base", p=p, split="val", user=u, cell=c, ba=b) for u, c, b in cells(d, z["val_idx"], z["prob_val_last"])]
                rows += [dict(seed=s, v=v or "base", p=p, split="test", user=u, cell=c, ba=b) for u, c, b in cells(d, z["test_idx"], z["prob_last"])]
    D = pd.DataFrame(rows)
    val = D[D.split == "val"].groupby("v").ba.mean()
    vu = D[D.split == "val"].groupby(["v", "user"]).ba.mean().unstack()
    print("[D420] kit VAL cell BA (selection): " + " · ".join(f"{k} {x:.4f}" for k, x in val.items()))
    print("[D420] kit VAL by user:\n" + vu.to_string(float_format=lambda x: f"{x:.4f}"))
    pick = max(VARS, key=lambda k: val[k])
    out = dict(val=val.to_dict(), pick=pick)
    if val[pick] <= val["base"]:
        out["verdict"] = "KILLED (selection: no variant beats no-TTA on kit val)"
    te = D[D.split == "test"]

    def grp(g):
        typ = g[g.user != "P04"].ba.mean()
        new = g[(g.user == "P09") | ((g.user == "P04") & (g.p == "kit"))].ba.mean()
        ret = g[(g.user == "P10") | ((g.user == "P04") & (g.p == "seen1"))].ba.mean()
        return pd.Series(dict(typical=typ, new=new, returning=ret, P04=g[g.user == "P04"].ba.mean(), W50=0.5 * ret + 0.5 * new))
    G = te.groupby(["v", "seed"]).apply(grp)
    U = te.groupby(["v", "seed", "user"]).ba.mean().unstack()
    for v in VARS:
        Dv = G.loc[v] - G.loc["base"]
        Du = (U.loc[v] - U.loc["base"]).mean()
        out[v] = dict(Dv.mean().to_dict(), up=int((Dv.typical > 0).sum()), users=Du.to_dict(), worst=float(Du.min()))
        print(f"[D420] {v} − no-TTA (judged test cells): typical {Dv.typical.mean():+.4f} ({int((Dv.typical > 0).sum())} / 12), new {Dv.new.mean():+.4f}, "
              f"returning {Dv.returning.mean():+.4f}, P04 {Dv.P04.mean():+.4f}, W50 {Dv.W50.mean():+.4f}; per user " + " ".join(f"{k} {x:+.4f}" for k, x in Du.items()))
    print("[D420] no-TTA level: " + " · ".join(f"{k} {x:.4f}" for k, x in G.loc["base"].mean().items()))
    if "verdict" not in out:
        o = out[pick]
        if o["typical"] >= 0.015 and o["W50"] > 0 and o["worst"] >= -0.020:
            out["verdict"] = f"GATE MET (pick {pick}) -> the package row T2-D420p (runtime priced first)"
        elif o["typical"] <= 0:
            out["verdict"] = f"KILLED (pick {pick}: typical <= 0)"
        else:
            out["verdict"] = f"IN BETWEEN (pick {pick}) -> one more dose on fresh seeds"
    print(f"[D420] VERDICT: {out['verdict']}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
