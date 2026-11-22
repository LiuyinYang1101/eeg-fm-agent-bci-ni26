#!/usr/bin/env python
"""Package arms read under the SHIPPED decoder (SKλ: λ 0.95, n0 3, iters 10; T2-D304), for T2-D310 and later stacks. A spec.json as t2p_pkgshape's
(members = {name: {offset, kit, seen1}} with a {s} seed placeholder; arms = {name: {member: weight}}; anchor), the n indices; each arm's mix decoded
causally per session (R0 order) by SKλ; T2-D285 metrics (new = (kit test + seen1 val) / 2, mix = (seen + new) / 2) vs the anchor per index;
bar d141 (mix >= +0.005 with >= 9 / n up, kit test >= -0.005) AND new >= 0 AND no kill (kit val idx 0-5 < -0.01) for the arms named in --candidates;
the window check: windows fixed / broken vs the anchor (scored windows, all indices) and the share of flips below the anchor's median SK margin.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_skl_pkg.py --spec <spec.json> --candidates S1,S2 [--lam 0.95]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "local"))
from ni26 import metrics  # noqa: E402
from t2_d309_skgrid import sk  # noqa: E402

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--candidates", default="")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--n0", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    S = json.load(open(a.spec))
    n, anchor, mem, arms = S["n"], S["anchor"], S["members"], S["arms"]
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    rows, flips = [], {}
    for proto in ("kit", "seen1"):
        for i in range(n):
            P, te, va = {}, None, None
            for m in {k for w in arms.values() for k in w}:
                z = np.load(mem[m][proto].format(s=mem[m]["offset"] + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (m, proto, i)
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            ok = np.zeros(len(idx), bool); ok[te] = ok[va] = True
            tm, vm = np.zeros(len(idx), bool), np.zeros(len(idx), bool); tm[te] = True; vm[va] = True
            masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), val=vm) if proto == "seen1" else dict(test=tm, val=vm)
            streams = []
            for st in dict.fromkeys(strm[ok]):
                r_ = np.flatnonzero(ok & (strm == st))
                streams.append(r_[np.argsort(ordr[r_], kind="stable")])
            for arm, w in arms.items():
                tot = sum(w.values())
                mix = sum((v / tot) * P[m] for m, v in w.items())
                pred, marg = np.full(len(idx), -1), np.full(len(idx), np.nan)
                for r_ in streams:
                    q = sk(mix[r_], a.iters, a.lam, a.n0)
                    pred[r_] = q.argmax(1); q = np.sort(q, 1); marg[r_] = q[:, -1] - q[:, -2]
                sc = np.zeros(len(idx), bool)
                for mk in masks.values():
                    sc |= mk
                flips[(proto, i, arm)] = (pred[sc] == y[sc], marg[sc])
                rows.append(dict(proto=proto, i=i, arm=arm, **{k: metrics.cell_balanced_accuracy(y[mk], pred[mk], cells[mk])[0] for k, mk in masks.items()}))
    df = pd.DataFrame(rows)

    def comp(arm):
        g = lambda p, c: df[(df.proto == p) & (df.arm == arm)].set_index("i")[c]  # noqa: E731
        kt, sv, ss, kv = g("kit", "test"), g("seen1", "val"), g("seen1", "seen"), g("kit", "val")
        new = (kt + sv) / 2
        return dict(mix=(ss + new) / 2, new=new, seen=ss, kit_test=kt, kit_val=kv)
    base = comp(anchor)
    res = {}
    print(f"under SKλ (λ {a.lam}, n0 {a.n0}, iters {a.iters}); vs {anchor}, mean over {n} indices:")
    for arm in arms:
        if arm == anchor:
            continue
        c = comp(arm)
        d = {k: c[k] - base[k] for k in c}
        r = {k: float(v.mean()) for k, v in d.items()}
        r["up"] = int((d["mix"] > 0).sum())
        r["kill_kv06"] = float(d["kit_val"].loc[list(range(min(6, n)))].mean())
        r["bar"] = bool(r["mix"] >= 0.005 and r["up"] >= 9 and r["kit_test"] >= -0.005 and r["new"] >= 0 and r["kill_kv06"] >= -0.01)
        fx = br = lo = 0
        for (pr, i, ar), (ok_a, _) in flips.items():
            if ar != arm:
                continue
            ok0, m0 = flips[(pr, i, anchor)]
            f_, b_ = ok_a & ~ok0, ~ok_a & ok0
            fx += int(f_.sum()); br += int(b_.sum()); lo += int(((f_ | b_) & (m0 < np.median(m0))).sum())
        r.update(fixed=fx, broken=br, low_margin_share=lo / max(fx + br, 1))
        res[arm] = r
        cand = arm in a.candidates.split(",")
        print(f"  {arm}: mix {r['mix']:+.4f} ({r['up']} / {n}), kit test {r['kit_test']:+.4f}, kit val {r['kit_val']:+.4f}, seen {r['seen']:+.4f}, "
              f"new {r['new']:+.4f}, kit val 0-5 {r['kill_kv06']:+.4f}; windows fixed {fx} / broken {br}, {r['low_margin_share']:.0%} below the median margin" + (f" -> {'PASS (d141 + new >= 0)' if r['bar'] else 'no'}" if cand else " (descriptive)"))
    out = Path(a.spec).with_name("skl_read.json")
    out.write_text(json.dumps(res, indent=1))
    df.to_csv(Path(a.spec).with_name("skl_rows.csv"), index=False)
    passing = [k for k in a.candidates.split(",") if k and res[k]["bar"]]
    print("VERDICT:", f"PASS: {', '.join(passing)}" if passing else "NOT ADOPTED (no candidate passes)")


if __name__ == "__main__":
    main()
