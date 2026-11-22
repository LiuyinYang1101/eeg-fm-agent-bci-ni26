#!/usr/bin/env python
"""T2-D313 read (pre-registered 2026-10-31 before any number; zero GPU): NeuroRVQ-EEG v1 as a T2 member, RESEARCH screen.

Inputs: NeuroRVQ runs (outputs/sofia_s1/ni26-t2p-d313-nrvq-5090/runs, seeds 24-35, kit + seen1) and T2-D312's ST-L k1 runs
(seeds 24-35, the paired reference). CNN context FIXED = the mean of all 24 routed seeds per family (as T2-D312).
(A) member alone, index i: NeuroRVQ seed 24+i vs ST-L seed 24+i: raw argmax cell-BA (kit test, kit val, seen1 seen / val) and
    the T2-D285 metrics under the shipped SKλ.
(B) packages under SKλ (λ 0.95, n0 3, iters 10): A0 = 4:3:2 {fbc, sh, ST-L}; B1 = 4:3:2 {fbc, sh, NeuroRVQ} (in the slot);
    B2 = 4:3:2:1 {fbc, sh, ST-L, NeuroRVQ} (added). Bar for each of B1 / B2 vs A0: d141 (mix >= +0.005 with >= 9 / 12 up,
    kit test >= -0.005) AND new >= 0. Prediction: B1 -0.02 to -0.05, B2 -0.003 to +0.002; window prediction for B2: fixes on
    NEW-user windows, |seen| <= 0.002.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d313_read.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "local"))
from ni26 import metrics  # noqa: E402
from t2_d309_skgrid import sk  # noqa: E402
from t2_d312_read import STL, stl_file  # noqa: E402
from t2_seedcurve import CPT, FAM  # noqa: E402

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
O = Path.home() / "workspace/outputs/neural_interface_challenge"
NR = "{p}__fm-neurorvq-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75__s{s}.npz"
ARMS = {"A0": dict(fbc=0.4, sh=0.3, stl=0.2), "B1": dict(fbc=0.4, sh=0.3, nr=0.2), "B2": dict(fbc=0.4, sh=0.3, stl=0.2, nr=0.1),
        "stl_alone": dict(stl=1.0), "nr_alone": dict(nr=1.0)}


RUNS = "ni26-t2p-d313-nrvq-*"                 # T2-D313b: --runs / --nr / --out select the LoRA arm's files and output


def nr_file(proto, s):
    hits = glob.glob(str(O / "sofia_s1" / RUNS / "runs" / NR.format(p=proto, s=s)))
    assert len(hits) == 1, (proto, s, hits)
    return hits[0]


def main():
    global RUNS, NR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS); ap.add_argument("--nr", default=NR); ap.add_argument("--out", default="t2_d313")
    a = ap.parse_args()
    RUNS, NR = a.runs, a.nr
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int); subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy(); ordr = idx["order"].to_numpy()
    rows_out, raw, win = [], [], {}
    for proto in ("kit", "seen1"):
        te = va = None
        cnn = {}
        for fam in ("fbc", "sh"):
            arr = []
            for d, pat, seeds in FAM[fam]:
                for s in seeds:
                    z = np.load(d / pat.format(p=proto, s=s, cpt=CPT[proto]))
                    if te is None:
                        te, va = z["test_idx"], z["val_idx"]
                    assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (fam, proto, s)
                    arr.append(np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64))
            cnn[fam] = np.mean(arr, 0)
        rows = np.concatenate([te, va])
        tm = np.zeros(len(idx), bool); vm = np.zeros(len(idx), bool); tm[te] = True; vm[va] = True
        masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), val=vm) if proto == "seen1" else dict(test=tm, val=vm)
        sc = np.zeros(len(idx), bool)
        for mk in masks.values():
            sc |= mk
        pos = {r: j for j, r in enumerate(rows)}
        streams = []
        for st in dict.fromkeys(strm[rows]):
            r_ = rows[strm[rows] == st]
            streams.append(np.array([pos[r] for r in r_[np.argsort(ordr[r_], kind="stable")]]))
        new_user = ~np.isin(subj[sc], ["P04", "P10"]) if proto == "seen1" else np.ones(int(sc.sum()), bool)
        for i in range(12):
            P = dict(cnn)
            for key, fn in (("stl", stl_file), ("nr", nr_file)):
                z = np.load(fn(proto, 24 + i))
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (key, proto, i)
                P[key] = np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64)
                full = np.full(len(idx), -1); full[rows] = P[key].argmax(1)
                raw.append(dict(proto=proto, i=i, member=key, **{k: metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0]
                                                                 for k, mk in masks.items()}))
            for arm, w in ARMS.items():
                mix = sum(v * P[k] for k, v in w.items()) / sum(w.values())
                pred = np.full(len(rows), -1)
                for r_ in streams:
                    pred[r_] = sk(mix[r_], 10, 0.95, 3).argmax(1)
                full = np.full(len(idx), -1); full[rows] = pred
                rows_out.append(dict(proto=proto, i=i, arm=arm, **{k: metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0]
                                                                     for k, mk in masks.items()}))
                win[(proto, i, arm)] = (full[sc] == y[sc], new_user)
    df = pd.DataFrame(rows_out); rw = pd.DataFrame(raw)

    def comp(arm):
        g = lambda p, c: df[(df.proto == p) & (df.arm == arm)].set_index("i")[c]  # noqa: E731
        kt, sv, ss, kv = g("kit", "test"), g("seen1", "val"), g("seen1", "seen"), g("kit", "val")
        new = (kt + sv) / 2
        return dict(mix=(ss + new) / 2, new=new, seen=ss, kit_test=kt, kit_val=kv)
    res = {"raw_alone": rw.groupby(["proto", "member"])[[c for c in ("test", "val", "seen") if c in rw]].mean().round(4).reset_index().to_dict("records")}
    abs_ = {arm: {k: float(v.mean()) for k, v in comp(arm).items()} for arm in ARMS}
    res["abs_under_skl"] = abs_
    print("raw member alone (argmax, last epoch), means over 12 seeds:")
    print(rw.groupby(["proto", "member"])[["test", "val", "seen"]].mean().round(4).to_string())
    print("under SKλ, absolute:", json.dumps({a: {k: round(v, 4) for k, v in d.items()} for a, d in abs_.items()}))
    base = comp("A0")
    for arm in ("B1", "B2"):
        c = comp(arm)
        d = {k: c[k] - base[k] for k in c}
        r = {k: float(v.mean()) for k, v in d.items()}
        r["up"] = int((d["mix"] > 0).sum())
        r["bar"] = bool(r["mix"] >= 0.005 and r["up"] >= 9 and r["kit_test"] >= -0.005 and r["new"] >= 0)
        fx_new = br_new = fx = br = 0
        for (pr, i, ar), (ok_a, nu) in win.items():
            if ar != arm:
                continue
            ok0, _ = win[(pr, i, "A0")]
            f_, b_ = ok_a & ~ok0, ~ok_a & ok0
            fx += int(f_.sum()); br += int(b_.sum()); fx_new += int((f_ & nu).sum()); br_new += int((b_ & nu).sum())
        r.update(fixed=fx, broken=br, net_fix_new_share=(fx_new - br_new) / (fx - br) if fx != br else None)
        res[arm] = r
        print(f"{arm} vs A0: mix {r['mix']:+.4f} ({r['up']} / 12), new {r['new']:+.4f}, seen {r['seen']:+.4f}, kit test {r['kit_test']:+.4f}, "
              f"kit val {r['kit_val']:+.4f}; windows fixed {fx} / broken {br} -> {'PASS' if r['bar'] else 'no'}")
    out = O / a.out; out.mkdir(exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1)); df.to_csv(out / "rows.csv", index=False); rw.to_csv(out / "raw_alone.csv", index=False)
    print("VERDICT:", "PASS: " + ", ".join(a for a in ("B1", "B2") if res[a]["bar"]) if any(res[a]["bar"] for a in ("B1", "B2"))
          else "NOT ADOPTED (neither B1 nor B2 meets d141 + new >= 0)")


if __name__ == "__main__":
    main()
