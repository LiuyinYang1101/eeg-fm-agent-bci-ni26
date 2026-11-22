#!/usr/bin/env python
"""T2-D312 read (pre-registered 2026-10-31 before any number; zero GPU): does a MULTI-SEED ST-L slot (k ST-L fine-tunes averaged
in probability inside the 0.2 FM share) lift the package on FRESH ST-L seeds?

Package (MOR-free, the Nov 5 shape 4:3:2, decoded by the SHIPPED SKλ λ 0.95 / n0 3 / iters 10, per stream in R0 order):
  CNN context, FIXED across indices = the mean of all 24 routed seeds per family (FBCNet rc75 192-215, ShallowNet vr 156-179;
  the seed curve shows CNN seeds saturated by ≈ 15, so this stands in for the shipped 15-seed families);
  ST-L, index i = 0..11: k1 = {24+i}; k2 = {24+i, 36+i}; k3 = {24+i, 36+i, 48+i} (D312's fresh runs, T2-D247r's recipe unchanged).
Metrics as scripts/local/t2_skl_pkg.py (new = (kit test + seen1 val) / 2, mix = (seen + new) / 2).
BAR (fixed now): k3 vs k1 mix >= +0.002 with >= 8 / 12 indices up AND new >= 0 AND kit test >= -0.005 -> PASS.
Prediction (written now, from outputs/t2_seedcurve): k2 +0.0017 to +0.0026, k3 +0.0025 to +0.0034; seen and new alike.
Window prediction: the fixes concentrate in windows where the three ST-L seeds disagree (top tercile of the across-seed sd of
the ST-L probability of the true class): >= 50 % of the net fixes there.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d312_read.py
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
from t2_seedcurve import CPT, FAM  # noqa: E402  (the CNN members, the same files)

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
O = Path.home() / "workspace/outputs/neural_interface_challenge"
STL = "{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt{cpt}__s{s}.npz"
W = dict(fbc=0.4, sh=0.3, stl=0.2)
ARMS = {"k1": (0,), "k2": (0, 12), "k3": (0, 12, 24)}       # offsets added to 24 + i


def stl_file(proto, s):
    hits = glob.glob(str(O / "sofia_s1" / "ni26-t2p-d312-stl-*" / "runs" / STL.format(p=proto, s=s, cpt=CPT[proto])))
    assert len(hits) == 1, (proto, s, hits)
    return hits[0]


def main():
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int); subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy(); ordr = idx["order"].to_numpy()
    rows_out, win = [], {}
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
        stl = {}
        for s in range(24, 60):
            z = np.load(stl_file(proto, s))
            assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), ("stl", proto, s)
            stl[s] = np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64)
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
        yr = y[rows]
        for i in range(12):
            sd3 = np.std([stl[24 + i + o][np.arange(len(rows)), yr] for o in ARMS["k3"]], 0)
            for arm, offs in ARMS.items():
                p_stl = np.mean([stl[24 + i + o] for o in offs], 0)
                mix = (W["fbc"] * cnn["fbc"] + W["sh"] * cnn["sh"] + W["stl"] * p_stl) / sum(W.values())
                pred = np.full(len(rows), -1)
                for r_ in streams:
                    pred[r_] = sk(mix[r_], 10, 0.95, 3).argmax(1)
                full = np.full(len(idx), -1); full[rows] = pred
                rows_out.append(dict(proto=proto, i=i, arm=arm, **{k: metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0]
                                                                     for k, mk in masks.items()}))
                win[(proto, i, arm)] = (full[sc] == y[sc], sd3[[pos[r] for r in np.flatnonzero(sc)]])
    df = pd.DataFrame(rows_out)

    def comp(arm):
        g = lambda p, c: df[(df.proto == p) & (df.arm == arm)].set_index("i")[c]  # noqa: E731
        kt, sv, ss, kv = g("kit", "test"), g("seen1", "val"), g("seen1", "seen"), g("kit", "val")
        new = (kt + sv) / 2
        return dict(mix=(ss + new) / 2, new=new, seen=ss, kit_test=kt, kit_val=kv)
    base = comp("k1")
    res = {"k1_abs": {k: float(v.mean()) for k, v in base.items()}}
    for arm in ("k2", "k3"):
        c = comp(arm)
        d = {k: c[k] - base[k] for k in c}
        r = {k: float(v.mean()) for k, v in d.items()}
        r["up"] = int((d["mix"] > 0).sum())
        fx = br = fx_hi = br_hi = 0
        for (pr, i, ar), (ok_a, sd) in win.items():
            if ar != arm:
                continue
            ok0, _ = win[(pr, i, "k1")]
            hi = sd >= np.quantile(sd, 2 / 3)
            f_, b_ = ok_a & ~ok0, ~ok_a & ok0
            fx += int(f_.sum()); br += int(b_.sum()); fx_hi += int((f_ & hi).sum()); br_hi += int((b_ & hi).sum())
        r.update(fixed=fx, broken=br, net_fix_top_tercile_share=(fx_hi - br_hi) / max(fx - br, 1))
        res[arm] = r
        print(f"{arm} vs k1: mix {r['mix']:+.4f} ({r['up']} / 12), new {r['new']:+.4f}, seen {r['seen']:+.4f}, kit test {r['kit_test']:+.4f}, "
              f"kit val {r['kit_val']:+.4f}; windows fixed {fx} / broken {br}, net-fix share in the top sd tercile {r['net_fix_top_tercile_share']:.0%}")
    k3 = res["k3"]
    res["bar"] = bool(k3["mix"] >= 0.002 and k3["up"] >= 8 and k3["new"] >= 0 and k3["kit_test"] >= -0.005)
    res["window_prediction_met"] = bool(k3["net_fix_top_tercile_share"] >= 0.5)
    out = O / "t2_d312"; out.mkdir(exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1)); df.to_csv(out / "rows.csv", index=False)
    print("VERDICT:", "PASS (k3 vs k1 meets the bar)" if res["bar"] else "NOT ADOPTED (k3 vs k1 misses the bar)",
          "| window prediction", "met" if res["window_prediction_met"] else "not met")


if __name__ == "__main__":
    main()
