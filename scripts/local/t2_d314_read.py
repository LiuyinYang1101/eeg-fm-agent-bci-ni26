#!/usr/bin/env python
"""T2-D314 read (pre-registered 2026-10-31 before any number; zero GPU): TA568 with T2's role-LoRA MOR recipe vs the 100M MOR.

Members, index i = 0..11 (seed i): CNN context FIXED = the mean of all 24 routed seeds per family (as T2-D312 / D313); ST-L = stl_rcll seed i
(T2-D247; kit CPT d89b, seen1 d111-with); MOR100 = the 100M role-LoRA MOR seed i (1459941 kit / 1460136 seen1); TA568 = this row's runs, seed i.
(A) member alone (raw argmax, last epoch): TA568 vs MOR100.
(B) packages under the shipped SKλ: A = W10's shape {fbc 0.4, sh 0.3, stl 0.2, mor100 0.1}; B = {fbc 0.4, sh 0.3, stl 0.2, ta568 0.1};
    descriptive: both vs the MOR-free 4:3:2 (W9's shape). Bar: B vs A d141 (mix >= +0.005 with >= 9 / 12 up, kit test >= -0.005) AND new >= 0.
Prediction: member kit test TA568 - MOR100 -0.02 to +0.02; B - A -0.003 to +0.003; window prediction: any gain in NEW-user windows, seen flat.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d314_read.py
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
from t2_seedcurve import CPT, FAM  # noqa: E402

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
O = Path.home() / "workspace/outputs/neural_interface_challenge"
STL = {p: str(O / "t2_d247/stl_rcll" / ("{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt{cpt}__s{s}.npz")) for p in ("kit", "seen1")}
MOR = {"kit": str(O / "sofia_s1/ni26-t2p-t2d130-lora-mor2-1459941/runs/kit__fm-mor_p1v2-mean__wz__n00__fmE12b64le0.001lh0.001wd0.05ll1ema0dp0li16roled0.1__s{s}.npz"),
       "seen1": str(O / "sofia_s1/ni26-t2p-t2d133-rolelora-1460136/runs/seen1__fm-mor_p1v2-mean__wz__n00__fmE12b64le0.001lh0.001wd0.05ll1ema0dp0li16roled0.1__s{s}.npz")}
TA = str(O / "sofia_s1" / "ni26-t2p-d314-ta568-{p}-*" / "runs" / "{p}__fm-ta568-mean__wz__*__s{s}.npz")
ARMS = {"W9": dict(fbc=0.4, sh=0.3, stl=0.2), "A": dict(fbc=0.4, sh=0.3, stl=0.2, mor=0.1), "B": dict(fbc=0.4, sh=0.3, stl=0.2, ta=0.1)}


def one(pat):
    hits = glob.glob(pat)
    assert len(hits) == 1, (pat, hits)
    return hits[0]


def main():
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
            for key, f in (("stl", STL[proto].format(p=proto, cpt=CPT[proto], s=i)), ("mor", MOR[proto].format(s=i)), ("ta", one(TA.format(p=proto, s=i)))):
                z = np.load(f)
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (key, proto, i)
                P[key] = np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64)
                if key in ("mor", "ta"):
                    full = np.full(len(idx), -1); full[rows] = P[key].argmax(1)
                    raw.append(dict(proto=proto, i=i, member=key, **{k: metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0] for k, mk in masks.items()}))
            for arm, w in ARMS.items():
                mix = sum(v * P[k] for k, v in w.items()) / sum(w.values())
                pred = np.full(len(rows), -1)
                for r_ in streams:
                    pred[r_] = sk(mix[r_], 10, 0.95, 3).argmax(1)
                full = np.full(len(idx), -1); full[rows] = pred
                rows_out.append(dict(proto=proto, i=i, arm=arm, **{k: metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0] for k, mk in masks.items()}))
                win[(proto, i, arm)] = (full[sc] == y[sc], new_user)
    df = pd.DataFrame(rows_out); rw = pd.DataFrame(raw)

    def comp(arm):
        g = lambda p, c: df[(df.proto == p) & (df.arm == arm)].set_index("i")[c]  # noqa: E731
        kt, sv, ss, kv = g("kit", "test"), g("seen1", "val"), g("seen1", "seen"), g("kit", "val")
        new = (kt + sv) / 2
        return dict(mix=(ss + new) / 2, new=new, seen=ss, kit_test=kt, kit_val=kv)
    print("raw member alone (argmax, last epoch), means over 12 seeds:")
    print(rw.groupby(["proto", "member"])[["test", "val", "seen"]].mean().round(4).to_string())
    res = {"raw_alone": rw.groupby(["proto", "member"])[["test", "val", "seen"]].mean().round(4).reset_index().to_dict("records"),
           "abs": {a: {k: float(v.mean()) for k, v in comp(a).items()} for a in ARMS}}
    for arm, ref in (("B", "A"), ("A", "W9"), ("B", "W9")):
        c, b0 = comp(arm), comp(ref)
        d = {k: c[k] - b0[k] for k in c}
        r = {k: float(v.mean()) for k, v in d.items()}
        r["up"] = int((d["mix"] > 0).sum())
        fx = br = fxn = brn = 0
        for (pr, i, ar), (ok_a, nu) in win.items():
            if ar != arm:
                continue
            ok0, _ = win[(pr, i, ref)]
            f_, b_ = ok_a & ~ok0, ~ok_a & ok0
            fx += int(f_.sum()); br += int(b_.sum()); fxn += int((f_ & nu).sum()); brn += int((b_ & nu).sum())
        r.update(fixed=fx, broken=br, net_new=fxn - brn, net_seen=(fx - fxn) - (br - brn))
        if (arm, ref) == ("B", "A"):
            r["bar"] = bool(r["mix"] >= 0.005 and r["up"] >= 9 and r["kit_test"] >= -0.005 and r["new"] >= 0)
        res[f"{arm}_vs_{ref}"] = r
        print(f"{arm} vs {ref}: mix {r['mix']:+.4f} ({r['up']} / 12), new {r['new']:+.4f}, seen {r['seen']:+.4f}, kit test {r['kit_test']:+.4f}; "
              f"windows fixed {fx} / broken {br} (net new {fxn - brn:+d}, net seen {r['net_seen']:+d})" + (f" -> {'PASS' if r['bar'] else 'no'}" if "bar" in r else " (descriptive)"))
    out = O / "t2_d314"; out.mkdir(exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1)); df.to_csv(out / "rows.csv", index=False); rw.to_csv(out / "raw_alone.csv", index=False)
    print("VERDICT:", "PASS (B vs A meets d141 + new >= 0)" if res["B_vs_A"]["bar"] else "NOT ADOPTED (B vs A misses the bar)")


if __name__ == "__main__":
    main()
