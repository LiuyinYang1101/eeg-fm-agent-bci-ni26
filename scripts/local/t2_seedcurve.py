#!/usr/bin/env python
"""T2 heavy-plan sizing (2026-10-31, descriptive, zero GPU): how the package-shape score grows with the number of CNN seeds
per family (k_cnn) and of ST-L seeds in the 0.2 slot (k_fm), under the SHIPPED decoder (SKλ: λ 0.95, n0 3, iters 10).

Members (same recipes as the Nov 5 line, MOR-free, 4:3:2): FBCNet rc75 routes seeds 192-215 (T2-D296r 192-203 + T2-D231 204-215),
ShallowNet vr routes 156-179 (T2-D296r 156-167 + T2-D215r 168-179), ST-L stl_rcll 0-23 (T2-D247 0-11 + T2-D247r 12-23).
A k-seed family = the mean of k distinct seeds' routed probabilities (= the package's mean over seeds and their copies).
Metrics as scripts/local/t2_skl_pkg.py (T2-D285): new = (kit test + seen1 val) / 2, mix = (seen + new) / 2.
For each (k_cnn, k_fm) R random seed subsets are drawn (fixed rng); the mean and sd over draws are reported, then
mix(k) = a - b / k is fitted per axis to extrapolate 15 -> 30 / 45 CNN seeds and 1 -> 2 / 3 / 4 ST-L seeds.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_seedcurve.py --out <dir>
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
O = Path.home() / "workspace/outputs/neural_interface_challenge"
FAM = {
    "fbc": [(O / "t2_d296r/route_fbc_r1_pulxx_vp2_e180_vr_rc75", "{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75fb3__s{s}.npz", range(192, 204)),
            (O / "t2_d231/route_fbc_r1_pulxx_vp2_e180_vr_rc75", "{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75fb3__s{s}.npz", range(204, 216))],
    "sh": [(O / "t2_d296r/route_sh_r1_pulx_vp2_e180_vr", "{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrfb2__s{s}.npz", range(156, 168)),
           (O / "t2_d215r/route_sh_r1_pulx_vp2_e180_vr", "{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrfb2__s{s}.npz", range(168, 180))],
    "stl": [(O / "t2_d247/stl_rcll", "{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt{cpt}__s{s}.npz", range(0, 12)),
            (O / "t2_d247r/stl_rcll", "{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt{cpt}__s{s}.npz", range(12, 24))],
}
W = dict(fbc=0.4, sh=0.3, stl=0.2)
CPT = dict(kit="stlarge-d89b", seen1="pt-d111-with")   # the ST-L CPT per protocol (as t2_d253/spec.json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--kcnn", default="1,2,4,8,12,16,24")
    ap.add_argument("--kfm", default="1,2,3,4,6,12")
    ap.add_argument("--kcnn-at-fm", type=int, default=16, help="k_cnn held while k_fm varies (≈ the shipped 15)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int); subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy(); ordr = idx["order"].to_numpy()
    data = {}
    for proto in ("kit", "seen1"):
        te = va = None; P = {}
        for fam, parts in FAM.items():
            arr = []
            for d, pat, seeds in parts:
                for s in seeds:
                    z = np.load(d / pat.format(p=proto, s=s, cpt=CPT[proto]))
                    if te is None:
                        te, va = z["test_idx"], z["val_idx"]
                    assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (fam, proto, s)
                    rows = np.concatenate([te, va])
                    arr.append(np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64))
            P[fam] = np.stack(arr)                       # (24, n_rows, 3) on rows = test ++ val
        rows = np.concatenate([te, va])
        tm = np.zeros(len(idx), bool); vm = np.zeros(len(idx), bool); tm[te] = True; vm[va] = True
        masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), val=vm) if proto == "seen1" else dict(test=tm, val=vm)
        pos = {r: i for i, r in enumerate(rows)}
        streams = []
        for st in dict.fromkeys(strm[rows]):
            r_ = rows[strm[rows] == st]
            r_ = r_[np.argsort(ordr[r_], kind="stable")]
            streams.append(np.array([pos[r] for r in r_]))
        data[proto] = (P, rows, masks, streams)
        print(proto, {k: v.shape for k, v in P.items()}, len(streams), "streams", flush=True)

    rng = np.random.default_rng(20261031)

    def score(sub):
        res = {}
        for proto, (P, rows, masks, streams) in data.items():
            mix = sum(W[f] * P[f][sub[f]].mean(0) for f in W) / sum(W.values())
            pred = np.full(len(rows), -1)
            for r_ in streams:
                pred[r_] = sk(mix[r_], 10, 0.95, 3).argmax(1)
            full = np.full(len(idx), -1); full[rows] = pred
            for k, mk in masks.items():
                res[f"{proto}_{k}"] = metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0]
        new = (res["kit_test"] + res["seen1_val"]) / 2
        return dict(mix=(res["seen1_seen"] + new) / 2, new=new, seen=res["seen1_seen"], kit_test=res["kit_test"], kit_val=res["kit_val"])

    recs = []

    def run(kc, kf):
        n = 1 if (kc == 24 and kf == 24) else a.draws
        for r in range(n):
            sub = {f: rng.choice(24, kc, replace=False) for f in ("fbc", "sh")}
            sub["stl"] = rng.choice(24, kf, replace=False)
            recs.append(dict(k_cnn=kc, k_fm=kf, draw=r, **score(sub)))
        g = pd.DataFrame([x for x in recs if x["k_cnn"] == kc and x["k_fm"] == kf])
        print(f"k_cnn {kc:2d} k_fm {kf:2d}: mix {g.mix.mean():.4f} ± {g.mix.std(ddof=1) if len(g) > 1 else 0:.4f}  new {g.new.mean():.4f}  "
              f"seen {g.seen.mean():.4f}  kit test {g.kit_test.mean():.4f} (n {len(g)})", flush=True)

    for kc in map(int, a.kcnn.split(",")):
        run(kc, 1)
    for kf in map(int, a.kfm.split(",")):
        if kf != 1 or a.kcnn_at_fm not in map(int, a.kcnn.split(",")):
            run(a.kcnn_at_fm, kf)
        elif kf == 1:
            pass
    df = pd.DataFrame(recs); df.to_csv(out / "seedcurve_rows.csv", index=False)

    def fit(sel, kcol):
        g = df[sel].groupby(kcol)[["mix", "new", "seen", "kit_test"]].mean()
        k = g.index.to_numpy(float); X = np.c_[np.ones_like(k), -1.0 / k]
        coef = {c: np.linalg.lstsq(X, g[c].to_numpy(), rcond=None)[0].tolist() for c in g.columns}
        return g, coef
    gc, cc = fit(df.k_fm == 1, "k_cnn")
    gf, cf = fit(df.k_cnn == a.kcnn_at_fm, "k_fm")
    pred = lambda c, k1, k2: c[1] * (1.0 / k1 - 1.0 / k2)  # noqa: E731
    summ = dict(
        cnn_curve=gc.reset_index().to_dict("records"), cnn_fit=cc,
        cnn_gain_15_30={c: pred(cc[c], 15, 30) for c in cc}, cnn_gain_15_45={c: pred(cc[c], 15, 45) for c in cc},
        fm_curve=gf.reset_index().to_dict("records"), fm_fit=cf,
        fm_gain_1_2={c: pred(cf[c], 1, 2) for c in cf}, fm_gain_1_3={c: pred(cf[c], 1, 3) for c in cf}, fm_gain_1_4={c: pred(cf[c], 1, 4) for c in cf},
    )
    (out / "seedcurve.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if "gain" in k}, indent=1))


if __name__ == "__main__":
    main()
