#!/usr/bin/env python
"""T2-D316 (pre-registered 2026-10-31 before any number; zero GPU): does an internal statistic predict a draw's held-out score?

60 draws = ST-L seeds 0-59 (T2-D247 0-11, D247r 12-23, D312 24-59), each in the FIXED 24-seed CNN context at 4:3:2 MOR-free under the shipped
SKλ. Targets: T_kit = kit test, T_new = (kit test + seen1 val) / 2, T_mix = (seen1 seen + T_new) / 2 (T2-D285). Proxies: P_val = the package's
kit val (P05 + P08: the CEILING, not available for --train all ZIPs), P_tl = the ST-L member's train-loss tail (mean of its last 5 epochs in the
kit run; in every build log), P_ent = the package's mean predictive entropy on the kit test windows (label-free; descriptive).
Bar: Spearman r(P_tl, T_new) >= 0.3 with a bootstrap 90 % CI > 0 AND r x 0.0050 x 1.163 >= +0.002. Prediction: every r in -0.15 to +0.2.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d316_proxy.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "local"))
from ni26 import metrics  # noqa: E402
from t2_d309_skgrid import sk  # noqa: E402
from t2_seedcurve import CPT, FAM  # noqa: E402

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
O = Path.home() / "workspace/outputs/neural_interface_challenge"
TAG = "{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt{cpt}__s{s}.npz"
DIRS = [O / "t2_d247/stl_rcll", O / "t2_d247r/stl_rcll"] + [Path(d) for d in sorted(glob.glob(str(O / "sofia_s1/ni26-t2p-d312-stl-*-*/runs")))]
W = dict(fbc=0.4, sh=0.3, stl=0.2)


def stl_npz(p, s):
    hits = [d / TAG.format(p=p, cpt=CPT[p], s=s) for d in DIRS if (d / TAG.format(p=p, cpt=CPT[p], s=s)).exists()]
    assert len(hits) == 1, (p, s, hits)
    return np.load(hits[0], allow_pickle=True)


def main():
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int); subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy(); ordr = idx["order"].to_numpy()
    per = {}
    for proto in ("kit", "seen1"):
        te = va = None; cnn = {}
        for fam in ("fbc", "sh"):
            arr = []
            for d, pat, seeds in FAM[fam]:
                for s in seeds:
                    z = np.load(d / pat.format(p=proto, s=s, cpt=CPT[proto]))
                    if te is None:
                        te, va = z["test_idx"], z["val_idx"]
                    arr.append(np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64))
            cnn[fam] = np.mean(arr, 0)
        rows = np.concatenate([te, va])
        tm = np.zeros(len(idx), bool); vm = np.zeros(len(idx), bool); tm[te] = True; vm[va] = True
        masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), val=vm) if proto == "seen1" else dict(test=tm, val=vm)
        pos = {r: j for j, r in enumerate(rows)}
        streams = []
        for st in dict.fromkeys(strm[rows]):
            r_ = rows[strm[rows] == st]
            streams.append(np.array([pos[r] for r in r_[np.argsort(ordr[r_], kind="stable")]]))
        for s in range(60):
            z = stl_npz(proto, s)
            assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (proto, s)
            pst = np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64)
            mix = (W["fbc"] * cnn["fbc"] + W["sh"] * cnn["sh"] + W["stl"] * pst) / sum(W.values())
            pred = np.full(len(rows), -1)
            for r_ in streams:
                pred[r_] = sk(mix[r_], 10, 0.95, 3).argmax(1)
            full = np.full(len(idx), -1); full[rows] = pred
            d = per.setdefault(s, {})
            for k, mk in masks.items():
                d[f"{proto}_{k}"] = metrics.cell_balanced_accuracy(y[mk], full[mk], cells[mk])[0]
            if proto == "kit":
                res = json.loads(str(z["res"]))
                d["P_tl"] = float(np.mean([c["train_loss"] for c in res["curves"][-5:]]))
                pt = mix[: len(te)]                                   # rows = test ++ val: the kit test windows first
                d["P_ent"] = float(-(pt * np.log(np.clip(pt, 1e-12, 1))).sum(1).mean())
    df = pd.DataFrame.from_dict(per, orient="index")
    df["T_kit"] = df["kit_test"]; df["T_new"] = (df["kit_test"] + df["seen1_val"]) / 2; df["T_mix"] = (df["seen1_seen"] + df["T_new"]) / 2
    df["P_val"] = df["kit_val"]
    rng = np.random.default_rng(20261031)
    out = {"n_draws": int(len(df)), "sd": {t: float(df[t].std(ddof=1)) for t in ("T_kit", "T_new", "T_mix")}}
    for pr in ("P_val", "P_tl", "P_ent"):
        for t in ("T_kit", "T_new", "T_mix"):
            sgn = -1.0 if pr in ("P_tl", "P_ent") else 1.0                  # lower loss / entropy = predicted better
            r = float(spearmanr(sgn * df[pr], df[t]).correlation)
            bs = [spearmanr(sgn * df[pr].to_numpy()[ix], df[t].to_numpy()[ix]).correlation for ix in (rng.integers(0, len(df), len(df)) for _ in range(2000))]
            lo, hi = np.nanpercentile(bs, [5, 95])
            out[f"{pr}->{t}"] = dict(r=r, ci90=[float(lo), float(hi)], implied_gain=r * 0.0050 * 1.163)
            print(f"{pr} -> {t}: Spearman r {r:+.3f} (90 % CI {lo:+.3f} .. {hi:+.3f}); implied gain at m 5 {r * 0.0050 * 1.163:+.4f}")
    k = out["P_tl->T_new"]
    out["bar"] = bool(k["r"] >= 0.3 and k["ci90"][0] > 0 and k["implied_gain"] >= 0.002)
    print("draw sd:", {t: round(v, 4) for t, v in out["sd"].items()}, "| n", len(df))
    o = O / "t2_d316"; o.mkdir(exist_ok=True)
    (o / "read.json").write_text(json.dumps(out, indent=1)); df.to_csv(o / "draws.csv")
    print("VERDICT:", "PASS (P_tl predicts T_new)" if out["bar"] else "NOT ADOPTED (no transferable proxy predicts the held-out draw score)")


if __name__ == "__main__":
    main()
