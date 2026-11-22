#!/usr/bin/env python
"""T2-D280 (zero compute, descriptive; pre-registered in docs/DECISIONS_T2.md): sigma_draw at the sealed set's size.

PLAN §6b item 10 (PROPOSED) steers between lines with se = 0.0056 * sqrt(1/n_V + 1/n_N). 0.0056 is T2-D138's t2p_seedsub sd on the kit
test (8 cells, 2 users) with 10-of-12 seeds drawn WITHOUT replacement. Here: independent draws of the L_V recipe, simulated from the
per-seed package-shape files of T2-D253's spec (fbc_rc75 204-215, sh_vr 168-179, stl_rcll 0-11, mor 0-11; w4321; prior (1, 3) per
stream). One draw = each CNN family's mean over k seeds drawn WITH replacement from its 12, ST-L and MOR one seed each (independent,
with replacement). Per draw: the cell BA of every scored cell. From the draws' cell covariance, within each protocol:
  v   = mean within-cell variance, c_u = mean covariance of two different cells of one user, c_x = of two different users' cells,
each x 12/11 (the bootstrap's variance factor undone). Sealed-size model (C cells evenly over U users):
  Var = v / C + c_u (1/U - 1/C) + c_x (1 - 1/U)
sigma* = the model at U = 10, C = 20, k = 15, both protocols' components averaged. Rule (fixed before any number): sigma* < 0.0037 ->
item 10 amended (still proposed); 0.0037-0.0084 -> unchanged; > 0.0084 -> amended upward.

  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d280_sigma.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26 import metrics  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OUT = O / "t2_d280"
W = {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mor": 0.1}
CNN = ("fbc_rc75", "sh_vr")
OFFS = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
NDRAW, RNG, NS = 1000, 20261026, 12
SCEN = [(10, 20), (10, 60), (10, 240)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(O / "t2_d253/spec.json"))
    mem = {m: spec["members"][m] for m in W}
    assert {m: d["offset"] for m, d in mem.items()} == OFFS, {m: d["offset"] for m, d in mem.items()}
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    rng = np.random.default_rng(RNG)
    res = dict(files={}, protos={})
    comps = {}
    for proto in ("kit", "seen1"):
        P, te = {}, None
        for m, d in mem.items():
            fs = [d[proto].format(s=OFFS[m] + i) for i in range(NS)]
            res["files"][f"{proto}:{m}"] = fs
            st = []
            for f in fs:
                z = np.load(f)
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), f
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                st.append(Pm)
            P[m] = np.stack(st)
        ok = np.zeros(len(idx), bool)
        ok[te], ok[va] = True, True
        tm = np.zeros(len(idx), bool)
        tm[te] = True
        cl = sorted(set(cells[ok]))
        user = np.array([c.split("|")[0] for c in cl])
        groups = {"kit": dict(test=sorted(set(cells[tm])), val=sorted(set(cells[ok & ~tm]))),
                  "seen1": dict(seen=sorted(set(cells[tm & np.isin(subj, ["P04", "P10"])])), p09=sorted(set(cells[tm & (subj == "P09")])),
                                val=sorted(set(cells[ok & ~tm])))}[proto]
        pr_ = {}

        def score(sel):
            mix = sum(W[m] * P[m][sel[m]].mean(0) for m in W)
            pc = causal_prior_correction(mix[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
            _, per = metrics.cell_balanced_accuracy(y[ok], pc.argmax(1), cells[ok])
            return np.array([per[c] for c in cl])

        for k in (10, 15):
            S = np.stack([score({m: rng.integers(0, NS, size=(k if m in CNN else 1)) for m in W}) for _ in range(NDRAW)])
            cov = np.cov(S, rowvar=False) * NS / (NS - 1)
            same = (user[:, None] == user[None, :]) & ~np.eye(len(cl), dtype=bool)
            diff = user[:, None] != user[None, :]
            c = dict(v=float(np.diag(cov).mean()), c_u=float(cov[same].mean()), c_x=float(cov[diff].mean()))
            comps[(proto, k)] = c
            direct = {}
            for g, gc in groups.items():
                ii = [cl.index(x) for x in gc]
                sd = float(S[:, ii].mean(1).std(ddof=1) * np.sqrt(NS / (NS - 1)))
                u = user[ii]
                n_same = int(sum((u == a).sum() * ((u == a).sum() - 1) for a in set(u)))
                C = len(ii)
                mod = np.sqrt((C * c["v"] + n_same * c["c_u"] + (C * C - C - n_same) * c["c_x"]) / C ** 2)
                direct[g] = dict(cells=C, users=len(set(u)), sd_direct=sd, sd_model=float(mod), mean=float(S[:, ii].mean()))
            pr_[k] = dict(components=c, groups=direct, mean_all=float(S.mean()))
            print(f"{proto} k={k}: v {c['v']:.3e} c_u {c['c_u']:.3e} c_x {c['c_x']:.3e} | "
                  + "; ".join(f"{g} ({d['cells']} cells, {d['users']} users) sd {d['sd_direct']:.4f} model {d['sd_model']:.4f}" for g, d in direct.items()),
                  flush=True)
        # T2-D138's scheme on today's members: 10-of-12 CNN seeds WITHOUT replacement, one FM seed each
        S = np.stack([score({m: (rng.choice(NS, 10, replace=False) if m in CNN else rng.integers(0, NS, 1)) for m in W}) for _ in range(NDRAW)])
        ii = [cl.index(x) for x in groups["test" if proto == "kit" else "seen"]]
        pr_["d138_scheme_sd"] = float(S[:, ii].mean(1).std(ddof=1))
        print(f"{proto}: T2-D138 scheme (10 of 12 w/o replacement, FM 1) sd on {'kit test' if proto == 'kit' else 'seen'} {pr_['d138_scheme_sd']:.4f}")
        res["protos"][proto] = pr_
    res["sealed"] = {}
    for k in (10, 15):
        c = {q: float(np.mean([comps[(p, k)][q] for p in ("kit", "seen1")])) for q in ("v", "c_u", "c_x")}
        sc = {f"U{U}_C{C}": float(np.sqrt(c["v"] / C + c["c_u"] * (1 / U - 1 / C) + c["c_x"] * (1 - 1 / U))) for U, C in SCEN}
        res["sealed"][k] = dict(components=c, sigma=sc)
        print(f"sealed model k={k}: " + ", ".join(f"{s} {v:.4f}" for s, v in sc.items()) + f"  (floor sqrt(c_x) {np.sqrt(max(c['c_x'], 0)):.4f})")
    s = res["sealed"][15]["sigma"]["U10_C20"]
    verdict = ("AMEND item 10 (sigma* < 0.0037)" if s < 0.0037 else "AMEND upward (sigma* > 0.0084)" if s > 0.0084
               else "item 10 UNCHANGED (0.0037 <= sigma* <= 0.0084)")
    res["sigma_star"], res["verdict"] = s, verdict
    print(f"sigma* (U 10, C 20, k 15) = {s:.4f} -> {verdict}; best of 25 vs 5 at sigma*: +{0.80 * s:.4f}")
    (OUT / "sigma.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
