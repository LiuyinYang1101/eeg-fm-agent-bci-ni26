#!/usr/bin/env python
"""T2-D535 nomination read (written 2026-11-16 before any XGBoost / CatBoost number): in W17's context on K = 9 draws 810-812, exactly as
T2-D522 read its dc8 bag (scripts/local/t2_d522_read.py's head and mix() exec'd: FINAL + the S5 side nets 'thirds' + the window-mean family at
share 0.075 through its own causal Sinkhorn, cross-fitted start). Arms, each IN PLACE of W17's LightGBM bag (pred_s1-2-3_dc.npz, seeds 1 / 2 / 3):
  XGB  = the XGBoost bag (outputs/t2_d535/pred_s1-2-3_dc_xgb.npz);   MIX  = (LightGBM bag + XGB bag) / 2  (6 models, equal weights);
  CAT  = the CatBoost bag (pred_s1-2-3_dc_cat.npz), if present;      MIX3 = (LightGBM + XGB + CAT) / 3.
Rule (D535): NOMINATED iff ten-stranger mean > 0 on EACH of the 3 draws (3 / 3), no stranger < -0.020, returning >= -0.003; sealed mix reported.
    mem-guard python scripts/local/t2_d535_read.py
"""
import json

import numpy as np

src = open("scripts/local/t2_d522_read.py").read().split("\nDC8, DC8V = bag(")[0]
exec(compile(src, "t2_d522_read_head", "exec"))     # D518 / D512 head, score, DC (W17's LightGBM bag), DCVEC, bag(), mix(), U, TRU, O_


def combine(paths):
    """the equal-weight mean of several bags' per-fold probabilities (same test rows), with its cross-fitted start vectors."""
    zs = [np.load(p) for p in paths]
    D_ = {}
    for f in range(10):
        te = zs[0][f"te_{f}"]
        assert all((z[f"te_{f}"] == te).all() for z in zs)
        D_[f] = (te, np.mean([z[f"bag_{f}"] for z in zs], 0))
    V = {}
    for f in range(10):
        v = np.mean([D_[g][1][U[D_[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f], 0)
        V[f] = v / v.sum()
    return D_, V


LGB = O_ / "t2_d515/pred_s1-2-3_dc.npz"
R5 = O_ / "t2_d535"
arms = {"XGB": [R5 / "pred_s1-2-3_dc_xgb.npz"], "MIX": [LGB, R5 / "pred_s1-2-3_dc_xgb.npz"]}
if (R5 / "pred_s1-2-3_dc_cat.npz").exists():
    arms.update({"CAT": [R5 / "pred_s1-2-3_dc_cat.npz"], "MIX3": [LGB, R5 / "pred_s1-2-3_dc_xgb.npz", R5 / "pred_s1-2-3_dc_cat.npz"]})
D0, V0 = combine([LGB])
assert all(np.allclose(D0[f][1], DC[f][1], atol=0, rtol=0) and (D0[f][0] == DC[f][0]).all() for f in range(10)), "the LightGBM bag != W17's DC"
REF = mix("S5", 0.075, DC, DCVEC)                    # W17
res = {}
for name, paths in arms.items():
    Dx, Vx = combine(paths)
    e = score(mix("S5", 0.075, Dx, Vx), REF)
    e["sealed_mix"] = round(0.57 * e["ten"] + 0.43 * e["returning"], 4)
    per = e.get("per_stranger", {})
    e["nominated"] = bool(all(d > 0 for d in e["draws"]) and e["ten"] > 0 and (min(per.values()) if per else e["min_user"]) >= -0.020
                          and e["returning"] >= -0.003)
    res[name] = e
    print(f"D535 {name} vs W17's LightGBM bag: {e}", flush=True)
json.dump(res, open(O_ / "t2_d535.json", "w"), indent=1, default=str)
print("D535 nominated: " + (", ".join(k for k, v in res.items() if v["nominated"]) or "none"), flush=True)
