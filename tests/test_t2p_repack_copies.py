#!/usr/bin/env python
"""t2p_repack.merge_copies (the S1 copies swap, 2026-11-01), CPU, no export or zip needed.
  1. a pv10 donor of the same draw (FBCNet r1_pulxx_vp2_e180_vr_rc75_pv10, ShallowNet copies r1_pulx_vp2_e180_vr_pv10) merges:
     the base keeps its route (r, every, users, mean / scale / W / mu) and file names; the donor's copy provenance comes over;
  2. a donor whose POOLED recipe differs (MixStyle), whose route differs (other data), or whose seeds differ is refused;
  3. _tensors flattens nested state dicts / lists in a fixed order.
  python tests/test_t2p_repack_copies.py  ->  "OK"
"""
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_repack as R  # noqa: E402
from ni26 import train as T  # noqa: E402

ROUTE = dict(users=["P01", "P02"], mean=[0.1, 0.2], scale=[1.0, 2.0], W=[[0.5, 0.1], [0.2, 0.3]], mu=[[1.0, 2.0], [3.0, 4.0]])
BASE = {"model": "fbcnet_bboth", "seeds": "0,1,2", "epochs": 180, "align": {"mode": "none"}, "train": "all", "train_subjects": ["P01", "P02"],
        "n_times": 480, "sfreq": 120.0, "ch_names": ["C3", "C4"], "export_sha": "abc", "recipe": "r1_pulxx_vp2_e180_vr_rc75",
        "cnn2": {"model": "shallow", "seeds": "0,1,2", "epochs": 180, "align": {"mode": "ea"}, "recipe": "r1_pulxx_vp2_e180_vr", "file": "weights2.pt", "weight": 1 / 3},
        "user_blend": dict(r=0.3333, file="weights_pu.pt", n_seeds=3, every=1, **ROUTE),
        "user_blend2": dict(r=0.5, file="weights2_pu.pt", n_seeds=3, users=ROUTE["users"], recipe_pu="r1_pulx", pu_lr=1e-3, pu_epochs=40)}


def donor():
    d = copy.deepcopy(BASE)
    d["recipe"] = "r1_pulxx_vp2_e180_vr_rc75_pv10"
    d["user_blend"]["r"], d["user_blend2"]["r"] = 0.3333, 0.5           # t2p_final's flags; the base's r is kept regardless
    d["user_blend"]["mean"] = [0.1 + 1e-9, 0.2]                         # data-derived: equal up to float noise
    d["user_blend2"]["recipe_pu"] = "r1_pulx_vp2_e180_vr_pv10"
    return d


def refused(d, why):
    try:
        R.merge_copies(copy.deepcopy(BASE), d, T.RECIPES)
    except AssertionError:
        return
    raise AssertionError(f"merge_copies must refuse: {why}")


def main():
    assert "r1_pulxx_vp2_e180_vr_rc75" in T.RECIPES and "r1_pulx_vp2_e180_vr_pv10" in T.RECIPES
    new, files = R.merge_copies(copy.deepcopy(BASE), donor(), T.RECIPES)
    assert files == ["weights_pu.pt", "weights2_pu.pt"]
    ub, ub2 = new["user_blend"], new["user_blend2"]
    assert ub["r"] == 0.3333 and ub["every"] == 1 and ub["mean"] == BASE["user_blend"]["mean"] and ub["file"] == "weights_pu.pt"
    assert ub["copies_from_recipe"] == "r1_pulxx_vp2_e180_vr_rc75_pv10"
    assert ub2["r"] == 0.5 and ub2["recipe_pu"] == "r1_pulx_vp2_e180_vr_pv10" and ub2["copies_from_recipe"] == "r1_pulx_vp2_e180_vr_pv10"
    assert T.peruser_for(T.RECIPES[ub2["recipe_pu"]]).vp_a == 1.0 and T.peruser_for(T.RECIPES["r1_pulx"]).vp == 0   # pv10 = copies through the bank

    d = donor(); d["recipe"] = "r1_pulxx_vp2_e180_vr_rc75_ms5"; refused(d, "a different pooled recipe (MixStyle)")
    d = donor(); d["user_blend"]["mu"] = [[9.0, 2.0], [3.0, 4.0]]; refused(d, "another fingerprint route")
    d = donor(); d["seeds"] = "20,21,22"; refused(d, "another draw's seeds")
    d = donor(); d["cnn2"]["recipe"] = "r1_pulx_vp2_e180_vr"; refused(d, "another ShallowNet pooled recipe")
    d = donor(); del d["user_blend2"]; refused(d, "no second-family copies")

    sd = [{"b": torch.ones(2), "a": torch.zeros(3)}, {"a": torch.full((1,), 2.0), "n": 5}]
    ts = R._tensors(sd)
    assert [tuple(t.shape) for t in ts] == [(3,), (2,), (1,)] and float(ts[2]) == 2.0
    print("OK")


if __name__ == "__main__":
    main()
