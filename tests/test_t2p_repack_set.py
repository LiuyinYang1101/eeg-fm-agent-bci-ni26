#!/usr/bin/env python
"""t2p_repack's --set edits (config-only repacks, 2026-11-01), CPU, no export needed.
  1. a dotted dict edit behaves as before (cnn2.weight; a missing key reads as None);
  2. a numeric part indexes a list (fm_members.0.weight), so the F edit can move the ST-L share;
  3. the U / F edits on configs shaped like the approved Nov 5 ZIPs (A1 MOR-free 4:3:2, A4 4:3:2:1) pass check_weights,
     with the intended family shares; an inconsistent edit fails it (the template asserts the same sum at load).
  python tests/test_t2p_repack_set.py  ->  "OK"
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_repack as R  # noqa: E402

A1 = {"fm_weight": 2 / 9, "fm_members": [{"family": "st", "weight": 2 / 9, "file": "fm_0.pt"}],
      "cnn2": {"weight": 1 / 3, "file": "cnn2.pt"}, "user_blend": {"r": 0.3333, "n_seeds": 15, "every": 1},
      "user_blend2": {"r": 0.5, "n_seeds": 15}}
A4 = {"fm_weight": 0.3, "fm_members": [{"family": "st", "weight": 0.2, "file": "fm_0.pt"}, {"family": "mor", "weight": 0.1, "file": "fm_1.pt"}],
      "cnn2": {"weight": 0.3, "file": "cnn2.pt"}, "user_blend": {"r": 0.3333, "n_seeds": 15, "every": 1},
      "user_blend2": {"r": 0.5, "n_seeds": 15}}


def shares(c):
    fm = {m["family"]: m["weight"] for m in c["fm_members"]}
    return dict(fbc=1 - c["fm_weight"] - c["cnn2"]["weight"], sh=c["cnn2"]["weight"], **fm)


def main():
    c = copy.deepcopy(A1)
    assert R.set_cfg(c, "cnn2.weight=0.3") == ("cnn2.weight", 1 / 3, 0.3)
    assert R.set_cfg(c, "cnn2.newkey=1")[1] is None and c["cnn2"]["newkey"] == 1
    k, was, now = R.set_cfg(c, "fm_members.0.weight=0.25")
    assert (k, was, now) == ("fm_members.0.weight", 2 / 9, 0.25) and c["fm_members"][0]["weight"] == 0.25
    try:
        R.set_cfg(copy.deepcopy(A1), "fm_members.3.weight=0.1")
        raise AssertionError("a bad list index must raise")
    except IndexError:
        pass

    c = copy.deepcopy(A1)                                  # U: copy odds x 2 in both families
    for kv in ("user_blend.r=0.5", "user_blend2.r=0.6667"):
        R.set_cfg(c, kv)
    R.check_weights(c)
    assert c["user_blend"]["r"] == 0.5 and c["user_blend2"]["r"] == 0.6667 and c["user_blend"]["n_seeds"] == 15

    c = copy.deepcopy(A1)                                  # F, MOR-free: 4:3:2 -> 4:3:3
    for kv in ("fm_weight=0.3", "fm_members.0.weight=0.3", "cnn2.weight=0.3"):
        R.set_cfg(c, kv)
    R.check_weights(c)
    s = shares(c)
    assert all(abs(s[f] - v) < 1e-12 for f, v in dict(fbc=0.4, sh=0.3, st=0.3).items()), s

    c = copy.deepcopy(A4)                                  # F, MOR base: fm_weight 0.3 -> 0.4, ST-L : MOR 2 : 1 and FBCNet : ShallowNet 4 : 3 kept
    for kv in (f"fm_weight={json.dumps(0.4)}", f"fm_members.0.weight={json.dumps(0.4 * 2 / 3)}",       # (amendment 2026-11-01 19:1x: a
               f"fm_members.1.weight={json.dumps(0.4 / 3)}", f"cnn2.weight={json.dumps(0.6 * 3 / 7)}"):  # within-FM ratio change fails the parts check)
        R.set_cfg(c, kv)
    R.check_weights(c)
    s = shares(c)
    assert all(abs(s[f] - v) < 1e-12 for f, v in dict(fbc=0.6 * 4 / 7, sh=0.6 * 3 / 7, st=0.4 * 2 / 3, mor=0.4 / 3).items()), s
    assert abs(s["st"] / s["mor"] - 2.0) < 1e-12 and abs(s["fbc"] / s["sh"] - 4 / 3) < 1e-12

    for bad in (("fm_weight=0.3",), ("cnn2.weight=0.8",)):  # member sum != fm_weight; FBCNet share <= 0
        c = copy.deepcopy(A1)
        for kv in bad:
            R.set_cfg(c, kv)
        try:
            R.check_weights(c)
            raise AssertionError(f"check_weights must reject {bad}")
        except AssertionError as e:
            assert "must reject" not in str(e), e
    R.check_weights(copy.deepcopy(A1)); R.check_weights(copy.deepcopy(A4))   # the approved shapes pass unchanged
    print("OK")


if __name__ == "__main__":
    main()
