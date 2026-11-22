#!/usr/bin/env python
"""T2-D232: the package builder's re-colouring wiring (t2p_final --recipe2, user_index, cnn2_recipe), CPU, no export needed:
  1. user_index = t2p_bench's training-user index (rank among the sorted training subjects);
  2. cnn2_recipe: --recipe2 unset = the earlier behaviour (replace(RECIPES[--recipe], epochs=--epochs2)); --recipe2 r1_pulxx_vp2_e180_vr with
     --recipe ..._rc5 gives ShallowNet the vr line (rc 0) and FBCNet keeps rc 0.5; a --recipe2 with another virtual-participant bank is refused;
  3. the builder passes `utr` to fit_eval only for an rc or user_adv (T2-D293) recipe and writes cfg["cnn2"]["recipe"] only when --recipe2 is set (flags off = the
     earlier build byte for byte), and t2p_final's training data feeds fit_eval's rc branch (a 2-epoch CPU fit with rc 0.5 runs and moves the weights).
  python tests/test_t2p_final_rc.py  ->  "OK"
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from ni26 import train as T  # noqa: E402
import t2p_final as F  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    subj = rng.choice(["P01", "P02", "P03", "P06", "P07", "P10"], 500)
    bench = pd.Series(subj).map({u: k for k, u in enumerate(sorted(pd.Series(subj).unique()))}).to_numpy()
    assert np.array_equal(F.user_index(subj), bench) and F.user_index(subj).dtype == np.int64
    print("user_index = t2p_bench's training-user index")
    rc5, vr = "r1_pulxx_vp2_e180_vr_rc5", "r1_pulxx_vp2_e180_vr"
    assert F.cnn2_recipe(rc5, "", 180) == replace(T.RECIPES[rc5], epochs=180), "unset = --recipe"
    assert F.cnn2_recipe(vr, "", 180) == replace(T.RECIPES[vr], epochs=180)
    got = F.cnn2_recipe(rc5, vr, 180)
    assert got == replace(T.RECIPES[vr], epochs=180) and got.rc == 0 and T.RECIPES[rc5].rc == 0.5
    for bad in ("r1_pulx", "r1_pulxx_vp2"):
        try:
            F.cnn2_recipe(rc5, bad, 180)
            raise SystemExit(f"cnn2_recipe must refuse {bad}")
        except AssertionError:
            pass
    print("cnn2_recipe: unset = --recipe; --recipe2 vr gives ShallowNet rc 0; another vp bank refused")
    src = (ROOT / "tracks/t2_bci/scripts/t2p_final.py").read_text()
    assert 'ap.add_argument("--recipe2", default=""' in src and "    if args.recipe2:" in src
    assert "cte=cells[ev], **rc_u, **wub)" in src and "cte=cells[ev], **rc_u2, **wub)" in src   # T2-D638: + the W24' user-balance spread
    assert "    wub = {}  " in src and "    if args.user_balance:" in src   # T2-D638: wub is {} unless --user-balance (absent = bitwise)
    assert "rc_u = {}\n    if recipe.rc > 0 or recipe.user_adv > 0:" in src and "if recipe2.rc > 0 or recipe2.user_adv > 0 else {}" in src   # T2-D293: + DANN
    d = data()
    d["utr"] = F.user_index(np.array(["P0%d" % (k + 1) for k in d["utr"]]))
    r = replace(T.RECIPES["r1"], epochs=2)
    a = T.fit_eval("fbcnet_bboth", d, r, 0, device="cpu", return_state=True)
    b = T.fit_eval("fbcnet_bboth", d, replace(r, rc=0.5), 0, device="cpu", return_state=True)
    moved = max(float((a["state_last"][k] - b["state_last"][k]).abs().max()) for k in a["state_last"] if a["state_last"][k].is_floating_point())
    assert moved > 1e-4, moved
    print(f"fit_eval with the builder's utr and rc 0.5 runs; weights move by {moved:.1e}")
    print("OK")


if __name__ == "__main__":
    main()
