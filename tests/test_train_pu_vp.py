#!/usr/bin/env python
"""T2-D200: per-user copies through the virtual-participant bank (Recipe.pu_vp / vp_a, ni26.train.finetune) on CPU:
  1. r1's fit_eval and PERUSER's finetune through the current train.py = the pre-D200 train.py (REF) bitwise (FBCNet B3; state and probabilities);
  2. r1_pulxx_vp2 (pu_vp 0) copies = REF's copies bitwise (the copies stay unaugmented);
  3. r1_pulxx_vp2_pv5 copies differ (they train through the bank at strength 0.5).
  python tests/test_train_pu_vp.py  ->  "OK"
"""
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = "c559970^"   # train.py before T2-D200 (pinned: later edits must keep r1 and the vp2 copies bitwise)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from ni26 import train as T  # noqa: E402
from test_train_ema import data, maxd  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "train_ref.py"
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF}:src/ni26/train.py"], capture_output=True, text=True, check=True).stdout)
        spec = importlib.util.spec_from_file_location("ni26.train_ref", ref)
        old = importlib.util.module_from_spec(spec); sys.modules["ni26.train_ref"] = old; spec.loader.exec_module(old)
    assert not hasattr(old.Recipe(), "pu_vp"), "the reference already has Recipe.pu_vp"
    d = data()
    rn, ro = T.fit_eval("fbcnet_bboth", d, replace(T.RECIPES["r1"], epochs=2), 0, device="cpu", return_state=True), \
        old.fit_eval("fbcnet_bboth", d, replace(old.RECIPES["r1"], epochs=2), 0, device="cpu", return_state=True)
    assert maxd(rn["state_last"], ro["state_last"]) == 0.0 and np.array_equal(rn["prob_last"], ro["prob_last"])
    net = T.build_model("fbcnet_bboth", d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0); net.load_state_dict(rn["state_last"])
    X, y = torch.as_tensor(d["Xtr"][:32]), torch.as_tensor(d["ytr"][:32], dtype=torch.long)
    C = X.shape[1]
    g = torch.Generator().manual_seed(0)
    M = torch.eye(C) + 0.05 * torch.randn(8, C, C, generator=g)
    T.VP_BANK.clear(); T.VP_BANK.update(mats=M, gains=1 + 0.1 * torch.rand(8, C, generator=g), shifts=torch.zeros(8, dtype=torch.long))
    a = T.finetune(net, X, y, replace(T.peruser_for(T.RECIPES["r1_pulxx_vp2"]), epochs=2), 5).state_dict()
    b = old.finetune(net, X, y, replace(old.peruser_for(old.RECIPES["r1_pulxx_vp2"]), epochs=2), 5).state_dict()
    assert maxd(a, b) == 0.0, maxd(a, b)
    c = T.finetune(net, X, y, replace(T.peruser_for(T.RECIPES["r1_pulxx_vp2_pv5"]), epochs=2), 5).state_dict()
    assert maxd(c, b) > 1e-6, maxd(c, b)
    T.VP_BANK.clear()
    print(f"r1 fit_eval + vp2 copies = {REF} bitwise; pv5 copies move by {maxd(c, b):.2e}")
    print("OK")


if __name__ == "__main__":
    main()
