#!/usr/bin/env python
"""T2-D154: per-user fine-tune recipe (Recipe.pu_lr / pu_epochs, ni26.train.peruser_for) on CPU:
  1. peruser_for(r) = PERUSER for every recipe but the three D154 ones, D155's r1_pulx, D156's r1_pulxx and D168's r1_pulxxx (no existing recipe changes its per-user copies);
  2. r1_pul / r1_pue / r1_pus / r1_pulx / r1_pulxx = PERUSER with lr 1e-3 / 40 epochs / 10 epochs / both / lr 1e-3 + 80 epochs, nothing else changed;
  3. the pooled training is untouched: fit_eval(r1_pu*) = fit_eval(r1) bitwise (FBCNet B3, small random data);
  4. the fine-tune follows the recipe: finetune(peruser_for(r1)) = finetune(PERUSER) bitwise, finetune(peruser_for(r1_pul)) differs;
  5. t2p_bench and t2p_final take the CNN copies' recipe from peruser_for (the only direct T.PERUSER left = T2-D166's FM heads).
  python tests/test_train_peruser.py  ->  "OK"
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from ni26 import train as T  # noqa: E402
from test_train_ema import data, maxd  # noqa: E402

D154 = {"r1_pul": dict(lr=1e-3), "r1_pue": dict(epochs=40), "r1_pus": dict(epochs=10),
        "r1_pulx": dict(lr=1e-3, epochs=40),                                 # T2-D155
        "r1_pulxx": dict(lr=1e-3, epochs=80),                                # T2-D156
        "r1_pulxxx": dict(lr=1e-3, epochs=160),                              # T2-D168
        "r1_pulxx_vp": dict(lr=1e-3, epochs=80), "r1_pulx_vp": dict(lr=1e-3, epochs=40),   # T2-D193 (pooled training NOT r1: augmented)
        "r1_pulxx_vp2": dict(lr=1e-3, epochs=80), "r1_pulx_vp2": dict(lr=1e-3, epochs=40),  # T2-D195 (10 mm jitter)
        "r1_pulxx_vp3": dict(lr=1e-3, epochs=80), "r1_pulx_vp3": dict(lr=1e-3, epochs=40),  # T2-D197 (15 mm)
        "r1_pulxx_vp4": dict(lr=1e-3, epochs=80), "r1_pulx_vp4": dict(lr=1e-3, epochs=40),  # T2-D197 (20 mm)
        "r1_pulxx_vp2_pv5": dict(lr=1e-3, epochs=80, vp=1, vp_a=0.5), "r1_pulx_vp2_pv5": dict(lr=1e-3, epochs=40, vp=1, vp_a=0.5),    # T2-D200
        "r1_pulxx_vp2_pv10": dict(lr=1e-3, epochs=80, vp=1, vp_a=1.0), "r1_pulx_vp2_pv10": dict(lr=1e-3, epochs=40, vp=1, vp_a=1.0),
        **{f"r1_pulxx_vp2_{a}": dict(lr=1e-3, epochs=80) for a in ("e90", "e120", "k256", "e180", "e240", "e360")},           # T2-D202..D204
        **{f"r1_pulx_vp2_{a}": dict(lr=1e-3, epochs=40) for a in ("e90", "e120", "k256", "e180", "e240", "e360")},
        "r1_pulxx_vp2_e180_pu40": dict(lr=1e-3, epochs=40), "r1_pulxx_vp2_e180_pu160": dict(lr=1e-3, epochs=160),   # T2-D213
        "r1_pulx_vp2_e180_pu20": dict(lr=1e-3, epochs=20), "r1_pulx_vp2_e180_pu80": dict(lr=1e-3, epochs=80),
        "r1_pulxx_vp2_e180_g2": dict(lr=1e-3, epochs=80), "r1_pulxx_vp2_e180_s6": dict(lr=1e-3, epochs=80),   # T2-D214
        "r1_pulx_vp2_e180_g2": dict(lr=1e-3, epochs=40), "r1_pulx_vp2_e180_s6": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr": dict(lr=1e-3, epochs=40),   # T2-D215
        "r1_pulxx_vp2_e180_lr2": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_lr2": dict(lr=1e-3, epochs=40),   # T2-D216: the copies unchanged
        "r1_pulxx_vp2_e180_wd5": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_wd5": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr10": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr10": dict(lr=1e-3, epochs=40),   # T2-D220: the copies unchanged
        "r1_pulxx_vp2_e180_vr_mx4": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_mx4": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr_ls0": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_ls0": dict(lr=1e-3, epochs=40),   # T2-D223: the copies unchanged (PERUSER LS 0.1)
        "r1_pulxx_vp2_e180_vr_ls2": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_ls2": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr_sam05": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_sam05": dict(lr=1e-3, epochs=40),   # T2-D224: the copies unchanged
        "r1_pulxx_vp2_e180_vr_sam2": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_sam2": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr_bs32": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_bs32": dict(lr=1e-3, epochs=40),   # T2-D225: the copies unchanged
        "r1_pulxx_vp2_e180_vr_bs128": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_bs128": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr_pls2": dict(lr=1e-3, epochs=80, label_smoothing=0.2), "r1_pulx_vp2_e180_vr_pls2": dict(lr=1e-3, epochs=40, label_smoothing=0.2),   # T2-D226
        "r1_pulxx_vp2_e180_vr_pls3": dict(lr=1e-3, epochs=80, label_smoothing=0.3), "r1_pulx_vp2_e180_vr_pls3": dict(lr=1e-3, epochs=40, label_smoothing=0.3),
        "r1_pulxx_vp2_e180_vr_psw25": dict(lr=1e-3, epochs=80, swa=0.25), "r1_pulx_vp2_e180_vr_psw25": dict(lr=1e-3, epochs=40, swa=0.25),   # T2-D227
        "r1_pulxx_vp2_e180_vr_psw50": dict(lr=1e-3, epochs=80, swa=0.5), "r1_pulx_vp2_e180_vr_psw50": dict(lr=1e-3, epochs=40, swa=0.5),
        "r1_pulxx_vp2_e180_vr_phead": dict(lr=1e-3, epochs=80, train_part="head"), "r1_pulx_vp2_e180_vr_phead": dict(lr=1e-3, epochs=40, train_part="head"),   # T2-D228
        "r1_pulxx_vp2_e180_vr_pfeat": dict(lr=1e-3, epochs=80, train_part="feat"), "r1_pulx_vp2_e180_vr_pfeat": dict(lr=1e-3, epochs=40, train_part="feat"),
        "r1_pulxx_vp2_e180_vr_fw05": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_fw05": dict(lr=1e-3, epochs=40),   # T2-D229: copies = the vr line's
        "r1_pulxx_vp2_e180_vr_fw10": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_fw10": dict(lr=1e-3, epochs=40),
        "r1_pulxx_vp2_e180_vr_rc5": dict(lr=1e-3, epochs=80), "r1_pulxx_vp2_e180_vr_rc10": dict(lr=1e-3, epochs=80),   # T2-D230: copies = the vr line's
        **{f"r1_pulxx_vp2_e180_vr_{t}": dict(lr=1e-3, epochs=80) for t in ("rc25", "rc75")},   # T2-D231: copies = the vr line's
        **{f"r1_pulx_vp2_e180_vr_{t}": dict(lr=1e-3, epochs=40) for t in ("rc75", "rc10")},   # T2-D233 (ShallowNet): copies = the vr line's
        **{f"r1_pulxx_vp2_e180_vr_{t}": dict(lr=1e-3, epochs=80) for t in ("rcu25", "rcu50")},   # T2-D236: copies = the vr line's
        **{f"r1_pulxx_vp2_e180_vr_{t}": dict(lr=1e-3, epochs=80) for t in ("rc75m1", "rc75m2")},   # T2-D237: copies = the vr line's
        "r1_pulxx_vp2_e180_vr_rc75_pux1": dict(lr=1e-3, epochs=40), "r1_pulxx_vp2_e180_vr_rc75_pux3": dict(lr=1e-3, epochs=20),   # T2-D239
        **{f"r1_pulxx_vp2_e180_vr_rc75_{t}": dict(lr=1e-3, epochs=80) for t in ("ua1", "ua3")},   # T2-D293: copies = the lines' (no adversary)
        **{f"r1_pulx_vp2_e180_vr_{t}": dict(lr=1e-3, epochs=40) for t in ("ua1", "ua3")},
        **{f"r1_pulxx_vp2_e180_vr_rc75_{t}": dict(lr=1e-3, epochs=80) for t in ("kd", "kdx")},   # backlog 69: copies = the lines'
        **{f"r1_pulx_vp2_e180_vr_{t}": dict(lr=1e-3, epochs=40) for t in ("kd", "kdx")},
        "r1_pulxx_vp2_e180_vr_rcx75": dict(lr=1e-3, epochs=80),   # backlog 65 (a): copies = the line's
        "r1_pulxx_vp2_e180_vr_rc75_cap10": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_cap10": dict(lr=1e-3, epochs=40),   # backlog 65 (b)
        "r1_pulxx_vp2_e180_vr_rc75_ms5": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_ms5": dict(lr=1e-3, epochs=40),   # backlog 70
        "r1_pulxx_vp2_e180_vr_rc75_pv10": dict(lr=1e-3, epochs=80, vp=1, vp_a=1.0), "r1_pulx_vp2_e180_vr_pv10": dict(lr=1e-3, epochs=40, vp=1, vp_a=1.0),   # backlog 72
        "r1_pulxx_vp2_e180_vr_rc75_xm5": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_xm5": dict(lr=1e-3, epochs=40),   # backlog 73
        "r1_pulxx_vp2_e180_vr_rc75_tl15": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_tl15": dict(lr=1e-3, epochs=40),   # backlog 74
        "r1_pulxx_vp2_e360_vr_rc75_tl15": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e360_vr": dict(lr=1e-3, epochs=40),   # T2-D382: copies = the lines'
        "r1_pulxx_vp2_e180_vr_rc75_tl15_pv10": dict(lr=1e-3, epochs=80, vp=1, vp_a=1.0),   # T2-D388: S1's pv10 copies on the tilt line
        "r1_pulxx_vp2_e180_vr_rc75cc_tl15": dict(lr=1e-3, epochs=80),   # T2-D412: copies = the tilt line's (rc_cls never reaches them)
        "r1_pulxx_vp2_e180_vr_rc75_tl15_mf50": dict(lr=1e-3, epochs=80),   # T2-D413: copies = the tilt line's (syn never reaches them)
        "r1_pulxx_vp2_e180_vr_rc75_tl15_mf20": dict(lr=1e-3, epochs=80),   # T2-D561: = mf50's (syn 0.2; copies = the tilt line's)
        "r1_pulx_vp2_e180_vr_mf20": dict(lr=1e-3, epochs=40), "r1_pulx_vp2_e180_vr_mf50": dict(lr=1e-3, epochs=40),   # T2-D561: = the line's copies
        "r1_pulx_vp2_e180_vr_smx4": dict(lr=1e-3, epochs=40),   # T2-D580: = the side line's copies (side_mix never reaches them)
        "r1_pulx_vp2_e180_vr_pmx4": dict(lr=1e-3, epochs=40, ft_mix=0.4),   # T2-D589: the copies get ft_mix = pu_mix (the copies' mixup)
        "r1_pulxx_vp2_e180_vr_rc75_mf20": dict(lr=1e-3, epochs=80),   # T2-D576: = the stack FBCNet line's copies (syn never reaches them)
        **{f"r1_pulxx_vp2_e180_vr_rc75_{t}": dict(lr=1e-3, epochs=80) for t in ("dm03", "dm1", "dc03", "dc1")},   # T2-D374: copies = the line's (no adversary)
        "r1_pulxx_vp2_e180_vr_rc75_vs5": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_vs5": dict(lr=1e-3, epochs=40),   # T2-D308
        "r1_pulxx_vp2_e180_vr_rc75_us5": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_us5": dict(lr=1e-3, epochs=40),   # T2-D365
        "r1_pulxx_vp2_e180_vr_rc75_gd3": dict(lr=1e-3, epochs=80), "r1_pulx_vp2_e180_vr_gd3": dict(lr=1e-3, epochs=40),   # T2-D315: copies = the lines' (plain CE)
        "r1_pulxx_vp3_e180_vr_rc75": dict(lr=1e-3, epochs=80), "r1_pulx_vp3_e180_vr": dict(lr=1e-3, epochs=40),   # T2-D339: copies = the vp2 lines' (pu_vp off)
        "r1_pulxx_vp2_e180_vr_rc75_kl1": dict(lr=1e-3, epochs=80, kl_other=1.0), "r1_pulx_vp2_e180_vr_kl1": dict(lr=1e-3, epochs=40, kl_other=1.0),   # T2-D430: the copies get kl_other = pu_kl
        # T2-D478 (on purpose, as pre-registered: "pooled nets and copies alike"): the copies inherit sbal; a copy's batches hold its user's streams
        "r1_pulxx_vp2_e180_vr_rc75_sb16": dict(lr=1e-3, epochs=80, sbal=16), "r1_pulx_vp2_e180_vr_sb16": dict(lr=1e-3, epochs=40, sbal=16),
        "r1_pulx_vp2_e65_vr": dict(lr=1e-3, epochs=40),   # T2-D547: the side ShallowNet nets at 65 epochs (copies = the e180 line's)
        "r1_pulxx_vp2_e240_vr_rc75": dict(lr=1e-3, epochs=80)}   # T2-D548: FBCNet at 240 epochs (copies = the e180 rc75 line's)


def main():
    for name, r in T.RECIPES.items():
        want = replace(T.PERUSER, **D154[name]) if name in D154 else T.PERUSER
        assert T.peruser_for(r) == want, name
    print(f"peruser_for = PERUSER for {len(T.RECIPES) - len(D154)} recipes; D154 variants as registered", flush=True)
    d = data()
    r1 = replace(T.RECIPES["r1"], epochs=2)
    ref = T.fit_eval("fbcnet_bboth", d, r1, 0, device="cpu", return_state=True)
    for name in [n for n in D154 if not T.RECIPES[n].vp]:
        got = T.fit_eval("fbcnet_bboth", d, replace(T.RECIPES[name], epochs=2), 0, device="cpu", return_state=True)
        assert maxd(got["state_last"], ref["state_last"]) == 0.0 and np.array_equal(got["prob_last"], ref["prob_last"]), name
    print(f"pooled fit_eval of {' / '.join(n for n in D154 if not T.RECIPES[n].vp)} = r1 bitwise", flush=True)
    net = T.build_model("fbcnet_bboth", d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
    net.load_state_dict(ref["state_last"])
    X, y = torch.as_tensor(d["Xtr"][:32]), torch.as_tensor(d["ytr"][:32], dtype=torch.long)
    a = T.finetune(net, X, y, replace(T.peruser_for(T.RECIPES["r1"]), epochs=2), 5).state_dict()
    b = T.finetune(net, X, y, replace(T.PERUSER, epochs=2), 5).state_dict()
    c = T.finetune(net, X, y, replace(T.peruser_for(T.RECIPES["r1_pul"]), epochs=2), 5).state_dict()
    assert maxd(a, b) == 0.0 and maxd(c, b) > 1e-6, (maxd(a, b), maxd(c, b))
    print(f"finetune: r1 = PERUSER bitwise; r1_pul moves the copy by {maxd(c, b):.2e}", flush=True)
    for f in ("tracks/t2_bci/scripts/t2p_bench.py", "tracks/t2_bci/scripts/t2p_final.py"):
        s = (ROOT / f).read_text()
        rest = s.replace("_replace(T.PERUSER, lr=1e-3, epochs=40)", "")      # T2-D166: the FM heads' own recipe (not a CNN copy)
        assert "T.PERUSER" not in rest and "T.peruser_for(recipe)" in s, f
    print("OK")


if __name__ == "__main__":
    main()
