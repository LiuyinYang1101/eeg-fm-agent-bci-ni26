#!/usr/bin/env python
"""T2-D374: the tuned DANN (Recipe.adv_ramp / adv_hidden / adv_lr_mult / adv_cond), CPU:
  1. the four rc75 DANN recipes differ from the FBCNet rc75 line only in name, user_adv and the adv_* fields; copies unchanged;
  2. fit_eval runs the marginal and the class-conditional MLP adversary with the ramp, moves the weights, leaves no hook behind (eval = the
     plain net), prints the engagement line; T2-D293's fixed linear adversary (adv_* off) and the plain line stay bitwise reproducible.
  python tests/test_dann.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ni26.train as T  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    b = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75"]
    for t, lam, cond in (("dm03", 0.03, 0), ("dm1", 0.1, 0), ("dc03", 0.03, 1), ("dc1", 0.1, 1)):
        r = T.RECIPES[f"r1_pulxx_vp2_e180_vr_rc75_{t}"]
        diff = {k for k, v in asdict(r).items() if v != asdict(b)[k]}
        assert diff == {"name", "user_adv", "adv_ramp", "adv_hidden", "adv_lr_mult"} | ({"adv_cond"} if cond else set()), (t, diff)
        assert (r.user_adv, r.adv_ramp, r.adv_hidden, r.adv_lr_mult, r.adv_cond) == (lam, 1, 128, 10.0, cond), t
        assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    print("recipes: the four DANN recipes differ from the rc75 line in the adversary fields only; copies unchanged")
    d = data()
    plain = replace(T.RECIPES["r1"], epochs=2)
    a0 = T.fit_eval("shallow", d, plain, 0, device="cpu", return_state=True)
    a0b = T.fit_eval("shallow", d, plain, 0, device="cpu", return_state=True)
    assert all(torch.equal(a0["state_last"][k], a0b["state_last"][k]) for k in a0["state_last"]), "the plain line must be bitwise reproducible"
    f1 = T.fit_eval("shallow", d, replace(plain, user_adv=0.1), 0, device="cpu", return_state=True)
    f2 = T.fit_eval("shallow", d, replace(plain, user_adv=0.1), 0, device="cpu", return_state=True)
    assert all(torch.equal(f1["state_last"][k], f2["state_last"][k]) for k in f1["state_last"]), "T2-D293's fixed adversary must stay reproducible"
    for cond in (0, 1):
        rec = replace(plain, user_adv=0.1, adv_ramp=1, adv_hidden=16, adv_lr_mult=10.0, adv_cond=cond)
        a1 = T.fit_eval("shallow", d, rec, 0, device="cpu", return_state=True)
        sa, sb = a0["state_last"], a1["state_last"]
        moved = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
        assert moved > 1e-6, (cond, moved)
        net = T.build_model("shallow", d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
        net.load_state_dict(sb)
        assert not net.final_layer._forward_pre_hooks
        assert np.allclose(T.predict_proba(net, torch.as_tensor(d["Xte"])), a1["prob_last"], atol=1e-5), "eval = the plain net"
        print(f"fit_eval tuned DANN (cond {cond}): runs, weights move by up to {moved:.1e}, no hook left")
    print("OK")


if __name__ == "__main__":
    main()
