#!/usr/bin/env python
"""T2-D293: user-adversarial (DANN) pooled training (Recipe.user_adv -> ni26.train.fit_eval), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_ua1 / _ua3 differ from the FBCNet rc75 line, and r1_pulx_vp2_e180_vr_ua1 / _ua3 from the ShallowNet vr line,
     in `user_adv` only (0.1 / 0.3); their per-user copies are the lines' (no adversary); only these four recipes set user_adv;
  2. GradReverse: identity forward, -lam x the gradient backward;
  3. fit_eval with user_adv 0 = r1 bitwise (FBCNet B3 and ShallowNet);
  4. fit_eval with user_adv 0.3 (plain, and with rc 0.75 for FBCNet) runs, moves the weights, and returns the plain net's state_dict keys and
     shapes (the adversary is discarded); a fresh net loaded with that state reproduces prob_last (no hook left in the predictions);
  5. the branch guard: user_adv with mixup is refused.
  python tests/test_user_adv.py  ->  "OK"
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
    want = {"r1_pulxx_vp2_e180_vr_rc75": ("r1_pulxx_vp2_e180_vr_rc75_ua1", "r1_pulxx_vp2_e180_vr_rc75_ua3"),
            "r1_pulx_vp2_e180_vr": ("r1_pulx_vp2_e180_vr_ua1", "r1_pulx_vp2_e180_vr_ua3")}
    for base, names in want.items():
        b = T.RECIPES[base]
        for n, lam in zip(names, (0.1, 0.3)):
            r = T.RECIPES[n]
            assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "user_adv"} and r.user_adv == lam, n
            assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).user_adv == 0, "the copies must be unchanged"
    d374 = [f"r1_pulxx_vp2_e180_vr_rc75_{t}" for t in ("dm03", "dm1", "dc03", "dc1")]   # T2-D374's tuned DANN (checked in tests/test_dann.py)
    b = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75"]
    for n in d374:
        r = T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} <= {"name", "user_adv", "adv_ramp", "adv_hidden", "adv_lr_mult", "adv_cond"} and r.user_adv > 0, n
        assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).user_adv == 0, "the copies must be unchanged"
    assert sorted(n for n, r in T.RECIPES.items() if r.user_adv) == sorted(sum(want.values(), ()) + tuple(d374)), "only D293's four + D374's four"
    print("recipes: the four ua recipes differ in user_adv only, D374's four in the adversary fields only; copies unchanged")

    x = torch.randn(5, 7, requires_grad=True)
    g = torch.randn(5, 7)
    y = T.GradReverse.apply(x, 0.3)
    assert torch.equal(y, x)
    y.backward(g)
    assert torch.allclose(x.grad, -0.3 * g), "GradReverse backward = -lam g"
    print("GradReverse: identity forward, -lam g backward")

    d = data()
    plain = replace(T.RECIPES["r1"], epochs=2)
    for m in ("fbcnet_bboth", "shallow"):
        a = T.fit_eval(m, d, plain, 0, device="cpu", return_state=True)
        b = T.fit_eval(m, d, replace(plain, user_adv=0.0), 0, device="cpu", return_state=True)
        assert all(torch.equal(a["state_last"][k], b["state_last"][k]) for k in a["state_last"]) and \
            np.array_equal(a["prob_last"], b["prob_last"]), f"{m}: user_adv 0 must be r1 bitwise"
        runs = [replace(plain, user_adv=0.3)] + ([replace(plain, user_adv=0.3, rc=0.75)] if m == "fbcnet_bboth" else [])
        for rec in runs:
            c = T.fit_eval(m, d, rec, 0, device="cpu", return_state=True)
            ref = a if rec.rc == 0 else T.fit_eval(m, d, replace(rec, user_adv=0.0), 0, device="cpu", return_state=True)
            sa, sc = ref["state_last"], c["state_last"]
            assert sa.keys() == sc.keys() and all(sa[k].shape == sc[k].shape for k in sa), f"{m}: the adversary must not enter the state"
            moved = max(float((sa[k].float() - sc[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
            assert moved > 1e-5, (m, moved)
            net = T.build_model(m, d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0)
            net.load_state_dict(sc)
            p = T.predict_proba(net, torch.as_tensor(d["Xte"]))
            assert np.allclose(p, c["prob_last"], atol=1e-6), f"{m}: a fresh net with the state must reproduce prob_last"
            print(f"{m}: user_adv 0 = r1 bitwise; user_adv 0.3{' + rc 0.75' if rec.rc else ''} moves the weights by up to {moved:.1e}, "
                  "state keys / shapes = the plain net's, prob_last reproduced by a fresh net")
    try:
        T.fit_eval("shallow", d, replace(plain, user_adv=0.3, mixup=0.4), 0, device="cpu")
        raise SystemExit("user_adv + mixup must be refused")
    except AssertionError as e:
        assert "T2-D293" in str(e), e
    print("guard: user_adv + mixup refused")
    print("OK")


if __name__ == "__main__":
    main()
