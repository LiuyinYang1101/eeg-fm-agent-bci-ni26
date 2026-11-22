#!/usr/bin/env python
"""T2-D412: CLASS-CONDITIONAL re-colouring (Recipe.rc_cls -> ni26.train.rc_bank_cls / augment), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75cc_tl15 = M6's tilt FBCNet line r1_pulxx_vp2_e180_vr_rc75_tl15 + rc_cls 1 only; its per-user copies are the
     tl15 line's; it is the only recipe with rc_cls;
  2. rc_bank_cls: shape (n_classes, n_users, n_users, C, C); M[c] = rc_bank over class c's windows bitwise; at rc 1, T C_{u,c} T = C_{v,c};
     a training user with no window of some class is refused;
  3. augment with rc_cls: each window of user u, class c is multiplied by M[c, u, v] for a drawn v != u (replayed from the same RNG state, the
     same v draw as the class-pooled branch); without yc it is refused; rc_cls 0 = the class-pooled branch bitwise;
  4. fit_eval with rc_cls runs (rc 1, 1 epoch) and moves the weights vs class-pooled rc on the same seed.
  python tests/test_rc_cls.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402
from test_rc import covs, data  # noqa: E402


def main():
    b, r = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_tl15"], T.RECIPES["r1_pulxx_vp2_e180_vr_rc75cc_tl15"]
    assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "rc_cls"} and r.rc_cls == 1 and r.rc == 0.75 and r.tilt == 0.15
    assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).rc_cls == 0, "the copies must be unchanged"
    assert [n for n, x in T.RECIPES.items() if x.rc_cls] == ["r1_pulxx_vp2_e180_vr_rc75cc_tl15"]
    d = data(n_users=3)
    X, u, y = torch.as_tensor(d["Xtr"]), torch.as_tensor(d["utr"]), torch.as_tensor(d["ytr"])
    Mc = T.rc_bank_cls(X, u, y, 3, 3, 1.0)
    assert tuple(Mc.shape) == (3, 3, 3, 41, 41), Mc.shape
    err = 0.0
    for c in range(3):
        k = y == c
        assert torch.equal(Mc[c], T.rc_bank(X[k], u[k], 3, 1.0)), c
        Cs = covs(X[k], u[k], 3)
        for a in range(3):
            for v in range(3):
                if a != v:
                    Tm = Mc[c, a, v].double().numpy()
                    err = max(err, np.abs(Tm @ Cs[a] @ Tm.T - Cs[v]).max() / np.abs(Cs[v]).max())
    assert err < 1e-4, err
    print(f"rc_bank_cls: per-class rc_bank bitwise; T C_uc T = C_vc (max rel err {err:.1e})")
    y_bad = y.clone(); y_bad[(u == 1) & (y == 2)] = 0
    try:
        T.rc_bank_cls(X, u, y_bad, 3, 3, 1.0)
        raise SystemExit("FAIL: a user without class-2 windows must be refused")
    except AssertionError:
        pass
    rec = replace(T.RECIPES["r1"], rc=1.0, rc_cls=1)
    xb, ub, yb = X[:12], u[:12], y[:12]
    torch.manual_seed(7)
    out = T.augment(xb.clone(), rec, ub, Mc, yc=yb)
    torch.manual_seed(7)
    v = (ub + torch.randint(1, 3, (12,))) % 3
    want = torch.bmm(Mc[yb, ub, v], xb)
    assert torch.allclose(out, want, atol=1e-5), (out - want).abs().max()
    try:
        T.augment(xb.clone(), rec, ub, Mc)
        raise SystemExit("FAIL: rc_cls without yc must be refused")
    except AssertionError:
        pass
    pooled = replace(T.RECIPES["r1"], rc=1.0)
    rcm = T.rc_bank(X, u, 3, 1.0)
    torch.manual_seed(7)
    a0 = T.augment(xb.clone(), pooled, ub, rcm)
    torch.manual_seed(7)
    a1 = T.augment(xb.clone(), pooled, ub, rcm, yc=None)
    assert torch.equal(a0, a1)
    torch.manual_seed(7)
    v = (ub + torch.randint(1, 3, (12,))) % 3
    assert torch.allclose(a0, torch.bmm(rcm[ub, v], xb), atol=1e-5)
    print("augment: rc_cls = M[c, u, v] x window (replayed); refused without yc; rc_cls 0 = class-pooled bitwise")
    plain = replace(T.RECIPES["r1"], epochs=2)
    p0 = T.fit_eval("fbcnet_bboth", d, replace(plain, rc=1.0), 0, device="cpu", return_state=True)
    c0 = T.fit_eval("fbcnet_bboth", d, replace(plain, rc=1.0, rc_cls=1), 0, device="cpu", return_state=True)
    sp, sc = p0["state_last"], c0["state_last"]
    moved = max(float((sp[k] - sc[k]).abs().max()) for k in sp if sp[k].is_floating_point())
    assert moved > 1e-4, moved
    print(f"fit_eval: rc_cls runs and moves the weights vs class-pooled rc (same seed) by up to {moved:.1e}")
    print("OK")


if __name__ == "__main__":
    main()
