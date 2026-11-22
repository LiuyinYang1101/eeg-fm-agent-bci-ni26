#!/usr/bin/env python
"""T2-D413: MeanFlow synthetic windows in the pooled CNN's batches (Recipe.syn -> ni26.train.SYN_BANK / fit_eval), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_tl15_mf50 = M6's tilt FBCNet line r1_pulxx_vp2_e180_vr_rc75_tl15 + syn 0.5 only; its per-user copies are
     the tl15 line's (syn 0); it is the only recipe with syn;
  2. syn > 0: every batch of B real windows reaches augment with round(syn B) bank windows appended (their labels / dominant-parent users
     appended too, so rc maps them from that user); refused without SYN_BANK and with a branch that indexes by b (user_adv);
  3. syn 0 with a bank loaded = syn 0 without one, bitwise (the bank is never touched); syn 0.5 moves the weights (same seed).
  python tests/test_syn.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    b, r = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_tl15"], T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_tl15_mf50"]
    assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "syn"} and r.syn == 0.5 and r.rc == 0.75 and r.tilt == 0.15
    assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).syn == 0, "the copies must be unchanged"
    r20 = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_tl15_mf20"]          # T2-D561: the same line at syn 0.2 (only name and syn differ)
    assert {k for k, v in asdict(r20).items() if v != asdict(b)[k]} == {"name", "syn"} and r20.syn == 0.2 and T.peruser_for(r20).syn == 0
    bp = T.RECIPES["r1_pulx_vp2_e180_vr"]                            # T2-D561: the ShallowNet / IFNet line at syn 0.2 / 0.5 (only name and syn differ)
    for nm_, sv in (("r1_pulx_vp2_e180_vr_mf20", 0.2), ("r1_pulx_vp2_e180_vr_mf50", 0.5)):
        rx = T.RECIPES[nm_]
        assert {k for k, v in asdict(rx).items() if v != asdict(bp)[k]} == {"name", "syn"} and rx.syn == sv and T.peruser_for(rx) == T.peruser_for(bp)
    rs = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_mf20"]                 # T2-D576: the stack's FBCNet line at syn 0.2 (only name and syn differ)
    bs_ = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75"]
    assert {k for k, v in asdict(rs).items() if v != asdict(bs_)[k]} == {"name", "syn"} and rs.syn == 0.2 and T.peruser_for(rs) == T.peruser_for(bs_)
    assert [n for n, x in T.RECIPES.items() if x.syn] == ["r1_pulxx_vp2_e180_vr_rc75_tl15_mf50", "r1_pulxx_vp2_e180_vr_rc75_tl15_mf20",
                                                          "r1_pulx_vp2_e180_vr_mf20", "r1_pulx_vp2_e180_vr_mf50", "r1_pulxx_vp2_e180_vr_rc75_mf20"]
    d = data(n_users=3)
    base = replace(T.RECIPES["r1"], epochs=2, bs=16, rc=1.0)
    s0 = T.fit_eval("fbcnet_bboth", d, base, 0, device="cpu", return_state=True)["state_last"]
    T.SYN_BANK.update(X=torch.full((40, 41, 480), 7.0).half(), y=torch.zeros(40, dtype=torch.long), u=torch.ones(40, dtype=torch.long))
    s0b = T.fit_eval("fbcnet_bboth", d, base, 0, device="cpu", return_state=True)["state_last"]
    assert all(torch.equal(s0[k], s0b[k]) for k in s0), "syn 0 must not touch the bank"
    seen, aug = [], T.augment

    def spy(xb, recipe, ub=None, rcm=None, yc=None):
        seen.append((xb.clone(), ub.clone()))
        return aug(xb, recipe, ub, rcm, yc)

    T.augment = spy
    try:
        s5 = T.fit_eval("fbcnet_bboth", d, replace(base, syn=0.5), 0, device="cpu", return_state=True)["state_last"]
    finally:
        T.augment = aug
    n_real = [16, 16, 16, 12] * 2                                    # 60 training windows, bs 16, 2 epochs
    assert [len(x) for x, _ in seen] == [n + round(0.5 * n) for n in n_real], [len(x) for x, _ in seen]
    for (x, u), n in zip(seen, n_real):
        assert torch.all(x[n:] == 7.0) and torch.all(u[n:] == 1) and not torch.all(x[:n] == 7.0)
    moved = max(float((s0[k] - s5[k]).abs().max()) for k in s0 if s0[k].is_floating_point())
    assert moved > 1e-4, moved
    print(f"syn 0.5: batches B + round(B / 2) with the bank windows / users appended; syn 0 bitwise with a bank loaded; weights moved {moved:.1e}")
    for bad in (replace(base, syn=0.5, user_adv=0.1),):
        try:
            T.fit_eval("fbcnet_bboth", d, bad, 0, device="cpu")
            raise SystemExit("FAIL: syn with user_adv must be refused")
        except AssertionError:
            pass
    T.SYN_BANK.clear()
    try:
        T.fit_eval("fbcnet_bboth", d, replace(base, syn=0.5), 0, device="cpu")
        raise SystemExit("FAIL: syn without SYN_BANK must be refused")
    except AssertionError:
        pass
    print("refused: syn with user_adv; syn without SYN_BANK")
    print("OK")


if __name__ == "__main__":
    main()
