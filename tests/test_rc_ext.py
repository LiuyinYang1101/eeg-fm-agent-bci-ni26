#!/usr/bin/env python
"""Backlog 65 (a): re-colouring toward an external covariance bank (Recipe.rc_ext -> ni26.train.rc_bank(ext=...) / augment), CPU:
  1. r1_pulxx_vp2_e180_vr_rcx75 differs from the FBCNet rc75 line in rc_ext only; its copies are the line's; only it sets rc_ext;
  2. rc_bank(ext=None) = rc_bank bitwise; with ext (K, C, C): shape (n_users, K, C, C), and at rc 1 each map sends user u's covariance onto the
     trace-matched, shrunk target (T C_u T = C_j tr(C_u) / tr(C_j) + shrink), T symmetric positive definite; at rc 0.75 M = I + 0.75 (T - I);
  3. augment with rc_ext draws one of the K maps per window (replayed from the same RNG state);
  4. fit_eval with rc_ext and RC_EXT_BANK set runs and moves the weights away from rc 0.75; without the bank it is refused.
  5. t2p_final.rc_ext_bank (the package builder, T2-D295's ship path): no rc_ext recipe -> nothing installed; an rc_ext recipe -> the bank
     installed as t2p_bench does; a bank on another montage, or no --rc-ext-bank, refused.
  python tests/test_rc_ext.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ni26.train as T  # noqa: E402
from test_rc import covs, data  # noqa: E402


def main():
    b, r = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75"], T.RECIPES["r1_pulxx_vp2_e180_vr_rcx75"]
    assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "rc_ext"} and r.rc_ext == 1
    assert T.peruser_for(r) == T.peruser_for(b), "the copies must be unchanged"
    assert [n for n, x in T.RECIPES.items() if x.rc_ext] == ["r1_pulxx_vp2_e180_vr_rcx75"]
    print("recipe: rcx75 = rc75 + rc_ext only; copies unchanged")

    d = data()
    X, u = torch.as_tensor(d["Xtr"]), torch.as_tensor(d["utr"])
    rng = np.random.default_rng(3)
    C = X.shape[1]
    ext = []
    for _ in range(4):
        A = np.eye(C) + 0.4 * rng.standard_normal((C, C))
        ext.append(A @ A.T * rng.uniform(0.1, 10.0))
    ext = np.stack(ext)
    assert torch.equal(T.rc_bank(X, u, 3, 0.75), T.rc_bank(X, u, 3, 0.75, ext=None)), "ext None = rc_bank bitwise"
    M1, M75 = T.rc_bank(X, u, 3, 1.0, ext=ext).double().numpy(), T.rc_bank(X, u, 3, 0.75, ext=ext).double().numpy()
    assert M1.shape == (3, 4, C, C)
    Cs = covs(X, u, 3)
    err = 0.0
    for a in range(3):
        for j in range(4):
            Ct = ext[j] * np.trace(Cs[a]) / np.trace(ext[j])
            Ct = Ct + 1e-3 * np.trace(Ct) / C * np.eye(C)
            Tm = M1[a, j]
            assert np.allclose(Tm, Tm.T, atol=1e-4) and np.linalg.eigvalsh((Tm + Tm.T) / 2).min() > 0
            err = max(err, float(np.abs(Tm @ Cs[a] @ Tm - Ct).max() / np.abs(Ct).max()))
            assert np.allclose(M75[a, j], np.eye(C) + 0.75 * (Tm - np.eye(C)), atol=1e-5)
    assert err < 1e-3, err
    print(f"rc_bank(ext): T C_u T = the trace-matched target (max rel err {err:.1e}); rc 0.75 = I + 0.75 (T - I)")

    rec = replace(r, vp=0, vp_r=0.0)
    rcm = T.rc_bank(X, u, 3, 0.75, ext=ext)
    xb, ub = X[:16], u[:16]
    torch.manual_seed(7)
    got = T.augment(xb, rec, ub, rcm)
    torch.manual_seed(7)
    j = torch.randint(rcm.shape[1], (16,))
    assert torch.allclose(got, torch.bmm(rcm[ub, j], xb), atol=1e-5), "augment: one external map per window"
    print("augment: each window x M[u, j], j drawn uniformly over the K external targets (replayed)")

    plain = replace(T.RECIPES["r1"], epochs=2)
    try:
        T.fit_eval("fbcnet_bboth", d, replace(plain, rc=0.75, rc_ext=1), 0, device="cpu")
        raise SystemExit("rc_ext without RC_EXT_BANK must be refused")
    except AssertionError as e:
        assert "RC_EXT_BANK" in str(e), e
    T.RC_EXT_BANK["C"] = ext
    a0 = T.fit_eval("fbcnet_bboth", d, replace(plain, rc=0.75), 0, device="cpu", return_state=True)
    c0 = T.fit_eval("fbcnet_bboth", d, replace(plain, rc=0.75, rc_ext=1), 0, device="cpu", return_state=True)
    sa, sc = a0["state_last"], c0["state_last"]
    moved = max(float((sa[k].float() - sc[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
    assert sa.keys() == sc.keys() and moved > 1e-5, moved
    T.RC_EXT_BANK.clear()
    print(f"fit_eval: rc_ext without the bank refused; with it the weights move by up to {moved:.1e} from rc 0.75")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/t2_bci/scripts"))
    import tempfile
    import t2p_final as F
    names = [f"E{i}" for i in range(C)]
    with tempfile.TemporaryDirectory() as td:
        good, bad = Path(td) / "bank.npz", Path(td) / "bad.npz"
        np.savez(good, C=ext, ch_names=np.array(names), corpus=np.array(["a"] * 4))
        np.savez(bad, C=ext, ch_names=np.array(names[::-1]), corpus=np.array(["a"] * 4))
        assert F.rc_ext_bank(str(good), names, [b, T.RECIPES["r1_pulx_vp2_e180_vr"]]) == 0 and not T.RC_EXT_BANK
        assert F.rc_ext_bank(str(good), names, [T.RECIPES["r1_pulx_vp2_e180_vr"], r]) == 4 and np.array_equal(T.RC_EXT_BANK["C"], ext)
        T.RC_EXT_BANK.clear()
        for path in (str(bad), ""):
            try:
                F.rc_ext_bank(path, names, [r])
                raise SystemExit("t2p_final.rc_ext_bank must refuse a bank on another montage / no bank")
            except AssertionError:
                pass
        assert not T.RC_EXT_BANK
    print("t2p_final.rc_ext_bank: nothing without an rc_ext recipe; installs the bank for one (recipe or recipe2); wrong montage / no bank refused")
    print("OK")


if __name__ == "__main__":
    main()
