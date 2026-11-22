#!/usr/bin/env python
"""T2-D230: re-colouring of the pooled training windows toward another training user (Recipe.rc -> ni26.train.rc_bank / augment), CPU:
  1. r1_pulxx_vp2_e180_vr_rc5 / _rc10 (T2-D230) / _rc25 / _rc75 (T2-D231) differ from the FBCNet vr line in `rc` only, and
     r1_pulx_vp2_e180_vr_rc75 / _rc10 (T2-D233) from the ShallowNet vr line; their per-user copies are the vr lines'; only these six re-colour;
  2. rc_bank: T_uv C_u T_uv = C_v (the users' covariances as rc_bank computes them), T_uv symmetric positive definite, M[u, u] = I,
     M at rc 0.5 = I + 0.5 (T - I);
  3. augment: each window of user u is multiplied by M[u, v] for a drawn v != u (replayed from the same RNG state);
  4. fit_eval with rc 0 = r1 bitwise; with rc 1 the weights move, the BatchNorm step counter matches (FBCNet B3; T2-D233: ShallowNet too);
  5. T2-D236: r1_pulxx_vp2_e180_vr_rcu25 / _rcu50 = _rc10 + rc_lo 0.25 / 0.5 only; augment with rc_lo = each window x I + s (T - I), s ~ U(rc_lo, rc)
     replayed from the same RNG state; rc_lo 0 = the fixed strength bitwise;
  6. T2-D237: r1_pulxx_vp2_e180_vr_rc75m1 / _rc75m2 = _rc75 + rc_mix 1 / 2 only (copies unchanged); rc_bank mix 0 = the default bitwise; mix 1 on
     4 users = 3 maps per user, each landing C_u on the Bures midpoint G C_v G (G = (I + T_vw) / 2) of a pair of the other users (= G' C_w G'
     from the other end); mix 2 = the real maps then the mix-1 maps; augment draws one of the K per window (replayed); fit_eval with rc_mix runs.
  python tests/test_rc.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402


def data(seed=0, C=41, L=480, n_users=3):
    rng = np.random.default_rng(seed)
    d = {}
    for k, m in dict(tr=60, va=12, te=12).items():
        mix = rng.standard_normal((n_users, C, C)) * 0.3 + np.eye(C)          # users differ in spatial colouring
        u = np.arange(m) % n_users
        d[f"X{k}"] = np.einsum("ncd,ndl->ncl", mix[u], rng.standard_normal((m, C, L))).astype(np.float32)
        d[f"y{k}"] = rng.integers(0, 3, m)
        d[f"c{k}"] = np.array(["a"] * m)
        if k == "tr":
            d["utr"] = u
    return d


def covs(X, u, n):
    out = []
    for k in range(n):
        x = X[u == k].double()
        x = x - x.mean(-1, keepdim=True)
        c = (torch.einsum("ncl,ndl->cd", x, x) / (x.shape[0] * x.shape[-1])).numpy()
        out.append(c + 1e-3 * np.trace(c) / len(c) * np.eye(len(c)))
    return out


def main():
    want = [("pulxx", t, a) for t, a in (("rc5", 0.5), ("rc10", 1.0), ("rc25", 0.25), ("rc75", 0.75))] + [("pulx", t, a) for t, a in (("rc75", 0.75), ("rc10", 1.0))]
    rcu = {"rcu25": 0.25, "rcu50": 0.5}                                 # T2-D236
    for t, lo in rcu.items():
        b, r = T.RECIPES["r1_pulxx_vp2_e180_vr_rc10"], T.RECIPES[f"r1_pulxx_vp2_e180_vr_{t}"]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "rc_lo"} and r.rc_lo == lo and r.rc == 1.0, t
        assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).rc_lo == 0, "the copies must be unchanged"
    assert sorted(n for n, r in T.RECIPES.items() if r.rc_lo) == sorted(f"r1_pulxx_vp2_e180_vr_{t}" for t in rcu)
    rcm_ = {"rc75m1": 1, "rc75m2": 2}                                   # T2-D237
    for t, m in rcm_.items():
        b, r = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75"], T.RECIPES[f"r1_pulxx_vp2_e180_vr_{t}"]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "rc_mix"} and r.rc_mix == m and r.rc == 0.75, t
        assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).rc_mix == 0, "the copies must be unchanged"
    assert sorted(n for n, r in T.RECIPES.items() if r.rc_mix) == sorted(f"r1_pulxx_vp2_e180_vr_{t}" for t in rcm_)
    for f, t, a in want:
        b = T.RECIPES[f"r1_{f}_vp2_e180_vr"]
        r = T.RECIPES[f"r1_{f}_vp2_e180_vr_{t}"]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "rc"} and r.rc == a, (f, t)
        assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).rc == 0, "the copies must be unchanged"
    on = sorted(n for n, r in T.RECIPES.items() if r.rc)
    assert on == sorted([f"r1_{f}_vp2_e180_vr_{t}" for f, t, _ in want] + [f"r1_pulxx_vp2_e180_vr_{t}" for t in (*rcu, *rcm_, "rc75_pux1", "rc75_pux3", "rc75_ua1", "rc75_ua3", "rc75_kd", "rc75_kdx", "rcx75", "rc75_cap10", "rc75_ms5", "rc75_us5", "rc75_pv10", "rc75_xm5", "rc75_tl15", "rc75cc_tl15", "rc75_vs5", "rc75_gd3", "rc75_kl1", "rc75_sb16")] + ["r1_pulxx_vp3_e180_vr_rc75", "r1_pulxx_vp2_e360_vr_rc75_tl15", "r1_pulxx_vp2_e180_vr_rc75_tl15_pv10", "r1_pulxx_vp2_e180_vr_rc75_tl15_mf50", "r1_pulxx_vp2_e180_vr_rc75_tl15_mf20", "r1_pulxx_vp2_e180_vr_rc75_mf20"] + [f"r1_pulxx_vp2_e180_vr_rc75_{t}" for t in ("dm03", "dm1", "dc03", "dc1")] + ["r1_pulxx_vp2_e240_vr_rc75"]), on   # + T2-D548 (FBCNet at 240 epochs, diagnosis only) + T2-D413 (tl15 + MeanFlow syn) + T2-D561 (tl15 + MeanFlow syn 0.2) + T2-D576 (rc75 + MeanFlow syn 0.2) + T2-D339 (vp3 on the rc75 line) + T2-D239 (rc75 pooled) + T2-D293 (rc75 + user_adv) + backlog 69 (rc75 + kd) + backlog 65 (b) + backlog 70 / 72 + T2-D478 (rc75 + sb16)
    d = data()
    X, u = torch.as_tensor(d["Xtr"]), torch.as_tensor(d["utr"])
    M1, M5 = T.rc_bank(X, u, 3, 1.0).double().numpy(), T.rc_bank(X, u, 3, 0.5).double().numpy()
    Cs = covs(X, u, 3)
    err = 0.0
    for a in range(3):
        assert np.array_equal(M1[a, a], np.eye(41)) and np.array_equal(M5[a, a], np.eye(41))
        for c in range(3):
            if a != c:
                Tm = M1[a, c]
                err = max(err, np.abs(Tm @ Cs[a] @ Tm.T - Cs[c]).max() / np.abs(Cs[c]).max())
                assert np.abs(Tm - Tm.T).max() < 1e-4 and np.linalg.eigvalsh((Tm + Tm.T) / 2).min() > 0, (a, c)
                assert np.abs(M5[a, c] - (np.eye(41) + 0.5 * (Tm - np.eye(41)))).max() < 1e-5
    assert err < 1e-4, err
    print(f"rc_bank: T C_u T = C_v (max rel err {err:.1e}); T symmetric PD; M[u, u] = I; rc 0.5 = I + 0.5 (T - I)")
    rec = replace(T.RECIPES["r1"], rc=1.0)
    rcm = T.rc_bank(X, u, 3, 1.0)
    xb, ub = X[:12], u[:12]
    torch.manual_seed(7)
    got = T.augment(xb, rec, ub, rcm)
    torch.manual_seed(7)
    v = (ub + torch.randint(1, 3, (12,))) % 3
    assert (v != ub).all() and torch.allclose(got, torch.einsum("bcd,bdl->bcl", rcm[ub, v], xb), atol=1e-5)
    assert torch.equal(T.augment(xb, T.RECIPES["r1"]), xb), "rc 0: identity"
    print("augment: each window x M[u, v], v != u; rc 0 = identity")
    torch.manual_seed(7)
    assert torch.equal(T.augment(xb, replace(rec, rc_lo=0.0), ub, rcm), got), "rc_lo 0 = the fixed strength bitwise"
    reu = replace(rec, rc_lo=0.25)
    torch.manual_seed(7)
    gu = T.augment(xb, reu, ub, rcm)
    torch.manual_seed(7)
    v = (ub + torch.randint(1, 3, (12,))) % 3
    s = 0.25 + 0.75 * torch.rand(12, 1, 1)
    assert (s >= 0.25).all() and (s <= 1.0).all() and s.std() > 0.05
    want_u = torch.einsum("bcd,bdl->bcl", torch.eye(41) + s * (rcm[ub, v] - torch.eye(41)), xb)
    assert torch.allclose(gu, want_u, atol=1e-5) and not torch.allclose(gu, got, atol=1e-3)
    print(f"T2-D236: rc_lo 0.25 -> x (I + s (T - I)), s in [{float(s.min()):.2f}, {float(s.max()):.2f}]; rc_lo 0 bitwise = fixed")
    plain = replace(T.RECIPES["r1"], epochs=2)
    a0 = T.fit_eval("fbcnet_bboth", d, plain, 0, device="cpu", return_state=True)
    b0 = T.fit_eval("fbcnet_bboth", d, replace(plain, rc=0.0), 0, device="cpu", return_state=True)
    c0 = T.fit_eval("fbcnet_bboth", d, replace(plain, rc=1.0), 0, device="cpu", return_state=True)
    sa, sb, sc = a0["state_last"], b0["state_last"], c0["state_last"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa) and np.array_equal(a0["prob_last"], b0["prob_last"]), "rc 0 must be r1 bitwise"
    moved = max(float((sa[k] - sc[k]).abs().max()) for k in sa if sa[k].is_floating_point())
    assert moved > 1e-4 and all(torch.equal(sa[k], sc[k]) for k in sa if "num_batches" in k), moved
    print(f"fit_eval: rc 0 = r1 bitwise; rc 1 moves the weights by up to {moved:.1e}")
    a1 = T.fit_eval("shallow", d, plain, 0, device="cpu", return_state=True)
    c1 = T.fit_eval("shallow", d, replace(plain, rc=1.0), 0, device="cpu", return_state=True)
    m1 = max(float((a1["state_last"][k] - c1["state_last"][k]).abs().max()) for k in a1["state_last"] if a1["state_last"][k].is_floating_point())
    assert m1 > 1e-4, m1
    print(f"T2-D233: ShallowNet fit_eval with rc 1 runs and moves the weights by up to {m1:.1e}")
    d4 = data(seed=1, n_users=4)                                        # T2-D237
    X4, u4 = torch.as_tensor(d4["Xtr"]), torch.as_tensor(d4["utr"])
    assert torch.equal(T.rc_bank(X4, u4, 4, 0.75, mix=0), T.rc_bank(X4, u4, 4, 0.75)), "mix 0 = the default bitwise"
    R1, B1, B2 = (T.rc_bank(X4, u4, 4, 1.0, mix=m).double().numpy() for m in (0, 1, 2))
    assert B1.shape == (4, 3, 41, 41) and B2.shape == (4, 6, 41, 41), (B1.shape, B2.shape)
    C4 = covs(X4, u4, 4)
    err, sym = 0.0, 0.0
    for a in range(4):
        pairs = [(b, c) for b in range(4) for c in range(b + 1, 4) if a not in (b, c)]
        assert len(pairs) == 3
        for k, (b, c) in enumerate(pairs):
            G, Gp = (np.eye(41) + R1[b, c]) / 2, (np.eye(41) + R1[c, b]) / 2
            mid = G @ C4[b] @ G
            sym = max(sym, np.abs(Gp @ C4[c] @ Gp - mid).max() / np.abs(mid).max())
            Tm = B1[a, k]
            err = max(err, np.abs(Tm @ C4[a] @ Tm.T - mid).max() / np.abs(mid).max())
            assert np.abs(B2[a, 3 + k] - Tm).max() < 1e-6
            assert np.abs(mid - C4[b]).max() > 1e-3 * np.abs(mid).max() and np.abs(mid - C4[c]).max() > 1e-3 * np.abs(mid).max(), "a midpoint is no real user"
        assert np.abs(B2[a, :3] - np.stack([R1[a, v] for v in range(4) if v != a])).max() < 1e-6
    assert err < 1e-4 and sym < 1e-3, (err, sym)
    print(f"T2-D237: mix 1 lands C_u on the pairwise Bures midpoints (max rel err {err:.1e}; midpoint from either end {sym:.1e}); mix 2 = real + mix 1")
    B1t = T.rc_bank(X4, u4, 4, 0.75, mix=1)
    rm = replace(T.RECIPES["r1"], rc=0.75, rc_mix=1)
    xb, ub = X4[:12], u4[:12]
    torch.manual_seed(3)
    gm = T.augment(xb, rm, ub, B1t)
    torch.manual_seed(3)
    k = torch.randint(3, (12,))
    assert k.unique().numel() > 1 and torch.allclose(gm, torch.einsum("bcd,bdl->bcl", B1t[ub, k], xb), atol=1e-5)
    try:
        T.augment(xb, replace(rm, rc_lo=0.25), ub, B1t)
        raise SystemExit("rc_mix with rc_lo must refuse")
    except AssertionError:
        pass
    r4 = T.fit_eval("fbcnet_bboth", d4, replace(plain, rc=0.75), 0, device="cpu", return_state=True)
    m4 = T.fit_eval("fbcnet_bboth", d4, replace(plain, rc=0.75, rc_mix=1), 0, device="cpu", return_state=True)
    dm = max(float((r4["state_last"][q] - m4["state_last"][q]).abs().max()) for q in r4["state_last"] if r4["state_last"][q].is_floating_point())
    assert dm > 1e-4, dm
    print(f"T2-D237: augment draws one of the K maps per window (replayed); rc_lo refused; fit_eval rc_mix 1 runs, {dm:.1e} from rc 0.75")
    print("OK")


if __name__ == "__main__":
    main()
