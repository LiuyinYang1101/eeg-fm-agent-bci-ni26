#!/usr/bin/env python
"""T2-D239: the per-user copies' data with the other users' windows mapped onto their user (Recipe.pu_rcx -> ni26.train.copy_data, t2p_bench), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_pux1 / _pux3 differ from _rc75 in pu_rcx (1 / 3) and pu_epochs (40 / 20: each copy takes rc75's 80 epochs' steps)
     only; their copy recipe = rc75's with those epochs (pu_rcx is not a copy field); only these two set pu_rcx;
  2. copy_data: user k's own windows first, in order, bitwise; then n = round(ratio x n_k) other users' windows (capped at their count), distinct,
     none of k's, = bank[v, k] @ X (replayed generator); with the full maps (rc_bank at 1.0) the mapped windows of each source user v take k's
     covariance (T C_v T = C_k); the labels follow their windows;
  3. t2p_bench: pu_rcx 0 = the earlier copy slicing (source); > 0 builds rc_bank(Xt, ut_, n, 1.0) once and feeds copy_data to finetune.
  python tests/test_pu_rcx.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import ni26.train as T  # noqa: E402
from test_rc import covs, data  # noqa: E402


def main():
    b = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75"]
    for t, x, e in (("pux1", 1.0, 40), ("pux3", 3.0, 20)):
        r = T.RECIPES[f"r1_pulxx_vp2_e180_vr_rc75_{t}"]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "pu_rcx", "pu_epochs"} and r.pu_rcx == x and r.pu_epochs == e, t
        assert T.peruser_for(r) == replace(T.peruser_for(b), epochs=e) and (1 + x) * e == b.pu_epochs, t
    assert sorted(n for n, r in T.RECIPES.items() if r.pu_rcx) == [f"r1_pulxx_vp2_e180_vr_rc75_{t}" for t in ("pux1", "pux3")]
    print("recipes: rc75 + pu_rcx 1 / 3 with pu_epochs 40 / 20 (the same copy steps); the copy recipe has no pu_rcx")
    d = data(seed=2, n_users=4)
    X, u = torch.as_tensor(d["Xtr"]), torch.as_tensor(d["utr"], dtype=torch.long)
    y = torch.as_tensor(d["ytr"], dtype=torch.long)
    bank = T.rc_bank(X, u, 4, 1.0)
    C4 = covs(X, u, 4)
    for k, ratio in ((0, 1.0), (2, 2.0), (1, 9.0)):
        Xk, yk = T.copy_data(X, y, u, k, bank, ratio, 123)
        own = int((u == k).sum())
        n = min(int(round(ratio * own)), int((u != k).sum()))
        assert len(Xk) == len(yk) == own + n, (k, len(Xk), own, n)
        assert torch.equal(Xk[:own], X[u == k]) and torch.equal(yk[:own], y[u == k]), "own windows first, bitwise"
        oth = torch.nonzero(u != k).flatten()
        pick = oth[torch.randperm(len(oth), generator=torch.Generator().manual_seed(123))[:n]]
        assert pick.unique().numel() == n and not (u[pick] == k).any()
        assert torch.allclose(Xk[own:], torch.einsum("bcd,bdl->bcl", bank[u[pick], k], X[pick]), atol=1e-5) and torch.equal(yk[own:], y[pick])
        print(f"copy_data user {k} ratio {ratio:g}: {own} own + {n} mapped (cap {int((u != k).sum())}); replayed, labels follow")
    Xall, _ = T.copy_data(X, y, u, 3, bank, 99.0, 0)                    # every other window, each mapped onto user 3
    own = int((u == 3).sum())
    vs = u[torch.nonzero(u != 3).flatten()[torch.randperm(int((u != 3).sum()), generator=torch.Generator().manual_seed(0))]]
    err = 0.0
    for v in range(3):                                                  # all of v's windows, mapped: cov = T C_v,raw T; with rc_bank's shrink
        xm = Xall[own:][vs == v].double()                               # term s_v T T added back, = C_3 as rc_bank computes it (T C_v T = C_3)
        xm = xm - xm.mean(-1, keepdim=True)
        c = (torch.einsum("ncl,ndl->cd", xm, xm) / (xm.shape[0] * xm.shape[-1])).numpy()
        Tm = bank[v, 3].double().numpy()
        sv = (C4[v].trace() / len(C4[v])) * 1e-3 / (1 + 1e-3)          # C4[v] = C_raw + 1e-3 mean-eig(C_raw) I -> s_v
        err = max(err, np.abs(c + sv * Tm @ Tm.T - C4[3]).max() / np.abs(C4[3]).max())
    assert err < 1e-4, err
    print(f"full maps: each source user's mapped windows take user 3's covariance (max rel err {err:.1e})")
    s = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert "Xk, yk = Xt[torch.as_tensor(utr_ == k, device=dev)], yt[torch.as_tensor(utr_ == k, device=dev)]" in s, "pu_rcx 0 = the earlier slicing"
    assert "rcb = T.rc_bank(Xt, ut_, len(tr_users), 1.0)" in s and "T.copy_data(Xt, yt, ut_, k, rcb, recipe.pu_rcx, seed * 100 + k + 50)" in s
    # T2-D430 added the strangers (None unless pu_kl); T2-D486 the copy's long windows (None unless --long-jitter: Lt stays None otherwise)
    assert ("m_u = T.finetune(net, Xk, yk, T.peruser_for(recipe), seed * 100 + k, X_other=x_oth, sid=sid_k,\n"
            "                                 Lx=None if Lt is None else Lt[torch.as_tensor(utr_ == k, device=dev)])") in s
    assert "Lt = None\n            if ltr_ is not None:" in s and "ltr_ = None\n        if LJ is not None:" in s
    assert "sid_k = None\n                if recipe.sbal:" in s     # T2-D478: sid stays None unless the stream-balanced recipe is on (pu_rcx 0 = the earlier path)
    print("t2p_bench: pu_rcx 0 = the earlier copy data; > 0 = copy_data with the full maps")
    print("OK")


if __name__ == "__main__":
    main()
