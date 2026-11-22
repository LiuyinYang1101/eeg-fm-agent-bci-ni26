#!/usr/bin/env python
"""T2-D308: virtual sessions (Recipe.vs -> ni26.train.augment with VS_BANK; t2p_bench.vs_bank), CPU:
  1. the vs5 recipes differ from the FBCNet rc75 / ShallowNet vr lines in vs (0.5) only; copies unchanged; only these two set vs;
  2. vs_bank on synthetic sessions: K = ordered same-user session pairs + 2 x sessions (halves); each map satisfies T C_a T = C_b (1e-6);
     on the real export's kit training rows the bank is built from those rows only;
  3. augment: vs 0 -> unchanged; vs 1 -> each window = M[k] x (replayed); fit_eval with vs runs and moves the weights.
  python tests/test_vs.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import ni26.train as T  # noqa: E402
import t2p_bench as B  # noqa: E402
from test_rc import data  # noqa: E402


def main():
    for base, n in (("r1_pulxx_vp2_e180_vr_rc75", "r1_pulxx_vp2_e180_vr_rc75_vs5"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e180_vr_vs5")):
        b, r = T.RECIPES[base], T.RECIPES[n]
        assert {k for k, v in asdict(r).items() if v != asdict(b)[k]} == {"name", "vs"} and r.vs == 0.5, n
        assert T.peruser_for(r) == T.peruser_for(b)
    assert sorted(n for n, x in T.RECIPES.items() if x.vs) == ["r1_pulx_vp2_e180_vr_vs5", "r1_pulxx_vp2_e180_vr_rc75_vs5"]
    print("recipes: the two vs5 recipes differ from their lines in vs only; copies unchanged")

    rng = np.random.default_rng(0)
    C, L = 6, 200
    rows_meta = []
    X = []
    for subj, sessions in (("A", ["1", "2", "3"]), ("B", ["1"])):
        for se in sessions:
            mix = np.eye(C) + 0.3 * rng.standard_normal((C, C))
            for t in range(40):
                X.append(mix @ rng.standard_normal((C, L)))
                rows_meta.append(dict(subject=subj, session=se, order=t))
    X = np.stack(X).astype(np.float32)
    idx = pd.DataFrame(rows_meta)
    rows = np.arange(len(X))
    bk = B.vs_bank(X, rows, idx).double().numpy()
    assert len(bk) == 6 + 2 * 4, len(bk)                          # A: 3 x 2 ordered session pairs; halves of 4 sessions
    def cov(r):
        x = X[r].astype(np.float64); x = x - x.mean(-1, keepdims=True)
        Cm = np.einsum("ncl,ndl->cd", x, x) / (x.shape[0] * x.shape[-1]); return Cm + 1e-3 * np.trace(Cm) / C * np.eye(C)
    a = np.flatnonzero((idx.subject == "A") & (idx.session == "1")); b_ = np.flatnonzero((idx.subject == "A") & (idx.session == "2"))
    Ca, Cb = cov(a), cov(b_)
    err = min(float(np.abs(M @ Ca @ M - Cb).max() / np.abs(Cb).max()) for M in bk)
    assert err < 1e-6, err
    print(f"vs_bank: {len(bk)} maps = 6 session pairs + 8 half maps; a pair map satisfies T C_a T = C_b (rel err {err:.1e})")

    torch.manual_seed(0)
    xb = torch.randn(16, C, L)
    plain = replace(T.RECIPES["r1"])
    T.VS_BANK.clear(); T.VS_BANK["mats"] = torch.as_tensor(bk, dtype=torch.float32)
    assert torch.equal(T.augment(xb.clone(), plain), xb)
    torch.manual_seed(5); got = T.augment(xb.clone(), replace(plain, vs=1.0))
    torch.manual_seed(5); k = torch.randint(len(bk), (16,)); _ = torch.rand(16)
    assert torch.allclose(got, torch.bmm(T.VS_BANK["mats"][k], xb), atol=1e-5)
    print("augment: vs 0 unchanged; vs 1 = M[k] x per window (replayed)")
    d = data()
    T.VS_BANK.clear(); T.VS_BANK["mats"] = torch.eye(41)[None].repeat(3, 1, 1) + 0.02 * torch.randn(3, 41, 41)
    p2 = replace(T.RECIPES["r1"], epochs=2)
    a0 = T.fit_eval("fbcnet_bboth", d, p2, 0, device="cpu", return_state=True)
    a1 = T.fit_eval("fbcnet_bboth", d, replace(p2, vs=1.0), 0, device="cpu", return_state=True)
    moved = max(float((a0["state_last"][k_].float() - a1["state_last"][k_].float()).abs().max()) for k_ in a0["state_last"] if a0["state_last"][k_].is_floating_point())
    assert moved > 1e-5, moved
    T.VS_BANK.clear()
    print(f"fit_eval with vs runs, weights move by up to {moved:.1e}")
    print("OK")


if __name__ == "__main__":
    main()
