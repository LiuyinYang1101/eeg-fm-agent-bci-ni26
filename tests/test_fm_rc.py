#!/usr/bin/env python
"""T2-D235: re-colouring on the FM members' target training windows (t2p_fm.fit_eval_fm(rc=), t2p_bench --fm-rc), CPU:
  1. fit_eval_fm's rc defaults to 0 (off = the earlier FM recipe); the rc block builds T2-D230's maps (ni26.train.rc_bank) from the target
     corpus's training windows and users, and re-colours target batches only (k == 0), before the vp bank, through ni26.train.augment;
  2. t2p_bench: --fm-rc (default 0) reaches both fit_eval_fm and the result's fm_args; the result-file tag gains rc<x> only when set;
  3. the map on sz-like inputs (each channel's power scaled per user, correlations left): at rc 1 user 0's re-coloured windows take user 1's
     mean covariance (the Bures map T C_0 T = C_1), at rc 0.75 they lie in between, and augment with an FM recipe = the bmm by hand.
  python tests/test_fm_rc.py  ->  "OK"
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import train as T  # noqa: E402


def body(src, name):
    i = src.index(f"def {name}(")
    j = src.find("\ndef ", i + 1)
    return src[i:j if j > 0 else len(src)]


def cov(x):
    x = x - x.mean(-1, keepdim=True)
    return (torch.einsum("ncl,ndl->cd", x.double(), x.double()) / (x.shape[0] * x.shape[-1])).numpy()


def main():
    fm_src = (ROOT / "tracks/t2_bci/scripts/t2p_fm.py").read_text()
    f = body(fm_src, "fit_eval_fm")
    assert "side=\"\", rc=0.0):" in f, "rc must default to 0"
    i_rc, i_vp = f.index("if rrc is not None and k == 0:"), f.index("if rvp is not None and k == 0:")
    assert i_rc < i_vp, "re-colouring comes before the vp bank (the CNN order)"
    assert "rcm = TV.rc_bank(Xtr, urc, int(data[\"utr\"].max()) + 1, float(rc))" in f
    assert "xb = TV.augment(xb, rrc, urc[b], rcm)" in f and f.count("TV.augment(") == 2
    print("fit_eval_fm: rc 0 by default; maps from the target training windows; target batches only, before vp")
    b = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert "ap.add_argument(\"--fm-rc\", type=float, default=0.0," in b
    assert b.count("rc=args.fm_rc") == 2, "fit_eval_fm kwargs + fm_args"
    assert "(f\"rc{args.fm_rc:g}\" if args.fm_rc else \"\")" in body(b, "task_rec")
    print("t2p_bench: --fm-rc reaches fit_eval_fm and fm_args; tag rc<x> only when set")
    g = torch.Generator().manual_seed(0)
    C, L, n = 6, 240, 200
    A = [torch.eye(C) + 0.3 * torch.randn(C, C, generator=g) for _ in range(2)]
    X = torch.cat([A[k] @ torch.randn(n, C, L, generator=g) for k in range(2)])
    X = X / X.pow(2).mean((0, 2), keepdim=True).sqrt()                  # sz-like: unit channel power over the pool
    u = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    C0, C1 = cov(X[u == 0]), cov(X[u == 1])
    for rc, tol in ((1.0, 0.02), (0.75, None)):
        rcm = T.rc_bank(X, u, 2, rc)
        torch.manual_seed(1)
        got = T.augment(X[u == 0], T.Recipe("fm_rc", rc=rc), u[u == 0], rcm)
        assert torch.allclose(got, torch.bmm(rcm[0, 1].expand(n, C, C), X[u == 0]), atol=1e-5), "two users: every window goes to the other"
        Cg = cov(got)
        d1, d0 = np.linalg.norm(Cg - C1) / np.linalg.norm(C1), np.linalg.norm(Cg - C0) / np.linalg.norm(C0)
        if tol is not None:
            assert d1 < tol, (rc, d1)
        else:
            assert d1 < d0 and d1 > 0.01, (rc, d0, d1)
        print(f"rc {rc}: re-coloured user 0 -> rel. distance {d1:.4f} to C_1, {d0:.4f} to C_0")
    print("OK")


if __name__ == "__main__":
    main()
