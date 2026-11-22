#!/usr/bin/env python
"""Backlog 96 (G3, P3 v2): the FM's PACKAGE loss, CPU:
  1. t2p_fm.pkg_nll = the NLL of the true class under (1 - w) pc + w softmax(lo): w 1 = plain CE; the member's gradient is larger
     where the fixed CNN mix is wrong than where it is right;
  2. fit_eval_fm takes pkg (default None = off: PK None, every package line guarded by `PK is not None and k == 0`), on the plain and
     the mixup paths (the mixed window's CNN mix = the same lambda's mix), target windows only;
  3. t2p_bench --fm-pkg (default 0) / --fm-pkg-w (default 0.35) need --fm-sw-oof, reach fit_eval_fm (pkg=pk_) and fm_args, and the stem
     gains pk<beta>w<w> only when set.
  python tests/test_fm_pkg.py  ->  "OK"
"""
import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm  # noqa: E402


def main():
    torch.manual_seed(0)
    lo, y = torch.randn(16, 3), torch.randint(0, 3, (16,))
    pc = torch.softmax(torch.randn(16, 3), 1)
    assert torch.allclose(t2p_fm.pkg_nll(lo, pc, y, 1.0), F.cross_entropy(lo, y), atol=1e-6), "w 1 = plain CE"
    right, wrong = F.one_hot(y, 3).float() * 0.98 + 0.01 / 1.5, None
    wrong = torch.roll(F.one_hot(y, 3).float(), 1, 1) * 0.98 + 0.01 / 1.5
    g = []
    for p_ in (right, wrong):
        l_ = lo.clone().requires_grad_(True)
        t2p_fm.pkg_nll(l_, p_, y, 0.35).backward()
        g.append(float(l_.grad.abs().sum()))
    assert g[1] > 3 * g[0], ("the gradient must concentrate where the CNN mix is wrong", g)
    print(f"pkg_nll: w 1 = CE; |grad| CNN-wrong {g[1]:.3f} vs CNN-right {g[0]:.3f}")
    sig = inspect.signature(t2p_fm.fit_eval_fm).parameters
    assert sig["pkg"].default is None, "pkg must default to off"
    src = inspect.getsource(t2p_fm.fit_eval_fm)
    assert "    PK = None\n" in src and src.count("if PK is not None and k == 0:") == 2, "both CE paths, guarded"
    assert "pcm = lam * PK[b] + (1 - lam) * PK[b][j]" in src, "mixup: the CNN mix mixed with the same lambda"
    assert "loss = loss + pk_beta * pkg_nll(lo, PK[b], yb, pk_w)" in src
    assert 'assert len(pk_p) == len(Xtr) and not data.get("ext") and supcon == 0 and int(micro) == 1 and not fuse' in src
    b = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert 'ap.add_argument("--fm-pkg", type=float, default=0.0,' in b and 'ap.add_argument("--fm-pkg-w", type=float, default=0.35,' in b
    assert "assert bool(args.fm_sw or args.fm_pkg) == bool(args.fm_sw_oof)" in b
    assert "sw=sw_, pkg=pk_," in b and "pkg=args.fm_pkg, pkg_w=args.fm_pkg_w," in b
    assert '(f"pk{args.fm_pkg:g}w{args.fm_pkg_w:g}" if getattr(args, "fm_pkg", 0) else "")' in b
    assert "oof_pcnn(args.fm_sw_oof, proto, y, tr, full=True)" in b
    print("fit_eval_fm: pkg off by default, both CE paths; t2p_bench: --fm-pkg reaches the kwargs + fm_args; tag only when set")
    print("OK")


if __name__ == "__main__":
    main()
