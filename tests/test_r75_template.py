#!/usr/bin/env python
"""Row 75 (the combined warm-up twin): the template's CNN test-time averaging (cfg cnn_tta_vp), CPU, eager path, small random nets:
  1. cnn_tta_vp absent / k 1: StreamModel.predict = the pre-row-75 template (git 87f3630) bitwise on a stream of windows (two CNN families);
  2. k 8: StreamModel._fam over _v(z) = the mean over the family's nets of ni26.train.predict_proba_vp(net, z, 8) with the same bank (the bench's
     TTA, T2-D296 / D302), for both families; the copies take the views only with "copies";
  3. set_tta_bank keeps the first k - 1 draws; predict runs with k 8 and differs from k 1.
  python tests/test_r75_template.py  ->  "OK"
"""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ni26.train as T  # noqa: E402

bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
bs.CompetSolver = object; bu.base_solver = bs
sys.modules.update({"benchmark_utils": bu, "benchmark_utils.base_solver": bs})
OLD_REF = "87f3630"                                  # the commit before the row 75 template change


def load_template(src, name):
    ns = {"__name__": name}
    exec(compile(src, f"{name}.py", "exec"), ns)
    return ns


def model(ns, cfg, nets, nets2, C):
    meta = dict(ch_names=[f"E{i}" for i in range(C)], sfreq=120.0)
    return ns["StreamModel"](nets, cfg, meta, (), nets2).eval().stack_members()


def main():
    torch.manual_seed(0)
    C, L, K = 8, 480, 8
    new = load_template((ROOT / "tracks/t2_bci/submission/t2p_submission.py").read_text(), "t2p_new")
    old = load_template(subprocess.run(["git", "show", f"{OLD_REF}:tracks/t2_bci/submission/t2p_submission.py"], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout, "t2p_old")
    nets = [T.build_model("fbcnet_bboth", C, 3, L, 120.0).eval() for _ in range(3)]
    nets2 = [T.build_model("shallow", C, 3, L, 120.0).eval() for _ in range(2)]
    cfg = dict(ch_names=[f"E{i}" for i in range(C)], sfreq=120.0, n_times=L, n_classes=3, align=dict(mode="none"),
               cnn2=dict(weight=0.3, align=dict(mode="none")))
    X = torch.randn(12, C, L) * 10
    m_old, m_new = model(old, cfg, nets, nets2, C), model(new, cfg, nets, nets2, C)
    a, b = [], []
    for x in X:
        a.append(m_old.predict(x[None])[0]); a.append(np.asarray(m_old.last_p) if hasattr(m_old, "last_p") else None)
        b.append(m_new.predict(x[None])[0]); b.append(np.asarray(m_new.last_p) if hasattr(m_new, "last_p") else None)
    assert a[0::2] == b[0::2], "tv off: the argmax stream differs from the old template"
    po = [old["StreamModel"]._fam(nets, m_old.stk[0], torch.as_tensor(x[None])) for x in X]
    pn = [new["StreamModel"]._fam(nets, m_new.stk[0], m_new._v(torch.as_tensor(x[None]))) for x in X]
    assert all(np.array_equal(p, q) for p, q in zip(po, pn)), "tv off: _fam must be the old value bitwise"
    print(f"tv off: predict = the template at {OLD_REF} (argmax on {len(X)} windows) and _fam bitwise")

    rng = np.random.default_rng(1)
    bank = dict(mats=torch.as_tensor(np.stack([np.eye(C) + 0.05 * rng.standard_normal((C, C)) for _ in range(10)]), dtype=torch.float32),
                gains=torch.as_tensor(rng.uniform(0.9, 1.1, (10, C)), dtype=torch.float32),
                shifts=torch.as_tensor(rng.integers(-1, 2, 10), dtype=torch.float32))
    for copies in (False, True):
        cfg_t = dict(cfg, cnn_tta_vp=dict(k=K, copies=copies, file="tta_vp.pt"))
        m = model(new, cfg_t, nets, nets2, C).set_tta_bank(bank, "cpu")
        assert m.tv_bank[0].shape[0] == K - 1 and torch.equal(m.tv_bank[0], bank["mats"][:K - 1])
        T.VP_BANK.clear(); T.VP_BANK.update(bank)
        z = torch.as_tensor(X[:1])
        for fam_nets, stk in ((nets, m.stk[0]), (nets2, m.stk[1])):
            got = new["StreamModel"]._fam(fam_nets, stk, m._v(z))
            want = np.mean([T.predict_proba_vp(n, z, K)[0] for n in fam_nets], 0)
            assert np.allclose(got, want, atol=1e-5), (float(np.abs(got - want).max()))
        assert m._v(z, copies=True).shape[0] == (K if copies else 1)
        ys = [m.predict(x[None])[0] for x in X]
        assert len(ys) == len(X)
    T.VP_BANK.clear()
    print(f"tv k {K}: each family = the mean of the bench's predict_proba_vp over its nets (max err < 1e-5); copies take the views only with 'copies'")
    pt = [m._fam(nets, m.stk[0], m._v(torch.as_tensor(x[None]))) for x in X]
    assert not all(np.allclose(p, q, atol=1e-4) for p, q in zip(pt, pn)), "tv on must change the family output"
    print("tv k 8 changes the family outputs; set_tta_bank keeps the first k - 1 draws")
    print("OK")


if __name__ == "__main__":
    main()
