#!/usr/bin/env python
"""T2-D352: template W11 (tracks/t2_bci/submission/t2p_submission_w11.py) = W9 (t2p_submission.py) + an optional third CNN family. CPU, eager path
(GPU graph path: tests/test_w11_template.py --gpu, on a CUDA device), small random nets, routed per-user copies (the route fixed by patching _user):
  1. no cnn3: W11's StreamModel.predict and last_p = W9's BITWISE on a stream (two families + copies of both + test-time views with copies);
  2. cnn3 (+ user_blend3): last_p = (1 - w2 - w3) p1 + w2 p2 + w3 p3 with each family's copies blended at its r (no FM members: fm_weight 0),
     computed by hand from the families' _fam outputs; parts[3] = p3;
  3. --gpu: the graph path (cfg gpu_graph) = the eager path within 1e-5 for the three-family package.
  python tests/test_w11_template.py [--gpu]  ->  "OK"
"""
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
SUB = ROOT / "tracks/t2_bci/submission"


def load(path, name):
    ns = {"__name__": name}
    exec(compile(path.read_text(), f"{name}.py", "exec"), ns)
    return ns


def build(ns, cfg, fam, C, dev="cpu"):
    meta = dict(ch_names=[f"E{i}" for i in range(C)], sfreq=120.0)
    m = ns["StreamModel"](fam["nets"], cfg, meta, (), fam["nets2"], fam["pu"], fam["pu2"],
                          *([fam["nets3"], fam["pu3"]] if "nets3" in fam else [])).to(dev).eval().stack_members()
    m._user = lambda xi: 1                              # the route: training user 1 from the first window on
    return m


def run(m, X):
    ys, ps = [], []
    for x in X:
        ys.append(int(m.predict(x[None])[0]))
        ps.append(np.asarray(m.last_p, np.float64).copy())
    return ys, ps


def main():
    gpu = "--gpu" in sys.argv
    dev = "cuda" if gpu else "cpu"
    torch.manual_seed(0)
    C, L, K, U = 8, 480, 4, 2
    w9, w11 = load(SUB / "t2p_submission.py", "w9"), load(SUB / "t2p_submission_w11.py", "w11")
    mk = lambda name, n: [T.build_model(name, C, 3, L, 120.0).eval() for _ in range(n)]  # noqa: E731
    fam = dict(nets=mk("fbcnet_bboth", 2), nets2=mk("shallow", 2), pu=[mk("fbcnet_bboth", 2) for _ in range(U)],
               pu2=[mk("shallow", 2) for _ in range(U)])
    ub = dict(r=0.3333, mean=[0.0], scale=[1.0], W=[[1.0]], mu=[[0.0]] * U, users=[f"u{k}" for k in range(U)])
    cfg = dict(ch_names=[f"E{i}" for i in range(C)], sfreq=120.0, n_times=L, n_classes=3, align=dict(mode="none"),
               cnn2=dict(weight=0.3, align=dict(mode="ea")), user_blend=ub, user_blend2=dict(r=0.5),
               cnn_tta_vp=dict(k=K, copies=True, file="tta_vp.pt"))
    rng = np.random.default_rng(1)
    bank = dict(mats=torch.as_tensor(np.stack([np.eye(C) + 0.05 * rng.standard_normal((C, C)) for _ in range(K)]), dtype=torch.float32),
                gains=torch.as_tensor(rng.uniform(0.9, 1.1, (K, C)), dtype=torch.float32),
                shifts=torch.as_tensor(rng.integers(-1, 2, K), dtype=torch.float32))
    X = torch.randn(10, C, L) * 10
    if not gpu:
        a = build(w9, cfg, fam, C).set_tta_bank(bank, "cpu")
        b = build(w11, cfg, fam, C).set_tta_bank(bank, "cpu")
        ya, pa = run(a, X)
        yb, pb = run(b, X)
        assert ya == yb and all(np.array_equal(p, q) for p, q in zip(pa, pb)), "no cnn3: W11 must equal W9 bitwise"
        print(f"1. no cnn3: W11 = W9 bitwise (predict + last_p) on {len(X)} windows (two families, copies of both, tv k {K} with copies)")

    fam3 = dict(fam, nets3=mk("ifnet", 2), pu3=[mk("ifnet", 2) for _ in range(U)])
    w2, w3 = 0.3, 0.1
    cfg3 = dict(cfg, cnn3=dict(weight=w3, align=dict(mode="ea")), user_blend3=dict(r=0.5))
    m = build(w11, cfg3, fam3, C, dev).set_tta_bank(bank, dev)
    if not gpu:
        S = w11["StreamModel"]
        for x in X[:4]:
            mm = m                                       # one stream; recompute each family by hand with fresh aligner copies
            al = [mm.aligner, mm.aligner2, mm.aligner3]
            import copy
            st = [copy.deepcopy(a_) for a_ in al]
            xi = x.numpy()
            z1, z2, z3 = (torch.from_numpy(a_(xi))[None] for a_ in st)
            p1 = (1 - 0.3333) * S._fam(fam3["nets"], mm.stk[0], mm._v(z1)).astype(np.float64) + \
                0.3333 * S._fam(fam3["pu"][1], mm.stk_pu[1], mm._v(z1, copies=True)).astype(np.float64)
            p2 = 0.5 * S._fam(fam3["nets2"], mm.stk[1], mm._v(z2)).astype(np.float64) + \
                0.5 * S._fam(fam3["pu2"][1], mm.stk_pu2[1], mm._v(z2, copies=True)).astype(np.float64)
            p3 = 0.5 * S._fam(fam3["nets3"], mm.stk[2], mm._v(z3)).astype(np.float64) + \
                0.5 * S._fam(fam3["pu3"][1], mm.stk_pu3[1], mm._v(z3, copies=True)).astype(np.float64)
            want = (1 - w2 - w3) * p1 + w2 * p2 + w3 * p3
            import os
            os.environ["NI26_PARTS"] = "1"
            mm.predict(x[None])
            assert np.allclose(mm.last_p, want, atol=1e-9), (mm.last_p, want)
        print("2. cnn3: last_p = (1 - w2 - w3) p1 + w2 p2 + w3 p3 with each family's copies at its r (max err < 1e-9)")
    else:
        cfg3g = dict(cfg3, gpu_graph=True)
        mg = build(w11, cfg3g, fam3, C, dev).set_tta_bank(bank, dev)
        me = build(w11, cfg3, fam3, C, dev).set_tta_bank(bank, dev)
        _, pg = run(mg, X)
        _, pe = run(me, X)
        d = max(float(np.abs(p - q).max()) for p, q in zip(pg, pe))
        assert d < 1e-5, d
        print(f"3. GPU: the graph path = the eager path for the three-family package (max |d| {d:.1e})")
    print("OK")


if __name__ == "__main__":
    main()
