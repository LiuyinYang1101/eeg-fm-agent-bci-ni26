#!/usr/bin/env python
"""T2-D188 / D189: the template's STRouted / STLinearD (per-user low-rank deltas inside the ST member's one forward) on CPU, st_small, random weights:
  1. delta(None) (zeros) = the plain STClassifier bitwise;
  2. delta(k) = the same net with every adapted Linear's weight merged W + s B A (allclose, fp32);
  3. a deep copy of the template's StreamModel-style sharing is not needed here; the graph path is checked on sofia (T2-D188's job).
  python tests/test_st_routed.py  ->  "OK"
"""
import copy
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26.fm.steeg_pt import STClassifier  # noqa: E402

bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
bs.CompetSolver = object; bu.base_solver = bs
sys.modules.update({"benchmark_utils": bu, "benchmark_utils.base_solver": bs})
src = (ROOT / "tracks/t2_bci/submission/t2p_submission.py").read_text()
ns = {"__name__": "t2p_sub_test"}
exec(compile(src, "t2p_submission.py", "exec"), ns)
STRouted, STLinearD = ns["STRouted"], ns["STLinearD"]


def main():
    torch.manual_seed(0)
    C, n_samples, r, s, K = 6, 128, 4, 0.5, 3
    net = STClassifier("st_small", list(range(C)), n_samples, "gain", -1, 3).eval()
    layout = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Linear) and ".blocks." in name:
            layout.append((name, m.in_features, m.out_features))
    N = sum(r * i + o * r for _, i, o in layout)
    bank = 0.05 * torch.randn(K, N)
    x = torch.randn(2, C, 480)
    with torch.no_grad():
        ref = net(x)
        routed = STRouted(copy.deepcopy(net), bank, layout, r, s).eval()
        assert sum(isinstance(m, STLinearD) for m in routed.modules()) == len(layout) == 4 * 8, len(layout)
        z = routed(x, routed.delta(None))
        assert torch.equal(z, ref), float((z - ref).abs().max())
        for k in range(K):
            merged = copy.deepcopy(net)
            off = 0
            for name, i, o in layout:
                a = bank[k, off:off + r * i].view(r, i); b = bank[k, off + r * i:off + r * i + o * r].view(o, r); off += r * i + o * r
                merged.get_submodule(name).weight += s * (b @ a)
            zk, mk = routed(x, routed.delta(k)), merged(x)
            assert torch.allclose(zk, mk, atol=1e-5, rtol=1e-4), (k, float((zk - mk).abs().max()))
            assert not torch.allclose(zk, ref, atol=1e-4), k
    print(f"STRouted: {len(layout)} adapted Linear layers, N = {N}; zero delta = plain bitwise; delta k = merged weights (K = {K})")
    print("OK")


if __name__ == "__main__":
    main()
