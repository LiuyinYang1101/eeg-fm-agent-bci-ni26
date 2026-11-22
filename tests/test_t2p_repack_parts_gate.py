#!/usr/bin/env python
"""t2p_repack.reconfig_parts_gate (2026-11-22, after the W30 one-hop build 1649496 exited before K1 on a check that could not pass), CPU, no data:
  1. STRICT (config-only / every repack except the case below): any part > 1e-6 fails, and so does a donor side-head deviation;
  2. FM SWAP ONTO AN FM-ADAPTING BASE: last night's measured values (p1 / p2 / p3 0.0, pf 0.2111, recombine 0.0781; exact before the first refit)
     PASS; a moving CNN part, a pre-refit FM / recombination / head deviation, or no pre-refit window at all still FAIL;
  3. the repack wires the gate (no bare all-parts assert left), and fm_adapt is keyed on the base's FM-reading adaptation only.
  python tests/test_t2p_repack_parts_gate.py  ->  "OK"
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_repack as R  # noqa: E402

G = R.reconfig_parts_gate
ZERO = dict(p1=0.0, p2=0.0, pf=0.0, recombine=0.0, p3=0.0)
PRE0 = dict(pf=0.0, recombine=0.0, in_vs_donor=0.0, head_vs_indep=0.0, windows=840)
SIDE0 = dict(in_vs_donor=0.0, head_vs_indep=0.0)
W30 = dict(p1=0.0, p2=0.0, pf=0.2110600927028567, recombine=0.0781082930369289, p3=0.0)   # build 1649496, 2026-11-21 20:14


def main():
    # 1. strict
    assert G(ZERO, PRE0, None, False) == [] and G(ZERO, PRE0, SIDE0, False) == []
    assert G(dict(ZERO, pf=0.2), PRE0, None, False), "strict: a moved FM part must fail"
    assert G(dict(ZERO, recombine=1e-3), PRE0, None, False), "strict: a moved recombination must fail"
    assert G(ZERO, PRE0, dict(SIDE0, head_vs_indep=1e-3), False), "strict: a side-head deviation must fail"
    assert G(W30, PRE0, SIDE0, False), "strict: last night's values must fail a config-only repack"
    # 2. FM swap onto an FM-adapting base
    assert G(W30, PRE0, SIDE0, True) == [], G(W30, PRE0, SIDE0, True)
    assert G(W30, PRE0, None, True) == []
    assert G(dict(W30, p2=1e-4), PRE0, SIDE0, True), "fm_adapt: a CNN part must stay strict"
    assert G(dict(W30, p3=1e-4), PRE0, SIDE0, True), "fm_adapt: the third CNN family must stay strict"
    for k in ("pf", "recombine", "in_vs_donor", "head_vs_indep"):
        assert G(W30, dict(PRE0, **{k: 1e-3}), SIDE0, True), f"fm_adapt: a pre-refit {k} deviation must fail"
    assert G(W30, dict(PRE0, windows=0), SIDE0, True), "fm_adapt: no pre-refit window must fail"
    assert G(W30, dict(PRE0, head_vs_indep=1e-3), None, True) == [], "without side heads the head keys are not checked"
    # 3. wiring
    src = (ROOT / "tracks/t2_bci/scripts/t2p_repack.py").read_text()
    assert "assert max(parts_dev.values()) <= 1e-6" not in src, "the bare all-parts assert is gone"
    assert "bad_ = reconfig_parts_gate(parts_dev, parts_pre, side_info, fm_adapt)" in src and "assert not bad_" in src
    assert R.FM_ADAPT_KEYS == ("fm_side_refit", "fm_proto", "cnn_lp")
    assert re.search(r"fm_adapt = bool\(args\.fm_from and any\(cfg\.get\(k\) for k in FM_ADAPT_KEYS\)\)", src), "fm_adapt needs --fm-from"
    assert src.count("if fm_adapt and t_ < fr0:") == 4, "pre-refit tracking: pf, recombine, in_vs_donor, head_vs_indep"
    print("OK")


if __name__ == "__main__":
    main()
