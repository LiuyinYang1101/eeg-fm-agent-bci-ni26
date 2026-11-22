#!/usr/bin/env python
"""T2-D595 / D603: t2p_repack's restart-aware reference recombination (found by the D603 GPU smoke, 2026-11-19: R6b's 'recombine' check failed
by 0.105 because the repack decoded the P1 families with a plain causal_sinkhorn while the package restarts them at the fired refits). CPU:
  1. _sinkhorn_restart(mode 'start') == Track3's D595 lab decoder sinkhorn_reset (d595.py, sha-pinned) <= 1e-12;
  2. _sinkhorn_restart(mode 'mean')  == Track3's D603 lab decoder sinkhorn_seg(..., 'marginal') (d60x.py, sha-pinned) <= 1e-12;
  3. no fired refit (every c >= the stream length) == t2p_report.causal_sinkhorn, both modes;
  on single streams of lengths 40 / 120 / 230, refits 60 / 100 / 150, the shipped decode 10 / 3 / 0.95 and a random start.
    python tests/test_t2p_repack_restart.py  ->  "OK"
"""
import ast
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RPK = ROOT / "tracks/t2_bci/scripts/t2p_repack.py"
REP = ROOT / "tracks/t2_bci/scripts/t2p_report.py"
LABD = Path.home() / "workspace/outputs/neural_interface_challenge/lead/diag_word_2026-11-18"
PINS = {"d595.py": "6fa75f95529586df2d66a0cbbc943f3de86ab5f1d0d52a72da46c4d533d264db",
        "d60x.py": "08205fab1226d0730f199b3f0a69ce4d710888142db8be0403465853564c878a"}


def grab(path, names):
    ns = {"np": np}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module([n], []), str(path), "exec"), ns)
    assert set(names) <= set(ns), set(names) - set(ns)
    return ns


def main():
    for f, sha in PINS.items():
        assert hashlib.sha256((LABD / f).read_bytes()).hexdigest() == sha, f"Track3's {f} changed: re-pin after a review"
    rs = grab(RPK, {"_sinkhorn_restart"})["_sinkhorn_restart"]
    cs = grab(REP, {"causal_sinkhorn"})["causal_sinkhorn"]
    reset = grab(LABD / "d595.py", {"sinkhorn_reset"})["sinkhorn_reset"]
    seg = grab(LABD / "d60x.py", {"sinkhorn_seg"})["sinkhorn_seg"]
    rng = np.random.default_rng(595603)
    cks = [60, 100, 150]
    worst = dict(start=0.0, mean=0.0, none=0.0, moved=0.0)
    for L in (40, 120, 230):
        P = rng.dirichlet(np.ones(3) * 0.7, L)
        start = rng.dirichlet(np.ones(3) * 5)
        keys = (np.zeros(L, int), np.arange(L))
        a = rs(P, 10, 3, 0.95, start, cks, "start")
        b = rs(P, 10, 3, 0.95, start, cks, "mean")
        worst["start"] = max(worst["start"], float(np.abs(a - reset(P, keys, 10, 3, 0.95, start, cks)).max()))
        worst["mean"] = max(worst["mean"], float(np.abs(b - seg(P, keys, 10, 3, 0.95, start, cks, "marginal")).max()))
        plain = cs(P, keys, 10, 3, 0.95, start=start)
        for mode in ("start", "mean"):
            worst["none"] = max(worst["none"], float(np.abs(rs(P, 10, 3, 0.95, start, [L, L + 10], mode) - plain).max()))
        if L > cks[0]:
            worst["moved"] = max(worst["moved"], float(np.abs(a - plain).max()), float(np.abs(b - plain).max()))
    assert worst["start"] <= 1e-12 and worst["mean"] <= 1e-12 and worst["none"] <= 1e-12 and worst["moved"] > 1e-6, worst
    src = RPK.read_text()
    assert src.count("_sinkhorn_restart(np.stack(seq[f])") == 1 and 'if rs_ and f in isf_:' in src, "the per-family branch's restart hook moved"
    print(f"1. mode 'start' == Track3's D595 sinkhorn_reset (max |d| {worst['start']:.1e})")
    print(f"2. mode 'mean' == Track3's D603 sinkhorn_seg 'marginal' (max |d| {worst['mean']:.1e})")
    print(f"3. no fired refit == causal_sinkhorn (max |d| {worst['none']:.1e}); a fired refit moves the decode ({worst['moved']:.3f})")
    print("OK")


if __name__ == "__main__":
    main()
