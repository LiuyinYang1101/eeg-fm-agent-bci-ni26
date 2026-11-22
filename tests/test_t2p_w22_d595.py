#!/usr/bin/env python
"""T2-D595 (W22 candidate, prepared as code only): cfg['p1_prior_reset'] in tracks/t2_bci/submission/t2p_submission_w22.py, CPU, no package:
  1. GOLDEN: the template's own SinkhornCorrector, reset when a refit fires (as the flag does, before the window is decoded), equals Track3's
     lab decoder sinkhorn_reset (lead/diag_word_2026-11-18/d595.py, its source pinned by sha256) on random streams of lengths 40 / 120 / 230
     with refits 60 / 100 / 150, the shipped decode args (10 iters, n0 3, lam 0.95) and a random start (<= 1e-12); and = the plain
     corrector before the first refit;
  2. the W22 template = W21's (t2p_submission_w21lp.py) byte for byte except the header line and the flag block, which sits inside the fired-
     refit branch and resets fam_prior over is_fams only (never is_tprior); absent flag = W21 bitwise.
  python tests/test_t2p_w22_d595.py  ->  "OK"
"""
import ast
import difflib
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W21 = ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py"
W22 = ROOT / "tracks/t2_bci/submission/t2p_submission_w22.py"
LAB = Path.home() / "workspace/outputs/neural_interface_challenge/lead/diag_word_2026-11-18/d595.py"
LAB_SHA = "6fa75f95529586df2d66a0cbbc943f3de86ab5f1d0d52a72da46c4d533d264db"


def grab(path, names):
    src = path.read_text()
    tree = ast.parse(src)
    ns = {"np": np}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            exec(compile(ast.Module([n], []), str(path), "exec"), ns)
    return ns


def main():
    assert hashlib.sha256(LAB.read_bytes()).hexdigest() == LAB_SHA, "the lab's d595.py changed: re-pin after a review"
    SC = grab(W22, {"SinkhornCorrector"})["SinkhornCorrector"]
    lab = grab(LAB, {"sinkhorn_reset"})["sinkhorn_reset"]
    rng = np.random.default_rng(595)
    cks = [60, 100, 150]
    lens = [40, 120, 230]
    streams = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
    order = np.concatenate([rng.permutation(n) for n in lens])
    P = rng.dirichlet(np.ones(3) * 0.7, len(streams))
    start = rng.dirichlet(np.ones(3) * 5)
    ref = lab(P, (streams, order), 10, 3, 0.95, start, cks)
    out = np.empty_like(ref)
    plain = np.empty_like(ref)
    for s_ in np.unique(streams):
        rows = np.flatnonzero(streams == s_)
        rows = rows[np.argsort(order[rows], kind="stable")]
        c, c0 = SC(10, 3, 3, 0.95, start), SC(10, 3, 3, 0.95, start)
        for t, r in enumerate(rows):
            if t in cks:                                   # the template: the refit fires at window index t, then the flag resets
                c.reset()
            out[r] = c(P[r])
            plain[r] = c0(P[r])
    d = float(np.abs(out - ref).max())
    assert d < 1e-12, d
    pre = np.concatenate([np.flatnonzero(streams == s_)[np.argsort(order[streams == s_], kind="stable")][:60] for s_ in np.unique(streams)])
    assert float(np.abs(out[pre] - plain[pre]).max()) == 0.0
    assert float(np.abs(out - plain).max()) > 1e-6
    print(f"1. golden: the template's corrector reset at the fired refits == the lab's sinkhorn_reset (max |d| {d:.1e}); == the plain decode "
          "before the first refit; differs after")
    a, b = W21.read_text().splitlines(), W22.read_text().splitlines()
    add = [l_[1:] for l_ in difflib.unified_diff(a, b, lineterm="", n=0) if l_.startswith("+") and not l_.startswith("+++")]
    rem = [l_ for l_ in difflib.unified_diff(a, b, lineterm="", n=0) if l_.startswith("-") and not l_.startswith("---")]
    assert not rem and len(add) == 6 and add[0].startswith("# T2-D595"), (rem, add)
    blk = "\n".join(add[1:])
    assert "self.cfg.get(\"p1_prior_reset\")" in blk and "self.fam_prior[f].reset()" in blk and "D595 p1 prior reset at" in blk and "is_tprior" not in blk.replace("never the teacher's is_tprior", "")
    src = W22.read_text()
    i = src.index("if self.cfg.get(\"p1_prior_reset\")")
    head = src[:i]
    assert head.rfind("if self.is_n in self.is_refits and not sh.get(\"is_off\"):") > head.rfind("def _instream(self"), "the flag is not inside the fired-refit branch"
    print("2. W22 = W21 + the header line + the 5-line flag block (with its K1 print) inside the fired-refit branch (fam_prior over is_fams only); absent = W21")
    print("OK")


if __name__ == "__main__":
    main()
