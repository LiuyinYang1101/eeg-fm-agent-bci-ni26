#!/usr/bin/env python
"""T2-D603 (W22-D603 candidate, prepared as code only): cfg['p1_prior_informed'] in tracks/t2_bci/submission/t2p_submission_w22d603.py,
CPU, no package. The template's OWN code is exercised: its SinkhornCorrector (with restart), its _fam_decode bound to a small stand-in model,
and its refit block extracted verbatim from _instream.
  1. GOLDEN: on random streams of lengths 40 / 120 / 230 with refits 60 / 100 / 150 (shipped decode 10 / 3 / 0.95, random family starts),
     the pooled output == fam_w-weighted [causal_sinkhorn for cnn / fm, Track3's sinkhorn_seg(..., 'marginal') for cnn2 / cnn3]
     (lead/diag_word_2026-11-18/d60x.py, its source pinned by sha256) <= 1e-12; == the plain pool before the first refit; differs after;
  2. flag absent: the template's _fam_decode == W21lp's _fam_decode bitwise (==), the same corrector states;
  3. the source: W22-D603 = W21lp + the header + the restart method + the p1_osum init + the flag-guarded _fam_decode branch (the original
     pool line moved into its else branch) + the flag-guarded refit block (fam_prior over is_fams only, never is_tprior).
    python tests/test_t2p_w22_d603.py  ->  "OK"
"""
import ast
import difflib
import hashlib
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W21 = ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py"
W22 = ROOT / "tracks/t2_bci/submission/t2p_submission_w22d603.py"
REP = ROOT / "tracks/t2_bci/scripts/t2p_report.py"
LAB = Path.home() / "workspace/outputs/neural_interface_challenge/lead/diag_word_2026-11-18/d60x.py"
LAB_SHA = "08205fab1226d0730f199b3f0a69ce4d710888142db8be0403465853564c878a"
FAMS = ("cnn", "cnn2", "fm", "cnn3")


def grab(path, names, cls=None):
    tree = ast.parse(path.read_text())
    ns = {"np": np}
    body = tree.body
    if cls:
        body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            exec(compile(ast.Module([n], []), str(path), "exec"), ns)
    assert set(names) <= set(ns), set(names) - set(ns)
    return ns


def refit_block(path):
    """the template's D603 refit block, verbatim (from its 'if' line through its K1 print), dedented."""
    L = path.read_text().split("\n")
    i = [k for k, x in enumerate(L) if 'if self.cfg.get("p1_prior_informed"):' in x]
    assert len(i) == 1, i
    j = [k for k in range(i[0], len(L)) if "D603 p1 informed restart at" in L[k]][0]
    return textwrap.dedent("\n".join(L[i[0]:j + 1]))


class Stand:
    """the attributes _fam_decode and the refit block read."""


def run(SC, fam_decode, block, P, starts, cks, w, flag):
    m = Stand()
    m.cfg = {"p1_prior_informed": True} if flag else {}
    m.is_cfg, m.is_fams, m.fam_w, m.dc, m.ic = {}, ["cnn2", "cnn3"], w, None, None
    m.fam_prior = {f: SC(10, 3, 3, 0.95, starts[f]) for f in FAMS}
    m.p1_osum, m.p1_on = {}, 0
    out = []
    for t in range(len(P["cnn"])):
        if flag and t in cks:
            m.is_n = t
            exec(block, {"np": np, "print": lambda *a, **k: None}, {"self": m})
        out.append(fam_decode(m, [P[f][t] for f in FAMS]))
    return np.array(out), m


def main():
    assert hashlib.sha256(LAB.read_bytes()).hexdigest() == LAB_SHA, "Track3's d60x.py changed: re-pin after a review"
    SC = grab(W22, {"SinkhornCorrector"})["SinkhornCorrector"]
    fd22 = grab(W22, {"_fam_decode"}, cls="StreamModel")["_fam_decode"]
    fd21 = grab(W21, {"_fam_decode"}, cls="StreamModel")["_fam_decode"]
    seg = grab(LAB, {"sinkhorn_seg"})["sinkhorn_seg"]
    cs = grab(REP, {"causal_sinkhorn"})["causal_sinkhorn"]
    block = refit_block(W22)
    rng = np.random.default_rng(603)
    cks = [60, 100, 150]
    w = {"cnn": 0.3143, "cnn2": 0.2357, "fm": 0.35, "cnn3": 0.10}
    worst = dict(golden=0.0, pre=0.0, post=0.0)
    for L in (40, 120, 230):
        P = {f: rng.dirichlet(np.ones(3) * 0.7, L) for f in FAMS}
        starts = {f: rng.dirichlet(np.ones(3) * 5) for f in FAMS}
        got, m = run(SC, fd22, block, P, starts, cks, w, True)
        keys = (np.zeros(L, int), np.arange(L))
        ref = sum(w[f] * (seg(P[f], keys, 10, 3, 0.95, starts[f], cks, "marginal") if f in ("cnn2", "cnn3")
                          else cs(P[f], keys, 10, 3, 0.95, start=starts[f])) for f in FAMS)
        worst["golden"] = max(worst["golden"], float(np.abs(got - ref).max()))
        plain, _ = run(SC, fd22, block, P, starts, cks, w, False)
        k = min(cks[0], L)
        worst["pre"] = max(worst["pre"], float(np.abs(got[:k] - plain[:k]).max()))
        if L > cks[0]:
            worst["post"] = max(worst["post"], float(np.abs(got - plain).max()))
            assert m.p1_on == L, (m.p1_on, L)
        o21, _ = run(SC, fd21, block, P, starts, cks, w, False)   # 2. flag absent: W22-D603's _fam_decode == W21's, bitwise
        assert np.array_equal(plain, o21)
    assert worst["golden"] <= 1e-12 and worst["pre"] == 0.0 and worst["post"] > 1e-6, worst
    print(f"1. golden: the template's restart (its own corrector, _fam_decode, refit block) == Track3's sinkhorn_seg 'marginal' for cnn2 / cnn3 "
          f"+ causal_sinkhorn for cnn / fm (max |d| {worst['golden']:.1e}); == the plain pool before the first refit; differs after ({worst['post']:.3f})")
    print("2. flag absent: _fam_decode == W21lp's, bitwise")
    a, b = W21.read_text().splitlines(), W22.read_text().splitlines()
    dl = list(difflib.unified_diff(a, b, lineterm="", n=0))
    rem = [x[1:] for x in dl if x.startswith("-") and not x.startswith("---")]
    add = [x[1:] for x in dl if x.startswith("+") and not x.startswith("+++")]
    pool = "        out = sum(self.fam_w[f] * c(P[f]) for f, c in self.fam_prior.items())"
    assert rem == [pool], rem
    assert add[0].startswith("# T2-D603 (W22-D603 candidate") and "    " + pool in add, add[:3]
    assert len(add) == 20, len(add)
    src = W22.read_text()
    assert src.count('self.cfg.get("p1_prior_informed")') == 2 and src.count("def restart(self, start):") == 1
    assert "is_tprior" not in block.replace("never is_tprior", "") and ".restart(m / m.sum())" in block
    i = src.index('if self.cfg.get("p1_prior_informed"):')
    head = src[:i]
    assert head.rfind('if self.is_n in self.is_refits and not sh.get("is_off"):') > head.rfind("def _instream(self"), "not in the fired-refit branch"
    print("3. W22-D603 = W21lp + the header + restart() + the p1_osum init + the guarded _fam_decode branch + the guarded refit block (fam_prior only)")
    print("OK")


if __name__ == "__main__":
    main()
