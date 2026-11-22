#!/usr/bin/env python
"""T2-D606 (registered 9daab875): past-only output calibration, CPU, no package:
  1. the multiplier form: normalize(p_t * sk_mult([start; p_1 .. p_t])) == t2p_report.causal_sinkhorn (today's output) <= 1e-12;
  2. a STATEFUL corrector that stores the multiplier after each update (the template form: emit normalize(p_t * b), then append p_t and
     recompute b) == the lab's batch causal_sinkhorn_past <= 1e-12, on streams of lengths 1 / 40 / 230, lam 0.95 and 1.0, a random start;
  3. the template's own SinkhornCorrector == causal_sinkhorn (the reference the multiplier form reproduces).
    python tests/test_t2_d606_past.py  ->  "OK"
"""
import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "scripts/local/t2_lab.py"
REP = ROOT / "tracks/t2_bci/scripts/t2p_report.py"
TPL = ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py"


def grab(path, names):
    tree = ast.parse(path.read_text())
    ns = {"np": np}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            exec(compile(ast.Module([n], []), str(path), "exec"), ns)
    missing = set(names) - set(ns)
    assert not missing, missing
    return ns


class PastCorrector:
    """the template form of T2-D606: one stored multiplier per corrector."""

    def __init__(self, iters, n0, k, lam, start, sk_mult):
        self.iters, self.n0, self.k, self.lam, self.sk = iters, n0, k, lam, sk_mult
        self.start = np.asarray(start, np.float64)
        self.rows = [self.start.copy() for _ in range(n0)]
        self.b = sk_mult(np.stack(self.rows) if self.rows else np.zeros((0, k)), iters, lam)

    def __call__(self, p):
        p = np.asarray(p, np.float64)
        q = p * self.b
        q = q / q.sum()
        self.rows.append(p)
        self.b = self.sk(np.stack(self.rows), self.iters, self.lam)
        return q


def main():
    lab = grab(LAB, {"sk_mult", "causal_sinkhorn_past"})
    rep = grab(REP, {"causal_sinkhorn"})
    tpl = grab(TPL, {"SinkhornCorrector"})
    sk, past, cs, SC = lab["sk_mult"], lab["causal_sinkhorn_past"], rep["causal_sinkhorn"], tpl["SinkhornCorrector"]
    rng = np.random.default_rng(606)
    lens = [1, 40, 230]
    streams = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
    order = np.concatenate([rng.permutation(n) for n in lens])
    P = rng.dirichlet(np.ones(3) * 0.7, len(streams))
    start = rng.dirichlet(np.ones(3) * 5)
    worst = dict(mult=0.0, state=0.0, tpl=0.0, diff_vs_today=0.0)
    for lam in (0.95, 1.0):
        today = cs(P, (streams, order), 10, 3, lam, start=start)
        pp = past(P, (streams, order), 10, 3, lam, start=start)
        for s_ in np.unique(streams):
            rows = np.flatnonzero(streams == s_)
            rows = rows[np.argsort(order[rows], kind="stable")]
            c, t_, M = PastCorrector(10, 3, 3, lam, start, sk), SC(10, 3, 3, lam, start), [start.copy() for _ in range(3)]
            for r in rows:
                M.append(P[r])
                b = sk(np.stack(M), 10, lam)
                q = P[r] * b
                worst["mult"] = max(worst["mult"], float(np.abs(q / q.sum() - today[r]).max()))
                worst["state"] = max(worst["state"], float(np.abs(c(P[r]) - pp[r]).max()))
                worst["tpl"] = max(worst["tpl"], float(np.abs(t_(P[r]) - today[r]).max()))
        worst["diff_vs_today"] = max(worst["diff_vs_today"], float(np.abs(pp - today).max()))
    assert worst["mult"] <= 1e-12, worst
    assert worst["state"] <= 1e-12, worst
    assert worst["tpl"] <= 1e-12, worst
    assert worst["diff_vs_today"] > 1e-4, worst                 # the timing change does change the output
    print(f"1. the multiplier form == today's causal_sinkhorn (max |d| {worst['mult']:.1e})")
    print(f"2. the stateful (template-form) corrector == the batch past-only decode (max |d| {worst['state']:.1e}); past-only vs today max |d| "
          f"{worst['diff_vs_today']:.3f}")
    print(f"3. the template's SinkhornCorrector == causal_sinkhorn (max |d| {worst['tpl']:.1e})")
    print("OK")


if __name__ == "__main__":
    main()
