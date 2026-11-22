#!/usr/bin/env python
"""T2-D654 (W31v): the template's per-window gate == the lab's (Track3's ask: the lab-vs-template bitwise gate check), CPU, no model:
  1. `_d654_gate_step` is extracted from the PATCHED template (t2p_w31v_patch.patch of t2p_submission_w21lp.py) and run per stream in recording
     order; its decisions == scripts/local/t2_d654_viewmix.py's causal_gate ('cnn') and prob_gate ('prob', Generator 65460 re-seeded per stream)
     BITWISE on synthetic margins (3 streams, shuffled rows, ties included), q in 0.1 / 0.2 / 0.3;
  2. 'off' never gates; the token bucket holds (gated <= q t + 1) for both gates.
  python tests/test_t2_w31v_gate.py  ->  "OK"
"""
import ast
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "scripts/local"))
import t2p_w31v_patch as P  # noqa: E402
import t2_d654_viewmix as VM  # noqa: E402


def main():
    src = P.patch((ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py").read_text())
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_d654_gate_step"]
    assert len(fn) == 1
    ns = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=fn, type_ignores=[]), "w31v", "exec"), ns)
    step = ns["_d654_gate_step"]
    rng = np.random.default_rng(654)
    lens = (40, 57, 80)
    streams = np.concatenate([np.full(n, s) for s, n in enumerate(lens)])
    order = np.concatenate([rng.permutation(n) for n in lens])
    m = np.round(rng.random(len(streams)), 2)             # rounded: ties occur (the strict-rank / <= quantile edge cases)
    perm = rng.permutation(len(m))                          # rows in an arbitrary order (the lab's), streams interleaved
    streams, order, m = streams[perm], order[perm], m[perm]

    def template(gate, q):
        out = np.zeros(len(m), bool)
        for s_ in np.unique(streams):
            r = np.flatnonzero(streams == s_)
            r = r[np.argsort(order[r], kind="stable")]
            st = dict(m=[], n=0, g=0, gen=torch.Generator().manual_seed(65460))
            for i in r:
                out[i] = step(st, float(m[i]), q, gate)
            assert st["g"] <= q * len(r) + 1, (gate, q, st["g"], len(r))
        return out
    for q in (0.1, 0.2, 0.3):
        a, b = template("cnn", q), VM.causal_gate(m, streams, order, q)
        assert np.array_equal(a, b), ("cnn", q, int((a != b).sum()))
        a, b = template("prob", q), VM.prob_gate(m, streams, order, q)
        assert np.array_equal(a, b), ("prob", q, int((a != b).sum()))
        print(f"q {q}: cnn gate {int(template('cnn', q).sum())} / {len(m)} and prob gate {int(a.sum())} / {len(m)} windows == viewmix bitwise")
    assert not template("off", 0.3).any()
    print("1-2. the template's gate step == viewmix.causal_gate / prob_gate bitwise (3 streams, ties, shuffled rows); 'off' never gates; the bucket holds")
    print("OK")


if __name__ == "__main__":
    main()
