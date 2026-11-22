#!/usr/bin/env python
"""T2-D219: t2p_bench.aligned_extra(frac=) on a synthetic external corpus (CPU):
  1. frac 1 = the old output bitwise (every labelled window, in keep order);
  2. frac 0.5 keeps half of each stream's labelled windows (+-1), evenly spaced in acquisition order, the same rows on every call;
  3. the kept rows are exactly rows of the frac-1 output (the alignment ran on the whole stream first), labels kept with them.
  python tests/test_extra_frac.py  ->  "OK"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    rows = []
    for s, n in (("A|1", 41), ("A|2", 40), ("B|1", 37)):
        order = rng.permutation(n)                      # the file order is not the acquisition order
        for o in order:
            rows.append(dict(stream=s, order=int(o), y=int(rng.integers(-1, 3))))   # -1 = an unlabelled window
    idx = pd.DataFrame(rows)
    idx["i"] = np.arange(len(idx))
    X = rng.standard_normal((len(idx), 4, 16)).astype(np.float32)
    keep = np.flatnonzero(idx["y"].to_numpy() >= 0)
    ext = dict(name="export_syn", X=X, idx=idx, keep=keep, names=["C3", "Cz", "C4", "Pz"], sha=None)
    X1, y1 = B.aligned_extra(ext, "none", 0.05)
    Xf, yf = B.aligned_extra(ext, "none", 0.05, frac=1.0)
    assert np.array_equal(X1, Xf) and np.array_equal(y1, yf), "frac 1 must be the old output"
    Xh, yh = B.aligned_extra(ext, "none", 0.05, frac=0.5)
    Xh2, _ = B.aligned_extra(ext, "none", 0.05, frac=0.5)
    assert np.array_equal(Xh, Xh2), "the subsample must be fixed"
    g = idx.iloc[keep].reset_index(drop=True)
    pos = {tuple(X1[k].ravel()[:3]): k for k in range(len(X1))}
    hit = [pos[tuple(x.ravel()[:3])] for x in Xh]
    assert np.array_equal(y1[hit], yh), "labels must travel with their windows"
    for s, gs in g.iloc[hit].groupby("stream"):
        nl = int((g["stream"] == s).sum())
        assert abs(len(gs) - nl / 2) <= 1, (s, len(gs), nl)
        ranks = g[g["stream"] == s]["order"].rank().astype(int).to_numpy()   # acquisition rank among the labelled windows
        r_all = dict(zip(g[g["stream"] == s].index, ranks))
        kr = np.sort([r_all[i] for i in gs.index])
        assert np.all(np.diff(kr) == 2), (s, kr)       # every second labelled window in acquisition order
    print(f"frac 0.5: {len(yh)} / {len(y1)} labelled windows kept, every second per stream; frac 1 bitwise")
    print("OK")


if __name__ == "__main__":
    main()
