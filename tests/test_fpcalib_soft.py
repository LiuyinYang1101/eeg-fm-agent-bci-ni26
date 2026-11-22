#!/usr/bin/env python
"""T2-D165: t2p_fpcalib's soft route (CPU, synthetic + one real blend):
  1. stream_route_soft(top=1) = one-hot of stream_route_every (incl. the no-route start and non-finite windows);
  2. blend_route_soft with those one-hot weights = blend_route exactly;
  3. top=2 weights: non-negative, sum to 1 on routed rows, the nearest centroid gets the larger weight;
  4. (optional) the CLI default (--route nearest) re-blends T2-D159's files bitwise.
  python tests/test_fpcalib_soft.py [--export E --fp FP --runs-kit R --runs-seen1 R --ref REFDIR --tmp DIR]  ->  "OK"
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t2p_fpcalib as FC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for a in ("--export", "--fp", "--runs-kit", "--runs-seen1", "--ref", "--tmp"):
        ap.add_argument(a, default="")
    args = ap.parse_args()
    r = np.random.default_rng(0)
    for every in (1, 3):
        Z, mu = r.normal(size=(200, 5)), r.normal(size=(6, 5))
        Z[:4] = np.nan; Z[50] = np.inf
        k = FC.stream_route_every(Z, mu, every)
        W1 = FC.stream_route_soft(Z, mu, every, top=1)
        oh = np.zeros_like(W1); m = k >= 0; oh[np.flatnonzero(m), k[m]] = 1.0
        assert np.array_equal(W1, oh), every
        sh, Pu = r.dirichlet(np.ones(3), 200), r.dirichlet(np.ones(3), (6, 200))
        assert np.array_equal(FC.blend_route_soft(sh, Pu, W1, 0.4), FC.blend_route(sh, Pu, k, 0.4)), every
        W2 = FC.stream_route_soft(Z, mu, every, top=2)
        rows = W2.sum(1) > 0
        assert (W2 >= 0).all() and np.allclose(W2[rows].sum(1), 1) and np.array_equal(rows, m)
        assert ((W2 > 0).sum(1)[rows] == 2).all() and (W2[np.flatnonzero(rows), k[rows]] >= 0.5).all()
    print("soft route: top=1 = nearest exactly; top=2 weights valid", flush=True)
    if args.runs_kit:
        out = Path(args.tmp); out.mkdir(parents=True, exist_ok=True)
        for p in ("kit", "seen1"):
            subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_fpcalib.py"), "blend", "--export", args.export, "--fp", args.fp,
                            "--runs", getattr(args, f"runs_{p}"), "--protocols", p, "--out", str(out), "--model", "fbcnet_bboth", "--recipe", "r1_pulxx",
                            "--seeds", "48,49,50"], check=True, capture_output=True)
        n = 0
        for f in sorted(out.glob("*.npz")):
            a, b = np.load(f), np.load(Path(args.ref) / f.name)
            assert all(np.array_equal(a[k], b[k]) for k in ("prob_last", "prob_val_last", "test_idx", "val_idx")), f.name
            n += 1
        assert n == 6, n
        print(f"CLI default re-blends {n} T2-D159 files bitwise", flush=True)
        soft = out / "inv2"; soft.mkdir(exist_ok=True)                 # the --route inv2 CLI path runs and differs from nearest
        for p in ("kit", "seen1"):
            subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_fpcalib.py"), "blend", "--export", args.export, "--fp", args.fp,
                            "--runs", getattr(args, f"runs_{p}"), "--protocols", p, "--out", str(soft), "--model", "fbcnet_bboth", "--recipe", "r1_pulxx",
                            "--seeds", "48", "--route", "inv2", "--tag", "3s"], check=True, capture_output=True)
        fs = sorted(soft.glob("*fb3s__s48.npz")); assert len(fs) == 2, fs
        for f in fs:
            a, b = np.load(f), np.load(out / f.name.replace("fb3s", "fb3"))
            assert a["prob_last"].shape == b["prob_last"].shape and np.isfinite(a["prob_last"]).all()
            assert np.allclose(a["prob_last"].sum(1), 1, atol=1e-5) and not np.array_equal(a["prob_last"], b["prob_last"])
        print("CLI --route inv2 writes valid, different files", flush=True)
    print("OK")


if __name__ == "__main__":
    main()
