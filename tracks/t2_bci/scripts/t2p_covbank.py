#!/usr/bin/env python
"""Track 2 backlog 65 (a): the external covariance bank for re-colouring (Recipe.rc toward real external users instead of the training users).
Per subject of each external corpus (cpt_pools, contract preprocessing: 120 Hz, uV), up to --max-per-subject windows drawn uniformly (rng
20261028) over all its sessions, channels mapped onto the 41 PROTEUS channels exactly as T2-D124b (t2p_bench.interp_matrix, spherical spline
over the standard_1005 channels both share), then the subject's mean covariance of its demeaned windows, as ni26.train.rc_bank computes C_u
(no shrinkage here: rc_bank adds its own). A subject enters the bank only if >= --min-keep of its corpus channels are standard_1005 (an
interpolated covariance from fewer source channels than 41 is rank-deficient). Free CPU (zen5_vis):
  python t2p_covbank.py --pools DIR1,DIR2 --names-from <export_proteus_v1> --out covbank.npz
Output: C (K, 41, 41) float64, corpus (K,), subject (K,), n (K,) windows used, n_keep (K,), ch_names (41,).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2p_bench as B  # noqa: E402


def subject_cov(X, rows, keep, W, L):
    """mean covariance of the demeaned windows X[rows][:, keep, :L] mapped by W (41 x len(keep)), accumulated in chunks."""
    S, n = np.zeros((W.shape[0], W.shape[0])), 0
    rows = np.sort(rows)
    for k in range(0, len(rows), 256):
        xs = np.asarray(X[rows[k:k + 256]], np.float64)[:, keep, :L]
        xs = np.einsum("dc,nct->ndt", W, xs)
        xs = xs - xs.mean(-1, keepdims=True)
        S += np.einsum("ncl,ndl->cd", xs, xs)
        n += len(xs)
    return S / (n * L), n


def main():
    import mne
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--names-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-subject", type=int, default=400)
    ap.add_argument("--min-keep", type=int, default=41)
    a = ap.parse_args()
    meta0 = json.loads((Path(a.names_from) / "meta.json").read_text())
    dst, L = meta0["ch_names"], int(meta0["n_times"])
    known = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    rng = np.random.default_rng(20261028)
    out = dict(C=[], corpus=[], subject=[], n=[], n_keep=[])
    for d in a.pools.split(","):
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        names = list(meta["ch_names"])
        keep = [i for i, c in enumerate(names) if c in known]
        if len(keep) < a.min_keep:
            print(f"{d.name}: {len(keep)} standard_1005 channels < {a.min_keep}: skipped", flush=True)
            continue
        W = B.interp_matrix([names[i] for i in keep], dst)
        idx = pd.read_csv(d / "index.csv", dtype=str, keep_default_na=False)
        X = np.load(d / "X.npy", mmap_mode="r")
        assert X.shape[-1] >= L, (d.name, X.shape)
        for s, g in idx.groupby("subject", sort=False):
            rows = np.flatnonzero(idx["subject"].to_numpy() == s)
            if len(rows) > a.max_per_subject:
                rows = rng.choice(rows, a.max_per_subject, replace=False)
            C, n = subject_cov(X, rows, keep, W, L)
            if not np.isfinite(C).all() or np.trace(C) <= 0:
                print(f"{d.name} {s}: degenerate covariance, skipped", flush=True)
                continue
            out["C"].append(C)
            out["corpus"].append(d.name)
            out["subject"].append(str(s))
            out["n"].append(n)
            out["n_keep"].append(len(keep))
        print(f"{d.name}: {sum(c == d.name for c in out['corpus'])} subjects, {len(keep)} source channels", flush=True)
    assert out["C"], "no subject in the bank"
    np.savez(a.out, C=np.stack(out["C"]), corpus=np.array(out["corpus"]), subject=np.array(out["subject"]), n=np.array(out["n"]),
             n_keep=np.array(out["n_keep"]), ch_names=np.array(dst))
    print(f"bank: {len(out['C'])} subjects -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
