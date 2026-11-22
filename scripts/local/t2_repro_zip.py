#!/usr/bin/env python
"""Run-to-run reproducibility of a packaged T2 ZIP (the manager's ask 2026-11-10 after two top entries were re-scored):
t2p_time_zip's measurement (fresh load of a read-only unzip, every public stream one window at a time), --reps times, keeping each
rep's predictions and probabilities. Prints, for every pair of reps: flipped predictions, max |dp|, and the cell-BA difference.
  gpu-submit python scripts/local/t2_repro_zip.py --zip Z --out outputs/t2_repro/M7_cuda [--device cuda] [--reps 3]
A second call with --compare A.npz B.npz compares two saved runs (e.g. GPU vs CPU = a stand-in for another GPU's kernels).
"""
import argparse
import json
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def cell_ba(idx, pred):
    d = idx.assign(p=pred)
    out = []
    for _, g in d.groupby("cell"):
        out.append(np.mean([(g.p[g.y == c] == c).mean() for c in sorted(g.y.unique())]))
    return float(np.mean(out))


def compare(idx, a, b, na, nb):
    fl = int((a["pred"] != b["pred"]).sum())
    print(f"{na} vs {nb}: flipped predictions {fl} / {len(a['pred'])}; max |dp| {np.abs(a['prob'] - b['prob']).max():.3e}; "
          f"cell BA {cell_ba(idx, a['pred']):.5f} vs {cell_ba(idx, b['pred']):.5f} (diff {cell_ba(idx, b['pred']) - cell_ba(idx, a['pred']):+.5f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    order = np.concatenate([g["i"].to_numpy() for g in streams])
    ix = idx.set_index("i").loc[order].reset_index()
    if a.compare:
        A, B = (np.load(f) for f in a.compare)
        for ra in range(len(A["pred"])):
            compare(ix, dict(pred=A["pred"][ra], prob=A["prob"][ra]), dict(pred=B["pred"][0], prob=B["prob"][0]), f"{Path(a.compare[0]).stem} rep {ra}", f"{Path(a.compare[1]).stem} rep 0")
        return
    import torch
    sys.path.insert(0, str(REPO / "tracks/t2_bci/scripts"))
    import t2p_repack as R
    meta_e = json.loads((E / "meta.json").read_text())
    X = np.load(E / "X.npy", mmap_mode="r")
    R.kit.use_track("bci_decoding")
    meta = dict(device=a.device, ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]), n_times=int(X.shape[-1]), n_classes=3)
    xss = [np.asarray(X[g["i"].to_numpy()], np.float32) for g in streams]
    P, Q = [], []
    with tempfile.TemporaryDirectory(dir=str(Path(a.out).parent)) as td:
        d = Path(td) / "ro"
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(d)
        for p in d.rglob("*"):
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for r in range(a.reps):
            m = R.load(d, f"t{r}", meta)
            pr, pb = [], []
            for xs in xss:
                pr.append(R.run_stream(m, xs, a.device, pb)[0])
            P.append(np.concatenate(pr))
            Q.append(np.stack(pb).reshape(len(P[-1]), -1))
            print(f"[repro] rep {r} done: cell BA {cell_ba(ix, P[-1]):.5f}", flush=True)
            del m
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
        for p in d.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    np.savez(a.out + ".npz", pred=np.stack(P), prob=np.stack(Q), order=order)
    print("device:", torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", "torch", torch.__version__)
    for i in range(a.reps):
        for j in range(i + 1, a.reps):
            compare(ix, dict(pred=P[i], prob=Q[i]), dict(pred=P[j], prob=Q[j]), f"rep {i}", f"rep {j}")


if __name__ == "__main__":
    main()
