#!/usr/bin/env python
"""T2-D517 (template W17): the post-build check of a W17 package against the W15 package it was repacked from (OLD = the step-B build), on real
public streams, on the GPU path the worker takes:
  (a) every output finite; (b) NEW moves against OLD on every stream (the window-mean tree family + side nets act from window 1);
  (c) a second run of NEW on the same stream gives the same outputs (<= 1e-6; the in-stream copies under deterministic cuDNN, the side nets frozen);
  (d) the time per window of NEW vs OLD (the runtime the audit sees; W15 left ≈ 13 ms per window under the 3,400 s bar at ≈ 23k windows).
The template's arithmetic itself is proven by t2p_w17_selftest.py and t2p_repack.py (inert step E; step F's independent recombination).
  gpu-submit python tracks/t2_bci/scripts/t2p_w17_pkgcheck.py --old <unzipped W15 dir> --new <unzipped W17 dir> [--streams P04|2,P05|1,P09|1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--streams", default="P04|2,P05|1,P09|1")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(EXP), help="the PROTEUS export (sofia: data/ni26/t2_proteus/export_proteus_v1)")
    a = ap.parse_args()
    RP.kit.use_track("bci_decoding")
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    cfg = json.loads((Path(a.new) / "config.json").read_text())
    assert cfg.get("side") and cfg.get("dcfam") and cfg.get("instream"), "W17 check: NEW must carry instream + dcfam + side"
    old = RP.load(Path(a.old), "old", m_meta)
    new = RP.load(Path(a.new), "new", m_meta)
    res = {}
    for st in a.streams.split(","):
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        xs = np.asarray(X[rows], np.float32)
        po, pn, pn2 = [], [], []
        t0 = time.time()
        RP.run_stream(old, xs, dev, po)
        t_old = time.time() - t0
        t0 = time.time()
        RP.run_stream(new, xs, dev, pn)
        t_new = time.time() - t0
        RP.run_stream(new, xs, dev, pn2)
        po, pn = np.stack(po), np.stack(pn)
        d = np.abs(po - pn).max(1)
        r = dict(n=len(rows), finite=bool(np.isfinite(pn).all()), moved_windows=int((d > 0).sum()), max_dp=float(d.max()),
                 labels_changed=int((po.argmax(1) != pn.argmax(1)).sum()), rerun_max_dp=float(np.abs(np.stack(pn2) - pn).max()),
                 ms_per_window_old=1e3 * t_old / len(rows), ms_per_window_new=1e3 * t_new / len(rows))
        res[st] = r
        print(f"[w17 check] {st}: {r}", flush=True)
        assert r["finite"] and r["moved_windows"] > 0, (st, r)
        assert r["rerun_max_dp"] <= 1e-6, (st, "a re-run of NEW differs", r["rerun_max_dp"])
    print("[w17 check] PASS: finite; moves against W15 on every stream; a re-run is reproducible", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
