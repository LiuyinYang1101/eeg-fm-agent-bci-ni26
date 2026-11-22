#!/usr/bin/env python
"""T2-D513 (template W15): the post-build check of a W15 package against the package it was repacked from (OLD = e.g. W14-lite), on real public
streams, on the GPU path the worker takes (gpu_graph):
  (a) every window BEFORE the first refit point: NEW's output probabilities == OLD's, bitwise (W15's fallback is W14's arithmetic);
  (b) a stream cut shorter than the first refit point: NEW == OLD on every window (adaptation never fires);
  (c) after the first refit: NEW moves (some window differs) and every output stays finite;
  (d) the time per window of NEW vs OLD, and each refit's wall time (the runtime the audit sees);
  (e) a second run of NEW on the same stream gives the same outputs (<= 1e-6): the in-stream copies are reproducible.
  gpu-submit python tracks/t2_bci/scripts/t2p_w15_pkgcheck.py --old <unzipped OLD dir> --new <unzipped NEW dir> [--streams P04|2,P05|1,P09|1]
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
    old = RP.load(Path(a.old), "old", m_meta)
    new = RP.load(Path(a.new), "new", m_meta)
    cfg = json.loads((Path(a.new) / "config.json").read_text())
    c0 = int(cfg["instream"]["refits"][0])
    res = {}
    for st in a.streams.split(","):
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        xs = np.asarray(X[rows], np.float32)
        po, pn = [], []
        t0 = time.time()
        RP.run_stream(old, xs, dev, po)
        t_old = time.time() - t0
        t0 = time.time()
        RP.run_stream(new, xs, dev, pn)
        t_new = time.time() - t0
        po, pn = np.stack(po), np.stack(pn)
        d = np.abs(po - pn).max(1)
        pn2 = []
        RP.run_stream(new, xs, dev, pn2)                 # (e) a second NEW run: the in-stream copies must be reproducible (the audit's re-run)
        rerun_dp = float(np.abs(np.stack(pn2) - pn).max())
        ps = []
        RP.run_stream(new, xs[:c0 - 1], dev, ps)
        short_eq = bool(np.array_equal(np.stack(ps), po[:c0 - 1]))
        r = dict(n=len(rows), pre_refit_bitwise=bool((d[:c0] == 0).all()), post_moved=bool((d[c0:] > 0).any()), finite=bool(np.isfinite(pn).all()),
                 short_stream_bitwise=short_eq, rerun_max_dp=rerun_dp, max_dp_post=float(d[c0:].max()) if len(d) > c0 else 0.0,
                 labels_changed=int((po.argmax(1) != pn.argmax(1)).sum()), ms_per_window_old=1e3 * t_old / len(rows), ms_per_window_new=1e3 * t_new / len(rows))
        res[st] = r
        print(f"[w15 check] {st}: {r}", flush=True)
        assert r["pre_refit_bitwise"] and r["short_stream_bitwise"] and r["finite"], (st, r)
        assert r["rerun_max_dp"] <= 1e-6, (st, "a re-run of NEW differs", r["rerun_max_dp"])
        assert r["post_moved"] or len(rows) <= c0, (st, "the refit did not move the output")
    print("[w15 check] PASS: bitwise before the first refit and on a short stream; adaptation moves the output after it; all finite", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
