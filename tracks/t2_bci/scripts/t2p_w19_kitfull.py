#!/usr/bin/env python
"""T2-D549 condition K, part K2: a packaged ZIP through the KIT's own streaming harness on ALL public PROTEUS streams (the kit's benchopt
warm-up, part K1 in the build job, covers only its test split P04 + P09). benchmark_utils.streaming.predict_streams = a fresh deepcopy +
reset_state per stream, one window at a time, exactly what Codabench runs; the kit's own metric (benchmark_utils.metrics.group_scores with
balanced_accuracy per subject x session x context cell, averaged). Passes only if: no exception; the template's side-refit totals report 0 errors
and 0 kept-trained fallbacks; every window's prediction == the post-build check's saved NEW prediction (<pkgcheck>.pred.npz); the wall time per
window is reported beside the H200 timing (the build's own t2p_time_zip; consistency is judged within +-20 %).
    python tracks/t2_bci/scripts/t2p_w19_kitfull.py --pkg <unzipped dir> --pred <pkgcheck>.pred.npz [--export ...] [--out res.json]
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
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--pred", required=True, help="the post-build check's NEW predictions (t2p_w19_pkgcheck --out X.json -> X.pred.npz)")
    ap.add_argument("--export", default=str(EXP))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    RP.kit.use_track("bci_decoding")
    from benchmark_utils.streaming import predict_streams
    from benchmark_utils.metrics import balanced_accuracy, group_scores
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    model = RP.load(Path(a.pkg), "kitfull", m_meta)
    st0 = dict(getattr(model, "side_refit_stats", {}) or {})
    fr0 = dict(getattr(model, "fm_refit_stats", {}) or {})          # T2-D553
    pr0 = dict(getattr(model, "fm_proto_stats", {}) or {})          # T2-D560 (W20; absent on W19 -> {})
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]

    def loader():                                         # one batch per stream: contiguous, in time order (the kit splits it into windows)
        for g in streams:
            ii = g["i"].to_numpy()
            yield (torch.as_tensor(np.asarray(X[ii], np.float32)), torch.as_tensor(g["y"].to_numpy()),
                   dict(stream_id=g["stream"].to_numpy(), record_id=g["stream"].to_numpy(), onset=g["order"].to_numpy().astype(np.float64)))
    t0 = time.time()
    y_true, y_pred, info = predict_streams(model, loader())
    wall = time.time() - t0
    order_i = np.concatenate([g["i"].to_numpy() for g in streams])
    cells = np.concatenate([g["cell"].to_numpy() for g in streams])
    cells = np.unique(cells.astype(str), return_inverse=True)[1].astype(np.int64)   # integer codes (the same groups): some numpy builds
    # (zen5_vis) refuse np.unique(axis=0) on object arrays inside the kit's group_scores
    sc = group_scores(balanced_accuracy, np.asarray(y_true), np.asarray(y_pred), cells)
    st1 = dict(getattr(model, "side_refit_stats", {}) or {})
    stats = {k: st1.get(k, 0) - st0.get(k, 0) for k in st1}
    fr1 = dict(getattr(model, "fm_refit_stats", {}) or {})
    fstats = {k: fr1.get(k, 0) - fr0.get(k, 0) for k in fr1}
    pr1 = dict(getattr(model, "fm_proto_stats", {}) or {})
    pstats = {k: pr1.get(k, 0) - pr0.get(k, 0) for k in pr1}
    ref = np.load(a.pred)
    pr = np.full(len(idx), -1, np.int64)
    pr[order_i] = np.asarray(y_pred).astype(np.int64)
    same = bool(np.array_equal(pr[ref["i"]], ref["pred"]))
    res = dict(windows=int(len(y_pred)), streams=len(streams), cells=int(len(sc)), cell_mean_ba=float(np.mean(sc)), wall_s=round(wall, 1),
               ms_per_window=round(1e3 * wall / max(len(y_pred), 1), 2), refit_stats=stats, fm_refit_stats=fstats, fm_proto_stats=pstats, preds_equal_pkgcheck=same,
               n_differ=int((pr[ref["i"]] != ref["pred"]).sum()))
    print(f"[w19 kitfull] {res}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    assert stats.get("fits", 0) > 0, "K2: no side refit ran (the refit switch is off?)"
    assert stats.get("errors", 0) == 0 and stats.get("kept_trained", 0) == 0, ("K2: side-refit fallbacks on the public streams", stats)
    assert fstats.get("fits", 0) > 0 and fstats.get("errors", 0) == 0 and fstats.get("kept_trained", 0) == 0, ("K2: T2-D553 FM-refit fallbacks / none ran", fstats)
    if getattr(model, "fm_proto", None) is not None:      # T2-D560: the kit's per-stream deepcopies must run the prototypes (hooks shared)
        assert pstats.get("refits", 0) > 0 and pstats.get("errors", 0) == 0 and pstats.get("kept_trained", 0) == 0, ("K2: T2-D560 proto fallbacks / none ran", pstats)
    assert same, ("K2: the kit harness's predictions differ from the post-build check's", res["n_differ"])
    print("[w19 kitfull] PASS: the kit harness on every public stream; 0 refit fallbacks; predictions == the post-build check's", flush=True)


if __name__ == "__main__":
    main()
