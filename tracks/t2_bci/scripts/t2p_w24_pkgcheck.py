#!/usr/bin/env python
"""T2-D638 W24' (PREPARED; only if rule (b) passes): the post-build check of a W24' package (the base W21 / W23 with its six CNN files replaced by
a user-balanced donor, tracks/t2_bci/scripts/t2p_w24_swap.py) against that BASE (OLD), on every public stream, on the worker's path
(t2p_repack.load / run_parts: a fresh deepcopy + reset_state per stream, one window at a time, NI26_PARTS=1 for the parts).
  (0) files: every package file except weights{,2,3}.pt / weights{,2,3}_pu.pt byte-identical to OLD's (config.json, submission.py, FM, side,
      dcfam, tta_vp, ...); those six all differ;
  (a) finite; (c) a re-run reproduces NEW (<= 1e-6);
  (M) the MECHANISM: on every window BEFORE the stream's first in-stream refit (cfg instream refits), NEW's FM part (parts[2]) and window-mean
      family (DC) == OLD's (<= 1e-12: the FM members, side heads and DC never see the CNN before a refit) and the copy route k == OLD's on
      EVERY window (the router is config); the CNN parts (parts[0] / [1] / [3]) MOVE (max |d| > 1e-4): only the CNN members / copies changed;
  NEW's argmax on every public window -> .pred.npz (K2).
  python tracks/t2_bci/scripts/t2p_w24_pkgcheck.py --old <unzipped base dir> --new <unzipped W24' dir> [--out res.json] [--streams N]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["NI26_PARTS"] = "1"                    # before any template import: the parts are exposed (diagnostics; the worker leaves it off)
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402

SWAP = ("weights.pt", "weights2.pt", "weights3.pt", "weights_pu.pt", "weights2_pu.pt", "weights3_pu.pt")
LOG = {"old": [], "new": []}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def wrap_router(model, tag):
    cls = type(model)
    orig = cls._user

    def _user(self, xi):
        k = orig(self, xi)
        LOG[tag].append(-1 if k is None else int(k))
        return k
    cls._user = _user


def run(model, xs, dev, tag):
    LOG[tag].clear()
    lp, parts, dcs, _ = RP.run_parts(model, xs, dev)
    return np.stack(lp), parts, dcs, np.array(LOG[tag])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    ap.add_argument("--streams", type=int, default=0)
    a = ap.parse_args()
    RP.kit.use_track("bci_decoding")
    old_d, new_d = Path(a.old), Path(a.new)
    fo = sorted(p.relative_to(old_d).as_posix() for p in old_d.rglob("*") if p.is_file())
    fn = sorted(p.relative_to(new_d).as_posix() for p in new_d.rglob("*") if p.is_file())
    same = {f: sha(old_d / f) == sha(new_d / f) for f in fo} if fo == fn else {}
    files_ok = fo == fn and all(same[f] for f in fo if f not in SWAP) and all(not same[f] for f in SWAP)
    cfg0 = dict(files=len(fo), kept_identical=sum(same.get(f, False) for f in fo if f not in SWAP), replaced=[f for f in SWAP if not same.get(f, True)],
                files_ok=files_ok)
    print(f"[w24 check] (0) {cfg0}", flush=True)
    assert files_ok, cfg0
    cfg = json.loads((old_d / "config.json").read_text())
    first = min(int(c) for c in cfg["instream"]["refits"])
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    old, new = RP.load(old_d, "old", m_meta), RP.load(new_d, "new", m_meta)
    wrap_router(old, "old")
    wrap_router(new, "new")
    streams = list(dict.fromkeys(idx.sort_values(["stream", "order"])["stream"].tolist()))
    if a.streams:
        streams = streams[:a.streams]
    pred_new = np.full(len(idx), -1)
    worst = dict(rerun=0.0, fm_pre=0.0, dc_pre=0.0, cnn_moved=0.0)
    bad_route, finite, n_win = 0, True, 0
    for st in streams:
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        xs = np.asarray(X[rows], np.float32)
        po, pto, dco, ko = run(old, xs, dev, "old")
        pn, ptn, dcn, kn = run(new, xs, dev, "new")
        pn2, _, _, _ = run(new, xs, dev, "new")
        finite &= bool(np.isfinite(pn).all())
        pred_new[rows] = pn.argmax(1)
        n_win += len(xs)
        bad_route += int((ko != kn).sum())
        worst["rerun"] = max(worst["rerun"], float(np.abs(pn2 - pn).max()))
        for t in range(min(first, len(xs))):           # before the stream's first in-stream refit
            if ptn[t][2] is not None:
                worst["fm_pre"] = max(worst["fm_pre"], float(np.abs(ptn[t][2] - pto[t][2]).max()))
            if dcn[t] is not None:
                worst["dc_pre"] = max(worst["dc_pre"], float(np.abs(dcn[t] - dco[t]).max()))
            for j in (0, 1, 3):
                if len(ptn[t]) > j and ptn[t][j] is not None:
                    worst["cnn_moved"] = max(worst["cnn_moved"], float(np.abs(ptn[t][j] - pto[t][j]).max()))
        print(f"[w24 check] {st}: {len(xs)} windows; argmax moved {int((pn.argmax(1) != po.argmax(1)).sum())}; worst {worst}; bad routes {bad_route}", flush=True)
    res = dict(config=cfg0, streams=len(streams), windows=n_win, first_refit=first, finite=finite, worst=worst, bad_routes=bad_route,
               gates=dict(a=finite, c=worst["rerun"] <= 1e-6, M_fm=worst["fm_pre"] <= 1e-12, M_dc=worst["dc_pre"] <= 1e-12,
                          M_route=bad_route == 0, M_cnn_moved=worst["cnn_moved"] > 1e-4))
    print(f"[w24 check] {json.dumps(res['gates'])}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        if not a.streams:
            np.savez(a.out.replace(".json", "") + ".pred.npz", i=np.arange(len(idx)), pred=pred_new)   # read by t2p_w19_kitfull.py (K2)
    assert all(res["gates"].values()), res["gates"]
    print("[w24 check] PASS: only the six CNN files differ from the base; finite; reproducible; before the first refit the FM part and the DC "
          "family == the base's and the CNN parts moved; the routes == the base's on every window", flush=True)


if __name__ == "__main__":
    main()
