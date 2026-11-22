#!/usr/bin/env python
"""T2-D352 (the manager's 06:5x ask): run-to-run noise of a packaged T2 ZIP's runtime. Exactly t2p_repack's measurement (its own load() and
run_stream(): every public stream, one window at a time from a fresh deepcopy + reset_state, the probabilities collected), repeated --reps
times, each from a FRESH load of a read-only unzip; also the load time. Writes <out>.json.
T2-D381: --set dotted.key=JSON (repeatable; t2p_repack.set_cfg) edits the unzipped config.json before the read-only chmod — a scratch
timing of a config variant (never a ship candidate); unset = the ZIP as packaged, byte for byte.
  python t2p_time_zip.py --zip Z --export E --out OUT [--reps 3] [--device cuda] [--set cnn_tta_vp.copies=false ...]
"""
import argparse
import json
import os
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2p_repack as R  # noqa: E402


def patch(cfg_path, kvs):
    """T2-D381: apply each dotted.key=JSON edit to config.json in place; returns [(key, old, new)]; no edits = the file untouched."""
    if not kvs:
        return []
    cfg = json.loads(cfg_path.read_text())
    out = [R.set_cfg(cfg, kv) for kv in kvs]
    cfg_path.write_text(json.dumps(cfg, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", action="append", default=[], help="T2-D381: dotted.key=JSON config edit (repeatable)")
    a = ap.parse_args()
    E = Path(a.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    R.kit.use_track("bci_decoding")
    meta = dict(device=a.device, ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]),
                n_times=int(X.shape[-1]), n_classes=3)
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    xss = [np.asarray(X[g["i"].to_numpy()], np.float32) for g in streams]
    n = sum(len(x) for x in xss)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "ro"
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(d)
        sets = patch(d / "config.json", a.set)
        for p in d.rglob("*"):
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        reps = []
        for r in range(a.reps):
            t0 = time.time()
            m = R.load(d, f"t{r}", meta)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            t_load = time.time() - t0
            t_run = 0.0
            for xs in xss:
                pb = []
                t1 = time.time()
                R.run_stream(m, xs, a.device, pb)
                t_run += time.time() - t1
            gs = getattr(m, "gshared", {}) or {}                # T2-D559: the in-stream refit seconds of this rep and whether a budget fired
            reps.append(dict(rep=r, load_s=round(t_load, 2), ms_per_window=round(1e3 * t_run / n, 2), instream_s=round(float(gs.get("is_s", 0.0)), 1),
                             run_budget_hit=bool(gs.get("is_off", False))))
            print(f"[time_zip] rep {r}: load {t_load:.1f}s, {1e3 * t_run / n:.2f} ms / window over {n} windows", flush=True)
            del m
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
    ms = [x["ms_per_window"] for x in reps]
    res = dict(zip=a.zip, sets=sets, n_windows=n, reps=reps, mean_ms=round(float(np.mean(ms)), 2), sd_ms=round(float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0, 3),
               max_ms=max(ms), gpu=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", host=os.uname().nodename)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
