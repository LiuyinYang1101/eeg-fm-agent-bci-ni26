#!/usr/bin/env python
"""T2 P1 student-rep ensemble (the manager's GO; this script first committed 2026-11-19 04:05:48 (git 6c25be95), BEFORE any number): the runtime of ONE P1 student fine-tune on the GPU path
of W21 as shipped. Loads W21 (the gated ZIP's unzipped copy) through t2p_repack's loader, wraps the template's _w15_finetune with a timer, runs
the first N public streams (the worker path, gpu_graph on) and prints the fine-tune seconds per call, the calls per stream, the stream seconds
and the windows. The extra cost of +k reps = k x (the fine-tune seconds) per fired refit per P1 family.
    gpu-submit python scripts/local/t2_rep_timing.py --zip <W21 zip> --streams 6   ->  one 'REPTIME {...}' line
    (sofia H200: tracks/t2_bci/sofia/t2p_reptime.slurm)
"""
import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[2] / "tracks/t2_bci/scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--streams", type=int, default=6)
ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
a = ap.parse_args()
RP.kit.use_track("bci_decoding")
X, idx, meta = B.load_export(a.export)
dev = "cuda"
assert torch.cuda.is_available()
m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
tmp = Path(tempfile.mkdtemp(prefix="t2_reptime_"))
zipfile.ZipFile(a.zip).extractall(tmp)
model = RP.load(tmp, "w21", m_meta)
G = type(model)._instream_fit.__globals__          # the template module's globals (loaded via an importlib spec, not in sys.modules)
f0 = G["_w15_finetune"]
CALLS = []


def timed(*args, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = f0(*args, **kw)
    torch.cuda.synchronize()
    CALLS.append(time.perf_counter() - t0)
    return out


G["_w15_finetune"] = timed
streams = list(dict.fromkeys(idx.sort_values(["stream", "order"])["stream"].tolist()))[:a.streams]
per, wins, tot = [], 0, 0.0
for st in streams:
    rows = idx.index[idx["stream"] == st].to_numpy()
    rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
    n0 = len(CALLS)
    t0 = time.perf_counter()
    RP.run_stream(model, np.asarray(X[rows], np.float32), dev, [])
    tot += time.perf_counter() - t0
    per.append(dict(stream=st, windows=len(rows), finetunes=len(CALLS) - n0, ft_s=round(float(np.sum(CALLS[n0:])), 3)))
    wins += len(rows)
res = dict(gpu=torch.cuda.get_device_name(0), streams=len(streams), windows=wins, stream_s=round(tot, 2), ms_per_window=round(1000 * tot / wins, 2),
           finetunes=len(CALLS), ft_s_total=round(float(np.sum(CALLS)), 3), ft_s_mean=round(float(np.mean(CALLS)), 4) if CALLS else None,
           ft_s_max=round(float(np.max(CALLS)), 4) if CALLS else None, ft_ms_per_window=round(1000 * float(np.sum(CALLS)) / wins, 3), per_stream=per)
print("REPTIME " + json.dumps(res), flush=True)
import shutil  # noqa: E402
shutil.rmtree(tmp, ignore_errors=True)   # the unzipped package copy (disposable)
