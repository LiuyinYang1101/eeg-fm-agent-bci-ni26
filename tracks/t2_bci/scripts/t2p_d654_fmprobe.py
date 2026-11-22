#!/usr/bin/env python
"""T2-D654 (FM view-TTA) TIMING PROBE (the manager 2026-11-20 ≈ 15:4x; no training, no score): on an H200 of zen4_h200 (D517's timing host), the
packaged FM members of a ZIP (the W25-ALL6 PREBUILD) at B = 1 / 2 / 3 / 4, each member alone and all six together, EAGER and inside a CUDA graph
(the template's path: config gpu_graph = True), under the template's FM forward (torch.autocast bf16, no_grad, the member nets as load_model builds
them), on real windows through the template's own FM aligners (_fm_x + fm_aligners; B distinct windows stand in for B views).
  f1 = the six members at B = 1 (ms per window-call); r_K = ms(B = K) / ms(B = 1), per member and for the six.
  python t2p_d654_fmprobe.py --zip Z --export E --out OUT [--n 300]   -> OUT.json
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2p_repack as R  # noqa: E402


def timed(fn, n, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts)), float(np.percentile(ts, 90))


def graphed(fwd, inputs):
    """capture fwd(inputs) in one CUDA graph (side-stream warm-up first, as the template's _run does); returns replay()."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd(inputs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fwd(inputs)
    return g.replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    dev = torch.device("cuda")
    E = Path(a.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    R.kit.use_track("bci_decoding")
    meta = dict(device="cuda", ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]), n_times=int(X.shape[-1]), n_classes=3)
    g0 = idx.groupby("stream", sort=False).get_group(idx["stream"].iloc[0]).sort_values("order")
    xs = np.asarray(X[g0["i"].to_numpy()[:8]], np.float32)
    res = dict(zip=a.zip, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, n=a.n, torch=torch.__version__)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "ro"
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(d)
        for p in d.rglob("*"):
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        m = R.load(d, "probe", meta)
        res["gpu_graph_cfg"] = bool(getattr(m, "gpu_graph", False))
        nets = list(m.fm_nets)
        assert nets and not any(type(n).__name__ == "STRouted" for n in nets), "the probe covers plain (unrouted) FM members"
        zs = []                                            # per member: (4, C, T) aligned FM views of 4 real windows (the template's own path)
        wins = [np.ascontiguousarray(x[m.idx], dtype=np.float32) for x in xs[:4]]   # = predict(): x[:, self.idx, :], one window xi at a time
        for j, al in enumerate(m.fm_aligners):
            zs.append(torch.from_numpy(np.stack([al(m._fm_x(j, xi)) for xi in wins]).astype(np.float32)).to(dev))
        res["fm_input_shapes"] = [list(z_.shape) for z_ in zs]

        def fwd_one(j):
            def f(inp):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return F.softmax(nets[j](inp[0]).float(), dim=1)
            return f

        def fwd_all(inps):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return [F.softmax(n_(z_).float(), dim=1) for n_, z_ in zip(nets, inps)]

        per = []
        for j in range(len(nets)):
            row = dict(member=j)
            for B in (1, 2, 3, 4):
                inp = [zs[j][:B].contiguous()]
                f = fwd_one(j)
                row[f"eager_ms_B{B}"] = timed(lambda: f(inp), a.n)[0]
                rp = graphed(f, inp)
                row[f"graph_ms_B{B}"] = timed(rp, a.n)[0]
            for k in ("eager", "graph"):
                for B in (2, 3, 4):
                    row[f"{k}_r{B}"] = row[f"{k}_ms_B{B}"] / row[f"{k}_ms_B1"]
            per.append(row)
            print(f"[d654 probe] member {j}: graph ms B1..4 " + " / ".join(f"{row[f'graph_ms_B{B}']:.3f}" for B in (1, 2, 3, 4))
                  + "; eager " + " / ".join(f"{row[f'eager_ms_B{B}']:.3f}" for B in (1, 2, 3, 4)), flush=True)
        six = {}
        for B in (1, 2, 3, 4):
            inps = [z_[:B].contiguous() for z_ in zs]
            six[f"eager_ms_B{B}"] = timed(lambda: fwd_all(inps), a.n)[0]
            rp = graphed(fwd_all, inps)
            six[f"graph_ms_B{B}"], six[f"graph_p90_B{B}"] = timed(rp, a.n)
        for k in ("eager", "graph"):
            for B in (2, 3, 4):
                six[f"{k}_r{B}"] = six[f"{k}_ms_B{B}"] / six[f"{k}_ms_B1"]
        res.update(members=per, six=six, f1_graph_ms=six["graph_ms_B1"], f1_eager_ms=six["eager_ms_B1"],
                   note="ms per call (median of n); B distinct real windows stand in for B views; one CUDA graph per (member set, B) as the template")
        print(f"[d654 probe] SIX members: graph ms B1..4 " + " / ".join(f"{six[f'graph_ms_B{B}']:.3f}" for B in (1, 2, 3, 4))
              + f" (r2 {six['graph_r2']:.2f}, r3 {six['graph_r3']:.2f}, r4 {six['graph_r4']:.2f}); eager B1..4 "
              + " / ".join(f"{six[f'eager_ms_B{B}']:.3f}" for B in (1, 2, 3, 4)), flush=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
