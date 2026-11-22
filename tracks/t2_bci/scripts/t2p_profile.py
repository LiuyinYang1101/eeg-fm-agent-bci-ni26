#!/usr/bin/env python
"""Track 2 (T2-D140 timing): where does a packaged StreamModel spend its per-window time? Loads a package dir (submission.py +
config + weights, as the kit does), streams the first --n windows of each of --streams public streams one at a time (B = 1,
fresh deepcopy + reset_state per stream, as Codabench) and times each component of StreamModel.predict with CUDA
synchronisation: main aligner, main CNN family (stacked), user fingerprint, per-user family, second CNN family + aligner,
every FM member (aligner + forward), prior, plus the whole predict() call for the same windows.

  python t2p_profile.py --pkg <package dir> --export <export_proteus_v1> [--n 200 --streams 3] [--out prof.json]
"""

import argparse
import copy
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from ni26 import kit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kit.use_track("bci_decoding")
    pkg = Path(args.pkg)
    spec = importlib.util.spec_from_file_location("t2p_prof_sub", pkg / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = json.loads((Path(args.export) / "meta.json").read_text())
    X = np.load(Path(args.export) / "X.npy", mmap_mode="r")
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    solver = mod.Solver.__new__(mod.Solver)
    m = solver.load_model(dict(submission_dir=pkg, device=dev, ch_names=list(meta["ch_names"]), sfreq=meta["sfreq"],
                               n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3))
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    T = defaultdict(float)
    n_tot = 0
    streams = list(idx.groupby("stream", sort=False))[: args.streams]
    for _, g in streams:                                   # component timings (mirrors StreamModel.predict)
        mm = copy.deepcopy(m)
        mm.reset_state()
        rows = g.sort_values("order")["i"].to_numpy()[: args.n]
        with torch.inference_mode():
            for r in rows:
                xi = np.asarray(X[int(r)], np.float32)[mm.idx]
                t = time.perf_counter()
                z = torch.from_numpy(mm.aligner(xi))[None].to(dev); sync(); T["aligner"] += time.perf_counter() - t
                t = time.perf_counter()
                p = mm._fam(mm.nets, mm.stk[0], z); T["cnn_main"] += time.perf_counter() - t
                k = None
                if len(mm.pu_nets):
                    t = time.perf_counter()
                    k = mm._user(xi); T["user_fingerprint"] += time.perf_counter() - t
                    if k is not None:
                        t = time.perf_counter()
                        mm._fam(mm.pu_nets[k], mm.stk_pu[k], z); T["cnn_per_user"] += time.perf_counter() - t
                if len(mm.nets2):
                    t = time.perf_counter()
                    z2 = torch.from_numpy(mm.aligner2(xi))[None].to(dev)
                    mm._fam(mm.nets2, mm.stk[1], z2); T["cnn2"] += time.perf_counter() - t
                    if len(getattr(mm, "pu_nets2", [])) and k is not None:   # T2-D161: the second family's routed copies
                        t = time.perf_counter()
                        mm._fam(mm.pu_nets2[k], mm.stk_pu2[k], z2); T["cnn2_per_user"] += time.perf_counter() - t
                for j, (net, al) in enumerate(zip(mm.fm_nets, mm.fm_aligners)):
                    fam = mm.cfg["fm_members"][j].get("family", "st")
                    t = time.perf_counter()
                    zf = torch.from_numpy(al(xi))[None].to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=zf.is_cuda):
                        o = net(zf, net.delta(k)) if hasattr(net, "delta") else net(zf)   # T2-D188: an STRouted member
                        F.softmax(o.float(), dim=1)[0].cpu()
                    sync(); T[f"fm_{j}_{fam}"] += time.perf_counter() - t
                if mm.prior is not None:
                    t = time.perf_counter()
                    mm.prior(np.full(3, 1 / 3)); T["prior"] += time.perf_counter() - t
                n_tot += 1
    n_full = 0
    t_full = 0.0
    for _, g in streams:                                   # the real predict() on the same windows
        mm = copy.deepcopy(m)
        mm.reset_state()
        for r in g.sort_values("order")["i"].to_numpy()[: args.n]:
            xw = torch.as_tensor(np.asarray(X[int(r)])[None], device=dev)
            sync()
            t = time.perf_counter()
            mm.predict(xw)
            sync()
            t_full += time.perf_counter() - t
            n_full += 1
    G = {}
    if getattr(m, "gpu_graph", False) and dev == "cuda":    # T2-D140: the graph path, piece by piece
        G = defaultdict(float)
        n_g = 0
        for _, g in streams:
            mm = copy.deepcopy(m)
            mm.reset_state()
            with torch.inference_mode():
                for r in g.sort_values("order")["i"].to_numpy()[: args.n]:
                    xi = np.asarray(X[int(r)], np.float32)[mm.idx]
                    sync(); t = time.perf_counter()
                    z = torch.from_numpy(mm.aligner(xi))[None].to(dev)
                    z2 = torch.from_numpy(mm.aligner2(xi))[None].to(dev) if len(mm.nets2) else torch.zeros(1, device=dev)
                    zfs = [torch.from_numpy(al(xi))[None].to(dev) for al in mm.fm_aligners]
                    sync(); G["aligners_h2d"] += time.perf_counter() - t; t = time.perf_counter()
                    k = mm._user(xi) if len(mm.pu_nets) else None
                    sync(); G["user_fingerprint"] += time.perf_counter() - t; t = time.perf_counter()
                    outs = mm._run(mm._gpu_eager, [z, z2] + zfs, "main")
                    sync(); G["main graph (replay + clone)"] += time.perf_counter() - t; t = time.perf_counter()
                    if k is not None:
                        fam = lambda zs, cache=True, k=k: [mm._fam_t(mm.pu_nets[k], mm.stk_pu[k], zs[0])]  # noqa: E731
                        mm._run(fam, [z], ("pu", k))
                        sync(); G["per-user graph (replay + clone)"] += time.perf_counter() - t; t = time.perf_counter()
                    [q.cpu().numpy() for q in outs]
                    G["d2h"] += time.perf_counter() - t
                    n_g += 1
            for slot, gg in mm.gshared["g"].items():
                if gg:
                    sync(); t = time.perf_counter()
                    for _ in range(50):
                        gg[0].replay()
                    sync(); G[f"replay_only {slot}"] = (time.perf_counter() - t) / 50 * n_g
        G = {k: round(1000 * v / n_g, 3) for k, v in G.items()}
        G["graphs"] = {str(k_): bool(v_) for k_, v_ in m.gshared["g"].items()}
        G["off"] = bool(m.gshared.get("off"))
    ms = {k: round(1000 * v / n_tot, 3) for k, v in sorted(T.items(), key=lambda kv: -kv[1])}
    res = dict(pkg=str(pkg), device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", windows=n_tot,
               ms_per_window_components=ms, ms_sum_components=round(sum(ms.values()), 3),
               ms_per_window_predict=round(1000 * t_full / n_full, 3), graph_path_ms=G)
    print(json.dumps(res, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
