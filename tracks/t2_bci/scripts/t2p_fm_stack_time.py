#!/usr/bin/env python
"""T2-D475e (the manager's ask, 2026-11-14): the MEASURED runtime of a larger foundation-model slot before the user moves the 40 ms bar.
A gated package is unpacked once; each variant adds k copies of its first k ST-L members (stand-ins for k CAR members: CAR itself is one
channel mean per window, the members' compute is the same) and is timed through t2p_repack's load() + run_stream(), the path that
measured M7's 39.4 ms / window, on every public stream, one window at a time, after an untimed warm-up stream (CUDA-graph capture).
Implied sealed seconds = M7's 2,250.21 s x (ms_k / ms_0) (ratio form) and 2,250.21 + (ms_k - ms_0) x 2,250.21 / 39.39 (additive form).
    python t2p_fm_stack_time.py --zip <gated zip> --export <export_proteus_v1> --out <dir> [--dups 0,2,3,5]
"""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_repack as RP  # noqa: E402
from ni26 import kit  # noqa: E402

M7_SEALED_S, M7_MS = 2250.21, 39.39          # T2-D435 (sealed duration) / T2-D384 (repack ms / window, H200)


def variant(base, k, out, cnn_keep=0):
    """base's files symlinked into out, + fm_{n+j}.pt -> fm_j.pt for j < k, config fm_members extended with those copies.
    T2-D475g: cnn_keep > 0 keeps the first cnn_keep seeds of every CNN family (pooled nets and each user's copies; written, not linked)."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base / "config.json").read_text())
    fm = cfg.get("fm_members", [])
    assert k <= len(fm) and all(m.get("weight") is None for m in fm), "copies assume equal-weight ST members (the mean over members)"
    written = set()
    if cnn_keep:
        import torch
        pooled = ["weights.pt"] + [cfg[c]["file"] for c in ("cnn2", "cnn3") if cfg.get(c)]
        per_user = [cfg[u]["file"] for u in ("user_blend", "user_blend2", "user_blend3") if cfg.get(u)]
        for f in pooled:
            L = torch.load(base / f, map_location="cpu", weights_only=True)
            assert len(L) >= cnn_keep, (f, len(L))
            torch.save(L[:cnn_keep], out / f)
            written.add(f)
        for f in per_user:
            L = torch.load(base / f, map_location="cpu", weights_only=True)
            assert all(len(u) >= cnn_keep for u in L), f
            torch.save([u[:cnn_keep] for u in L], out / f)
            written.add(f)
        for u in ("user_blend", "user_blend2", "user_blend3"):
            if cfg.get(u) and "n_seeds" in cfg[u]:
                cfg[u]["n_seeds"] = cnn_keep
    for f in base.iterdir():
        if f.name != "config.json" and f.name not in written:
            os.symlink(f.resolve(), out / f.name)
    n = len(fm)
    for j in range(k):
        os.symlink((base / fm[j]["file"]).resolve(), out / f"fm_{n + j}.pt")
        fm.append(dict(fm[j], file=f"fm_{n + j}.pt", note="T2-D475e timing stand-in (a copy)"))
    cfg["fm_members"] = fm
    (out / "config.json").write_text(json.dumps(cfg))
    return len(fm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dups", default="0,2,3,5", help="extra ST-L copies per variant (all CNN seeds kept)")
    ap.add_argument("--variants", default="", help="T2-D475g: 'k:n,...' = k extra ST-L copies with the first n CNN seeds per family (0 = all); "
                    "overrides --dups")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    base = out / "pkg"
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(base)
    E = Path(a.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    kit.use_track("bci_decoding")
    meta = dict(device=a.device, ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]),
                n_times=int(X.shape[-1]), n_classes=3)
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    res = {}
    n0 = len(json.loads((base / "config.json").read_text()).get("fm_members", []))
    vs = [tuple(int(x) for x in v.split(":")) for v in a.variants.split(",")] if a.variants else [(int(v), 0) for v in a.dups.split(",")]
    for k, nk in vs:
        d = base if (k, nk) == (0, 0) else out / f"dup{k}_cnn{nk}"
        n_fm = variant(base, k, d, nk) if (k, nk) != (0, 0) else n0
        m = RP.load(d, f"dup{k}_cnn{nk}", meta)
        RP.run_stream(m, np.asarray(X[streams[0]["i"].to_numpy()], np.float32), a.device)     # warm-up (graph capture), untimed
        t, n = 0.0, 0
        for g in streams:
            xs = np.asarray(X[g["i"].to_numpy()], np.float32)
            t0 = time.time()
            RP.run_stream(m, xs, a.device)
            t += time.time() - t0
            n += len(xs)
        key = str(k) if not nk else f"{k}:{nk}"
        res[key] = dict(fm_members=n_fm, cnn_seeds=nk or "all", ms_per_window=round(1e3 * t / n, 2), windows=n)
        print(f"[stack_time] +{k} ST-L copies ({n_fm} members), CNN seeds {nk or 'all'}: {res[key]['ms_per_window']} ms / window on {n} windows", flush=True)
        del m
        import torch
        torch.cuda.empty_cache()
    ms0 = res["0"]["ms_per_window"] if "0" in res else None
    for k, r in res.items():
        if ms0:
            r["sealed_s_ratio"] = round(M7_SEALED_S * r["ms_per_window"] / ms0)
            r["sealed_s_additive"] = round(M7_SEALED_S + (r["ms_per_window"] - ms0) * M7_SEALED_S / M7_MS)
            r["under_3400"] = max(r["sealed_s_ratio"], r["sealed_s_additive"]) <= 3400
            r["under_3600"] = max(r["sealed_s_ratio"], r["sealed_s_additive"]) <= 3600
    (out.parent / (out.name + ".stack_time.json")).write_text(json.dumps(dict(zip=a.zip, res=res), indent=1))
    print(json.dumps(res, indent=1), flush=True)
    shutil.rmtree(out)                       # the unpacked copies (the zip stays where it is)


if __name__ == "__main__":
    main()
