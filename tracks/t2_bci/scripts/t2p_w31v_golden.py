#!/usr/bin/env python
"""T2-D654 (W31v) the GOLDEN + TIMING check of the split-graph view template (the manager's first gate), on one H200, over every stream of the
PROTEUS export (one window at a time from a fresh per-stream copy = t2p_repack.run_stream, as the kit): A = W25-ALL6 (the base ZIP), B = A with
the W31v template and fm_views gate "off" (the split graphs only), C.. = gated variants. Reports per package: ms / window (reps interleaved A, B,
C, ...), predictions differing from A, max |p - p_A|, and for gated packages the realised view fraction (per stream, max and mean) vs q.
  python t2p_w31v_golden.py --export E --out OUT --zip A=a.zip --zip B=b.zip [--zip C=c.zip ...] [--reps 2]   -> OUT.json
GATE (pinned by the manager): B vs A = 0 differing predictions and ms within noise; a gated C: the realised fraction <= q + 1 / t by the bucket."""
import argparse
import copy
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
import t2p_fm as FMV  # noqa: E402  (the bench's view implementation = the reference for the template's re-implementation)


def run_all(m, xss, dev):
    preds, probs, fr, t_run = [], [], [], 0.0
    for xs in xss:
        mm = copy.deepcopy(m)
        mm.reset_state()
        t0 = time.time()
        for x in xs:
            preds.append(int(mm.predict(torch.as_tensor(x[None], device=dev))[0]))
            probs.append(np.asarray(mm.last_p, np.float64))
        torch.cuda.synchronize()
        t_run += time.time() - t0
        if getattr(mm, "views", None) is not None:
            fr.append((mm.vg["g"], mm.vg["n"]))
    return np.asarray(preds), np.stack(probs), fr, t_run


def view_checks(m, xs, dev, n_win=20):
    """Track3's MEDIUM (before any C number is trusted): (a) len(idx) == 41 and the template's view spec == t2p_fm.fm_view_spec BITWISE; (b) the
    template's _view_batch(zf)[1:] == t2p_fm.fm_apply_view per view BITWISE; (c) on n_win real windows, the fmK graph's per-member probabilities vs
    an independent eager reference softmax(out(mean_k features(fm_apply_view(zf)))) under the same autocast, max |d| <= 5e-3."""
    K = int(m.views["K"])
    assert len(m.idx) == 41, len(m.idx)
    ref = FMV.fm_view_spec(41, K)
    assert len(ref) == len(m.view_spec) == K - 1
    for (g1, s1), (g2, s2) in zip(m.view_spec, ref):
        assert torch.equal(g1, g2) and int(s1) == int(s2), ("(a) view spec differs", s1, s2)
    zr = torch.randn(1, 41, int(m.n_times), device=dev)
    vb = m._view_batch(zr)
    for v, (g, s) in enumerate(ref, 1):
        assert torch.equal(vb[v:v + 1], FMV.fm_apply_view(zr, g, s)), ("(b) view batch differs", v)
    assert torch.equal(vb[:1], zr)
    mm = copy.deepcopy(m)
    mm.reset_state()
    dmax = 0.0
    for x in xs[:n_win]:
        xi = np.asarray(x, np.float32)[mm.idx]
        zfs = [torch.from_numpy(al(mm._fm_x(j, xi)).astype(np.float32))[None].to(dev) for j, al in enumerate(mm.fm_aligners)]
        got = mm._run(mm._gpu_fmk, [mm._view_batch(zf) for zf in zfs], "fmK")
        for net, zf, pg in zip(mm.fm_nets, zfs, got):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vs = [zf] + [FMV.fm_apply_view(zf, g, s) for g, s in ref]
                f_ = torch.stack([net.features(v_).float()[0] for v_ in vs]).mean(0, keepdim=True)
                pr = torch.softmax(net.out(f_).float(), dim=1)[0]
            dmax = max(dmax, float((pg.float() - pr).abs().max()))
    assert dmax <= 5e-3, ("(c) fmK graph vs the eager reference", dmax)
    return dict(a_spec_bitwise=True, b_view_batch_bitwise=True, c_fmk_vs_eager_max_abs=dmax, n_win=min(n_win, len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", action="append", required=True, help="NAME=path (the first one = the base A)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--save-pred", default="", help="NAME: also write OUT.<NAME>.pred.npz (i = export row, pred) for K2 (t2p_w19_kitfull.py)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    E = Path(a.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    R.kit.use_track("bci_decoding")
    meta = dict(device="cuda", ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]), n_times=int(X.shape[-1]), n_classes=3)
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    rows = np.concatenate([g["i"].to_numpy() for g in streams])   # the export row of every prediction, in run_all's order
    xss = [np.asarray(X[g["i"].to_numpy()], np.float32) for g in streams]
    n = sum(len(x) for x in xss)
    zips = [z.split("=", 1) for z in a.zip]
    res = dict(n_windows=n, n_streams=len(xss), gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, packages={})
    with tempfile.TemporaryDirectory() as td:
        models = {}
        for name, zp in zips:
            d = Path(td) / name
            with zipfile.ZipFile(zp) as z:
                z.extractall(d)
            for p in d.rglob("*"):
                p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            models[name] = R.load(d, f"g{name}", meta)
            res["packages"][name] = dict(zip=zp, fm_views=json.loads((d / "config.json").read_text()).get("fm_views"), ms=[])
            if getattr(models[name], "views", None) is not None:   # Track3's MEDIUM: the template's views == the bench's, checked first
                res["packages"][name]["view_checks"] = view_checks(models[name], xss[0], dev)
                print(f"[w31v golden] {name}: view checks {res['packages'][name]['view_checks']}", flush=True)
        base = {}
        for r in range(a.reps):
            for name, _ in zips:
                pr, pb, fr, t_run = run_all(models[name], xss, dev)
                P = res["packages"][name]
                P["ms"].append(round(1e3 * t_run / n, 3))
                if r == 0:
                    if not base:
                        base = dict(pr=pr, pb=pb)
                    if name == a.save_pred:
                        np.savez(a.out + f".{name}.pred.npz", i=rows, pred=pr)
                    P["differ_vs_base"] = int((pr != base["pr"]).sum())
                    P["max_abs_dp_vs_base"] = float(np.abs(pb - base["pb"]).max())
                    if fr:
                        q = float(P["fm_views"]["q"]) if P["fm_views"].get("gate", "cnn") != "off" else 0.0
                        f_ = [g / max(t, 1) for g, t in fr]
                        P["view_frac"] = dict(mean=float(np.mean(f_)), max=float(np.max(f_)), total=float(sum(g for g, _ in fr) / n), q=q,
                                              bucket_ok=bool(all(g <= q * t + 1 for g, t in fr)))
                print(f"[w31v golden] rep {r} {name}: {P['ms'][-1]:.2f} ms / window; differ {P.get('differ_vs_base')} (max |dp| {P.get('max_abs_dp_vs_base', 0):.2e})"
                      + (f"; views on {P['view_frac']['total']:.1%} (max stream {P['view_frac']['max']:.1%})" if "view_frac" in P else ""), flush=True)
        for name, P in res["packages"].items():
            P["mean_ms"] = float(np.mean(P["ms"]))
            P["line_s"] = 62.0 * P["mean_ms"] - 163.0                    # the D517 line (the registered affine timing line)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: dict(mean_ms=v["mean_ms"], line_s=v["line_s"], differ=v.get("differ_vs_base"), view_frac=v.get("view_frac")) for k, v in res["packages"].items()}), flush=True)


if __name__ == "__main__":
    main()
