#!/usr/bin/env python
"""T2-D517 / W17 build input: the window-mean SIDE NETS' final models for a package (the D517 dcmean arm, trained on every public window).
Per family (cnn2 = ShallowNet, cnn3 = IFNet): pooled nets on the family's own aligned input (EA, n0 0, shrink 0.05, the package's global scale) with
the 82 window means (delivered + CAR view, t2p_bench.side_raw 'dcmean'), causally z-scored per stream (t2p_bench.side_z, n0 10, the all-window
prior), entering a zero-init linear head added at the logits (ni26.train side head, plain CE branch); recipe r1_pulx_vp2_e180_vr (D517's line).
Output <out>/side.pt (torch.load(weights_only=True)-safe: {"kind", "n0", "mu", "sd", "scale", "seeds", "recipe", "export_sha", "cnn2" / "cnn3":
[{"net": state_dict, "head": {"weight", "bias"}} per seed]}) + <out>/side_golden.npz (every public window: the side z, and each family's per-seed
research-path probabilities softmax(net(XA) + head(Z)), the reference the W17 selftest must reproduce window by window).
    gpu-submit python tracks/t2_bci/scripts/t2p_side_final.py --out ~/workspace/outputs/neural_interface_challenge/t2_d517_final \
        --expect-scale 14.742759704589844 --seeds 51700,51701,51702,51703,51704
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402

FAMS = {"cnn2": "shallow", "cnn3": "ifnet"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="r1_pulx_vp2_e180_vr")
    ap.add_argument("--seeds", default="51700,51701,51702,51703,51704")
    ap.add_argument("--expect-scale", type=float, required=True, help="the package's cnn2 / cnn3 aligner scale (config.json): asserted")
    ap.add_argument("--kind", default="dcmean", choices=("dcmean",))
    args = ap.parse_args()
    import torch
    from ni26 import train as T
    t0 = time.time()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    X, idx, meta = B.load_export(args.export)
    y = idx["y"].to_numpy().astype(int)
    trm = np.ones(len(idx), bool)                                      # --train all (t2p_final's sealed-phase package)
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    assert abs(scale - args.expect_scale) < 1e-6 * max(1.0, abs(scale)), (scale, args.expect_scale)
    tr = np.flatnonzero(trm)
    ev = np.arange(len(idx))
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    XA = B.aligned_all(X, idx, "ea", trm, 0.0, 0.05, scale)
    Fr = B.side_raw(X, args.kind)
    mu0, sd0 = np.nanmean(Fr[tr], 0), np.nanstd(Fr[tr], 0) + 1e-6       # = t2p_bench.side_z's prior
    Z = B.side_z(Fr, idx, tr, args.kind)
    recipe = replace(T.RECIPES[args.recipe])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if recipe.vp:
        T.VP_BANK.update({k_: v_.to(dev) for k_, v_ in B.vp_bank(list(meta["ch_names"]), K=recipe.vp_k, sigma=0.005 * recipe.vp, gain=recipe.vp_g,
                                                                  shift=recipe.vp_s, vref=recipe.vp_r, cap=recipe.vp_cap).items()})
        print(f"[side_final] virtual participants: {len(T.VP_BANK['mats'])} draws", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    blob = dict(kind=args.kind, n0=10.0, mu=torch.as_tensor(mu0, dtype=torch.float64), sd=torch.as_tensor(sd0, dtype=torch.float64), scale=scale,
                seeds=seeds, recipe=args.recipe, export_sha=str(meta.get("x_sha256")))
    gold = dict(rows=ev, Z=Z, mu=mu0, sd=sd0)
    S0 = np.zeros((0, Z.shape[1]), np.float32)
    for fam, model in FAMS.items():
        blob[fam], P = [], []
        for s in seeds:
            data = dict(Xtr=XA[tr], ytr=y[tr], Xva=XA[:0], yva=y[:0], cva=cells[:0], Xte=XA[ev], yte=y[ev], cte=cells[ev],
                        side_tr=Z[tr], side_te=Z[ev], side_va=S0)
            r = T.fit_eval(model, data, recipe, s, device=dev, sfreq=float(meta["sfreq"]), log_every=30, return_state=True)
            assert r["side_state"] is not None
            blob[fam].append(dict(net={k: v.float() if v.is_floating_point() else v for k, v in r["state_last"].items()},
                                  head={k: v.float() for k, v in r["side_state"].items()}))
            P.append(r["prob_last"])
            print(f"[side_final] {fam} {model} seed {s}: train-window acc {float((r['prob_last'].argmax(1) == y).mean()):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        gold[f"{fam}_prob"] = np.stack(P)
    torch.save(blob, out / "side.pt")
    np.savez(out / "side_golden.npz", **gold)
    sha = hashlib.sha256((out / "side.pt").read_bytes()).hexdigest()
    (out / "side_meta.json").write_text(json.dumps(dict(sha256=sha, seeds=seeds, recipe=args.recipe, scale=scale, kind=args.kind,
                                                        export_sha=str(meta.get("x_sha256")), ch_names=list(meta["ch_names"]),
                                                        n_windows=int(len(idx)), seconds=round(time.time() - t0, 1)), indent=1))
    print(f"[side_final] wrote {out / 'side.pt'} sha256 {sha[:16]} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
