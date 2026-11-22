#!/usr/bin/env python
"""T2-D318 H-a: HEEGNet and its EEGNet control on the PROTEUS bench protocols, bench-format result files (t2p_pkgshape reads them).
  - input: the bench's causal 'ea' (n0 0) on every stream, as the ShallowNet member;
  - training: ni26.heegnet.fit (the reference demo config: domains = (subject, session) of the training windows, batches of
    5 domains x 10, Adam 1e-3 / L2 1e-4, <= 100 epochs, early stopping on an internal 20 % of the TRAINING windows; HEEGNet adds
    DSMDBN with the AdaMom schedule and 0.01 x the horospherical SW loss); the kit val is never read during training;
  - test / val: per stream in kit order; HEEGNet's DSMDBN statistics at window t = those of the stream's windows 1..t.
  python t2_d318_heeg.py --export E --out DIR [--models heeg,eegnetctl] [--protocols kit,seen1] [--seeds 0-11] [--list-outputs F]
  -> DIR/{proto}__{model}__ea__n00__heeg_ref__s{seed}.npz (test_idx, prob_last, val_idx, prob_val_last, prob_best, res)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))


def seeds_of(s):
    if "-" in s and "," not in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="heeg,eegnetctl")
    ap.add_argument("--protocols", default="kit,seen1")
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--list-outputs", default="")
    args = ap.parse_args()
    models, protos, seeds = args.models.split(","), args.protocols.split(","), seeds_of(args.seeds)
    assert set(models) <= {"heeg", "eegnetctl"}, models
    tag = lambda p, m, s: f"{p}__{m}__ea__n00__heeg_ref__s{s}"          # noqa: E731
    if args.list_outputs:
        Path(args.list_outputs).write_text("".join(f"{tag(p, m, s)}.npz\n" for p in protos for m in models for s in seeds))
        return
    import torch
    import t2p_bench as B
    from ni26 import heegnet as H
    from ni26 import metrics
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, idx, meta = B.load_export(args.export)
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    t0 = time.time()
    XA = B.aligned_all(X, idx, "ea", np.ones(len(idx), bool), 0.0, args.shrink, 1.0)   # n0 0: no training prior, mask unused
    print(f"[d318] causal EA on {len(idx)} windows ({time.time() - t0:.0f}s); device {dev}", flush=True)
    for proto in protos:
        (fname, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        doms = (idx["subject"] + "|" + idx["session"]).to_numpy()[tr]
        dnames = sorted(set(doms))
        dtr = np.array([dnames.index(d) for d in doms])
        s_te, s_va = B.stream_groups(idx, te), B.stream_groups(idx, va)
        for model in models:
            kind = "heeg" if model == "heeg" else "eegnet"
            for seed in seeds:
                fp = out / f"{tag(proto, model, seed)}.npz"
                if fp.exists():
                    continue
                t1 = time.time()
                print(f"[d318] {fp.name}: {len(tr)} training windows, {len(dnames)} domains {dnames}", flush=True)
                net, info = H.fit(kind, XA[tr], y[tr], dtr, 3, seed, device=dev, log=lambda s: print(s, flush=True))
                Pte = H.predict_streams(net, kind, XA, s_te, device=dev)
                Pva = H.predict_streams(net, kind, XA, s_va, device=dev)
                pte = np.stack([Pte[int(r)] for r in te]).astype(np.float32)
                pva = np.stack([Pva[int(r)] for r in va]).astype(np.float32)
                bt = metrics.cell_balanced_accuracy(y[te], pte.argmax(1), cells[te])[0]
                bv = metrics.cell_balanced_accuracy(y[va], pva.argmax(1), cells[va])[0]
                res = dict(protocol=proto, fold=fname, model=model, align="ea", n0=0.0, shrink=args.shrink, recipe="heeg_ref", seed=seed,
                           export=str(args.export), export_sha=meta.get("x_sha256"), n_train=len(tr), n_val=len(va), n_test=len(te),
                           domains=dnames, test_cell_bacc=bt, val_cell_bacc=bv, seconds=round(time.time() - t1, 1),
                           n_params=int(sum(p.numel() for p in net.parameters())),
                           **{k: v for k, v in info.items() if k != "curves"}, curves=info["curves"])
                np.savez_compressed(str(fp) + ".tmp.npz", test_idx=te, prob_last=pte, val_idx=va, prob_val_last=pva, prob_best=pte,
                                    res=json.dumps(res))
                os.replace(str(fp) + ".tmp.npz", fp)
                print(f"[d318] {fp.name}: test {bt:.4f} val {bv:.4f} (best epoch {info['best_epoch']} / {info['epochs_run']}, "
                      f"{time.time() - t1:.0f}s)", flush=True)
                del net
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
