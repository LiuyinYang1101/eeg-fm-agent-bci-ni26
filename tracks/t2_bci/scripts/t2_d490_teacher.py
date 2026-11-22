#!/usr/bin/env python
"""T2-D490 (seat G10 item 4; registered 2026-11-14 before any number): OUT-OF-FOLD CROSS-USER soft targets for the CNN families.
Teacher = T2-D253's leave-one-user-out CNN mix ON DISK (sofia 1535295: per training user j, the protocol's pooled FBCNet
r1_pulxx_vp2_e180_vr_rc75 / none and ShallowNet r1_pulx_vp2_e180_vr / ea = M7's two pooled recipes, trained on the OTHER training users,
predict j's training windows; seeds 0-2; FBCNet 4/7 + ShallowNet 3/7 by t2p_bench.oof_pcnn), so no target comes from a model that saw its user.
Balanced per training stream (user x session) before use: a full Sinkhorn makes each stream's column sums equal its TRUE class counts and
each row sum 1 (the decoder deletes the stream's class offset at test; the teacher's offset under a cross-user model is not knowledge).
Writes <out>/kdteacher__{kit,seen1}__s0.npz (train_idx, p_train) = the --kd-teacher contract of t2p_bench (backlog 69 plumbing).
    mem-guard python tracks/t2_bci/scripts/t2_d490_teacher.py --out ~/workspace/outputs/neural_interface_challenge/t2_d490
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OOF = O / "sofia_s1/ni26-t2p-d253-oof-1535295/runs"


def balance(P, y, it=200):
    """rows sum 1, column sums = the stream's true class counts (classes absent from the stream get 0 mass)."""
    tgt = np.bincount(y, minlength=P.shape[1]).astype(float)
    Q = P.astype(float).clip(1e-6) * (tgt > 0)
    for _ in range(it):
        Q *= np.where(tgt > 0, tgt / Q.sum(0).clip(1e-12), 0.0)
        Q /= Q.sum(1, keepdims=True)
    assert np.abs(Q.sum(1) - 1).max() < 1e-9 and np.abs(Q.sum(0) - tgt).max() < 1e-3 * len(y), (Q.sum(0), tgt)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = pd.read_csv(EXP / "index.csv")
    assert (idx["i"].to_numpy() == np.arange(len(idx))).all()
    y = idx["y"].to_numpy().astype(int)
    out, rep = Path(a.out), {}
    out.mkdir(parents=True, exist_ok=True)
    for proto in ("kit", "seen1"):
        f = B.folds(idx, proto)
        assert len(f) == 1, proto
        rows = np.flatnonzero(f[0][1])
        P = B.oof_pcnn(OOF, proto, y, rows, full=True)
        assert P.shape == (len(rows), 3) and np.isfinite(P).all()
        P = P / P.sum(1, keepdims=True)
        Q = np.empty_like(P)
        st = idx["stream"].to_numpy()[rows]
        for s in np.unique(st):
            m = st == s
            Q[m] = balance(P[m], y[rows][m])
        yr = y[rows]
        sub = idx["subject"].astype(str).str[-3:].to_numpy()[rows]
        r = dict(n=len(rows), acc_raw=float((P.argmax(1) == yr).mean()), acc_bal=float((Q.argmax(1) == yr).mean()),
                 maxp_raw=float(P.max(1).mean()), maxp_bal=float(Q.max(1).mean()), py_raw=float(P[np.arange(len(yr)), yr].mean()),
                 py_bal=float(Q[np.arange(len(yr)), yr].mean()),
                 per_user_bal={u: round(float((Q.argmax(1) == yr)[sub == u].mean()), 3) for u in np.unique(sub)})
        rep[proto] = r
        np.savez_compressed(out / f"kdteacher__{proto}__s0.npz", train_idx=rows, p_train=Q.astype(np.float32))
        print(f"[d490] {proto}: {json.dumps(r)}", flush=True)
    json.dump(rep, open(out / "teacher.json", "w"), indent=1)


if __name__ == "__main__":
    main()
