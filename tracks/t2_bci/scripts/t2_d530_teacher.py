#!/usr/bin/env python
"""T2-D530 (registered 2026-11-16 before any number): NESTED out-of-fold cross-user soft targets on the K = 9 panel.
For lc9 fold f, the bench's 'lc9_<f>oof' protocol trains, per training user j, M7's two pooled recipes (FBCNet r1_pulxx_vp2_e180_vr_rc75 /
none, ShallowNet r1_pulx_vp2_e180_vr / ea; seed 0) on fold f's training rows minus j's, predicting j's training windows; so no teacher saw
fold f's stranger or the user it labels. This assembles them exactly as T2-D490's teacher (t2_d490_teacher.py): t2p_bench.oof_pcnn
(FBCNet 4/7 + ShallowNet 3/7), rows normalised, then Sinkhorn-balanced per training stream to its TRUE class counts.
Writes <out>/kdteacher__lc9_<f>__s0.npz (train_idx, p_train) = the --kd-teacher contract ('{fold}' form) + teacher.json.
    mem-guard python tracks/t2_bci/scripts/t2_d530_teacher.py --oof <runs dir> --out ~/workspace/outputs/neural_interface_challenge/t2_d530
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
from t2_d490_teacher import balance  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="the runs dir holding the lc9_<f>oof_<j>__... result files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9", help="lc9 folds to assemble (smoke: 0)")
    a = ap.parse_args()
    idx = pd.read_csv(EXP / "index.csv")
    assert (idx["i"].to_numpy() == np.arange(len(idx))).all()
    y = idx["y"].to_numpy().astype(int)
    sub_all = idx["subject"].astype(str).str[-3:].to_numpy()
    out, rep = Path(a.out), {}
    out.mkdir(parents=True, exist_ok=True)
    for f in [int(x) for x in a.folds.split(",")]:
        proto = f"lc9_{f}"
        ((_, trn, _, tem),) = B.folds(idx, proto)
        rows = np.flatnonzero(trn)
        stranger = sorted(set(sub_all) - set(sub_all[rows]))
        assert len(stranger) == 1
        P = B.oof_pcnn(a.oof, proto, y, rows, full=True)           # every (model, seed) covers each training row exactly once
        assert P.shape == (len(rows), 3) and np.isfinite(P).all()
        P = P / P.sum(1, keepdims=True)
        Q = np.empty_like(P)
        st = idx["stream"].to_numpy()[rows]
        for s in np.unique(st):
            m = st == s
            Q[m] = balance(P[m], y[rows][m])
        yr, sub = y[rows], sub_all[rows]
        r = dict(n=len(rows), stranger=stranger[0], acc_raw=float((P.argmax(1) == yr).mean()), acc_bal=float((Q.argmax(1) == yr).mean()),
                 maxp_raw=float(P.max(1).mean()), maxp_bal=float(Q.max(1).mean()), py_bal=float(Q[np.arange(len(yr)), yr].mean()),
                 per_user_bal={u: round(float((Q.argmax(1) == yr)[sub == u].mean()), 3) for u in np.unique(sub)})
        rep[proto] = r
        np.savez_compressed(out / f"kdteacher__{proto}__s0.npz", train_idx=rows, p_train=Q.astype(np.float32))
        print(f"[d530] {proto}: {json.dumps(r)}", flush=True)
    json.dump(rep, open(out / "teacher.json", "w"), indent=1)


if __name__ == "__main__":
    main()
