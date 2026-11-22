#!/usr/bin/env python
"""T2-D397 (backlog 93 step 0): the ORACLE ceiling of a per-user ST-L copy for RETURNING users, on the seen1 protocol.
Per seed: the pooled ST-L trained exactly as the shipped line (aligned 'sz', calib gain, the seen1 CPT checkpoint, mixup 0.4, rc 0.75,
llrd 0.75, 30 epochs, t2p_bench's data contract); then for each returning test user (P04, P10: session 1 in training, sessions 2-3 =
the seen1 test) a copy fine-tuned FROM the pooled state on that user's session-1 windows, at two a-priori settings (light / moderate);
each scored on the user's own test cells under causal SK (10 it, n0 3, λ 0.95): pooled vs copy vs the 0.5 blend (oracle own-copy routing).
  python t2_r93_stl_copy.py --export E --ckpt CPT --out OUT.json [--seeds 120,121,122,123] [--device cuda] [--epochs 0 = shipped]
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
import t2p_fm  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

SETTINGS = {"light": dict(epochs=10, lr_enc=1e-5, lr_head=1e-4), "moderate": dict(epochs=20, lr_enc=3e-5, lr_head=3e-4)}
SHIP = dict(epochs=30, bs=64, lr_enc=1e-4, lr_head=1e-3, wd=0.05, llrd=0.75, ls=0.1, mixup=0.4, calib="gain", rc=0.75)


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def cells_ba(P, rows, idx):
    pred = causal_sinkhorn(P, (idx["stream"].to_numpy()[rows], idx["order"].to_numpy()[rows]), 10, 3, 0.95).argmax(1)
    y, cells = idx["y"].to_numpy()[rows].astype(int), idx["cell"].to_numpy()[rows]
    return {c: float(bal(y[cells == c], pred[cells == c])) for c in np.unique(cells)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True, help="the seen1 CPT checkpoint (the shipped line's: d111-with)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="120,121,122,123")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=0, help="smoke only: overrides every epoch count (0 = as pre-registered)")
    a = ap.parse_args()
    X, idx, meta = B.load_export(a.export)
    names = list(meta["ch_names"])
    (_, trm, vam, tem), = B.folds(idx, "seen1")
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    XA = B.aligned_all(X, idx, "sz", trm, 0.0, 0.05, scale)
    y = idx["y"].to_numpy().astype(int)
    if "cell" not in idx:                                 # the rescreen's cell = subject | session | interface (as t2_d375_member)
        idx = idx.assign(cell=(idx["subject"].astype(str) + "|" + idx["session"].astype(str) + "|" + idx["interface"].astype(str)))
    cells = idx["cell"].to_numpy()
    subj = idx["subject"].astype(str).str[-3:].to_numpy()
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    users = sorted(set(subj[tr]))
    utr = np.array([users.index(u) for u in subj[tr]])
    log = lambda m: print(f"[r93] {m}", flush=True)  # noqa: E731
    out = dict(settings=SETTINGS, ship=SHIP, seeds=[], ckpt=a.ckpt)
    for seed in [int(s) for s in a.seeds.split(",")]:
        ship = dict(SHIP, epochs=a.epochs or SHIP["epochs"])
        data = dict(Xtr=XA[tr], ytr=y[tr], Xva=XA[:0], yva=y[:0], cva=cells[:0], Xte=XA[te], yte=y[te], cte=cells[te], ext=[], utr=utr,
                    str=idx.iloc[tr]["session"].astype(str).to_numpy())
        r = t2p_fm.fit_eval_fm("st_large", data, seed, names, fm_ckpt=a.ckpt, device=a.device, return_state=True, log=log, **ship)
        p_pool = dict(zip(te, r["prob_last"]))
        rec = dict(seed=seed, users={})
        for u in ("P04", "P10"):
            tr_u, te_u = tr[subj[tr] == u], te[subj[te] == u]
            assert len(tr_u) and len(te_u), (u, len(tr_u), len(te_u))
            P0 = np.stack([p_pool[k] for k in te_u])
            res = {"pooled": cells_ba(P0, te_u, idx)}
            for nm, st in SETTINGS.items():
                st = dict(st, epochs=a.epochs or st["epochs"])
                du = dict(Xtr=XA[tr_u], ytr=y[tr_u], Xva=XA[:0], yva=y[:0], cva=cells[:0], Xte=XA[te_u], yte=y[te_u], cte=cells[te_u], ext=[],
                          utr=np.zeros(len(tr_u), int), str=idx.iloc[tr_u]["session"].astype(str).to_numpy())
                rc_ = t2p_fm.fit_eval_fm("st_large", du, seed, names, fm_ckpt=a.ckpt, device=a.device, init_state=r["state_last"], log=log,
                                         bs=64, wd=0.05, llrd=0.75, ls=0.1, mixup=0.0, calib="gain", rc=0.0, **st)
                Pc = rc_["prob_last"]
                res[f"copy_{nm}"] = cells_ba(Pc, te_u, idx)
                res[f"blend_{nm}"] = cells_ba(0.5 * P0 + 0.5 * Pc, te_u, idx)
            rec["users"][u] = res
            log(f"seed {seed} {u}: " + " · ".join(f"{k} {np.mean(list(v.values())):.4f}" for k, v in res.items()))
        out["seeds"].append(rec)
        Path(a.out).write_text(json.dumps(out, indent=1))
    arms = [k for k in out["seeds"][0]["users"]["P04"]]
    summ = {}
    for k in arms:
        per_seed = [np.mean([np.mean(list(s["users"][u][k].values())) for u in ("P04", "P10")]) for s in out["seeds"]]
        summ[k] = dict(mean=float(np.mean(per_seed)), per_seed=[round(float(v), 4) for v in per_seed])
    base = summ["pooled"]["per_seed"]
    for k in arms:
        d = np.array(summ[k]["per_seed"]) - np.array(base)
        summ[k].update(delta=float(d.mean()), up=int((d > 0).sum()))
    out["summary"] = summ
    Path(a.out).write_text(json.dumps(out, indent=1))
    log("summary (returning cells, mean over P04 / P10 cells): " + " · ".join(f"{k} {v['mean']:.4f} ({v['delta']:+.4f}, {v['up']}/{len(base)})" for k, v in summ.items()))


if __name__ == "__main__":
    main()
