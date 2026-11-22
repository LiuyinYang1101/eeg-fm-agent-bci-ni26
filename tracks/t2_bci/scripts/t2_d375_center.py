#!/usr/bin/env python
"""T2-D375 step 1: per-user feature centring, 12 seeds per family. Per seed, the pooled CNN trained exactly as t2p_bench's kit task
(t2_d369_userprobe.features), then on final_layer's input (standardised + PCA <= 64, fit on the kit-TRAIN users) a linear head on
  (i) raw features, vs (ii) per-user-centred training features applied with a CAUSAL per-stream running mean (prior = the training
  mean, weight N0 = 10; the stream = subject | session in index order of the kit 'order' column),
scored by cell-BA on the held-out (kit val + test) users. Writes per-seed rows + a summary.
  python t2_d375_center.py --export E --out DIR [--models fbcnet_bboth,shallow] [--device cuda]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2_d369_userprobe as P  # noqa: E402
import t2p_bench as B  # noqa: E402
from ni26 import metrics  # noqa: E402

N0 = 10
SEEDS = {"fbcnet_bboth": range(192, 204), "shallow": range(156, 168)}


def evaluate(F, y, subj, tr, ho, cells, stream, order):
    sc = StandardScaler().fit(F[tr])
    pca = PCA(min(64, F.shape[1], len(tr) - 1), random_state=0).fit(sc.transform(F[tr]))
    Z = pca.transform(sc.transform(F))
    Z = Z / Z[tr].std(0, keepdims=True)
    mu0 = Z[tr].mean(0)
    Zc = Z.copy()
    for u in np.unique(subj[tr]):
        k = subj == u
        Zc[k] = Z[k] - Z[k].mean(0)
    Zt = np.empty_like(Z)
    for s in np.unique(stream):
        r = np.flatnonzero(stream == s)
        r = r[np.argsort(order[r], kind="stable")]
        Zt[r] = Z[r] - (np.cumsum(Z[r], 0) + N0 * mu0) / (np.arange(1, len(r) + 1)[:, None] + N0)
    cb = lambda p: float(metrics.cell_balanced_accuracy(y[ho], p, cells[ho])[0])  # noqa: E731
    raw = cb(LogisticRegression(max_iter=3000).fit(Z[tr], y[tr]).predict(Z[ho]))
    cen = cb(LogisticRegression(max_iter=3000).fit(Zc[tr], y[tr]).predict(Zt[ho]))
    return raw, cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="fbcnet_bboth,shallow")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, idx, meta = B.load_export(a.export)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    stream, order = idx["stream"].to_numpy(), idx["order"].to_numpy()
    rows = []
    for m in a.models.split(","):
        align, rname, _ = P.SHIPPED[m]
        for s in SEEDS[m]:
            P.SHIPPED[m] = (align, rname, s)
            F, y, subj, tr, ho, _ = P.features(m, X, idx, meta, a.device, lambda t: None)
            raw, cen = evaluate(F, y, subj, tr, ho, cells, stream, order)
            rows.append(dict(model=m, seed=s, raw=raw, centred=cen, d=cen - raw))
            print(f"[d375] {m} seed {s}: raw {raw:.4f} centred {cen:.4f} d {cen - raw:+.4f}", flush=True)
    summ = {}
    for m in a.models.split(","):
        d = np.array([r["d"] for r in rows if r["model"] == m])
        summ[m] = dict(mean_d=float(d.mean()), sd_d=float(d.std(ddof=1)), up=int((d > 0).sum()), n=int(len(d)))
    (out / "d375_step1.json").write_text(json.dumps(dict(rows=rows, summary=summ, n0=N0), indent=1))
    print("[d375] summary " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
