#!/usr/bin/env python
"""Aggregate t2p_bench results: the sealed metric per (protocol, config), per seed, and seed-ensembled.

For a protocol, every fold's test windows are pooled and scored with the sealed metric (balanced accuracy per
subject x session x interface cell, mean over cells): loso = 14 sessions x 2 interfaces = 28 cells over 10 new
participants; xsess = the 2 multi-session participants' sessions 2-3 (8 cells); kit = the warm-up test (P04, P09).
A config is reported only when every fold of its protocol is present for that seed.

Usage: python t2p_report.py --export <dir> --runs <dir>[,<dir>...] [--which best|last] [--ref eegnet/none] [--csv out.csv]
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from ni26 import metrics  # noqa: E402

N_FOLDS = {"kit": 1}


def causal_prior_correction(prob, order_keys, alpha=1.0, n0=3.0):
    """Streaming output re-balancing (legal: label-free, current + earlier windows of the same session only).
    Within each stream, in chronological order: q_t = p_t / m_t^alpha, m_t = (n0 * uniform + sum_{i<=t} p_i) / (n0 + t).
    order_keys: (stream id, order) per row. Returns corrected probabilities (rows in the input order)."""
    out = np.empty_like(prob, dtype=np.float64)
    K = prob.shape[1]
    streams, order = order_keys
    for s_ in np.unique(streams):
        rows = np.flatnonzero(streams == s_)
        rows = rows[np.argsort(order[rows], kind="stable")]
        cum = np.cumsum(prob[rows].astype(np.float64), 0)
        m = (n0 / K + cum) / (n0 + np.arange(1, len(rows) + 1))[:, None]
        q = prob[rows] / m ** alpha
        out[rows] = q / q.sum(1, keepdims=True)
    return out


def causal_sinkhorn(prob, order_keys, iters=10, n0=3, lam=1.0, start=None):
    """T2-D285 / D286: causal Sinkhorn re-balancing (label-free, current + earlier windows of the same session only). Within each
    stream, in chronological order, row t of [n0 uniform pseudo-rows; p_1 .. p_t] after `iters` rounds of (columns scaled to equal
    class mass, rows normalised). iters = 1 equals causal_prior_correction(alpha 1, n0). Batch reference for the template's
    SinkhornCorrector (t2p_sinkhorn_selftest.py). T2-D304: lam < 1 = the forgetting variant, row j of the n-row matrix weighs lam^(n-1-j) in the
    column balance (the pseudo-rows are the oldest); lam 1.0 = the code below unchanged (bitwise). T2-D459: start = the class vector
    the n0 pseudo-rows hold (None = uniform, bitwise as before)."""
    out = np.empty_like(prob, dtype=np.float64)
    K = prob.shape[1]
    streams, order = order_keys
    for s_ in np.unique(streams):
        rows = np.flatnonzero(streams == s_)
        rows = rows[np.argsort(order[rows], kind="stable")]
        P = prob[rows].astype(np.float64)
        for t in range(len(rows)):
            M = np.vstack([np.full((int(n0), K), 1.0 / K) if start is None else np.tile(np.asarray(start, np.float64), (int(n0), 1)), P[:t + 1]])
            if lam == 1.0:
                for _ in range(int(iters)):
                    M = M * (M.shape[0] / K) / M.sum(0, keepdims=True)
                    M = M / M.sum(1, keepdims=True)
            else:                                         # T2-D304
                w = float(lam) ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
                for _ in range(int(iters)):
                    M = M * (w.sum() / K) / (M * w[:, None]).sum(0, keepdims=True)
                    M = M / M.sum(1, keepdims=True)
            out[rows[t]] = M[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="dir with index.csv (the export the runs used)")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--which", default="last", choices=("best", "last"))   # last = the decision mode (D88); best is val-selected
    ap.add_argument("--ref", default="eegnet|none", help="model|align of the paired reference")
    ap.add_argument("--csv", default="")
    ap.add_argument("--md", default="", help="write a markdown table with the leaderboard's three columns")
    ap.add_argument("--per-subject", action="store_true")
    ap.add_argument("--tta", default="", help="'prior[:alpha[:n0]]' = causal per-session output prior correction")
    args = ap.parse_args()

    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    tta = None
    if args.tta.startswith("prior"):
        parts = args.tta.split(":")
        tta = (float(parts[1]) if len(parts) > 1 else 1.0, float(parts[2]) if len(parts) > 2 else 3.0)
    n_folds = {"loso": idx["subject"].nunique(),
               "xsess": int((idx.groupby("subject")["session"].nunique() >= 2).sum()), "kit": 1}

    runs = defaultdict(lambda: defaultdict(dict))   # (protocol, config) -> seed -> fold -> (test_idx, prob)
    vals = defaultdict(lambda: defaultdict(dict))   # (protocol, config) -> seed -> fold -> inner-val cell-BA (plain)
    meta = {}
    for d in args.runs.split(","):
        for f in [f for f in glob.glob(str(Path(d) / "*.npz")) if not f.endswith(".tmp.npz")]:
            z = np.load(f)
            r = json.loads(str(z["res"]))
            cfg = f.split("/")[-1].split("__")
            ext = "+".join(e["name"].replace("export_", "") for e in r.get("extra", [])) or "-"
            config = "|".join([r["model"], r["align"], cfg[3], cfg[4], ext])     # model|align|n0|recipe|extra data
            runs[(r["protocol"], config)][r["seed"]][r["fold"]] = (z["test_idx"], z[f"prob_{args.which}"])
            # the inner-val participants' cell-BA at the evaluated epoch (kit: P05 + P08 = the start kit's val split)
            cv = r.get("curves") or []
            vv = (r.get("best_val") if args.which == "best" else (cv[-1].get("val_cell_bacc") if cv else None))
            if args.which == "last" and "prob_val_last" in z.files and len(z["prob_val_last"]):
                vi, pv = z["val_idx"], z["prob_val_last"]       # saved val probabilities: score them (incl. --tta)
                if tta is not None:
                    pv = causal_prior_correction(pv, (strm[vi], ordr[vi]), *tta)
                vv = metrics.cell_balanced_accuracy(y[vi], pv.argmax(1), cells[vi])[0]
            vals[(r["protocol"], config)][r["seed"]][r["fold"]] = vv if (vv is not None and vv >= 0) else np.nan
            meta[(r["protocol"], config)] = r
    rows, per_subject = [], {}
    for (proto, config), by_seed in sorted(runs.items()):
        if proto not in n_folds:          # kit2 / l2o = TSF cross-fitting inputs (t2p_tsf.py), not a report protocol
            continue
        seeds = sorted(s for s, fo in by_seed.items() if len(fo) == n_folds[proto])
        if not seeds:
            continue
        scores, pooled, accs, probs = [], [], [], None
        for s in seeds:
            ti = np.concatenate([v[0] for v in by_seed[s].values()])
            pr = np.concatenate([v[1] for v in by_seed[s].values()])
            o = np.argsort(ti)
            ti, pr = ti[o], pr[o]
            if tta is not None:
                pr = causal_prior_correction(pr, (strm[ti], ordr[ti]), *tta)
            scores.append(metrics.cell_balanced_accuracy(y[ti], pr.argmax(1), cells[ti])[0])
            pooled.append(metrics.pooled_balanced_accuracy(y[ti], pr.argmax(1)))
            accs.append(float(np.mean(y[ti] == pr.argmax(1))))
            probs = pr if probs is None else probs + pr
        ens, per = metrics.cell_balanced_accuracy(y[ti], probs.argmax(1), cells[ti])
        ps = defaultdict(list)
        for c, v in per.items():
            ps[c.split("|")[0].split("/")[-1]].append(v)
        per_subject[(proto, config)] = {k: float(np.mean(v)) for k, v in ps.items()}
        m, a, n0, rec, ext = config.split("|")
        # the three Codabench leaderboard columns (phase config `scoring`): balanced_accuracy_cell_mean (ranking),
        # balanced_accuracy (pooled), accuracy -- per-seed means, and the same three for the seed ensemble
        rows.append(dict(protocol=proto, model=m, align=a, n0=n0, recipe=rec, extra=ext, n_seeds=len(seeds),
                         cellBA=float(np.mean(scores)), sd=float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
                         pooledBA=float(np.mean(pooled)), acc=float(np.mean(accs)),
                         ens_cellBA=float(ens), ens_pooledBA=metrics.pooled_balanced_accuracy(y[ti], probs.argmax(1)),
                         ens_acc=float(np.mean(y[ti] == probs.argmax(1))),
                         seeds=" ".join(f"{v:.3f}" for v in scores), n_cells=len(per),
                         val_cellBA=float(np.nanmean([np.nanmean(list(vals[(proto, config)][s].values())) for s in seeds]))
                         if any(vals[(proto, config)][s] for s in seeds) else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no complete config")
        return
    rm, ra = args.ref.split("|")
    out = []
    for proto, g in df.groupby("protocol"):
        ref = g[(g["model"] == rm) & (g["align"] == ra) & (g["extra"] == "-")]
        g = g.copy()
        if len(ref):
            r0 = ref.iloc[0]
            g["d_ref"] = g["cellBA"] - r0["cellBA"]
            g["d_val"] = g["val_cellBA"] - r0["val_cellBA"]
            k0 = (proto, "|".join([rm, ra, r0["n0"], r0["recipe"], r0["extra"]]))
            g["users_up"] = [
                f"{sum(per_subject[(proto, '|'.join([m, a, n, rc, ex]))][s] > per_subject[k0][s] for s in per_subject[k0])}/{len(per_subject[k0])}"
                for m, a, n, rc, ex in zip(g["model"], g["align"], g["n0"], g["recipe"], g["extra"])]
        out.append(g.sort_values("cellBA", ascending=False))
    df = pd.concat(out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 500)
    print(df.drop(columns=["recipe"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.per_subject:
        for k, v in sorted(per_subject.items()):
            print(k, " ".join(f"{s}:{x:.3f}" for s, x in sorted(v.items())))
    if args.csv:
        df.to_csv(args.csv, index=False)
    if args.md:
        names = {"kit": "warm-up mirror = the Codabench leaderboard (test P04 + P09, 8 cells)",
                 "loso": "new-participant mirror (leave-one-participant-out, 10 users, 28 cells)",
                 "xsess": "later sessions of a seen participant (P04 / P10 sessions 2-3, 8 cells)"}
        L = [f"<!-- generated by t2p_report.py --md (which={args.which}, tta={args.tta or 'none'}); do not edit -->",
             "Columns = the Codabench leaderboard's: **cell-mean balanced accuracy** (ranking), pooled balanced accuracy,"
             " accuracy. Mean over seeds; `ens` = the seed-averaged ensemble.", ""]
        for proto in ("kit", "loso", "xsess"):
            g = df[df["protocol"] == proto]
            if g.empty:
                continue
            L += [f"### {names[proto]}", "",
                  "| model | recipe | align | extra data | seeds | cell-BA | pooled BA | acc | ens cell-BA | ens pooled BA | ens acc | Δ ref | users up | val cell-BA (P05+P08) | Δ val |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            fm0 = "fmE30b64le0.0001lh0.001wd0.05ll1ema0dp0"      # the D82 FM recipe; arms show only what they change
            for _, r in g.iterrows():
                rec = str(r["recipe"])
                rec = ("fm" + ("+" + rec[len(fm0):] if len(rec) > len(fm0) else "")) if rec.startswith(fm0) else rec
                L.append(f"| {r['model']} | {rec} | {r['align']} | {r['extra']} | {r['n_seeds']} | {r['cellBA']:.4f} ± {0 if r['sd'] != r['sd'] else r['sd']:.4f} | "
                         f"{r['pooledBA']:.4f} | {r['acc']:.4f} | {r['ens_cellBA']:.4f} | {r['ens_pooledBA']:.4f} | "
                         f"{r['ens_acc']:.4f} | {r.get('d_ref', float('nan')):+.4f} | {r.get('users_up', '')} | "
                         f"{r['val_cellBA']:.4f} | {r.get('d_val', float('nan')):+.4f} |")
            L.append("")
        Path(args.md).write_text("\n".join(L))


if __name__ == "__main__":
    main()
