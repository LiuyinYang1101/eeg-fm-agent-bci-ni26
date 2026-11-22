#!/usr/bin/env python
"""T2-D364 step 0 (outlier bet B5; zero GPU): the LABELLED ceiling of per-stream family re-weighting, replayed on the stored
per-window member probabilities of a t2p_pkgshape spec.

Per draw, protocol and split: family mixes F_f = sum of the arm's member weight x probs within family f (FAM as t2_d362_ccc).
For every multiplier vector m in {0.5, 1, 2}^F: mix_m = sum_f m_f F_f / sum_f m_f W_f (W_f = the family's weight total),
decoded by the shipped SKλ (causal Sinkhorn, 10 its, n0 3, λ 0.95) over the whole stream. Per stream: m* = argmax of the cell-mean
BA on the stream's FIRST half (by order; ties -> the base m = 1, then the lowest index); scored on the SECOND half against the base
(m = 1) on the same windows. A cross-fit with labels = more than any label-free rule can know.
Groups as t2_d320_rescreen: typical (non-P04), P04, P10, newT; per-user Δ + a user bootstrap over the typical users.
  python t2_d364_ceiling.py --export E --spec SPEC --arm nm35 --out DIR [--jobs 6]
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_d362_ccc import FAM, SK, bal  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

GRID = (0.5, 1.0, 2.0)
G = {}


def init(export):
    d = pd.read_csv(f"{export}/index.csv")
    G.update(strm=d.stream.to_numpy(), ordr=d.order.to_numpy(), y=d.y.to_numpy(), cells=d.cell.to_numpy(),
             subj=d.subject.str[-3:].to_numpy())


def task(args):
    spec, arm, proto, i = args
    J = json.load(open(spec))
    wts = J["arms"][arm]
    out = []
    for split in ("test", "val"):
        parts, ix = {}, None
        for mname in wts:
            M = J["members"][mname]
            z = np.load(M[proto].format(s=M.get("offset", 0) + i))
            ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
            assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all()), "index mismatch"
            assert M.get("temp", 1) == 1, "tempered members not supported"
            ix, parts[mname] = ix_, p.astype(np.float64)
        fams = [f for f, names in FAM.items() if any(m in names for m in parts)]
        assert sum(len([m for m in parts if m in FAM[f]]) for f in fams) == len(parts), f"member outside FAM: {sorted(parts)}"
        F = {f: sum(wts[m] * parts[m] for m in parts if m in FAM[f]) for f in fams}
        W = {f: sum(wts[m] for m in parts if m in FAM[f]) for f in fams}
        st, od, y, cl = G["strm"][ix], G["ordr"][ix], G["y"][ix], G["cells"][ix]
        ok = (st, od)
        rank = np.empty(len(ix))
        for s_ in np.unique(st):
            r_ = np.flatnonzero(st == s_)
            rank[r_[np.argsort(od[r_], kind="stable")]] = np.arange(len(r_)) / len(r_)
        first = rank < 0.5
        combos = list(itertools.product(GRID, repeat=len(fams)))
        base_k = combos.index(tuple([1.0] * len(fams)))
        preds = []
        for c in combos:
            mix = sum(m * F[f] for m, f in zip(c, fams)) / sum(m * W[f] for m, f in zip(c, fams))
            preds.append(causal_sinkhorn(mix, ok, SK["iters"], SK["n0"], SK["lam"]).argmax(1))
        preds = np.stack(preds)
        for s_ in np.unique(st):
            ms = st == s_
            a = ms & first
            sc = []
            for k in range(len(combos)):
                sc.append(np.mean([bal(y[a & (cl == c_)], preds[k][a & (cl == c_)]) for c_ in np.unique(cl[a])]))
            sc = np.array(sc)
            best = np.flatnonzero(sc >= sc.max() - 1e-12)
            k_ = base_k if base_k in best else int(best[0])
            b = ms & ~first
            for c_ in np.unique(cl[b]):
                mk = b & (cl == c_)
                out.append(dict(proto=proto, split=split, i=i, cell=c_, subj=G["subj"][ix][mk][0], n=int(mk.sum()),
                                ba_base=bal(y[mk], preds[base_k][mk]), ba_orc=bal(y[mk], preds[k_][mk]),
                                m="/".join(f"{f}{m:g}" for f, m in zip(fams, combos[k_])), is_base=bool(k_ == base_k)))
    return out


def summarise(R, n_boot=2000):
    R = R.copy()
    R["d"] = R.ba_orc - R.ba_base
    R["grp"] = np.select([R.subj == "P04", R.subj == "P10"], ["P04", "P10"], "newT")
    rows = []
    for split, Rs in [("all (primary: test + val pooled, as t2_d320 / t2_d362)", R)] + list(R.groupby("split")):
        r = dict(split=split)
        for name, mask in (("typical", Rs.grp != "P04"), ("P04", Rs.grp == "P04"), ("P10", Rs.grp == "P10"), ("newT", Rs.grp == "newT")):
            dd = Rs[mask].groupby(["proto", "i"]).d.mean()
            r[name], r[name + "_up"] = dd.mean(), f"{int((dd > 0).sum())}/{len(dd)}"
        du = Rs[Rs.grp != "P04"].groupby("subj").d.mean()
        rng = np.random.default_rng(0)
        bs = [du.to_numpy()[rng.integers(0, len(du), len(du))].mean() for _ in range(n_boot)]
        r["user_ci90"] = f"[{np.quantile(bs, 0.05):+.4f}, {np.quantile(bs, 0.95):+.4f}]"
        r["users_up"] = f"{int((du > 0).sum())}/{len(du)}"
        r["base_frac"] = float(Rs.is_base.mean())          # the share of (stream, cell) picks that stayed at m = 1
        rows.append(r)
    return pd.DataFrame(rows), pd.concat([R[R.grp != "P04"].assign(split="all")] + [R[R.grp != "P04"]]).groupby(["split", "subj"]).d.mean().unstack("subj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--arm", default="nm35")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n", type=int, default=0, help="draws (0 = the spec's n)")
    a = ap.parse_args()
    J = json.load(open(a.spec))
    n = a.n or int(J.get("n", 12))
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.spec, a.arm, p, i) for p in ("kit", "seen1") for i in range(n)]
    with Pool(a.jobs, initializer=init, initargs=(a.export,)) as P:
        R = pd.DataFrame([r for rr in P.imap_unordered(task, jobs) for r in rr])
    R.to_csv(f"{a.out}/cells.csv", index=False)
    S, U = summarise(R)
    S.to_csv(f"{a.out}/summary.csv", index=False)
    json.dump(dict(spec=a.spec, arm=a.arm, n=n, grid=GRID, sk=SK, fam=FAM), open(f"{a.out}/config.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(f"== T2-D364 step 0: cross-fit (first half -> second half) family re-weighting vs {a.arm} + SKλ, {n} draws")
    print(S.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("== per typical user (Δ)")
    print(U.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("== chosen multipliers (test, top 6)")
    print(R[R.split == "test"].m.value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
