#!/usr/bin/env python
"""T2-D425 STEP 0 (pre-registered in docs/DECISIONS_T2.md; backlog 117 = the panel seed (a)'s untested piece): can a MODEL statistic tell a returning
user's stream from a new user's, where the fingerprint distance could not (T2-D124b AUC 0.51 / 0.56; T2-D317 0.47)?
Statistic (label-free, causal, no fitted constant): per window, which training user's per-user copy is the MOST CONFIDENT (max class probability, the
copies through the same tv8c views); a_t = the running fraction of windows 1..t on which that copy is the ROUTED one (the package's route: nearest
training-user centroid of the running-mean fingerprint, every window). Gate open iff a_t > 1/2 (chance ≈ 1 / K, K = 6 or 7 training users).
Uses NO class label: identity only (a stream is RETURNING iff its user is in that protocol's training users = seen1 test P04 / P10 sessions 2-3).
Out: <out>/streams.csv (one row per protocol x seed x stream) + a printed summary (stream-level AUC of a_end, a_20, a_60; window open rates).
    mem-guard python tracks/t2_bci/scripts/t2_d425_gate.py --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75tv8c \
        --runs-kit <dir> --runs-seen1 <dir> --seeds 192..203 --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FC  # noqa: E402


def own_top_running(Pu, k):
    """Pu (K, T, 3) copies' probabilities in stream order; k (T,) routed copy index (-1 = no route yet) -> (a (T,), hit (T,))."""
    top = Pu.max(-1).argmax(0)
    hit = (top == k) & (k >= 0)
    return np.cumsum(hit) / np.arange(1, len(k) + 1), hit


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    ap.add_argument("--fp", default=str(Path.home() / "workspace/outputs/neural_interface_challenge/t2_d124b/fp_proteus.npy"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--recipe", required=True, help="the pu files' recipe stem, e.g. r1_pulxx_vp2_e180_vr_rc75tv8c")
    ap.add_argument("--runs-kit", required=True)
    ap.add_argument("--runs-seen1", required=True)
    ap.add_argument("--seeds", required=True, help="a..b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(".."))
    X, idx, meta = B.load_export(a.export)
    Fp = np.load(a.fp)
    subj = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
    subj_raw = idx["subject"].to_numpy()
    rows = []
    for proto, runs in (("kit", a.runs_kit), ("seen1", a.runs_seen1)):
        (_, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = map(np.flatnonzero, (trm, vam, tem))
        proj, mu, users = FC.fit_space(Fp[tr], subj_raw[tr])
        train_users = {str(u).split("/")[-1] for u in users}
        for s in range(lo, hi + 1):
            pu = np.load(Path(runs) / f"{proto}__{a.model}__{a.align}__n00__{a.recipe}pu__s{s}.npz")
            assert list(pu["users"]) == list(users), (list(pu["users"]), users)
            for name, rr, ikey, ukey in (("test", te, "test_idx", "prob_users_test"), ("val", va, "val_idx", "prob_users_val")):
                assert np.array_equal(pu[ikey], rr), (proto, s, name)
                P = pu[ukey].astype(np.float64)
                pos = {int(r): j for j, r in enumerate(rr)}
                for g in B.stream_groups(idx, rr):
                    jj = np.array([pos[int(r)] for r in g])
                    k = FC.stream_route_every(proj(Fp[g]), mu, 1)
                    at, hit = own_top_running(P[:, jj], k)
                    u = subj[g[0]]
                    T = len(g)
                    rows.append(dict(proto=proto, seed=s, split=name, stream=str(idx["stream"].to_numpy()[g[0]]), user=u, returning=u in train_users, n=T,
                                     a_end=at[-1], a20=at[min(19, T - 1)], a60=at[min(59, T - 1)], open_t20=float((at[19:] > 0.5).mean()) if T > 20 else np.nan,
                                     routed_own=bool(k[-1] >= 0 and str(users[k[-1]]).split("/")[-1] == u)))
    d = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    d.to_csv(Path(a.out) / "streams.csv", index=False)
    y = d.returning.to_numpy().astype(int)
    st = dict(model=a.model, n_streams=len(d), n_returning=int(y.sum()), users_returning=sorted(d[d.returning].user.unique()),
              auc_end=auc(y, d.a_end), auc_20=auc(y, d.a20), auc_60=auc(y, d.a60),
              auc_end_per_seed_median=float(np.nanmedian([auc(g.returning.astype(int), g.a_end) for _, g in d.groupby("seed")])),
              open_t20_returning=float(d[d.returning].open_t20.mean()), open_t20_new=float(d[~d.returning].open_t20.mean()),
              a_end_returning_median=float(d[d.returning].a_end.median()), a_end_new_median=float(d[~d.returning].a_end.median()),
              routed_own_returning=float(d[d.returning].routed_own.mean()))
    json.dump(st, open(Path(a.out) / "summary.json", "w"), indent=1)
    print(json.dumps(st, indent=1))
    print(d.groupby(["proto", "split", "user", "returning"])[["a_end", "open_t20"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
