#!/usr/bin/env python
"""T2-D516 / W16 build input (zero GPU): the window-mean family's FINAL models for a package. The 82 features (per-channel means of the delivered
window and of its CAR view) of EVERY public window, z-scored causally per stream with the all-window prior (n0 10, as t2_d515_read.causal_z); three
LightGBM multiclass models (seeds 1-3, D515's parameters, 300 rounds; all 82 features = the in-fold top 300 of 82) trained on every labelled public
window; their trees dumped as padded arrays (split feature, threshold, children, leaf values, default-left) that a template evaluates with numpy;
plus the z prior (mu, sd) and the family's start vector (T2-D459's rule: the mean posterior of the K = 9 folds' strangers, D516's bag).
Output: outputs/t2_d516/dcfam.npz + dcfam_meta.json; checks: the array evaluator == LightGBM predict_proba on every public window (<= 1e-9).
    mem-guard python scripts/local/t2_d516_final.py
"""
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import t2p_bench as B  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
OUT = O / "t2_d516"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
X, idx, meta = B.load_export(str(EXP))
STREAM, ORD, Y, U = idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy().astype(int), idx.subject.to_numpy()
PRM = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.3, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=8)
N0 = 10.0


def dc_features(Xw):
    m = np.asarray(Xw, np.float64).mean(-1)
    return np.hstack([m, m - m.mean(1, keepdims=True)])


def causal_z(F, mu0, sd0):
    Z = np.empty_like(F)
    for st in np.unique(STREAM):
        r = np.flatnonzero(STREAM == st)
        r = r[np.argsort(ORD[r], kind="stable")]
        x = np.where(np.isfinite(F[r]), F[r], mu0)
        n = N0 + np.arange(1, len(r) + 1)[:, None]
        m = (N0 * mu0 + np.cumsum(x, 0)) / n
        v = (N0 * (sd0 ** 2 + mu0 ** 2) + np.cumsum(x * x, 0)) / n - m ** 2
        Z[r] = (x - m) / np.sqrt(np.maximum(v, 1e-12))
    return Z


def dump(booster):
    """LightGBM model -> padded arrays over trees: feat, thr, left, right (child node index; < 0 = leaf ~idx), leaf values, default_left."""
    js = booster.dump_model()
    trees = js["tree_info"]
    nmax = max(t["num_leaves"] for t in trees)
    T = len(trees)
    feat = np.zeros((T, nmax), np.int32)
    thr = np.zeros((T, nmax), np.float64)
    left = np.zeros((T, nmax), np.int32)
    right = np.zeros((T, nmax), np.int32)
    dleft = np.zeros((T, nmax), bool)
    leaf = np.zeros((T, nmax), np.float64)
    for ti, t in enumerate(trees):
        st = t["tree_structure"]
        if "leaf_value" in st:                      # a single-leaf tree
            left[ti, 0] = right[ti, 0] = -1
            leaf[ti, 0] = st["leaf_value"]
            feat[ti, 0], thr[ti, 0] = 0, np.inf
            continue
        stack = [st]
        while stack:
            nd = stack.pop()
            i = nd["split_index"]
            assert nd["decision_type"] == "<=", nd["decision_type"]
            feat[ti, i], thr[ti, i], dleft[ti, i] = nd["split_feature"], nd["threshold"], nd["default_left"]
            for side, arr in (("left_child", left), ("right_child", right)):
                ch = nd[side]
                if "leaf_index" in ch:
                    arr[ti, i] = -(ch["leaf_index"] + 1)
                    leaf[ti, ch["leaf_index"]] = ch["leaf_value"]
                else:
                    arr[ti, i] = ch["split_index"]
                    stack.append(ch)
    return feat, thr, left, right, dleft, leaf


def eval_arrays(z, feat, thr, left, right, dleft, leaf, K=3):
    """raw scores (n, K) of the padded trees for rows z (n, F); tree t belongs to class t % K; NaN -> the default side."""
    n, T = len(z), len(feat)
    node = np.zeros((n, T), np.int64)
    done = np.zeros((n, T), bool)
    out = np.zeros((n, T))
    tt = np.arange(T)[None, :]
    for _ in range(64):
        f = feat[tt, node]
        x = np.take_along_axis(z, f.reshape(n, -1), 1).reshape(n, T)
        go = np.where(np.isnan(x), dleft[tt, node], x <= thr[tt, node])
        nxt = np.where(go, left[tt, node], right[tt, node])
        leafhit = (nxt < 0) & ~done
        out[leafhit] = leaf[np.broadcast_to(tt, (n, T))[leafhit], -(nxt[leafhit]) - 1]
        done |= nxt < 0
        node = np.where(nxt < 0, 0, nxt)
        if done.all():
            break
    assert done.all()
    return np.stack([out[:, k::K].sum(1) for k in range(K)], 1)


def softmax(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = dc_features(X)
    mu0, sd0 = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-6
    Z = causal_z(F, mu0, sd0)
    arrs, probs = {}, []
    for sd in (1, 2, 3):
        m = lgb.LGBMClassifier(n_estimators=300, seed=sd, **PRM).fit(Z, Y)
        a = dump(m.booster_)
        p_arr = softmax(eval_arrays(Z, *a))
        p_lgb = m.predict_proba(Z)
        d = float(np.abs(p_arr - p_lgb).max())
        print(f"[d516 final] seed {sd}: {len(a[0])} trees, <= {a[0].shape[1]} nodes; array evaluator vs LightGBM max |dp| {d:.2e}", flush=True)
        assert d <= 1e-9, d
        for nm, v in zip(("feat", "thr", "left", "right", "dleft", "leaf"), a):
            arrs[f"s{sd}_{nm}"] = v
        probs.append(p_arr)
    zp = np.load(O / "t2_d515/pred_s1-2-3_dm.npz")
    per = []
    for f in range(10):
        te, p = zp[f"te_{f}"], zp[f"bag_{f}"]
        held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
        per.append(p[U[te] == held].mean(0))
    start = np.mean(per, 0)
    start = start / start.sum()
    np.savez_compressed(OUT / "dcfam.npz", mu=mu0, sd=sd0, start=start, **arrs)
    json.dump(dict(features="per-channel window means: delivered (41) then CAR (41), the template's channel order", n0=N0, seeds=[1, 2, 3], rounds=300,
                   ch_names=list(meta["ch_names"]), start=[round(float(v), 4) for v in start], lgb_params=PRM, rule="T2-D516 / W16"),
              open(OUT / "dcfam_meta.json", "w"), indent=1)
    print(f"[d516 final] dcfam.npz written; start vector {np.round(start, 4).tolist()}; in-sample bag accuracy {np.mean(np.mean(probs, 0).argmax(1) == Y):.3f}", flush=True)


if __name__ == "__main__":
    main()
