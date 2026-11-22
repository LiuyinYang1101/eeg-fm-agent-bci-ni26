#!/usr/bin/env python
"""T2-D535 (registered 2026-11-16 before any XGBoost / CatBoost number): the window-mean (DC) family's bag with ANOTHER tree learner, on D515 /
D516's exact inputs and schedule. scripts/local/t2_d515_read.py's head (features_dc.npz = the 82 window means, the K = 9 folds, causal_z) is
exec'd unchanged. The learner replaces LightGBM in its fit_predict: a first fit of 150 rounds on every feature -> the top 300 by the learner's
total gain (all 82 here; only the column order follows the gain) -> the refit with 300 rounds -> predict_proba on the fold's test rows.
  xgb: XGBoost hist, lossguide, max_leaves 15, max_depth 0, eta 0.05, min_child_weight 4, colsample_bytree 0.3, subsample 0.8, lambda 1
  cat: CatBoost depth 4, lr 0.05, l2_leaf_reg 1, rsm 0.3, Bernoulli subsample 0.8, MultiClass
Writes outputs/t2_d535/pred_s<seeds>_dc_<learner>.npz in D515's format (te_<f>, bag_<f>) plus seed<sd>_<f> per seed.
    mem-guard python scripts/local/t2_d535_learners.py --learner xgb|cat --seeds 1,2,3
"""
import argparse
import sys
from pathlib import Path

ap0 = argparse.ArgumentParser()
ap0.add_argument("--learner", required=True, choices=("xgb", "cat"))
ap0.add_argument("--seeds", default="1,2,3")
A0 = ap0.parse_args()
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "--features", "features_dc.npz", "--seeds", A0.seeds, "--no-blocks"]
src = (HERE / "t2_d515_read.py").read_text()
head = src.split("\ndef fit_predict(")[0]
assert "def causal_z(" in head and "PRM = dict(" in head
exec(compile(head, "t2_d515_read_head", "exec"))          # F, idx, U, Y, STREAM, ORD, B, causal_z, SEEDS, D, O, np

THR = 8


def make(n, seed):
    if A0.learner == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=n, tree_method="hist", grow_policy="lossguide", max_leaves=15, max_depth=0, learning_rate=0.05,
                                 min_child_weight=4.0, colsample_bytree=0.3, subsample=0.8, reg_lambda=1.0, objective="multi:softprob",
                                 n_jobs=THR, random_state=seed, verbosity=0)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=n, depth=4, learning_rate=0.05, l2_leaf_reg=1.0, rsm=0.3, bootstrap_type="Bernoulli", subsample=0.8,
                              loss_function="MultiClass", random_seed=seed, verbose=0, thread_count=THR, allow_writing_files=False)


def gain_of(m, n_cols):
    if A0.learner == "xgb":
        g = m.get_booster().get_score(importance_type="total_gain")
        return np.array([g.get(f"f{j}", 0.0) for j in range(n_cols)])
    return np.asarray(m.get_feature_importance(), np.float64)


def fit_predict(Z, tr, te, cols, seed):
    m1 = make(150, seed).fit(Z[tr][:, cols], Y[tr])
    top = cols[np.argsort(-gain_of(m1, len(cols)), kind="stable")[:300]]
    m2 = make(300, seed).fit(Z[tr][:, top], Y[tr])
    return m2.predict_proba(Z[te][:, top])


out = {}
for f in range(10):
    (_, trm, _, tem), = B.folds(idx, f"lc9_{f}")
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    Z = causal_z(F, tr)
    ps = []
    for sd in SEEDS:
        p = fit_predict(Z, tr, te, np.arange(F.shape[1]), sd)
        assert p.shape == (len(te), 3) and np.isfinite(p).all()
        ps.append(p)
        out[f"seed{sd}_{f}"] = p
    out[f"te_{f}"], out[f"bag_{f}"] = te, np.mean(ps, 0)
    held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
    print(f"[d535 {A0.learner}] fold {f} (held {held}) fitted, {len(SEEDS)} seeds", flush=True)
R = O / "t2_d535"
R.mkdir(parents=True, exist_ok=True)
fn = R / f"pred_s{'-'.join(map(str, SEEDS))}_dc_{A0.learner}.npz"
np.savez_compressed(fn, **out)
print(f"[d535 {A0.learner}] wrote {fn}", flush=True)
