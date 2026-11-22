#!/usr/bin/env python
"""T2-D515 read (written 2026-11-15 before any D515 number; CPU only). On T2-D498's K = 9 panel (lc9_0..9: train = nine users' session 1, test = the
held-out user + the returning P04 / P10 sessions 2-3):
  * causal per-stream standardisation of every feature (the stream's running mean / sd of windows <= t, shrunk toward the fold's TRAINING mean / sd
    with 10 pseudo-windows; training rows standardised the same way inside their own streams);
  * LightGBM multiclass: a first fit on every feature -> the top 300 by gain (inside the fold) -> the refit; test probabilities;
  * (a) level alone: each test stream decoded by the causal Sinkhorn (10 / 3 / 0.95, uniform start), cell BA;
  * (b) error correlation (phi) with T2-D498's stack (outputs/t2_d510/teacher: its decoded posterior, draws 750-752) on the strangers' windows;
  * (c) the GBM as a FAMILY in D498's stack under W13's per-family decode (its own cross-fitted start), share w in 0.05 / 0.10 / 0.20, the other
    shares x (1 - w); vs the stack; per draw; part rule: positive, >= 2 / 3 draws, no stranger < -0.020, returning >= -0.003;
  * (d) gain importance per feature family (the first fit), and each block alone (tsfel / catch22 / spec / asym / sync): level and phi.
    mem-guard python scripts/local/t2_d515_read.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import t2p_bench as B  # noqa: E402
from t2_cluster_ci import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
D = O / "t2_d515"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
_, idx, _ = B.load_export(str(EXP))
U, SES, STREAM, ORD, Y, CELL = (idx.subject.to_numpy(), idx.session.astype(str).to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(),
                                idx.y.to_numpy().astype(int), idx.cell.to_numpy())
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0", help="T2-D515b: LightGBM seeds (each read alone + their bagged mean)")
ap.add_argument("--shares", default="0.05,0.10,0.20")
ap.add_argument("--no-blocks", action="store_true")
ap.add_argument("--features", default="features.npz", help="T2-D515d: features_dm.npz = the de-meaned windows' features")
ap.add_argument("--topk", default="300", help="T2-D515c: comma list of in-fold top-K sizes (one first fit; each K refit); K != 300 -> variant names bag_k<K>")
ap.add_argument("--drop-fam", default="", help="T2-D515c: comma list of feature families dropped BEFORE the in-fold selection")
ap.add_argument("--drop-pat", default="", help="T2-D515c: '|'-joined name substrings dropped before the selection (e.g. Autocorrelation|MFCC|LPCC)")
AR = ap.parse_args()
SEEDS = [int(v) for v in AR.seeds.split(",")]
KS = [int(v) for v in AR.topk.split(",")]
SHARES = [float(v) for v in AR.shares.split(",")]
SFX = "" if SEEDS == [0] else "_s" + "-".join(map(str, SEEDS))
SFX += "_restricted" if (AR.drop_fam or AR.drop_pat) else ""
SFX += "_trunc" if "," in AR.topk else ""
# T2-D516 / D522: the suffix names the feature file (features_dm -> _dm as before; features_dc -> _dc; features_dc8 -> _dc8). Until 2026-11-15
# 21:5x EVERY non-default file wrote _dm: D516's DC run overwrote D515d's _dm predictions, so pred_s1-2-3_dm.npz holds the DC bag
# (identical copy: pred_s1-2-3_dc.npz, sha256 c24e2536...); the D516 / D517 / W17 readers load that file and are correct.
if AR.features != "features.npz":
    _stem = Path(AR.features).stem.replace("features_", "")
    SFX += "_dm" if _stem == "dm" else f"_{_stem}"
z = np.load(D / AR.features)
F, NAMES, FAM = z["F"].astype(np.float32), z["names"], z["family"]
if AR.drop_fam or AR.drop_pat:                          # T2-D515c: the shippable subset, restricted before any fit
    keep_ = ~np.isin(FAM, [v for v in AR.drop_fam.split(",") if v])
    for pat in [v for v in AR.drop_pat.split("|") if v]:
        keep_ &= np.array([pat not in str(n) for n in NAMES])
    F, NAMES, FAM = F[:, keep_], NAMES[keep_], FAM[keep_]
    print(f"[d515] restricted to {keep_.sum()} of {len(keep_)} features", flush=True)
BLOCK = np.array([("tsfel" if f.startswith("tsfel") else "catch22" if f.startswith("catch22") else f.split("_")[1] if f.split("_")[1] in ("asym",)
                   else "sync" if f.split("_")[1] in ("plv", "coh") else "spec") for f in FAM])
W = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}
NC6 = ("P01", "P03", "P05", "P07", "P09", "P10")
DRAWS = (750, 751, 752)
PRM = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.3, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=8, seed=0)


def causal_z(X, tr, n0=10.0):
    """per-stream causal z-scores of every row (training rows inside their own streams too), prior = the training rows' mean / sd."""
    mu0 = np.nanmean(X[tr], 0)
    sd0 = np.nanstd(X[tr], 0) + 1e-6
    Z = np.empty_like(X)
    for st in np.unique(STREAM):
        r = np.flatnonzero(STREAM == st)
        r = r[np.argsort(ORD[r], kind="stable")]
        x = np.where(np.isfinite(X[r]), X[r], mu0)
        n = n0 + np.arange(1, len(r) + 1)[:, None]
        m = (n0 * mu0 + np.cumsum(x, 0)) / n
        v = (n0 * (sd0 ** 2 + mu0 ** 2) + np.cumsum(x * x, 0)) / n - m ** 2
        Z[r] = ((x - m) / np.sqrt(np.maximum(v, 1e-12))).astype(np.float32)
    return Z


def cells(pred, rows):
    out = {}
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c
        ks = np.unique(Y[rows][m])
        if len(ks) >= 2:
            out[c] = float(np.mean([np.mean(pred[m][Y[rows][m] == q] == q) for q in ks]))
    return out


def fit_predict(Z, tr, te, cols, seed=0, ks=(300,)):
    prm = dict(PRM, seed=seed)
    m1 = lgb.LGBMClassifier(n_estimators=150, **prm).fit(Z[tr][:, cols], Y[tr])
    gain = m1.booster_.feature_importance("gain")
    out = {}
    for k_ in ks:
        top = cols[np.argsort(-gain)[:k_]]
        m2 = lgb.LGBMClassifier(n_estimators=300, **prm).fit(Z[tr][:, top], Y[tr])
        out[k_] = m2.predict_proba(Z[te][:, top])
    return (out[ks[0]] if len(ks) == 1 else out), dict(zip(cols, gain))


FOLD, GAIN, BLK = {}, np.zeros(F.shape[1]), {}
for f in range(10):
    (_, trm, _, tem), = B.folds(idx, f"lc9_{f}")
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    Z = causal_z(F, tr)
    ps = []
    for sd in SEEDS:
        p, g = fit_predict(Z, tr, te, np.arange(F.shape[1]), sd, KS)
        GAIN += np.array([g.get(j, 0.0) for j in range(F.shape[1])])
        if len(KS) > 1:                                 # T2-D515c: the truncation curve, bagged over the seeds
            for k_, pk in p.items():
                BLK.setdefault((f, f"bag_k{k_}"), []).append(pk)
            p = p[300] if 300 in p else p[KS[-1]]
        ps.append(p)
        if len(SEEDS) > 1:
            BLK[(f, f"seed{sd}")] = p
    if len(KS) > 1:
        for k_ in KS:
            BLK[(f, f"bag_k{k_}")] = np.mean(BLK[(f, f"bag_k{k_}")], 0)
    FOLD[f] = (te, np.mean(ps, 0))
    if not AR.no_blocks:
        for b in np.unique(BLOCK):
            BLK[(f, b)] = fit_predict(Z, tr, te, np.flatnonzero(BLOCK == b))[0]
    held = [u for u in set(U[te]) if (U[te] == u).sum() == (U == u).sum()][0]
    print(f"[d515] fold {f} (held {held}) fitted", flush=True)


def decode(te, p):
    p = p / p.sum(1, keepdims=True)
    return causal_sinkhorn(p, (STREAM[te], ORD[te]), 10, 3, 0.95)


TRU = {f: [u for u in set(U[FOLD[f][0]]) if (U[FOLD[f][0]] == u).sum() == (U == u).sum()][0] for f in range(10)}
res, lev = {}, {}
# (a) / (b) / (d-alone)
KEYS = sorted({k[1] for k in BLK})
for name, src in [("all", {f: FOLD[f][1] for f in range(10)})] + [(b, {f: BLK[(f, b)] for f in range(10)}) for b in KEYS]:
    rec, phis = [], []
    for f in range(10):
        te, p = FOLD[f][0], src[f]
        dg = decode(te, p).argmax(1)
        hm = U[te] == TRU[f]
        for c, v in cells(dg[hm], te[hm]).items():
            rec.append(dict(user=TRU[f], ba=v))
        for d in DRAWS:
            t = np.load(O / f"t2_d510/teacher/lc9_{f}__s{d + 40}.npz")
            pos = {int(r): k for k, r in enumerate(te)}
            sel = np.array([pos[int(r)] for r in t["rows"][t["held"]]])
            eg = (dg[sel] != Y[te][sel]).astype(float)
            es = (t["post"][t["held"]].argmax(1) != Y[t["rows"][t["held"]]]).astype(float)
            if eg.std() > 0 and es.std() > 0:
                phis.append(float(np.corrcoef(eg, es)[0, 1]))
    pu = pd.DataFrame(rec).groupby("user").ba.mean()
    lev[name] = pu
    res[f"level_{name}"] = dict(ten=round(float(pu.mean()), 4), nc6=round(float(pu[[u for u in NC6 if u in pu.index]].mean()), 4),
                                per_user={u: round(float(v), 3) for u, v in pu.items()}, phi_stack=round(float(np.mean(phis)), 3))
    print(f"D515 (a/b) {name:8s}: strangers {res[f'level_{name}']['ten']:.4f} | not-covered six {res[f'level_{name}']['nc6']:.4f} | phi with the stack "
          f"{res[f'level_{name}']['phi_stack']:.3f} | " + " ".join(f"{u} {v:.3f}" for u, v in pu.items()), flush=True)
# (c) the GBM family inside D498's stack
def stack_read(PRED, w):
    vec = {}
    for f in range(10):
        per = [PRED[g][U[FOLD[g][0]] == TRU[g]].mean(0) for g in range(10) if g != f]
        v = np.mean(per, 0)
        vec[f] = v / v.sum()
    recs = []
    for f in range(10):
        te, p = FOLD[f][0], PRED[f]
        pos = {int(r): k for k, r in enumerate(te)}
        for d in DRAWS:
            t = np.load(O / f"t2_d510/teacher/lc9_{f}__s{d + 40}.npz")
            rows = t["rows"]
            j = np.array([pos[int(r)] for r in rows])
            base = sum(W[k] * t[f"D_{k}"].astype(np.float64) for k in W)
            pg = p[j] / p[j].sum(1, keepdims=True)
            dg = causal_sinkhorn(pg, (STREAM[rows], ORD[rows]), 10, 3, 0.95, start=vec[f])
            new = (1 - w) * base + w * dg
            kind = np.where(t["held"], "stranger", "returning")
            for kd in ("stranger", "returning"):
                for u in np.unique(U[rows][kind == kd]):
                    m = (kind == kd) & (U[rows] == u)
                    a, b = cells(base.argmax(1)[m], rows[m]), cells(new.argmax(1)[m], rows[m])
                    for c in a:
                        recs.append(dict(w=w, draw=d, kind=kd, user=u, d=b[c] - a[c]))
    return pd.DataFrame(recs)


np.savez_compressed(D / f"pred{SFX}.npz", **{f"te_{f}": FOLD[f][0] for f in range(10)}, **{f"bag_{f}": FOLD[f][1] for f in range(10)},
                    **{f"{k[1]}_{k[0]}": v for k, v in BLK.items()})
VARIANTS = {"bag" if len(SEEDS) > 1 else "gbm": {f: FOLD[f][1] for f in range(10)}}
if len(SEEDS) > 1:
    VARIANTS.update({f"seed{sd}": {f: BLK[(f, f"seed{sd}")] for f in range(10)} for sd in SEEDS})
if len(KS) > 1:
    VARIANTS.update({f"bag_k{k_}": {f: BLK[(f, f"bag_k{k_}")] for f in range(10)} for k_ in KS})
for (vn, PRED), w in [(kv, w) for kv in VARIANTS.items() for w in SHARES]:
    R = stack_read(PRED, w)
    st, rt = R[R.kind == "stranger"], R[R.kind == "returning"]
    pu = st.groupby("user").d.mean()
    dr = [round(float(x), 4) for x in st.groupby(["draw", "user"]).d.mean().groupby("draw").mean()]
    ok = bool(pu.mean() > 0 and sum(x > 0 for x in dr) >= 2 and pu.min() >= -0.020 and rt.groupby("user").d.mean().mean() >= -0.003)
    res[f"stack_{vn}_w{w}"] = dict(ten=round(float(pu.mean()), 4), nc6=round(float(pu[[u for u in NC6 if u in pu.index]].mean()), 4), draws=dr,
                              min_user=round(float(pu.min()), 4), min_who=str(pu.idxmin()), returning=round(float(rt.groupby("user").d.mean().mean()), 4), part=ok)
    e = res[f"stack_{vn}_w{w}"]
    print(f"D515 (c) {vn} at w {w:.2f}: strangers {e['ten']:+.4f} draws {dr} | six {e['nc6']:+.4f} | min {e['min_user']:+.4f} ({e['min_who']}) | "
          f"returning {e['returning']:+.4f} -> " + ("a PART" if ok else "not a part"), flush=True)
# (d) gain by family
gf = pd.Series(GAIN).groupby(FAM).sum()
gf = (gf / gf.sum()).sort_values(ascending=False)
res["gain_share_by_family"] = {k: round(float(v), 4) for k, v in gf.items()}
top = np.argsort(-GAIN)[:25]
res["top25"] = [str(NAMES[j]) for j in top]
print("D515 (d) gain share by family: " + ", ".join(f"{k} {v:.3f}" for k, v in gf.items()), flush=True)
print("D515 (d) top features: " + ", ".join(str(NAMES[j]) for j in top[:15]), flush=True)
part = any(v["part"] for k, v in res.items() if k.startswith("stack_"))
stack_lev = {u: None for u in NC6}
res["verdict"] = ("(c) a PART -> a member worth a template" if part else
                  "(c) not a part -> see (d): a block with signal on the not-covered six beyond the stack opens the IFNet side-input test; else closed")
print("D515 VERDICT: " + res["verdict"], flush=True)
if len(SEEDS) > 1:                                      # T2-D515b: the registered replicate rule at w 0.10
    sv = [res[f"stack_seed{sd}_w0.1"]["ten"] for sd in SEEDS]
    rep = bool(np.mean(sv) > 0 and sum(v > 0 for v in sv) >= 2)
    res["replicate_w0.10"] = dict(per_seed=sv, mean=round(float(np.mean(sv)), 4), replicates=rep)
    print(f"D515b REPLICATE at w 0.10: per seed {sv} -> mean {np.mean(sv):+.4f} -> " + ("REPLICATES (guards: see the per-seed lines)" if rep else "does not replicate"), flush=True)
json.dump(res, open(D / f"read{SFX}.json", "w"), indent=1)
