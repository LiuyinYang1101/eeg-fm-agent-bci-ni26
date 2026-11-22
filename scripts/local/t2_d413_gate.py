#!/usr/bin/env python
"""T2-D413 STEP 0 = the gate before any member run (pre-registered in docs/DECISIONS_T2.md row T2-D413, 2026-11-09 10:08; this script
written before any D413 number). Per protocol (kit, seen1), on the generator's own_ / interp_ npz (tracks/t2_bci/scripts/t2p_meanflow.py):
  (a) real-vs-synthetic AUC <= 0.80: L2 logistic regression (C 1) on standardised log band power (41 ch x 5 bands 4-8 / 8-13 / 13-20 /
      20-30 / 30-45 Hz, Welch 1 s), 5-fold stratified CV (shuffle, seed 0), real training windows vs own-user synthetic 1 : 1;
  (b) no copies [AMENDED 2026-11-09 before the D413 generators ran: a 500-step smoke of this code showed the pre-registered Euclidean
      version marks LOW-ENERGY samples as copies (a near-zero window is closer to every real window than real windows are to each other),
      so it is replaced by its scale-invariant form]: share of own-user synthetic windows whose MAXIMUM Pearson correlation with any real
      training window (flattened, z-scored as the generator) exceeds the 99th percentile of the real -> real leave-self-out maximum
      correlations <= 0.05 (the Euclidean version is still reported);
  (c) per class c: || mean log band power syn - real || <= the median over training users of || user-u class-c mean - real class-c mean ||
      AND AIRM(syn class-c mean 8-30 Hz covariance, real class-c mean) <= the median over users of AIRM(C_{u,c}, real class-c mean)
      (covariances: 4th-order Butterworth 8-30 Hz, zero-phase, per-window demeaned, X X^T / L, class means arithmetic, each mean shrunk by
      1e-3 tr / C before the AIRM); all 3 classes; own-user decides, interpolated reported only;
  (d) TSTR: fbcnet_bboth, recipe r1, seed 0, trained on the n_train INTERPOLATED synthetic windows only, kit val cell BA (last epoch)
      >= 0.45; own-user TSTR and real TRTR (same recipe / seed) reported only.
BOTH protocols must pass ALL four, else STOP (D413 closed, no member run).
    gpu-submit ... python scripts/local/t2_d413_gate.py --export E --gen-dir outputs/t2_d413 [--seed 0] [--protocols kit,seen1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigvalsh
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))
import ni26.train as T  # noqa: E402
from ni26 import metrics  # noqa: E402
from t2p_bench import folds, load_export  # noqa: E402

FS, BANDS = 120.0, ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
SOS = butter(4, (8, 30), btype="bandpass", fs=FS, output="sos")


def logbp(X):
    f, P = welch(X, fs=FS, nperseg=120, axis=-1)
    return np.log(np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1)).reshape(len(X), -1)


def covs(X):
    Z = sosfiltfilt(SOS, X.astype(np.float64), axis=-1)
    Z = Z - Z.mean(-1, keepdims=True)
    return np.einsum("nct,ndt->ncd", Z, Z) / Z.shape[-1]


def shrink(C):
    return C + 1e-3 * np.trace(C) / len(C) * np.eye(len(C))


def airm(A, B):
    return float(np.sqrt((np.log(eigvalsh(shrink(B), shrink(A))) ** 2).sum()))


def tstr(Xs, ys, us, Xva, yva, cva, Xte, yte, cte):
    d = dict(Xtr=Xs.astype(np.float32), ytr=ys.astype(np.int64), utr=us.astype(np.int64), Xva=Xva, yva=yva, cva=cva, Xte=Xte, yte=yte, cte=cte)
    r = T.fit_eval("fbcnet_bboth", d, T.RECIPES["r1"], 0, device="cuda", sfreq=FS, log_every=1000)
    return float(metrics.cell_balanced_accuracy(yva, r["prob_val_last"].argmax(1), cva)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="T2-D413b: the generator stem tag (default g<seed> = T2-D413)")
    ap.add_argument("--protocols", default="kit,seen1")
    a = ap.parse_args()
    G = Path(a.gen_dir)
    X, idx, meta = load_export(a.export)
    y_all = idx["y"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    va = np.flatnonzero((idx["split"] == "val").to_numpy()); te = np.flatnonzero((idx["split"] == "test").to_numpy())
    Xva, Xte = np.asarray(X[va], dtype=np.float32), np.asarray(X[te], dtype=np.float32)
    out, ok_all = {}, True
    for proto in a.protocols.split(","):
        (_, trm, vam, _), = folds(idx, proto)
        assert np.array_equal(np.flatnonzero(vam), va), "the TSTR val = the kit val (P05 P08) for both protocols"
        tr = np.flatnonzero(trm)
        users = sorted(idx.iloc[tr]["subject"].unique())
        u = idx.iloc[tr]["subject"].map({s: k for k, s in enumerate(users)}).to_numpy()
        Xr, yr = np.asarray(X[tr], dtype=np.float32), y_all[tr]
        tg = a.tag or f"g{a.seed}"
        own, itp = np.load(G / f"own_{proto}_{tg}.npz"), np.load(G / f"interp_{proto}_{tg}.npz")
        for z in (own, itp):
            assert np.array_equal(z["train_idx"], tr) and list(z["users"]) == users, proto
        assert np.array_equal(own["y"], yr) and np.array_equal(own["u"], u)
        Xo, Xi, yi = own["X"], itp["X"], itp["y"]
        R = dict(n_train=int(len(tr)), users=users)
        # (a)
        Fr, Fo, Fi = logbp(Xr), logbp(Xo), logbp(Xi)
        assert np.isfinite(Fo).all() and np.isfinite(Fi).all(), "non-finite synthetic band power"
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        lab = np.r_[np.zeros(len(Fr)), np.ones(len(Fo))]
        R["auc_own"] = float(cross_val_score(clf, np.r_[Fr, Fo], lab, cv=cv, scoring="roc_auc").mean())
        R["auc_interp"] = float(cross_val_score(clf, np.r_[Fr, Fi], np.r_[np.zeros(len(Fr)), np.ones(len(Fi))], cv=cv, scoring="roc_auc").mean())
        R["a_pass"] = R["auc_own"] <= 0.80
        # (b)
        mu, sd = Xr.mean((0, 2), keepdims=True), Xr.std((0, 2), keepdims=True)
        Zr = torch.as_tensor(((Xr - mu) / sd).reshape(len(Xr), -1), dtype=torch.float64)
        Zo = torch.as_tensor(((Xo - mu) / sd).reshape(len(Xo), -1), dtype=torch.float64)
        Drr = torch.cdist(Zr, Zr); Drr.fill_diagonal_(float("inf"))
        nn_rr = Drr.min(1).values.numpy(); nn_or = torch.cdist(Zo, Zr).min(1).values.numpy()
        thr = float(np.percentile(nn_rr, 1))
        R.update(nn_rr_median=float(np.median(nn_rr)), nn_syn_median=float(np.median(nn_or)), copy_thr_euclid=thr,
                 copy_share_euclid=float((nn_or < thr).mean()))
        def unit(Z):
            Z = Z - Z.mean(1, keepdim=True)
            return Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Ur, Uo = unit(Zr), unit(Zo)
        Krr = Ur @ Ur.T; Krr.fill_diagonal_(-float("inf"))
        mc_rr, mc_or = Krr.max(1).values.numpy(), (Uo @ Ur.T).max(1).values.numpy()
        thr_c = float(np.percentile(mc_rr, 99))
        R.update(maxcorr_rr_median=float(np.median(mc_rr)), maxcorr_syn_median=float(np.median(mc_or)), copy_thr=thr_c,
                 copy_share=float((mc_or > thr_c).mean()))
        R["b_pass"] = R["copy_share"] <= 0.05
        # (c)
        Cr, Co, Ci = covs(Xr), covs(Xo), covs(Xi)
        cls = {}
        for c in range(3):
            k = yr == c
            mr, Cm = Fr[k].mean(0), Cr[k].mean(0)
            spread_bp = float(np.median([np.linalg.norm(Fr[k & (u == j)].mean(0) - mr) for j in range(len(users))]))
            spread_cv = float(np.median([airm(Cr[k & (u == j)].mean(0), Cm) for j in range(len(users))]))
            ko, ki = own["y"] == c, yi == c
            cls[c] = dict(bp_own=float(np.linalg.norm(Fo[ko].mean(0) - mr)), bp_interp=float(np.linalg.norm(Fi[ki].mean(0) - mr)), bp_spread=spread_bp,
                          cov_own=airm(Co[ko].mean(0), Cm), cov_interp=airm(Ci[ki].mean(0), Cm), cov_spread=spread_cv)
            cls[c]["pass"] = cls[c]["bp_own"] <= spread_bp and cls[c]["cov_own"] <= spread_cv
        R["class"] = cls
        R["c_pass"] = all(v["pass"] for v in cls.values())
        # (d) on the bench's align-none scale: X / the protocol's median training-window channel std (t2p_bench.aligned_all)
        s_ = float(np.median(np.asarray(X[tr[::7]]).std(-1)))
        R["align_none_scale"] = s_
        ev = (Xva / s_, y_all[va], cells[va], Xte / s_, y_all[te], cells[te])
        R["tstr_interp"] = tstr(Xi / s_, yi, itp["u"], *ev)
        R["tstr_own"] = tstr(Xo / s_, own["y"], own["u"], *ev)
        R["trtr"] = tstr(Xr / s_, yr, u, *ev)
        R["d_pass"] = R["tstr_interp"] >= 0.45
        R["pass"] = bool(R["a_pass"] and R["b_pass"] and R["c_pass"] and R["d_pass"])
        ok_all &= R["pass"]
        out[proto] = R
        print(f"== {proto}: (a) AUC own {R['auc_own']:.3f} (interp {R['auc_interp']:.3f}) {'ok' if R['a_pass'] else 'FAIL'} | "
              f"(b) copy share {R['copy_share']:.3f} (max corr syn median {R['maxcorr_syn_median']:.3f} vs real-real {R['maxcorr_rr_median']:.3f}, p99 "
              f"{R['copy_thr']:.3f}; Euclid share {R['copy_share_euclid']:.3f}) "
              f"{'ok' if R['b_pass'] else 'FAIL'} | (c) " + "; ".join(
                  f"c{c} bp {v['bp_own']:.2f}/{v['bp_spread']:.2f} cov {v['cov_own']:.2f}/{v['cov_spread']:.2f} (interp bp {v['bp_interp']:.2f} "
                  f"cov {v['cov_interp']:.2f})" for c, v in cls.items()) + f" {'ok' if R['c_pass'] else 'FAIL'} | (d) TSTR interp "
              f"{R['tstr_interp']:.3f} (own {R['tstr_own']:.3f}, TRTR {R['trtr']:.3f}) {'ok' if R['d_pass'] else 'FAIL'} -> "
              f"{'PASS' if R['pass'] else 'FAIL'}", flush=True)
    out["gate_pass"] = bool(ok_all)
    json.dump(out, open(G / f"gate_{a.tag or f'g{a.seed}'}.json", "w"), indent=1)
    print(f"== T2-D413 STEP 0 GATE (pre-registered): {'PASS -> Step 1' if ok_all else 'FAIL -> STOP (D413 closed, no member run)'}", flush=True)


if __name__ == "__main__":
    main()
