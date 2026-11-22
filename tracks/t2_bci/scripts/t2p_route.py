#!/usr/bin/env python
"""Track 2 T2-D111 (iii): route a stream's windows to a per-user fine-tuned model when a causal user router is confident,
else keep the shared model; check the pre-registered router gate; write the routed probabilities as bench-style result
files that t2p_paired.py scores like any arm (CPU only).

  python t2p_route.py --export E --runs R --protocol seen1 --pu <A0 stem>pu --moe <moe stem> --seeds 0-11 --out DIR

Routers (posteriors over the fold's TRAINING users; running mean over windows 1..t of the stream, routed iff max >= tau):
  R1  band5_log shrinkage-LDA: log-variance per channel in 4-8 / 8-13 / 13-20 / 20-30 / 30-45 Hz (window-local
      zero-phase band-pass), fit on the fold's training windows' user ids (seed-independent)
  R2  the soft user-expert head's gate posteriors (fbcnet_moe uid_val / uid_test, same seed)
Gate (DECISIONS_T2 T2-D111): tau = the smallest of TAUS with a false-route rate <= 0.10 on the unseen val users (windows
>= 30); the router passes iff at that tau the correct-route rate from window 30 on is >= 0.90 for BOTH seen users (seen1:
P04 / P10 sessions 2-3). Per seed for R2; the pass uses the seed-mean rates. Writes <stem>rt1 / rt2 / orc files
(orc = ORACLE: the true user's model for a represented user's stream, shared otherwise; diagnostic only), and the T2-D114
soft variants (no router): uni = mean of the per-user models, half = 0.5 shared + 0.5 uni, sr1 / sr2 = the per-user
models weighted by R1's / R2's running-mean posteriors.
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
from t2p_paired import seeds_arg  # noqa: E402

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
T_MIN = 30          # windows >= 30 of a stream (1-based) enter the rates


def band5_log(X, fs):
    return np.concatenate([np.log(B._bandpass(X, lo, hi, fs).var(-1) + 1e-10) for lo, hi in BANDS], 1)


def fingerprint(X, fs, names):
    """T2-D124 session-stable fingerprints per window: IAF (peak of the mean posterior-channel PSD in 7-13 Hz), 1/f slope
    (log-log fit of the channel-mean PSD over 2-40 Hz without 7-13 Hz), channel-normalised relative-band topography (log
    relative power per channel and band minus its channel mean), trace-normalised covariance shape (upper triangle of
    logm(C / tr C))."""
    from scipy.signal import welch
    X = np.asarray(X, np.float64)
    X = X - X.mean(-1, keepdims=True)
    f, P = welch(X, fs=fs, nperseg=min(240, X.shape[-1]), axis=-1)            # (N, C, F)
    post = [i for i, c in enumerate(names) if c.upper().startswith("P")]
    a = (f >= 7) & (f <= 13)
    iaf = f[a][np.argmax(np.log(P[:, post][:, :, a].mean(1) + 1e-20), 1)]
    fit = (f >= 2) & (f <= 40) & ~a
    lf, lp = np.log10(f[fit]), np.log10(P[:, :, fit].mean(1) + 1e-20)
    slope = np.polyfit(lf, lp.T, 1)[0]
    tot = P[:, :, (f >= 1) & (f <= 45)].sum(-1) + 1e-20
    topo = []
    for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        r = np.log(P[:, :, (f >= lo) & (f < hi)].sum(-1) / tot + 1e-20)
        topo.append(r - r.mean(1, keepdims=True))
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    C = C / np.trace(C, axis1=1, axis2=2)[:, None, None]
    C = C + 1e-6 * np.eye(C.shape[1])
    w, V = np.linalg.eigh(C)
    L = np.einsum("nij,nj,nkj->nik", V, np.log(np.maximum(w, 1e-12)), V)
    iu = np.triu_indices(C.shape[1])
    return np.concatenate([iaf[:, None], slope[:, None]] + topo + [L[:, iu[0], iu[1]]], 1)


def running(post, idx, rows):
    """post (len(rows), K) -> (running-mean posteriors in the same row order, 1-based position of each row in its stream)."""
    pos = {int(r): k for k, r in enumerate(rows)}
    m, t = np.zeros_like(post), np.zeros(len(rows), int)
    for g in B.stream_groups(idx, rows):
        k = [pos[int(r)] for r in g]
        m[k] = np.cumsum(post[k], 0) / np.arange(1, len(g) + 1)[:, None]
        t[k] = np.arange(1, len(g) + 1)
    return m, t


def gate(mv, tv, mt, tt, subj_t, users, seen):
    """-> dict(tau, false_route_val, correct[user]) for one router (val / test running means)."""
    late_v = tv >= T_MIN
    out = dict(tau=None, false_route_val={}, correct={})
    for tau in TAUS:
        out["false_route_val"][tau] = float((mv[late_v].max(1) >= tau).mean()) if late_v.any() else 0.0
    ok = [tau for tau in TAUS if out["false_route_val"][tau] <= 0.10]
    if not ok:
        return out
    tau = out["tau"] = ok[0]
    for u in seen:
        m = (subj_t == u) & (tt >= T_MIN)
        routed = mt[m].max(1) >= tau
        out["correct"][u] = float((routed & (np.array(users)[mt[m].argmax(1)] == u)).mean()) if m.any() else None
    return out


def routed_probs(shared, per_user, m, tau):
    """window-wise: per_user[argmax m] if max m >= tau, else shared."""
    out = shared.copy()
    r = m.max(1) >= tau
    if r.any():
        out[r] = per_user[m[r].argmax(1), np.flatnonzero(r)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--protocol", required=True, choices=("kit", "seen1"))
    ap.add_argument("--pu", required=True, help="stem of the A0 per-user twin (t2p_bench --peruser), without __s<seed>")
    ap.add_argument("--moe", required=True, help="stem of the fbcnet_moe runs (R2's gate posteriors)")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X, idx, meta = B.load_export(args.export)
    fs = float(meta["sfreq"])
    subj = idx["subject"].to_numpy()
    (_, trm, vam, tem), = B.folds(idx, args.protocol)
    tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
    users = sorted(set(subj[tr]))
    seen = [u for u in ("P04", "P10") if u in users and np.isin(te, np.flatnonzero(subj == u)).any()]
    print(f"[route] {args.protocol}: {len(users)} training users {users}; seen users in test {seen}; "
          f"val users {sorted(set(subj[va]))}; test users {sorted(set(subj[te]))}", flush=True)

    # R1: band5_log LDA on the fold's training windows (user ids only)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(band5_log(np.asarray(X[tr]), fs), subj[tr])
    assert list(lda.classes_) == users
    r1 = {}
    for name, rows in (("val", va), ("test", te)):
        r1[name] = running(lda.predict_proba(band5_log(np.asarray(X[rows]), fs)), idx, rows)
    g1 = gate(*r1["val"], *r1["test"], subj[te], users, seen)
    print(f"[route] R1 band5_log LDA: {json.dumps(g1)}", flush=True)
    # R3 (T2-D124): session-stable fingerprints, standardised, shrinkage LDA (user ids only)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    names = list(meta["ch_names"])
    lda3 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    lda3.fit(fingerprint(X[tr], fs, names), subj[tr])
    assert list(lda3.classes_) == users
    r3 = {name: running(lda3.predict_proba(fingerprint(X[rows], fs, names)), idx, rows) for name, rows in (("val", va), ("test", te))}
    g3 = gate(*r3["val"], *r3["test"], subj[te], users, seen)
    print(f"[route] R3 fingerprint LDA: {json.dumps(g3)}", flush=True)

    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=args.protocol, users=users, seen=seen, R1=g1, R2={}, R3=g3)
    for s in seeds_arg(args.seeds):
        pu = np.load(Path(args.runs) / f"{args.pu}__s{s}.npz")
        mo = np.load(Path(args.runs) / f"{args.moe}__s{s}.npz")
        assert list(pu["users"]) == users and json.loads(str(mo["res"]))["train_users"] == users
        assert np.array_equal(pu["test_idx"], te) and np.array_equal(pu["val_idx"], va)
        r2 = {name: running(mo[f"uid_{name}"].astype(np.float64), idx, rows) for name, rows in (("val", va), ("test", te))}
        g2 = gate(*r2["val"], *r2["test"], subj[te], users, seen)
        report["R2"][s] = g2
        res = json.loads(str(pu["res"]))
        for suf, g, rr in (("rt1", g1, r1), ("rt2", g2, r2), ("rt3", g3, r3), ("orc", None, None), ("uni", None, None),
                           ("half", None, None), ("sr1", None, r1), ("sr2", None, r2)):
            P = {}
            for name, rows, key, ukey in (("test", te, "prob_last", "prob_users_test"), ("val", va, "prob_val_last", "prob_users_val")):
                sh, Pu = pu[key].astype(np.float64), pu[ukey].astype(np.float64)
                if suf in ("uni", "half"):
                    P[name] = Pu.mean(0) if suf == "uni" else 0.5 * sh + 0.5 * Pu.mean(0)
                elif suf in ("sr1", "sr2"):
                    P[name] = np.einsum("nk,knc->nc", rr[name][0], Pu)
                elif suf == "orc":
                    P[name] = sh.copy()
                    for k, u in enumerate(users):
                        m = subj[rows] == u
                        P[name][m] = Pu[k][m]
                elif g["tau"] is None:
                    P[name] = sh
                else:
                    P[name] = routed_probs(sh, Pu, rr[name][0], g["tau"])
            stem = args.pu[:-2] + suf      # '<A0 stem>pu' -> '<A0 stem>rt1'
            np.savez_compressed(outd / f"{stem}__s{s}.npz", test_idx=te, prob_last=P["test"].astype(np.float32), val_idx=va,
                                prob_val_last=P["val"].astype(np.float32), prob_best=P["test"].astype(np.float32),
                                res=json.dumps(dict(res, routed=suf, tau=None if g is None else g["tau"])))
        print(f"[route] seed {s} R2 gate posteriors: {json.dumps(g2)}", flush=True)
    for name, gs in (("R1", [g1]), ("R2", list(report["R2"].values())), ("R3", [g3])):
        taus = [g["tau"] for g in gs]
        corr = {u: (float(np.mean([g["correct"].get(u) or 0.0 for g in gs])) if all(g["tau"] is not None for g in gs) else 0.0)
                for u in seen}
        passed = bool(seen) and all(t is not None for t in taus) and all(c >= 0.90 for c in corr.values())
        report[f"{name}_pass"] = dict(taus=taus, correct_seed_mean=corr, passed=passed)
        print(f"[route] {name} GATE: taus {taus}, correct-route (seed mean) {corr} -> {'PASS' if passed else 'FAIL'}", flush=True)
    (outd / f"route_{args.protocol}.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
