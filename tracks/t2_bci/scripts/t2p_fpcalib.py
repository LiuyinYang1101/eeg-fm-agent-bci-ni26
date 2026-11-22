#!/usr/bin/env python
"""Track 2 T2-D124b: open-set rejection for the session-fingerprint user router (DECISIONS_T2 T2-D124b).

  export: fingerprints (t2p_route.fingerprint) of external multi-session corpora, channels mapped onto the 41 PROTEUS
          channels (t2p_bench.interp_matrix, spherical spline over the standard_1005 channels both share), <= --max-per-stream
          windows per subject x session stream in time order (CPU; run where the corpora live):
     python t2p_fpcalib.py export --pools DIR1,DIR2 --names-from <export_proteus_v1> --out fp_ext.npz
  fit:    statistic s_t = min_k |zbar_t - mu_k| in the LDA space of the training users (StandardScaler + shrinkage LDA,
          eigen solver), zbar_t = running mean over the stream's windows 1..t; calibration P(seen | log s_t, log t) by
          logistic regression on (i) 50 simulated deployments per external corpus (K random subjects' first session =
          training users; seen = their later sessions, new = first sessions of up to K other subjects) and (ii) PROTEUS
          leave-one-user-out over P01 P02 P03 P06 P07 P10-s1; one calibration per K (seen1 K = 7, kit K = 6); then routes
          the FBCNet member: soft p = (1 - r) shared + r per-user_{k*}, r = P(seen); hard twin r = 1[P >= 0.5]:
     python t2p_fpcalib.py fit --export <export_proteus_v1> --ext fp_ext.npz --runs <job-70 runs> --out <route dir>
Writes <stem>fs / <stem>fh result files (t2p_paired scores them) and fpcalib_<protocol>.json.
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
import t2p_route as RT  # noqa: E402

FS = 120.0
T_MIN = 10
CAL_USERS = [("P01", None), ("P02", None), ("P03", None), ("P06", None), ("P07", None), ("P10", "1")]


def export(args):
    import mne
    dst = json.loads((Path(args.names_from) / "meta.json").read_text())["ch_names"]
    known = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    out = dict(F=[], corpus=[], subject=[], session=[], pos=[])
    for d in args.pools.split(","):
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        names = list(meta["ch_names"])
        keep = [i for i, c in enumerate(names) if c in known]
        W = B.interp_matrix([names[i] for i in keep], dst)
        idx = pd.read_csv(d / "index.csv", dtype=str, keep_default_na=False)
        X = np.load(d / "X.npy", mmap_mode="r")
        sort_cols = [c for c in ("run", "order", "onset_s") if c in idx]
        for c in sort_cols:
            idx[c + "_n"] = pd.to_numeric(idx[c], errors="coerce").fillna(0)
        idx["i_"] = np.arange(len(idx))
        n = 0
        for (s, ss), g in idx.groupby(["subject", "session"], sort=False):
            g = g.sort_values([c + "_n" for c in sort_cols] or ["i_"]).head(args.max_per_stream)
            rows = g["i_"].to_numpy()
            xs = np.asarray(X[np.sort(rows)], np.float64)[np.argsort(np.argsort(rows))][:, keep, :480]
            xs = np.einsum("dc,nct->ndt", W, xs)
            out["F"].append(RT.fingerprint(xs, FS, dst).astype(np.float32))
            out["corpus"] += [d.name] * len(rows)
            out["subject"] += [str(s)] * len(rows)
            out["session"] += [str(ss)] * len(rows)
            out["pos"] += list(range(1, len(rows) + 1))
            n += len(rows)
        print(f"[fpcalib] {d.name}: {len(keep)}/{len(names)} channels mapped, {idx['subject'].nunique()} subjects, {n} windows", flush=True)
    np.savez_compressed(args.out, F=np.concatenate(out["F"]), **{k: np.asarray(v) for k, v in out.items() if k != "F"})


def fit_space(F, users):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(sc.transform(F), users)
    Z = lda.transform(sc.transform(F))
    mu = np.stack([Z[users == u].mean(0) for u in lda.classes_])
    return lambda G: lda.transform(sc.transform(G)), mu, list(lda.classes_)


def stream_route_every(Z, mu, every=1):
    """the package's route (t2p_submission StreamModel._user): the running mean of the projected fingerprints of windows
    0, every, 2 every, ... (1-based t = 1, 1 + every, ...); k* held between updates -> (T,) nearest-centroid index.
    As in the package, a non-finite fingerprint never enters the mean, and before the first usable one there is no route
    (the package's None = no blend): -1 here (review 2026-10-16 T2 P1d; consumers go through blend_route)."""
    k = np.empty(len(Z), int)
    s_, n_, cur = None, 0, None
    for t in range(len(Z)):
        if t % every == 0:
            if np.isfinite(Z[t]).all():
                s_ = Z[t] if s_ is None else s_ + Z[t]
                n_ += 1
            cur = None if n_ == 0 else int(np.argmin(np.linalg.norm(s_ / n_ - mu, axis=1)))
        k[t] = -1 if cur is None else cur
    return k


def stream_route_soft(Z, mu, every=1, top=2, eps=1e-9):
    """T2-D165: as stream_route_every, but -> (T, K) weights over the `top` nearest centroids of the running-mean fingerprint,
    w_i = (1 / d_i) / sum_j (1 / d_j) over those `top` (inverse distance, no tuning); all-zero rows = no route yet."""
    W = np.zeros((len(Z), len(mu)))
    s_, n_, cur = None, 0, None
    for t in range(len(Z)):
        if t % every == 0:
            if np.isfinite(Z[t]).all():
                s_ = Z[t] if s_ is None else s_ + Z[t]
                n_ += 1
            if n_:
                d = np.linalg.norm(s_ / n_ - mu, axis=1)
                nn = np.argsort(d, kind="stable")[:top]
                w = 1.0 / np.maximum(d[nn], eps)
                cur = (nn, w / w.sum())
        if cur is not None:
            W[t, cur[0]] = cur[1]
    return W


def blend_route_soft(sh, Pu, W, r):
    """T2-D165: (1 - r) sh + r sum_k W[t, k] Pu[k, t] per window; rows of W that are all zero (no route yet) keep sh."""
    out = np.array(sh, np.float64, copy=True)
    m = np.flatnonzero(W.sum(1) > 0)
    out[m] = (1 - r) * out[m] + r * np.einsum("tk,ktc->tc", W[m], Pu[:, m])
    return out


def blend_route(sh, Pu, k, r):
    """(1 - r) sh + r Pu[k] per window; k = -1 (no route yet) keeps sh, as the package does."""
    out = np.array(sh, np.float64, copy=True)
    m = np.flatnonzero(np.asarray(k) >= 0)
    out[m] = (1 - r) * out[m] + r * Pu[np.asarray(k)[m], m]
    return out


def stream_stat(proj, mu, G):
    """windows of ONE stream in order -> (s_t, k*_t) with the running mean of the projected windows."""
    Z = proj(G)
    zb = np.cumsum(Z, 0) / np.arange(1, len(Z) + 1)[:, None]
    d = np.linalg.norm(zb[:, None, :] - mu[None], axis=-1)
    return d.min(1), d.argmin(1)


def simulate(ext, K, n_dep, rng):
    """external deployments -> rows (log s, log t, label, corpus, dep)."""
    rows = []
    for corpus in np.unique(ext["corpus"]):
        m = ext["corpus"] == corpus
        F, S, SS, P = ext["F"][m], ext["subject"][m], ext["session"][m], ext["pos"][m]
        sess = {s: sorted(set(SS[S == s]), key=lambda v: (len(v), v)) for s in np.unique(S)}
        multi = [s for s, v in sess.items() if len(v) >= 2]
        for dep in range(n_dep):
            tr = rng.choice(multi, size=K, replace=False)
            others = rng.permutation([s for s in sess if s not in tr])[:K]
            sel = np.isin(S, tr) & np.array([SS[i] == sess[S[i]][0] for i in range(len(S))])
            proj, mu, _ = fit_space(F[sel], S[sel])
            streams = [(s, v, 1) for s in tr for v in sess[s][1:4]] + [(s, sess[s][0], 0) for s in others]
            for s, v, lab in streams:
                k = (S == s) & (SS == v)
                order = np.argsort(P[k])
                st, _ = stream_stat(proj, mu, F[k][order])
                t = np.arange(1, len(st) + 1)
                ok = t >= T_MIN
                rows += [(np.log(a), np.log(b), lab, corpus, dep) for a, b in zip(st[ok], t[ok])]
    return rows


def louo(Fp, idx, K, rng):
    rows = []
    subj, ses = idx["subject"].to_numpy(), idx["session"].astype(str).to_numpy()
    cal = [(u, sv) for u, sv in CAL_USERS]
    for u, sv in cal:
        rest = [c for c in cal if c[0] != u]
        if len(rest) > K:
            rest = [rest[i] for i in sorted(rng.choice(len(rest), K, replace=False))]
        trm = np.zeros(len(idx), bool)
        for v, svv in rest:
            trm |= (subj == v) & ((ses == svv) if svv else True)
        proj, mu, _ = fit_space(Fp[trm], subj[trm])
        k = (subj == u) & ((ses == sv) if sv else True)
        g = idx[k].sort_values("order")
        st, _ = stream_stat(proj, mu, Fp[g["i"].to_numpy()])
        t = np.arange(1, len(st) + 1)
        ok = t >= T_MIN
        rows += [(np.log(a), np.log(b), 0, "proteus_louo", 0) for a, b in zip(st[ok], t[ok])]
    return rows


def calibrate(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    df = pd.DataFrame(rows, columns=["ls", "lt", "y", "corpus", "dep"])
    df["w"] = 1.0 / df.groupby(["corpus", "y"])["y"].transform("size")
    hold = (df["dep"] >= 40) & (df["corpus"] != "proteus_louo")
    m = LogisticRegression(C=1.0).fit(df.loc[~hold, ["ls", "lt"]], df.loc[~hold, "y"], sample_weight=df.loc[~hold, "w"])
    auc = float(roc_auc_score(df.loc[hold, "y"], m.predict_proba(df.loc[hold, ["ls", "lt"]])[:, 1])) if hold.any() else None
    m = LogisticRegression(C=1.0).fit(df[["ls", "lt"]], df["y"], sample_weight=df["w"])     # final: every deployment
    louo_route = float((m.predict_proba(df.loc[df.corpus == "proteus_louo", ["ls", "lt"]])[:, 1] >= 0.5).mean())
    per = {c: dict(n=int((df.corpus == c).sum()), seen_frac=float(df.loc[df.corpus == c, "y"].mean())) for c in df.corpus.unique()}
    return m, dict(coef=m.coef_.tolist(), intercept=m.intercept_.tolist(), heldout_auc=auc, louo_false_route_hard=louo_route, data=per)


def fit(args):
    from ni26 import metrics  # noqa: F401  (import check)
    X, idx, meta = B.load_export(args.export)
    names = list(meta["ch_names"])
    cache = Path(args.out) / "fp_proteus.npy"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if cache.exists():
        Fp = np.load(cache)
    else:
        Fp = RT.fingerprint(np.asarray(X), FS, names).astype(np.float32)
        np.save(cache, Fp)
    z = np.load(args.ext)
    ext = {k: z[k] for k in z.files}
    subj = idx["subject"].to_numpy()
    for proto in args.protocols.split(","):
        (_, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = map(np.flatnonzero, (trm, vam, tem))
        users = sorted(set(subj[tr]))
        K = len(users)
        rng = np.random.default_rng(20261015 + K)
        rows = simulate(ext, K, args.n_dep, rng) + louo(Fp, idx, K, rng)
        model, rep = calibrate(rows)
        print(f"[fpcalib] {proto}: K {K}, calibration {json.dumps(rep)}", flush=True)
        proj, mu, cls = fit_space(Fp[tr], subj[tr])
        assert cls == users
        stat = {}
        for name, rr in (("val", va), ("test", te)):
            s_all, k_all, t_all = np.zeros(len(rr)), np.zeros(len(rr), int), np.zeros(len(rr), int)
            pos = {int(r): j for j, r in enumerate(rr)}
            for g in B.stream_groups(idx, rr):
                st, kk = stream_stat(proj, mu, Fp[g])
                j = [pos[int(r)] for r in g]
                s_all[j], k_all[j], t_all[j] = st, kk, np.arange(1, len(g) + 1)
            P = model.predict_proba(np.c_[np.log(s_all), np.log(t_all)])[:, 1]
            stat[name] = (P, k_all, t_all)
        late = {}
        for name, rr in (("val", va), ("test", te)):
            P, kk, tt = stat[name]
            for u in sorted(set(subj[rr])):
                m_ = (subj[rr] == u) & (tt >= 30)
                late[f"{name}:{u}"] = dict(p_seen=round(float(P[m_].mean()), 3), hard_route=round(float((P[m_] >= 0.5).mean()), 3),
                                           routed_to=dict(zip(*np.unique(np.array(users)[kk[m_]], return_counts=True))) if m_.any() else {})
        print(f"[fpcalib] {proto}: per user (windows >= 30): {json.dumps(late, default=int)}", flush=True)
        for s in [int(v) for v in args.seeds.split(",")]:
            pu = np.load(Path(args.runs) / f"{proto}__fbcnet__none__n00__r1pu__s{s}.npz")
            assert list(pu["users"]) == users
            res = json.loads(str(pu["res"]))
            for suf, hard in (("fs", False), ("fh", True)):
                out = {}
                for name, key, ukey in (("test", "prob_last", "prob_users_test"), ("val", "prob_val_last", "prob_users_val")):
                    P, kk, _ = stat[name]
                    r = (P >= 0.5).astype(float) if hard else P
                    sh, Pu = pu[key].astype(np.float64), pu[ukey].astype(np.float64)
                    out[name] = (1 - r)[:, None] * sh + r[:, None] * Pu[kk, np.arange(len(kk))]
                np.savez_compressed(Path(args.out) / f"{proto}__fbcnet__none__n00__r1{suf}__s{s}.npz", test_idx=te,
                                    prob_last=out["test"].astype(np.float32), val_idx=va, prob_val_last=out["val"].astype(np.float32),
                                    prob_best=out["test"].astype(np.float32), res=json.dumps(dict(res, routed=suf)))
        (Path(args.out) / f"fpcalib_{proto}.json").write_text(json.dumps(dict(calibration=rep, per_user=late), indent=1, default=int))


def blend(args):
    """T2-D124c: p = (1 - r) shared + r per-user_{k*}, r constant, k*_t = the nearest training-user centroid of the stream's
    running-mean fingerprint (T2-D124b's statistic; no novelty gate). Writes <stem>fb<tag> files."""
    X, idx, meta = B.load_export(args.export)
    Fp = np.load(args.fp) if args.fp and Path(args.fp).exists() else RT.fingerprint(np.asarray(X), FS, list(meta["ch_names"])).astype(np.float32)
    subj = idx["subject"].to_numpy()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for proto in args.protocols.split(","):
        (_, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = map(np.flatnonzero, (trm, vam, tem))
        proj, mu, users = fit_space(Fp[tr], subj[tr])
        ks, ws = {}, {}
        for name, rr in (("val", va), ("test", te)):
            k_all = np.zeros(len(rr), int)
            w_all = np.zeros((len(rr), len(mu)))
            pos = {int(r): j for j, r in enumerate(rr)}
            for g in B.stream_groups(idx, rr):
                kk = stream_route_every(proj(Fp[g]), mu, args.every)
                k_all[[pos[int(r)] for r in g]] = kk
                if args.route == "inv2":                      # T2-D165: soft top-2 inverse-distance route
                    w_all[[pos[int(r)] for r in g]] = stream_route_soft(proj(Fp[g]), mu, args.every, top=2)
            ks[name], ws[name] = k_all, w_all
        later = {u: dict(zip(*np.unique(np.array(users)[ks["test"][subj[te] == u]], return_counts=True))) for u in sorted(set(subj[te]))}
        print(f"[fpcalib blend] {proto}: r {args.r:.4f}; test routed to (all windows): {json.dumps(later, default=int)}", flush=True)
        for s_ in [int(v) for v in args.seeds.split(",")]:
            pu = np.load(Path(args.runs) / f"{proto}__{args.model}__{args.align}__{args.n0tag}__{args.recipe}pu{args.xtag}__s{s_}.npz")
            assert list(pu["users"]) == users
            out = {}
            for name, key, ukey in (("test", "prob_last", "prob_users_test"), ("val", "prob_val_last", "prob_users_val")):
                sh, Pu = pu[key].astype(np.float64), pu[ukey].astype(np.float64)
                out[name] = blend_route(sh, Pu, ks[name], args.r) if args.route == "nearest" else blend_route_soft(sh, Pu, ws[name], args.r)
            np.savez_compressed(Path(args.out) / f"{proto}__{args.model}__{args.align}__{args.n0tag}__{args.recipe}fb{args.tag}{args.xtag}__s{s_}.npz", test_idx=te,
                                prob_last=out["test"].astype(np.float32), val_idx=va, prob_val_last=out["val"].astype(np.float32),
                                prob_best=out["test"].astype(np.float32),
                                res=json.dumps(dict(json.loads(str(pu["res"])), routed=f"blend r={args.r}" + ("" if args.route == "nearest" else f" route={args.route}"))))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("--pools", required=True)
    e.add_argument("--names-from", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max-per-stream", type=int, default=120)
    f = sp.add_parser("fit")
    f.add_argument("--export", required=True)
    f.add_argument("--ext", required=True)
    f.add_argument("--runs", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--protocols", default="seen1,kit")
    f.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    f.add_argument("--n-dep", type=int, default=50)
    b = sp.add_parser("blend")
    b.add_argument("--export", required=True)
    b.add_argument("--runs", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--fp", default="", help="cached PROTEUS fingerprints (fit's fp_proteus.npy)")
    b.add_argument("--r", type=float, default=1 / 3)
    b.add_argument("--tag", default="3")
    b.add_argument("--n0tag", default="n00", help="T2-D494: the bench stems' n0 field (n00 = EA without a population prior; n010 / n030)")
    b.add_argument("--xtag", default="", help="T2-D219: the --extra field of the bench stems (e.g. __xscherer2015_v1), kept in the output stem")
    b.add_argument("--model", default="fbcnet", help="the CNN of the per-user twins (fbcnet | fbcnet_bboth)")
    b.add_argument("--recipe", default="r1", help="the recipe stem of the shared + per-user files (T2-D146: r1_mx4)")
    b.add_argument("--align", default="none", help="the align stem of the files (T2-D148: ShallowNet = ea)")
    b.add_argument("--every", type=int, default=1, help="fingerprint update every N windows (the package's ub_every)")
    b.add_argument("--route", default="nearest", choices=("nearest", "inv2"), help="T2-D165: nearest centroid (the package) or "
                   "inv2 = inverse-distance weights over the 2 nearest centroids")
    b.add_argument("--protocols", default="seen1,kit")
    b.add_argument("--seeds", default="12,13,14,15,16,17,18,19,20,21,22,23")
    args = ap.parse_args()
    {"export": export, "fit": fit, "blend": blend}[args.cmd](args)


if __name__ == "__main__":
    main()
