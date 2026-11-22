#!/usr/bin/env python
"""T2-D409 (pre-registered 2026-11-07 17:38, before any number): ERROR ANATOMY of M6 — wrong vs correct held-out windows, physiologically.
Error table: M6 (outputs/t2_endphase arm M6, causal SKλ 10 it / n0 3 / λ 0.95, 12 draws) on held-out windows, each window once: kit test + val for
P04 / P05 / P08 / P09 (new) + seen1 test + val for P10 s2-3 (returning); WRONG := wrong-rate over the 12 draws > 0.5. seen1 P04 s2-3 = confusion only.
Signatures F1-F12 (fixed), within-user standardised wrong - correct D (mean over users), within-user permutation p; explore = even within-stream
rank, confirm = odd; top-5 by |D| in explore confirmed iff same sign, p < 0.05 and >= 4 / 5 users agree in confirm. Pre-specified tests (b) / (d) /
(e) / (a) on all windows; LOUO logistic P(wrong) on causally z-scored features: AUC (i) F1-F12, (ii) confidence, (iii) both.
  python t2_d409_errors.py --export E [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
CLS = {0: "MI", 1: "sub", 2: "word"}
FEATS = ["F1_mu_lat", "F2_beta_lat", "F3_fm_theta", "F4_lr_frontotemp", "F5_eog", "F6_emg", "F7_line", "F8_var_outlier", "F9_offmanifold",
         "F10_global", "F11_post_alpha", "F12_position"]
MARKER = {0: ["F1_mu_lat", "F2_beta_lat"], 1: ["F3_fm_theta"], 2: ["F4_lr_frontotemp"]}
ARTEFACT = ["F5_eog", "F6_emg", "F7_line", "F8_var_outlier"]


def m6_table(d):
    """per held-out window: wrong-rate, modal prediction, confidence; plus the full per-draw confusion rows."""
    spec = json.load(open(O / "t2_endphase/spec.json")); M, w = spec["members"], spec["arms"]["M6"]
    strm, ordr, y = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy()
    subj, sess = d.subject.str[-3:].to_numpy(), d.session.astype(str).to_numpy()
    acc = {}                                       # (proto, window) -> list of (pred, conf)
    for i in range(spec["n"]):
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                ix, mix = None, 0.0
                for m, wm in w.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or np.array_equal(ix, ix_); ix, mix = ix_, mix + wm * p.astype(np.float64)
                Q = causal_sinkhorn(mix, (strm[ix], ordr[ix]), 10, 3, 0.95)
                for j, q in zip(ix, Q):
                    acc.setdefault((proto, int(j)), []).append((int(q.argmax()), float(q.max())))
    rows = []
    for (proto, j), v in acc.items():
        u, s = subj[j], sess[j]
        if proto == "kit" and u in ("P04", "P05", "P08", "P09"):
            role = "main"
        elif proto == "seen1" and u == "P10" and s in ("2", "3"):
            role = "main"
        elif proto == "seen1" and u == "P04" and s in ("2", "3"):
            role = "conf_only"
        else:
            continue
        pr = np.array([a for a, _ in v]); cf = np.array([c for _, c in v])
        rows.append(dict(proto=proto, j=j, role=role, user=u, session=s, y=int(y[j]), pred=int(np.bincount(pr, minlength=3).argmax()),
                         wrong_rate=float(np.mean(pr != y[j])), conf=float(cf.mean()), n_draws=len(v),
                         group="returning" if (u == "P10" or (u == "P04" and proto == "seen1")) else "new",
                         preds=pr.tolist()))
    T = pd.DataFrame(rows)
    T["wrong"] = T.wrong_rate > 0.5
    return T


def features(X, d, T, ch):
    c = {n: k for k, n in enumerate(ch)}
    idx = T.j.to_numpy()
    f, P = welch(np.asarray(X[idx], np.float64), fs=120.0, nperseg=120, noverlap=60, axis=-1)   # (n, ch, freq)

    def bp(chs, lo, hi):
        k = (f >= lo) & (f <= hi)
        return np.log(P[:, [c[x] for x in chs]][:, :, k].mean(-1) + 1e-30).mean(1)
    F = pd.DataFrame(index=T.index)
    F["F1_mu_lat"] = bp(["C3"], 8, 13) - bp(["C4"], 8, 13)
    F["F2_beta_lat"] = bp(["C3"], 13, 30) - bp(["C4"], 13, 30)
    F["F3_fm_theta"] = bp(["Fz", "FCz"], 4, 8)
    F["F4_lr_frontotemp"] = bp(["F5", "FC5", "C5", "TP7"], 4, 30) - bp(["F6", "FC6", "C6", "TP8"], 4, 30)
    F["F5_eog"] = bp(["AF3", "AF4"], 0.5, 4)
    F["F6_emg"] = bp(["TP7", "TP8"], 30, 55)
    F["F7_line"] = bp(ch, 48, 52) - bp(ch, 40, 46)
    lv = np.log(np.asarray(X[idx], np.float64).var(-1) + 1e-30)
    F["F8_var_outlier"] = lv.max(1) - np.median(lv, 1)
    F["F10_global"] = bp(ch, 1, 40)
    F["F11_post_alpha"] = bp(["PO7", "PO8", "Pz"], 8, 13)
    # F9 off-manifold + F12 position: per stream in recording order, over ALL of the stream's windows (the causal running mean)
    strm, ordr = d.stream.to_numpy(), d.order.to_numpy()
    F["F9_offmanifold"] = np.nan; F["F12_position"] = np.nan
    for (proto, s_), g in T.groupby(["proto", T.j.map(lambda j: strm[j])]):
        all_j = np.flatnonzero(strm == s_); all_j = all_j[np.argsort(ordr[all_j], kind="stable")]
        pos = {int(j): r for r, j in enumerate(all_j)}
        S, run, out = None, None, {}
        for r, j in enumerate(all_j):
            x = np.asarray(X[j], np.float64); x = x - x.mean(-1, keepdims=True)
            C = x @ x.T / x.shape[-1]; C = 0.95 * C + 0.05 * np.trace(C) / len(C) * np.eye(len(C))
            if run is not None:
                lam = eigvalsh(C, run)
                out[int(j)] = float(np.sqrt(np.sum(np.log(np.clip(lam, 1e-12, None)) ** 2)))
            S = C if S is None else S + C
            run = S / (r + 1)
        for k in g.index:
            j = int(T.at[k, "j"])
            F.at[k, "F9_offmanifold"] = out.get(j, np.nan)
            F.at[k, "F12_position"] = pos[j] / max(len(all_j) - 1, 1)
    return F


def within_D(v, wrong, users, n_perm=5000, seed=0):
    """mean over users of (mean wrong - mean correct) / pooled within-user SD, with a within-user permutation p; per-user d."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(v)
    v, wrong, users = v[ok], wrong[ok], users[ok]
    us = [u for u in np.unique(users) if 0 < wrong[users == u].sum() < (users == u).sum()]

    def stat(wf):
        ds = []
        for u in us:
            k = users == u; a, b = v[k][wf[k]], v[k][~wf[k]]
            sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1)) if len(a) > 1 and len(b) > 1 else np.nan
            ds.append((a.mean() - b.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan)
        return np.array(ds)
    du = stat(wrong); D = np.nanmean(du)
    null = np.empty(n_perm)
    for p in range(n_perm):
        wf = wrong.copy()
        for u in us:
            k = np.flatnonzero(users == u); wf[k] = rng.permutation(wf[k])
        null[p] = np.nanmean(stat(wf))
    pval = (1 + np.sum(np.abs(null) >= abs(D))) / (n_perm + 1)
    return float(D), float(pval), {u: float(x) for u, x in zip(us, du)}


def causal_z(F, T, d, min_t=6):
    strm = d.stream.to_numpy()
    Z = pd.DataFrame(np.nan, index=F.index, columns=F.columns)
    for (proto, s_), g in T.groupby(["proto", T.j.map(lambda j: strm[j])]):
        g = g.assign(pos=F.loc[g.index, "F12_position"]).sort_values("pos")
        vals = F.loc[g.index].to_numpy(float)
        for r, k in enumerate(g.index):
            if r < min_t:
                continue
            past = vals[:r]
            mu, sd = np.nanmean(past, 0), np.nanstd(past, 0) + 1e-9
            Z.loc[k] = (vals[r] - mu) / sd
    Z["F12_position"] = F["F12_position"]
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(O / "t2_d409"))
    ap.add_argument("--perm", type=int, default=5000)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    meta = json.load(open(f"{a.export}/meta.json")); ch = list(meta["ch_names"])
    X = np.load(f"{a.export}/X.npy", mmap_mode="r"); assert X.shape[0] == len(d)
    T = m6_table(d)
    res = {"n": {k: int(v) for k, v in T.groupby("role").size().items()}}
    # (2) confusion over all draws
    conf = {}
    for nm, sel in (("pooled_main", T.role == "main"), ("new", (T.role == "main") & (T.group == "new")),
                    ("returning", T.group == "returning")) + tuple((f"user_{u}", (T.role == "main") & (T.user == u)) for u in sorted(T[T.role == "main"].user.unique())):
        Cm = np.zeros((3, 3), int)
        for _, r in T[sel].iterrows():
            for p in r.preds:
                Cm[r.y, p] += 1
        off = {f"{CLS[i]}-{CLS[j]}": int(Cm[i, j] + Cm[j, i]) for i, j in ((0, 1), (0, 2), (1, 2))}
        conf[nm] = dict(matrix=Cm.tolist(), recall={CLS[i]: float(Cm[i, i] / Cm[i].sum()) for i in range(3)}, pairs=off,
                        top_pair=max(off, key=off.get), top_share=float(max(off.values()) / max(sum(off.values()), 1)))
    res["confusion"] = conf
    M_ = T[T.role == "main"].copy()
    res["error_rate"] = {u: float(g.wrong.mean()) for u, g in M_.groupby("user")} | {"all": float(M_.wrong.mean()),
                                                                                    "ambiguous_0.5": float((M_.wrong_rate == 0.5).mean())}
    F = features(X, d, M_, ch)
    users, wrong = M_.user.to_numpy(), M_.wrong.to_numpy()
    strm_all, ord_all = d.stream.to_numpy(), d.order.to_numpy()
    rank_all = np.empty(len(d), int)
    for s_ in np.unique(strm_all):
        k = np.flatnonzero(strm_all == s_); k = k[np.argsort(ord_all[k], kind="stable")]; rank_all[k] = np.arange(len(k))
    par = rank_all[M_.j.to_numpy()] % 2 == 0                                   # even within-stream rank = EXPLORE, odd = CONFIRM
    # (3) explore / confirm
    ex = {f: within_D(F[f].to_numpy(float)[par], wrong[par], users[par], a.perm) for f in FEATS}
    top = sorted(FEATS, key=lambda f: -abs(ex[f][0]))[:5]
    cf = {f: within_D(F[f].to_numpy(float)[~par], wrong[~par], users[~par], a.perm) for f in top}
    sig = []
    for f in top:
        De, pe, _ = ex[f]; Dc, pc, du = cf[f]
        agree = sum(np.sign(x) == np.sign(De) for x in du.values())
        sig.append(dict(feature=f, D_explore=De, p_explore=pe, D_confirm=Dc, p_confirm=pc, users_agree=f"{agree} / {len(du)}", per_user=du,
                        confirmed=bool(np.sign(Dc) == np.sign(De) and pc < 0.05 and agree >= 4)))
    res["signatures_top5"] = sig
    res["explore_all"] = {f: dict(D=v[0], p=v[1]) for f, v in ex.items()}
    # pre-specified tests on all windows
    yv = M_.y.to_numpy()
    b = {}
    for c_, fs in MARKER.items():
        for f in fs:
            v = F[f].to_numpy(float)
            Sc = within_D(v[~wrong], (yv[~wrong] == c_), users[~wrong], a.perm)     # correct-c vs correct-other
            Dc = within_D(v[yv == c_], wrong[yv == c_], users[yv == c_], a.perm)     # wrong vs correct among true-c
            b[f"{CLS[c_]}:{f}"] = dict(S_c=Sc[0], p_S=Sc[1], D_c=Dc[0], p_D=Dc[1],
                                       supported=bool(Sc[1] < 0.05 and Dc[1] < 0.05 and np.sign(Sc[0]) != np.sign(Dc[0])))
    b_cls = {CLS[c_]: any(b[f"{CLS[c_]}:{f}"]["supported"] for f in fs) for c_, fs in MARKER.items()}
    dpos = within_D(F["F12_position"].to_numpy(float), wrong, users, a.perm)
    pooled = conf["pooled_main"]; per_user_top = [conf[k]["top_pair"] for k in conf if k.startswith("user_")]
    e_ok = pooled["top_share"] >= 0.5 and sum(t == pooled["top_pair"] for t in per_user_top) >= 4
    a_ok = [s["feature"] for s in sig if s["confirmed"] and s["feature"] in ARTEFACT and s["D_confirm"] > 0]
    res["root_cause"] = {"a_artefact": dict(supported=bool(a_ok), confirmed=a_ok),
                         "b_marker_absent": dict(supported=sum(b_cls.values()) >= 2, per_class=b_cls, detail=b),
                         "c_shifted": "not testable per window (D229 IAF warp = the closed proxy)",
                         "d_drift": dict(supported=bool(dpos[1] < 0.05 and dpos[0] > 0), D=dpos[0], p=dpos[1]),
                         "e_pair": dict(supported=bool(e_ok), top_pair=pooled["top_pair"], share=pooled["top_share"], per_user_top=per_user_top)}
    # (4) predictability: LOUO logistic on causal z features
    Z = causal_z(F, M_, d)
    keep = Z.drop(columns="F12_position").notna().all(1).to_numpy()
    auc = {}
    for nm, cols in (("i_features", FEATS), ("ii_confidence", ["conf"]), ("iii_both", FEATS + ["conf"])):
        A = Z.assign(conf=M_.conf.to_numpy())[cols].to_numpy(float)[keep]; yw = wrong[keep]; uu = users[keep]
        pr = np.full(len(yw), np.nan)
        for u in np.unique(uu):
            tr, te = uu != u, uu == u
            mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((A[tr] - mu) / sd, yw[tr])
            pr[te] = clf.predict_proba((A[te] - mu) / sd)[:, 1]
        auc[nm] = dict(pooled=float(roc_auc_score(yw, pr)),
                       per_user={u: float(roc_auc_score(yw[uu == u], pr[uu == u])) for u in np.unique(uu) if 0 < yw[uu == u].sum() < (uu == u).sum()})
    res["auc"] = auc
    res["auc_n"] = int(keep.sum())
    M_.assign(**{f: F[f] for f in FEATS}).drop(columns="preds").to_csv(Path(a.out) / "error_table.csv", index=False)
    (Path(a.out) / "anatomy.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("n", "error_rate")}, default=float))
    for k in ("pooled_main", "new", "returning"):
        print(f"[D409] confusion {k}: {conf[k]['matrix']} recall {({c: round(v, 3) for c, v in conf[k]['recall'].items()})} pairs {conf[k]['pairs']}")
    for k in conf:
        if k.startswith("user_"):
            print(f"   {k}: recall {({c: round(v, 3) for c, v in conf[k]['recall'].items()})} top pair {conf[k]['top_pair']} ({conf[k]['top_share']:.2f})")
    print("[D409] top-5 signatures (explore -> confirm):")
    for s in sig:
        print(f"   {s['feature']:18s} D_ex {s['D_explore']:+.3f} (p {s['p_explore']:.3f}) | D_cf {s['D_confirm']:+.3f} (p {s['p_confirm']:.3f}) users {s['users_agree']} "
              f"-> {'CONFIRMED' if s['confirmed'] else 'not confirmed'}")
    print("[D409] root cause:", json.dumps({k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items() if kk != 'detail'}) for k, v in res['root_cause'].items()}, default=float))
    for k, v in b.items():
        print(f"   (b) {k}: S_c {v['S_c']:+.3f} (p {v['p_S']:.3f}), D_c {v['D_c']:+.3f} (p {v['p_D']:.3f}) -> {'supported' if v['supported'] else '-'}")
    print("[D409] AUC (LOUO, n =", res["auc_n"], "):", {k: round(v["pooled"], 3) for k, v in auc.items()})
    for k, v in auc.items():
        print(f"   {k}: " + ", ".join(f"{u} {x:.3f}" for u, x in v["per_user"].items()))


if __name__ == "__main__":
    main()
