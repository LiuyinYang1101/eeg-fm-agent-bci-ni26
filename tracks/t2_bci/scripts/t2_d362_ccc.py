#!/usr/bin/env python
"""T2-D362 (outlier bet B3; zero GPU): anchored class-conditional calibration before SKλ, a probability-space surrogate of
BTTA-DG / T3A / ADAPT, replayed causally on the stored per-window member probabilities of a t2p_pkgshape spec.

Per stream (recording order; a fresh state per stream, as the template's per-session reset):
  anchor a_t  = SKλ(mix)_t (causal Sinkhorn, 10 its, n0 3, λ 0.95: the shipped decoder, unchanged)
  bank        = earlier windows j < t with max a_j ≥ τ, labelled argmax a_j (the ANCHOR's labels: the adapted output never feeds
                back, unlike T2-D322), FIFO ≤ K per class
  once every class has ≥ m bank windows: z = additive-log-ratio features (fam: per family posterior; mem: per member; mix: the
  package mix), class means + one shared covariance (shrunk to its diagonal, ridge) or an isotropic variance (iso, T3A-like);
  ℓ_c = Gaussian log-likelihood, centred by logsumexp; log q_t = log mix_t + β ℓ_c (renormalised); else q_t = mix_t
  decision     = argmax SKλ(q)_t over the stream's fused q (the shipped decoder on top)
Reads (all vs base + SKλ on the same draws): full streams; the first third of each stream (early session); a class-imbalance replay
(⅔ of one class's windows dropped per stream, the class rotating with the stream; anchor and adapter both re-run on the reduced
stream). Groups as t2_d320_rescreen: typical (non-P04, 16 cells), P04, P10, newT; per-user Δ + a user bootstrap (typical users).
  python t2_d362_ccc.py --export E --spec SPEC --arm nm35 --out DIR [--jobs 6]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2p_report import causal_sinkhorn  # noqa: E402

FAM = {"stl": ("r0", "r12", "r24", "vp12", "lp10", "stl_ea", "sess", "stl_lora"), "fbc": ("fbc_tv8c", "fbc16", "fbc_vrmx"),
       "sh": ("sh_tv8c", "sh16", "sh_n75", "sh_vrmx"), "ifn": ("ifn",)}
PRIMARY = dict(feat="fam", cov="shared", beta=0.5, tau=0.5, K=40, m=5)
VARIANTS = {"primary": {}, "f_mix": dict(feat="mix"), "f_mem": dict(feat="mem"), "c_iso": dict(cov="iso"),
            "b025": dict(beta=0.25), "b1": dict(beta=1.0), "t04": dict(tau=0.4), "t06": dict(tau=0.6)}
SK = dict(iters=10, n0=3, lam=0.95)
G = {}


def init(export):
    d = pd.read_csv(f"{export}/index.csv")
    G.update(strm=d.stream.to_numpy(), ordr=d.order.to_numpy(), y=d.y.to_numpy(), cells=d.cell.to_numpy(),
             subj=d.subject.str[-3:].to_numpy())


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def alr(P):
    L = np.log(np.clip(P, 1e-6, None))
    return L[:, :-1] - L[:, -1:]


def sk(P, ok):
    return causal_sinkhorn(P, ok, SK["iters"], SK["n0"], SK["lam"])


def features(kind, mix, parts, wts):
    if kind == "mix":
        return alr(mix)
    if kind == "mem":
        return np.hstack([alr(parts[m]) for m in sorted(parts)])
    cols = []
    for fam, names in FAM.items():
        ms = [m for m in parts if m in names]
        if ms:
            w = np.array([wts[m] for m in ms]) / sum(wts[m] for m in ms)
            cols.append(alr(sum(wi * parts[m] for wi, m in zip(w, ms))))
    assert sum(len([m for m in parts if m in n]) for n in FAM.values()) == len(parts), f"member outside FAM: {sorted(parts)}"
    return np.hstack(cols)


def adapt(mix, Z, anchor, ok, cfg):
    """the fused q (before the second SKλ); causal: window t sees the bank of windows < t only."""
    beta, tau, K, m, cov = cfg["beta"], cfg["tau"], int(cfg["K"]), int(cfg["m"]), cfg["cov"]
    q = mix.astype(np.float64).copy()
    C, d = mix.shape[1], Z.shape[1]
    strm, ordr = ok
    for s_ in np.unique(strm):
        rows = np.flatnonzero(strm == s_)
        rows = rows[np.argsort(ordr[rows], kind="stable")]
        A, Zs = anchor[rows], Z[rows]
        lab, conf = A.argmax(1), A.max(1)
        bank = [[] for _ in range(C)]
        for t in range(len(rows)):
            if t > 0 and conf[t - 1] >= tau:
                b = bank[lab[t - 1]]
                b.append(t - 1)
                if len(b) > K:
                    b.pop(0)
            if min(len(b) for b in bank) < m:
                continue
            mu = np.stack([Zs[b].mean(0) for b in bank])
            R = np.vstack([Zs[b] - mu[c] for c, b in enumerate(bank)])
            dz = Zs[t] - mu
            if cov == "shared":
                S = R.T @ R / max(len(R) - C, 1)
                S = 0.5 * S + 0.5 * np.diag(np.diag(S)) + 1e-3 * np.eye(d)
                ll = -0.5 * np.einsum("cd,de,ce->c", dz, np.linalg.inv(S), dz)
            else:
                ll = -0.5 * (dz ** 2).sum(1) / ((R ** 2).mean() + 1e-3)
            ll = ll - (ll.max() + np.log(np.exp(ll - ll.max()).sum()))
            lq = np.log(np.clip(mix[rows[t]], 1e-12, None)) + beta * ll
            e = np.exp(lq - lq.max())
            q[rows[t]] = e / e.sum()
    return q


def imbalance_keep(ix):
    """⅔ of one class's windows dropped per stream; the class = the stream's rank mod 3; a fixed rng per stream."""
    keep = np.ones(len(ix), bool)
    st, y = G["strm"][ix], G["y"][ix]
    for r, s_ in enumerate(np.unique(st)):
        c = np.unique(y)[r % len(np.unique(y))]
        w = np.flatnonzero((st == s_) & (y == c))
        rng = np.random.default_rng(1000 + r)
        keep[rng.choice(w, size=int(round(len(w) * 2 / 3)), replace=False)] = False
    return keep


def task(args):
    spec, arm, proto, i = args
    J = json.load(open(spec))
    wts = J["arms"][arm]
    rows = []
    for split in ("test", "val"):
        parts, ix = {}, None
        for mname, w in wts.items():
            M = J["members"][mname]
            z = np.load(M[proto].format(s=M.get("offset", 0) + i))
            ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
            assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all()), "index mismatch"
            ix, parts[mname] = ix_, p.astype(np.float64)
            assert M.get("temp", 1) == 1, "tempered members not supported"
        for mode in ("full", "imb"):
            keep = imbalance_keep(ix) if mode == "imb" else np.ones(len(ix), bool)
            jx = ix[keep]
            P = {m: p[keep] for m, p in parts.items()}
            mix = sum(wts[m] * P[m] for m in P)
            ok = (G["strm"][jx], G["ordr"][jx])
            anchor = sk(mix, ok)
            rank = np.empty(len(jx))
            for s_ in np.unique(ok[0]):
                r_ = np.flatnonzero(ok[0] == s_)
                rank[r_[np.argsort(ok[1][r_], kind="stable")]] = np.arange(len(r_)) / len(r_)
            decs = {"base": anchor}
            feats = {}
            for vn, ch in VARIANTS.items():
                cfg = {**PRIMARY, **ch}
                if cfg["feat"] not in feats:
                    feats[cfg["feat"]] = features(cfg["feat"], mix, P, wts)
                decs[vn] = sk(adapt(mix, feats[cfg["feat"]], anchor, ok, cfg), ok)
            for vn, D in decs.items():
                pr = D.argmax(1)
                for c in np.unique(G["cells"][jx]):
                    mk = G["cells"][jx] == c
                    subj = G["subj"][jx][mk][0]
                    rows.append(dict(mode=mode, proto=proto, split=split, i=i, var=vn, cell=c, subj=subj, ba=bal(G["y"][jx][mk], pr[mk])))
                    e = mk & (rank < 1 / 3)
                    if mode == "full" and e.sum() >= 6:
                        rows.append(dict(mode="early", proto=proto, split=split, i=i, var=vn, cell=c, subj=subj,
                                         ba=bal(G["y"][jx][e], pr[e])))
    return rows


def summarise(R, n_boot=2000):
    R = R.copy()
    R["grp"] = np.select([R.subj == "P04", R.subj == "P10"], ["P04", "P10"], "newT")
    out, users = [], []
    for mode, Rm in R.groupby("mode"):
        per = {}
        for name, mask in (("typical", Rm.grp != "P04"), ("P04", Rm.grp == "P04"), ("P10", Rm.grp == "P10"), ("newT", Rm.grp == "newT")):
            per[name] = Rm[mask].groupby(["var", "i"]).ba.mean().unstack("var")
        U = Rm.groupby(["var", "subj", "i"]).ba.mean().unstack("var")
        for v in VARIANTS:
            r = dict(mode=mode, var=v)
            for name, T in per.items():
                dd = T[v] - T["base"]
                r[name], r[name + "_up"] = dd.mean(), int((dd > 0).sum())
            r["base_typ"] = per["typical"]["base"].mean()
            du = (U[v] - U["base"]).groupby(level="subj").mean()
            typ = du.drop("P04", errors="ignore")
            rng = np.random.default_rng(0)
            bs = [typ.to_numpy()[rng.integers(0, len(typ), len(typ))].mean() for _ in range(n_boot)]
            r["user_ci90"] = f"[{np.quantile(bs, 0.05):+.4f}, {np.quantile(bs, 0.95):+.4f}]"
            r["users_up"] = f"{int((typ > 0).sum())}/{len(typ)}"
            out.append(r)
            users += [dict(mode=mode, var=v, subj=s, d=x) for s, x in du.items()]
    return pd.DataFrame(out), pd.DataFrame(users)


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
    U.to_csv(f"{a.out}/users.csv", index=False)
    json.dump(dict(spec=a.spec, arm=a.arm, n=n, primary=PRIMARY, variants=VARIANTS, sk=SK, fam=FAM), open(f"{a.out}/config.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    f = lambda v: f"{v:+.4f}" if abs(v) < 0.5 else f"{v:.3f}"  # noqa: E731
    for mode in ("full", "early", "imb"):
        print(f"== {mode} (vs {a.arm} + SKλ, {n} draws)")
        print(S[S["mode"] == mode].drop(columns="mode").to_string(index=False, float_format=f))
    print("== per user (full)")
    print(U[U["mode"] == "full"].pivot(index="var", columns="subj", values="d").to_string(float_format=f))


if __name__ == "__main__":
    main()
