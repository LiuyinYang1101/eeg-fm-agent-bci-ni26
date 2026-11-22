#!/usr/bin/env python
"""P3 RUN NOW screens (zero GPU; pre-registered as T2-D393 / D394 / D395 before any code or number).
  --screen sk : T2-D393. M6's shape (outputs/t2_endphase/spec.json, arm M6), 12 draws: the shipped CAUSAL SKλ (10 it, n0 3, λ 0.95) vs a
                NON-causal Sinkhorn-Knopp over each whole stream (uniform class marginal, 50 it; a headroom diagnostic, not deployable).
  --screen b1 : T2-D394 (G2 B1). On outputs/t2_d369/feat_*.npz (kit protocol, one shipped seed per family): standardise + PCA 64 on the
                training users (= T2-D369's analyse), a logistic probe head; held-out kit users (val P05 P08, test P04 P09) under causal SK.
                Arms raw / mean (T2-D375's causal centring, n0 10) / dens / dens_res (the bounded target − source class log-likelihood ratio
                of shared-covariance Gaussians; target means updated causally from EARLIER windows, prior weight 10, α 1, |adj| ≤ 2), each
                density arm with the anchor's pseudo-labels and (diagnostic) the true labels of earlier windows.
  --screen b2 : T2-D395 (G2 B2). An ARM context head on the same features: logits = (W0 + Σ_r g_r(c) u_r v_rᵀ) x + b + H c, c = the causal
                mean of the stream's earlier windows (prior = the training mean, n0 10); trained on the training users' windows (each
                window's own causal context; support labels never used), rank 4, fixed a priori; arms genuine / shuffled-user prefix /
                bias-only / context-free, 5 head seeds averaged; all under causal SK.
  python t2_p3_screens.py --screen sk|b1|b2 --export E [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
SK = dict(iters=10, n0=3, lam=0.95)


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def sinkhorn_full(P, it=50):
    Q = np.asarray(P, np.float64) + 1e-12
    for _ in range(it):
        Q = Q / Q.sum(1, keepdims=True)
        Q = Q / Q.sum(0, keepdims=True) * (len(Q) / Q.shape[1])
    return Q


def decode(P, ix, d, how):
    strm, ordr = d.stream.to_numpy()[ix], d.order.to_numpy()[ix]
    if how == "causal":
        return causal_sinkhorn(P, (strm, ordr), SK["iters"], SK["n0"], SK["lam"]).argmax(1)
    out = np.empty(len(ix), int)
    for s in np.unique(strm):
        k = np.flatnonzero(strm == s)
        out[k] = sinkhorn_full(P[k]).argmax(1)
    return out


def cell_scores(pred, ix, d):
    """-> {group: mean cell BA}: typical (non-P04 cells), P04, and per user."""
    y, cells, subj = d.y.to_numpy()[ix], d.cell.to_numpy()[ix], d.subject.str[-3:].to_numpy()[ix]
    rows = [(subj[cells == c][0], bal(y[cells == c], pred[cells == c])) for c in np.unique(cells)]
    df = pd.DataFrame(rows, columns=["user", "ba"])
    out = dict(typical=float(df[df.user != "P04"].ba.mean()), P04=float(df[df.user == "P04"].ba.mean()) if (df.user == "P04").any() else np.nan)
    out.update({u: float(g.ba.mean()) for u, g in df.groupby("user")})
    return out


# ---------------------------------------------------------------- D393
def screen_sk(a, d):
    spec = json.load(open(O / "t2_endphase/spec.json"))
    M, w = spec["members"], spec["arms"]["M6"]
    res = {"causal": [], "full": []}
    for i in range(spec["n"]):
        acc = {k: [] for k in res}
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                ix, mix = None, 0.0
                for m, wm in w.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or np.array_equal(ix, ix_)
                    ix, mix = ix_, mix + wm * p.astype(np.float64)
                for how in res:
                    acc[how].append((ix, decode(mix, ix, d, how)))
        for how in res:
            ixs = np.concatenate([x for x, _ in acc[how]]); prs = np.concatenate([p for _, p in acc[how]])
            # cells are unique per (protocol, split) block; score each block's cells then pool them
            rows = []
            for ix, pr in acc[how]:
                y, cells, subj = d.y.to_numpy()[ix], d.cell.to_numpy()[ix], d.subject.str[-3:].to_numpy()[ix]
                rows += [(subj[cells == c][0], bal(y[cells == c], pr[cells == c])) for c in np.unique(cells)]
            df = pd.DataFrame(rows, columns=["user", "ba"])
            res[how].append(dict(typical=df[df.user != "P04"].ba.mean(), P04=df[df.user == "P04"].ba.mean(),
                                 new=df[df.user.isin(["P05", "P08", "P09"])].ba.mean()))
    C, F = pd.DataFrame(res["causal"]), pd.DataFrame(res["full"])
    D = F - C
    print(f"[D393] causal SKλ typical {C.typical.mean():.4f} · non-causal SK {F.typical.mean():.4f} → Δ typical {D.typical.mean():+.4f} "
          f"({int((D.typical > 0).sum())} / {len(D)}), P04 {D.P04.mean():+.4f}, new {D.new.mean():+.4f}")
    return dict(causal=C.mean().to_dict(), full=F.mean().to_dict(), delta=D.mean().to_dict(), up=int((D.typical > 0).sum()))


# ---------------------------------------------------------------- D394 / D395 shared
def load_feats(model, d):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    z = np.load(O / f"t2_d369/feat_{model}.npz", allow_pickle=True)
    F, y, tr, ho = z["F"].astype(np.float64), z["y"].astype(int), z["tr"], z["ho"]
    assert len(F) == len(d) and np.array_equal(y, d.y.to_numpy()), "dump rows must be the export's rows"
    sc = StandardScaler().fit(F[tr])
    pca = PCA(min(64, F.shape[1], len(tr) - 1), random_state=0).fit(sc.transform(F[tr]))
    Z = pca.transform(sc.transform(F))
    Z = Z / Z[tr].std(0, keepdims=True)
    return Z, y, tr, ho


def causal_ctx(Z, rows, d, mu0, n0=10):
    """per row: the causal mean of its stream's EARLIER windows (prior mu0, weight n0); rows grouped by stream, in order."""
    C = np.empty((len(rows), Z.shape[1]))
    strm, ordr = d.stream.to_numpy(), d.order.to_numpy()
    pos = {r: k for k, r in enumerate(rows)}
    for s in np.unique(strm[rows]):
        r = np.array([x for x in rows if strm[x] == s]); r = r[np.argsort(ordr[r], kind="stable")]
        cs = np.cumsum(Z[r], 0) - Z[r]                       # sum of the EARLIER windows only
        n = np.arange(len(r))[:, None]
        for k, x in enumerate(r):
            C[pos[x]] = (cs[k] + n0 * mu0) / (n[k] + n0)
    return C


def probe(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000).fit(Xtr, ytr)


# ---------------------------------------------------------------- D394
def screen_b1(a, d):
    out = {}
    subj = d.subject.to_numpy()
    for model in ("fbcnet_bboth", "shallow"):
        Z, y, tr, ho = load_feats(model, d)
        mu0 = Z[tr].mean(0)
        # raw probe
        pr_raw = probe(Z[tr], y[tr]).predict_proba(Z[ho])
        # mean (T2-D375): train rows per-user centred; eval rows causal per-stream centred incl. the current window (as t2_d375_member)
        Zc = Z.copy()
        for u in np.unique(subj[tr]):
            k = np.flatnonzero(subj == u); k = k[np.isin(k, tr)]
            Zc[k] = Z[k] - Z[k].mean(0)
        Zt = Z.copy()
        strm, ordr = d.stream.to_numpy(), d.order.to_numpy()
        for s in np.unique(strm[ho]):
            r = np.flatnonzero(strm == s); r = r[np.argsort(ordr[r], kind="stable")]
            Zt[r] = Z[r] - (np.cumsum(Z[r], 0) + 10 * mu0) / (np.arange(1, len(r) + 1)[:, None] + 10)
        cen = probe(Zc[tr], y[tr]); pr_mean = cen.predict_proba(Zt[ho])

        def density(Xs_tr, Xev, anchor, labels):
            mus = np.stack([Xs_tr[y[tr] == c].mean(0) for c in range(3)])
            R = Xs_tr - mus[y[tr]]
            Prec = np.linalg.pinv(np.cov(R.T) + 1e-3 * np.eye(R.shape[1]))
            ll = lambda x, m: -0.5 * np.einsum("ij,jk,ik->i", x - m, Prec, x - m)  # noqa: E731
            adj = np.zeros((len(ho), 3))
            pos = {r: k for k, r in enumerate(ho)}
            for s in np.unique(strm[ho]):
                r = np.array([x for x in ho if strm[x] == s]); r = r[np.argsort(ordr[r], kind="stable")]
                S, N = np.zeros((3, Xev.shape[1])), np.zeros(3)
                for x in r:
                    k = pos[x]; xv = Xev[k][None]
                    mt = (10 * mus + S) / (10 + N)[:, None]
                    adj[k] = np.clip(np.array([ll(xv, mt[c])[0] - ll(xv, mus[c])[0] for c in range(3)]), -2, 2)
                    c_ = y[x] if labels else int(np.argmax(anchor[k]))          # only THIS window's label / pseudo-label, used for LATER windows
                    S[c_] += Xev[k]; N[c_] += 1
            return adj
        arms = {"raw": pr_raw, "mean": pr_mean}
        for nm, Xs_tr, Xev, anc in (("dens", Z[tr], Z[ho], pr_raw), ("dens_res", Zc[tr], Zt[ho], pr_mean)):
            for lab in (False, True):
                A = density(Xs_tr, Xev, anc, lab)
                P = anc * np.exp(A); P /= P.sum(1, keepdims=True)
                arms[f"{nm}{'_lab' if lab else ''}"] = P
        res = {k: cell_scores(decode(P, ho, d, "causal"), ho, d) for k, P in arms.items()}
        out[model] = res
        base = res["mean"]["typical"]
        print(f"[D394] {model}: " + " · ".join(f"{k} {v['typical']:.4f} ({v['typical'] - base:+.4f} vs mean; P04 {v['P04']:.3f})" for k, v in res.items()))
    return out


# ---------------------------------------------------------------- D395
def screen_b2(a, d):
    import torch
    torch.set_num_threads(4)
    out = {}
    strm = d.stream.to_numpy()
    for model in ("fbcnet_bboth", "shallow"):
        Z, y, tr, ho = load_feats(model, d)
        mu0 = Z[tr].mean(0)
        Ctr, Cho = causal_ctx(Z, list(tr), d, mu0), causal_ctx(Z, list(ho), d, mu0)
        hs = sorted(np.unique(strm[ho]))
        shuf = {s: hs[(k + 1) % len(hs)] for k, s in enumerate(hs)}          # a different held-out stream
        Csh = np.empty_like(Cho)
        ordr = d.order.to_numpy()
        for s in hs:
            mine = np.flatnonzero(strm[ho] == s); mine = mine[np.argsort(ordr[ho][mine], kind="stable")]
            oth = np.flatnonzero(strm[ho] == shuf[s]); oth = oth[np.argsort(ordr[ho][oth], kind="stable")]
            Csh[mine] = Cho[oth[np.minimum(np.arange(len(mine)), len(oth) - 1)]]
        Xtr, ytr = torch.tensor(Z[tr], dtype=torch.float32), torch.tensor(y[tr])
        ctr = torch.tensor(Ctr - mu0, dtype=torch.float32)
        res = {}
        for kind in ("arm", "bias", "free"):
            P_g, P_s = 0.0, 0.0
            for seed in range(5):
                torch.manual_seed(seed)
                D_, R = Z.shape[1], 4
                W0 = torch.nn.Linear(D_, 3); H = torch.nn.Linear(D_, 3, bias=False); G = torch.nn.Linear(D_, R)
                U = torch.nn.Parameter(0.01 * torch.randn(R, 3)); V = torch.nn.Parameter(0.01 * torch.randn(R, D_))
                params = list(W0.parameters()) + ([*H.parameters()] if kind in ("arm", "bias") else []) + ([*G.parameters(), U, V] if kind == "arm" else [])
                opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-3)

                def logits(x, c):
                    z = W0(x)
                    if kind in ("arm", "bias"):
                        z = z + H(c)
                    if kind == "arm":
                        g = torch.tanh(G(c))                                  # (n, R)
                        z = z + torch.einsum("nr,rk,rd,nd->nk", g, U, V, x)  # Σ_r g_r(c) u_r v_rᵀ x
                    return z
                for _ in range(400):
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(logits(Xtr, ctr), ytr)
                    loss.backward(); opt.step()
                with torch.no_grad():
                    xh = torch.tensor(Z[ho], dtype=torch.float32)
                    P_g = P_g + torch.softmax(logits(xh, torch.tensor(Cho - mu0, dtype=torch.float32)), 1).numpy() / 5
                    P_s = P_s + torch.softmax(logits(xh, torch.tensor(Csh - mu0, dtype=torch.float32)), 1).numpy() / 5
            res[f"{kind}_genuine"] = cell_scores(decode(P_g, ho, d, "causal"), ho, d)
            if kind != "free":
                res[f"{kind}_shuffled"] = cell_scores(decode(P_s, ho, d, "causal"), ho, d)
        out[model] = res
        base = res["free_genuine"]["typical"]
        print(f"[D395] {model}: " + " · ".join(f"{k} {v['typical']:.4f} ({v['typical'] - base:+.4f} vs context-free)" for k, v in res.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, choices=("sk", "b1", "b2"))
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(O / "t2_p3_screens"))
    a = ap.parse_args()
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    res = {"sk": screen_sk, "b1": screen_b1, "b2": screen_b2}[a.screen](a, d)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.screen}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
