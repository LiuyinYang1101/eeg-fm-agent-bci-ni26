"""T2 FAST LAB ideas (scripts/local/t2_lab.py). An idea = (name, fn, params); fn(ctx, **params) returns any of
  {"P": parts}  - the four family parts per (fold, draw) before decoding (default: W19's),
  {"dec": {family: (iters, n0, lam)}}  - the causal Sinkhorn per family (default 10 / 3 / 0.95),
  {"W": weights}  - the family weights.
Every idea is label-free at test time (labels only through the stream's own P1 teacher posteriors, as W19's refits); no oracle input.
`registry()` lists every idea a confirm may name; `batch1()` is the first screen.
"""
import numpy as np

_FM = {}


def _conf(post, frac):
    """the confident `frac` (ceil) of each predicted class (frac 0.5 = W19's conf_half)."""
    ya, cf = post.argmax(1), post.max(1)
    keep = np.zeros(len(ya), bool)
    for c in np.unique(ya):
        jj = np.flatnonzero(ya == c)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) * frac))]]] = True
    return ya, keep


def _member(ctx, f, d, path, lam=1.0, frac=0.5, points=None, mode="side", centre=None, temp=1.0, bias_only=False, centre_n0=10.0,
            proto=0.0, hl="refit", lam_hl=None, proto_scope="all", proto_frac=None, proto_space="cos", lda_shrink=0.5):
    """one FM member's probabilities on IX[f] for draw d. mode 'none' = no refit; 'side' = W19 / D553 (the side head re-fitted from the
    trained head, pull lam); 'joint' = main + side heads re-fitted on [features, z] (D555); bias_only = only the bias moves (W fixed by a
    huge pull on W). centre = None | 'mean' | 'whiten': per-stream causal centring (/ diag whitening) of the pooled features before the
    main head (prior = the other folds' feature mean, n0 10). temp = the member logit temperature (after any refit)."""
    points = tuple(points) if points else None
    key = (str(path), d, lam, frac, points, mode, centre, temp, bias_only, centre_n0, proto, hl, lam_hl, proto_scope, proto_frac, proto_space, lda_shrink)
    # batch 3 (T2-D560 variants): lam_hl = the head-less member's refit pull (None = lam); proto_scope 'all' | 'hl' (the head-less member only)
    # | 'headed'; proto_frac = the prototypes' own confident fraction (None = frac, the refit's rows)
    if key in _FM:
        return _FM[key]
    dm = ctx.dumps(path)
    if dm.get("p") is not None:                          # no dumps: the member's stored probabilities (no head idea applies)
        _FM[key] = _tp(dm["p"], temp) if temp != 1.0 else dm["p"]
        return _FM[key]
    pts = tuple(points or ctx.CKS)
    feat = dm["feat"]
    if feat is None and (mode == "joint" or centre is not None or proto):
        raise ValueError("this idea needs the pooled features (sets A / B)")
    if centre is not None:
        feat = _centre(ctx, f, path, feat, centre, n0=centre_n0)
    L = feat @ dm["Wm"].T + dm["bm"] if feat is not None else dm["L"]
    has_s = "Ws" in dm
    lg = L + (dm["z"] @ dm["Ws"].T + dm["bs"] if has_s else 0.0)
    lg0 = lg.copy()                                      # the unadapted logits (every refit starts from them)
    # hl (the head-less member in mode 'side', T2-D560's decomposition): 'refit' = its main head re-fitted at lam (the lab's behaviour when
    # proto is on, = what D560 screened and confirmed), 'skip' = no refit (prototypes only), 'only' = the refit without prototypes
    if (mode != "none" and (has_s or mode in ("joint", "biaslog"))) or proto or (hl == "only" and not has_s):
        pos, post = ctx.teacher(f, d)
        rows = dm["rows"]
        st, od = ctx.stream[rows], ctx.order[rows]
        for s in np.unique(st):
            j = np.flatnonzero(st == s)
            j = j[np.argsort(od[j], kind="stable")]
            if len(j) <= pts[0]:
                continue
            for k, c in enumerate(pts):
                if len(j) <= c:
                    break
                e = pts[k + 1] if k + 1 < len(pts) else len(j)
                a = j[:c]
                pp = np.stack([post[c][pos[int(r)]] for r in rows[a]])
                fin = np.isfinite(pp).all(1)
                ya, keep = _conf(pp[fin], frac)
                sel, ys = a[fin][keep], ya[keep].astype(np.int64)
                use_p = bool(proto) and (proto_scope == "all" or (proto_scope == "hl") == (not has_s))
                if use_p and len(sel):              # per-stream class prototypes of the confident windows' features (cosine, x 10)
                    fz = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
                    if proto_frac is not None:
                        yp, kp = _conf(pp[fin], proto_frac)
                        selp, ysp = a[fin][kp], yp[kp].astype(np.int64)
                    else:
                        selp, ysp = sel, ys
                    mu = np.stack([fz[selp][ysp == q].mean(0) if (ysp == q).any() else np.zeros(fz.shape[1]) for q in range(3)])
                    projA = None
                    if proto_space == "lda":            # T2-D562 (2'): a shrinkage LDA on the stream's confident rows (Woodbury), prototypes there
                        Xc = fz[selp] - mu[ysp]
                        n_, D_ = Xc.shape
                        lam_s = lda_shrink * float((Xc ** 2).sum()) / n_ / D_ + 1e-9
                        beta_ = (1.0 - lda_shrink) / n_
                        Dm = (mu - mu.mean(0)).T
                        Mn = (lam_s / beta_) * np.eye(n_) + Xc @ Xc.T
                        projA = (Dm - Xc.T @ np.linalg.solve(Mn, Xc @ Dm)) / lam_s
                        mu = mu @ projA
                    mu = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
                    base = lg[j[c:e]] if mode == "none" else None
                if mode == "biaslog":                # T2 lab: a per-stream class-bias refit on the member's current logits (no features)
                    _, b_, _ = ctx.fit_head(lg0[sel], np.zeros((len(sel), 1)), ys, np.zeros((3, 1)), np.zeros(3), lam)
                    lg[j[c:e]] = lg0[j[c:e]] + b_
                    continue
                if mode == "side" and has_s:
                    W_, b_, _ = ctx.fit_head(L[sel], dm["z"][sel], ys, dm["Ws"], dm["bs"], lam)
                    lg[j[c:e]] = L[j[c:e]] + dm["z"][j[c:e]] @ W_.T + b_
                elif mode == "none":
                    lg[j[c:e]] = base if proto else lg[j[c:e]]
                elif mode == "side" and not has_s and hl == "skip":
                    lg[j[c:e]] = lg0[j[c:e]]
                else:
                    X = np.concatenate([feat, dm["z"]], 1) if has_s else feat
                    W0 = np.concatenate([dm["Wm"], dm["Ws"]], 1) if has_s else dm["Wm"]
                    b0 = dm["bm"] + dm["bs"] if has_s else dm["bm"]
                    if bias_only:
                        W_, b_, _ = ctx.fit_head(X[sel] @ W0.T, np.zeros((len(sel), 1)), ys, np.zeros((W0.shape[0], 1)), b0 * 0.0, lam)
                        lg[j[c:e]] = X[j[c:e]] @ W0.T + b0 + b_
                    else:
                        lam_ = lam_hl if (lam_hl is not None and not has_s and mode == "side") else lam
                        W_, b_, _ = ctx.fit_head(np.zeros((len(sel), W0.shape[0])), X[sel], ys, W0, b0, lam_)
                        lg[j[c:e]] = X[j[c:e]] @ W_.T + b_
                if use_p and len(sel):
                    zq = fz[j[c:e]] if projA is None else fz[j[c:e]] @ projA
                    if projA is not None:
                        zq = zq / (np.linalg.norm(zq, axis=1, keepdims=True) + 1e-9)
                    lg[j[c:e]] = lg[j[c:e]] + proto * 10.0 * (zq @ mu.T)
    p = ctx.softmax(lg / temp)
    _FM[key] = p
    return p


_MU = {}


def _centre(ctx, f, path, feat, how, n0=10.0):   # n0 = the prior's weight in windows
    """causal per-stream centring (/ diag whitening) of the pooled features; prior = the mean / sd of the OTHER folds' features of the same
    member slot (cross-fitted, label-free)."""
    slot = [i for i, p in enumerate(ctx.fm_paths[f]) if str(p) == str(path)][0]
    if (slot, f) not in _MU:
        others = np.concatenate([ctx.dumps(ctx.fm_paths[g][slot])["feat"] for g in ctx.folds if g != f])
        _MU[(slot, f)] = (others.mean(0), others.std(0) + 1e-6)
    mu0, sd0 = _MU[(slot, f)]
    rows = ctx.dumps(path)["rows"]
    st, od = ctx.stream[rows], ctx.order[rows]
    out = np.empty_like(feat)
    for s in np.unique(st):
        j = np.flatnonzero(st == s)
        j = j[np.argsort(od[j], kind="stable")]
        x = feat[j]
        n = n0 + np.arange(1, len(j) + 1)[:, None]
        m = (n0 * mu0 + np.cumsum(x, 0)) / n
        if how == "whiten":
            v = (n0 * (sd0 ** 2 + mu0 ** 2) + np.cumsum(x * x, 0)) / n - m ** 2
            out[j] = (x - m) / np.sqrt(np.maximum(v, 1e-6)) * sd0 + mu0
        else:
            out[j] = x - m + mu0
    return out


def _fm_part(ctx, drop=(), **kw):
    P = {}
    for (f, d), parts in ctx.P.items():
        ps = [_member(ctx, f, d, p, **kw) for i, p in enumerate(ctx.fm_paths[f]) if i not in drop]
        P[(f, d)] = dict(parts, fm=np.mean(ps, 0))
    return P


def fm(ctx, **kw):
    return {"P": _fm_part(ctx, **kw)}


def fm_temp_part(ctx, T=1.0):
    return {"P": {k: dict(v, fm=_tp(v["fm"], T)) for k, v in ctx.P.items()}}


def _tp(p, T):
    q = np.exp(np.log(np.maximum(p, 1e-30)) / T)
    return q / q.sum(1, keepdims=True)


def dec(ctx, iters=10, n0=3, lam=0.95, fams=("fm", "fbc", "sh", "ifn")):
    return {"dec": {f_: (iters, n0, lam) for f_ in fams}}


def weights(ctx, fm_w=0.35):
    W = dict(ctx.W)
    rest = sum(v for k, v in W.items() if k != "fm")
    W = {k: (fm_w if k == "fm" else v * (1.0 - fm_w) / rest) for k, v in W.items()}
    return {"W": W}


def lam_mix(ctx, lams=(0.3, 1.0, 3.0)):
    P = {}
    for (f, d), parts in ctx.P.items():
        ps = [np.mean([_member(ctx, f, d, p, lam=l_) for l_ in lams], 0) for p in ctx.fm_paths[f]]
        P[(f, d)] = dict(parts, fm=np.mean(ps, 0))
    return {"P": P}


def registry():
    return batch1() + batch1_nofeat() + batch2() + batch1b() + batch_d560dec() + batch_d560c() + batch3() + batch4() + batch5() + batch_d563() + batch_d563c() + batch_d566() + batch_d566c() + batch_d572() + batch_d575() + batch_d579() + batch_d581() + batch_d584() + batch_d585() + batch_d591() + batch_d655() + batch_d658()


def batch1_nofeat():
    """batch1 without the ideas that need pooled features (set A0)."""
    return [b for b in batch1() if b[0] not in ("joint", "bias_only", "centre", "centre_whiten", "centre_joint")]


def batch1():
    B = [("w19_reimpl", fm, {}),                                         # sanity: must read ~0 vs W19
         ("side_lam", fm, {"lam": 0.3}), ("side_lam", fm, {"lam": 3.0}), ("side_lam", fm, {"lam": 10.0}),
         ("side_conf", fm, {"frac": 0.3}), ("side_conf", fm, {"frac": 0.7}),
         ("side_points", fm, {"points": (60,)}), ("side_points", fm, {"points": (100, 150)}),
         ("joint", fm, {"mode": "joint", "lam": 1.0}), ("joint", fm, {"mode": "joint", "lam": 10.0}),
         ("bias_only", fm, {"mode": "joint", "bias_only": True, "lam": 1.0}),
         ("centre", fm, {"centre": "mean"}), ("centre_whiten", fm, {"centre": "whiten"}),
         ("centre_joint", fm, {"centre": "mean", "mode": "joint", "lam": 10.0}),
         ("fm_temp", fm_temp_part, {"T": 0.8}), ("fm_temp", fm_temp_part, {"T": 1.25}),
         ("member_temp", fm, {"temp": 1.5}),
         ("dec", dec, {"lam": 0.9}), ("dec", dec, {"lam": 0.98}), ("dec", dec, {"n0": 6}), ("dec", dec, {"iters": 3}),
         ("w_fm", weights, {"fm_w": 0.40}), ("w_fm", weights, {"fm_w": 0.30}),
         ("drop827", fm, {"drop": (1,)}),
         ("lam_mix", lam_mix, {})]
    return B


def batch2():
    """feature-level mechanisms (set A / B: D555's members with pooled features)."""
    return [("w19_reimpl", fm, {}),
            ("joint", fm, {"mode": "joint", "lam": 1.0}), ("joint", fm, {"mode": "joint", "lam": 3.0}),
            ("joint", fm, {"mode": "joint", "lam": 10.0}), ("joint", fm, {"mode": "joint", "lam": 30.0}),
            ("bias_only", fm, {"mode": "joint", "bias_only": True, "lam": 0.3}), ("bias_only", fm, {"mode": "joint", "bias_only": True, "lam": 1.0}),
            ("bias_only", fm, {"mode": "joint", "bias_only": True, "lam": 3.0}),
            ("centre", fm, {"centre": "mean"}), ("centre", fm, {"centre": "mean", "mode": "none"}),
            ("centre", fm, {"centre": "mean", "centre_n0": 3.0}), ("centre", fm, {"centre": "mean", "centre_n0": 30.0}),
            ("centre_whiten", fm, {"centre": "whiten"}), ("centre_whiten", fm, {"centre": "whiten", "mode": "none"}),
            ("centre_joint", fm, {"centre": "mean", "mode": "joint", "lam": 3.0}), ("centre_joint", fm, {"centre": "mean", "mode": "joint", "lam": 10.0}),
            ("proto", fm, {"proto": 0.5}), ("proto", fm, {"proto": 1.0}), ("proto", fm, {"proto": 0.5, "mode": "none"}),
            ("joint_points", fm, {"mode": "joint", "lam": 10.0, "points": (100, 150)}), ("joint_conf", fm, {"mode": "joint", "lam": 10.0, "frac": 0.7}),
            ("joint_drop827", fm, {"mode": "joint", "lam": 10.0, "drop": (1,)}), ("centre_drop827", fm, {"centre": "mean", "drop": (1,)})]


# ---------------- logit-level mechanisms (no features: set A0 too) ----------------
def _streams(ctx, f):
    rows = ctx.IX[f]
    st, od = ctx.stream[rows], ctx.order[rows]
    for s_ in np.unique(st):
        j = np.flatnonzero(st == s_)
        yield j[np.argsort(od[j], kind="stable")]


def bias_part(ctx, lam=1.0, fams=("fm", "fbc", "sh", "ifn"), frac=0.5):
    """per-stream CLASS-BIAS adaptation of each family part: log p + b, b re-fitted (pull lam to 0) on the stream's confident windows (the P1
    teacher's labels at 60 / 100 / 150, as W19's refits), applied to the windows after each refit point."""
    P = {}
    for (f, d), parts in ctx.P.items():
        pos, post = ctx.teacher(f, d)
        rows = ctx.IX[f]
        newp = dict(parts)
        for fam in fams:
            lg = np.log(np.maximum(parts[fam], 1e-30)).copy()
            base = lg.copy()
            for j in _streams(ctx, f):
                for k, c in enumerate(ctx.CKS):
                    if len(j) <= c:
                        break
                    e = ctx.CKS[k + 1] if k + 1 < len(ctx.CKS) else len(j)
                    a = j[:c]
                    pp = np.stack([post[c][pos[int(r)]] for r in rows[a]])
                    fin = np.isfinite(pp).all(1)
                    ya, keep = _conf(pp[fin], frac)
                    sel, ys = a[fin][keep], ya[keep].astype(np.int64)
                    _, b_, _ = ctx.fit_head(base[sel], np.zeros((len(sel), 1)), ys, np.zeros((3, 1)), np.zeros(3), lam)
                    lg[j[c:e]] = base[j[c:e]] + b_
            newp[fam] = ctx.softmax(lg)
        P[(f, d)] = newp
    return {"P": P}


def famweight(ctx, beta=5.0):
    """per-stream ONLINE family weights: each family's agreement with the running mixture's argmax over the stream's delivered prefix
    (label-free), w = W_fam * exp(beta * (agree_fam - mean agree)), renormalised; window t uses windows < t."""
    def post(f, d, key, D_, Wt, dcd):
        fams = list(Wt)
        base = sum(Wt[q] * D_[q] for q in fams)
        out = np.empty_like(base)
        st, od = key
        for s_ in np.unique(st):
            j = np.flatnonzero(st == s_)
            j = j[np.argsort(od[j], kind="stable")]
            am = base[j].argmax(1)
            ag = np.stack([np.concatenate([[0.0], np.cumsum(D_[q][j].argmax(1) == am)[:-1]]) for q in fams], 1)
            n = np.arange(len(j))[:, None]
            rate = np.where(n > 0, ag / np.maximum(n, 1), 0.0)
            w = np.array([Wt[q] for q in fams])[None] * np.exp(beta * (rate - rate.mean(1, keepdims=True)))
            w = w / w.sum(1, keepdims=True) * sum(Wt.values())
            out[j] = sum(w[:, i:i + 1] * D_[q][j] for i, q in enumerate(fams))
        return (1 - ctx.W_DC) * out + ctx.W_DC * dcd
    return {"post": post}


def enttemp(ctx, fam="fm", lo=0.5, hi=2.0):
    """per-stream temperature by entropy for one family part: T_t = (the stream's running mean entropy) / (the cross-fitted mean entropy of
    the other folds' strangers), clipped to [lo, hi]; the part's logits / T_t (causal: windows < t)."""
    P = {}
    ent = lambda p: -(p * np.log(np.maximum(p, 1e-30))).sum(1)  # noqa: E731
    ref = {}
    for (f, d), parts in ctx.P.items():
        ref.setdefault(d, {})[f] = ent(parts[fam]).mean()
    for (f, d), parts in ctx.P.items():
        tgt = np.mean([v for g, v in ref[d].items() if g != f])
        lg = np.log(np.maximum(parts[fam], 1e-30))
        out = lg.copy()
        for j in _streams(ctx, f):
            h = ent(parts[fam][j])
            run = np.concatenate([[tgt], np.cumsum(h)[:-1] / np.arange(1, len(j))])
            T = np.clip(run / tgt, lo, hi)
            out[j] = lg[j] / T[:, None]
        P[(f, d)] = dict(parts, **{fam: ctx.softmax(out)})
    return {"P": P}


def abstain(ctx, gamma=0.5):
    """disagreement-gated abstain-to-prior: where the FM part and the CNN mixture disagree on the argmax, shrink the final probabilities
    toward the stream's running mean of the final probabilities over earlier windows (causal) by gamma."""
    def post(f, d, key, D_, Wt, dcd):
        mx = (1 - ctx.W_DC) * sum(Wt[q] * D_[q] for q in Wt) + ctx.W_DC * dcd
        cnn = sum(Wt[q] * D_[q] for q in Wt if q != "fm")
        dis = D_["fm"].argmax(1) != cnn.argmax(1)
        out = mx.copy()
        st, od = key
        for s_ in np.unique(st):
            j = np.flatnonzero(st == s_)
            j = j[np.argsort(od[j], kind="stable")]
            prior = np.vstack([np.full((1, mx.shape[1]), 1.0 / mx.shape[1]), np.cumsum(mx[j], 0)[:-1] / np.arange(1, len(j))[:, None]])
            g = gamma * dis[j][:, None]
            out[j] = (1 - g) * mx[j] + g * prior
        return out
    return {"post": post}


def batch1b():
    """the manager's logit-level mechanisms (set A0)."""
    return [("w19_reimpl", fm, {}),
            ("bias_fm", fm, {"mode": "biaslog", "lam": 0.3}), ("bias_fm", fm, {"mode": "biaslog", "lam": 1.0}), ("bias_fm", fm, {"mode": "biaslog", "lam": 3.0}),
            ("bias_part", bias_part, {"lam": 0.3}), ("bias_part", bias_part, {"lam": 1.0}), ("bias_part", bias_part, {"lam": 3.0}),
            ("bias_part", bias_part, {"lam": 1.0, "fams": ("fm",)}), ("bias_part", bias_part, {"lam": 1.0, "fams": ("fbc", "sh", "ifn")}),
            ("famweight", famweight, {"beta": 2.0}), ("famweight", famweight, {"beta": 5.0}), ("famweight", famweight, {"beta": 10.0}),
            ("enttemp", enttemp, {"fam": "fm"}), ("enttemp", enttemp, {"fam": "fbc"}), ("enttemp", enttemp, {"fam": "fm", "lo": 0.8, "hi": 1.25}),
            ("abstain", abstain, {"gamma": 0.3}), ("abstain", abstain, {"gamma": 0.6}), ("abstain", abstain, {"gamma": 1.0})]


def batch_d560dec():
    """T2-D560's DESCRIPTIVE decomposition of proto 0.5 (never re-selects): prototypes without the head-less refit, the refit alone."""
    return [("w19_reimpl", fm, {}), ("proto", fm, {"proto": 0.5}), ("proto_hlskip", fm, {"proto": 0.5, "hl": "skip"}),
            ("hl_refit_only", fm, {"hl": "only"})]


def batch_d560c():
    """T2-D560's registered THIRD read on set C (831-833, fresh FM fits): the B passer only, judged by the second-read rule (ten > 0, >= 2 / 3
    draws > 0, no stranger < -0.020, returning >= -0.003) = the lab's 'guards'. Run with --log d560_C.jsonl."""
    return [("w19_reimpl", fm, {}), ("proto", fm, {"proto": 0.5})]


def batch3():
    """T2-D560 variants (set A, K = 3 to B then C): the MI refit's pull, the prototypes' scope / weight / rows."""
    P = {"proto": 0.5}
    return [("w19_reimpl", fm, {}), ("proto", fm, dict(P)),
            ("p_lamhl", fm, dict(P, lam_hl=0.3)), ("p_lamhl", fm, dict(P, lam_hl=3.0)), ("p_lamhl", fm, dict(P, lam_hl=10.0)),
            ("p_scope", fm, dict(P, proto_scope="hl")), ("p_scope", fm, dict(P, proto_scope="headed")),
            ("p_w", fm, {"proto": 0.25}), ("p_w", fm, {"proto": 0.75}),
            ("p_frac", fm, dict(P, proto_frac=0.3)), ("p_frac", fm, dict(P, proto_frac=0.7)),
            ("hl_only", fm, {"hl": "only", "lam_hl": 0.3}), ("hl_only", fm, {"hl": "only", "lam_hl": 3.0}),
            ("p_scope_lamhl", fm, dict(P, proto_scope="hl", lam_hl=3.0)), ("p_w_lamhl", fm, {"proto": 0.25, "lam_hl": 3.0})]


def batch4():
    """W21 candidates ON TOP OF W20 (run with --base w20; every FM-part idea keeps proto 0.5): the stack's knobs re-tuned after the FM part moved."""
    P = {"proto": 0.5}
    return [("w20_reimpl", fm, dict(P)),
            ("fm_temp", fm_temp_part, {"T": 0.8}), ("fm_temp", fm_temp_part, {"T": 1.25}),
            ("w_fm", weights, {"fm_w": 0.40}), ("w_fm", weights, {"fm_w": 0.45}), ("w_fm", weights, {"fm_w": 0.30}),
            ("dec", dec, {"lam": 0.9}), ("dec", dec, {"lam": 0.98}), ("dec", dec, {"n0": 6}), ("dec", dec, {"iters": 3}),
            ("side_lam", fm, dict(P, lam=0.3)), ("side_lam", fm, dict(P, lam=3.0)),
            ("side_points", fm, dict(P, points=(100, 150))), ("side_conf", fm, dict(P, frac=0.7)),
            ("drop827", fm, dict(P, drop=(1,))), ("member_temp", fm, dict(P, temp=1.25)), ("member_temp", fm, dict(P, temp=0.8)),
            ("lam_mix", lam_mix, {})]


def znorm(ctx, fams=("fm", "fbc", "sh", "ifn"), sd=False, n0=10.0):
    """T2-D562 (1'): per-stream causal normalisation of each family's raw class log-scores (label-free): x - mu_t + m0 (mean), or
    (x - mu_t) / sd_t * s0 + m0 (mean + sd); mu_t / sd_t = the stream's windows < t with the prior m0 / s0 (the OTHER folds' mean / sd) at weight n0."""
    lp = {(f, d, q): np.log(np.maximum(parts[q], 1e-30)) for (f, d), parts in ctx.P.items() for q in fams}
    P = {}
    for (f, d), parts in ctx.P.items():
        newp = dict(parts)
        for q in fams:
            oth = [lp[(g, d, q)] for g in ctx.folds if g != f]
            m0 = np.mean([o.mean(0) for o in oth], 0)
            s0 = np.mean([o.std(0) for o in oth], 0) + 1e-6
            x = lp[(f, d, q)]
            out = x.copy()
            for j in _streams(ctx, f):
                xs = x[j]
                n = np.arange(len(j))[:, None].astype(np.float64)
                c1 = np.vstack([np.zeros((1, xs.shape[1])), np.cumsum(xs, 0)[:-1]])
                mu = (n0 * m0 + c1) / (n0 + n)
                if sd:
                    c2 = np.vstack([np.zeros((1, xs.shape[1])), np.cumsum(xs ** 2, 0)[:-1]])
                    var = np.maximum((n0 * (s0 ** 2 + m0 ** 2) + c2) / (n0 + n) - mu ** 2, 1e-8)
                    out[j] = (xs - mu) / np.sqrt(var) * s0 + m0
                else:
                    out[j] = xs - mu + m0
            newp[q] = ctx.softmax(out)
        P[(f, d)] = newp
    return {"P": P}


def lrfuse(ctx, C=1.0):
    """T2-D562 (4): multinomial LR on the decoded family log-probs + the DC part (per-class offsets = its intercepts), CROSS-FITTED per scored
    fold on the OTHER folds' stranger rows, folds whose stranger appears anywhere in the scored fold excluded; deployable as constants."""
    from sklearn.linear_model import LogisticRegression
    cap = {}

    def capture(f, d, key, D_, Wt, dcd):
        cap[(f, d)] = ({q: v.copy() for q, v in D_.items()}, dcd.copy())
        return (1 - ctx.W_DC) * sum(Wt[q] * D_[q] for q in Wt) + ctx.W_DC * dcd
    ctx.lab_mix(ctx.P, ctx.W, post=capture)
    fams = list(ctx.W)

    def feats(D_, dcd):
        return np.log(np.maximum(np.hstack([D_[q] for q in fams] + [dcd]), 1e-30))
    models = {}
    for (f, d) in cap:
        excl = set(np.unique(ctx.U[ctx.IX[f]]).tolist())
        Xs, ys = [], []
        for g in ctx.folds:
            if g == f or ctx.TRU[g] in excl:
                continue
            ix = ctx.IX[g]
            m = ctx.U[ix] == ctx.TRU[g]
            D_, dcd = cap[(g, d)]
            Xs.append(feats(D_, dcd)[m])
            ys.append(ctx.Y[ix][m])
        models[(f, d)] = LogisticRegression(C=C, max_iter=3000).fit(np.vstack(Xs), np.concatenate(ys))

    def post(f, d, key, D_, Wt, dcd):
        lr = models[(f, d)]
        assert list(lr.classes_) == [0, 1, 2], lr.classes_
        return lr.predict_proba(feats(D_, dcd))
    return {"post": post}


def batch5():
    """T2-D562: the speaker-verification back-end on top of W20 (run with --base w20)."""
    P = {"proto": 0.5}
    return [("w20_reimpl", fm, dict(P)),
            ("znorm_mean", znorm, {}), ("znorm_mean_fm", znorm, {"fams": ("fm",)}), ("znorm_meansd", znorm, {"sd": True}),
            ("lda_proto", fm, dict(proto=0.5, proto_space="lda", lda_shrink=0.5)), ("lda_proto", fm, dict(proto=0.5, proto_space="lda", lda_shrink=0.9)),
            ("lda_proto", fm, dict(proto=1.0, proto_space="lda", lda_shrink=0.5)), ("lda_proto", fm, dict(proto=1.0, proto_space="lda", lda_shrink=0.9)),
            ("lrfuse", lrfuse, {"C": 0.1}), ("lrfuse", lrfuse, {"C": 1.0})]


# ---------------- T2-D563: the in-stream kNN datastore (the fresh-eyes hunt's ideas 1 + 2) ----------------
_KF = {}


def _knn_feats(ctx, f, space):
    """fold f's per-window vectors on IX[f]: 'fm' = the members' L2-normalised pooled features concatenated; 'side' = the 82-d window means."""
    if (f, space) not in _KF:
        if space == "fm":
            fs = [ctx.dumps(p)["feat"] for p in ctx.fm_paths[f]]
            assert all(x is not None for x in fs), "D563 fm space needs the pooled features (sets A / B / C)"
            _KF[(f, space)] = np.hstack([x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9) for x in fs])
        else:
            zs = [ctx.dumps(p).get("z") for p in ctx.fm_paths[f]]
            _KF[(f, space)] = next(z for z in zs if z is not None)
    return _KF[(f, space)]


def _knn_stream(ctx, X, P, space, k, t0, floor):
    """the causal vote on one stream (rows in acquisition order): (n, 3) p_kNN, nan where no term. Labels = P (the pre-kNN outputs)."""
    n = len(X)
    out = np.full((n, P.shape[1]), np.nan)
    if space == "fm":
        G = X @ X.T                                      # centred cosine from the Gram matrix: c_t = mean of the windows <= t
    for t in range(t0, n):
        if space == "fm":
            Gt = G[:t + 1, :t + 1]
            xc = Gt.mean(1)                              # x_i . c_t
            cc = xc.mean()                               # c_t . c_t
            nrm = np.sqrt(np.maximum(np.diag(Gt) - 2 * xc + cc, 1e-18))
            sim = (Gt[:t, t] - xc[:t] - xc[t] + cc) / (nrm[:t] * nrm[t])
        else:
            Z = X[:t + 1]
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            sim = Z[:t] @ Z[t]
        pp = P[:t]
        _, keep = ctx.conf_half(pp)
        ks = np.flatnonzero(keep)
        if floor is not None:
            ks = ks[sim[ks] > floor]
        if len(ks) == 0:
            continue
        nn = ks[np.argsort(-sim[ks], kind="stable")[:k]]
        v = (np.maximum(sim[nn], 0)[:, None] * pp[nn]).sum(0) + 1e-3
        out[t] = v / v.sum()
    return out


def knn(ctx, beta=0.5, k=10, t0=30, space="fm", floor=None):
    """T2-D563: p_final ∝ p_W20 · p_kNN^beta per window (both = the fm and side votes at beta / 2 each); labels = W20's own pre-kNN outputs."""
    spaces = ["fm", "side"] if space == "both" else [space]
    b_ = beta / len(spaces)

    def post(f, d, key, D_, Wt, dcd):
        base = (1 - ctx.W_DC) * sum(Wt[q] * D_[q] for q in Wt) + ctx.W_DC * dcd
        st, od = key
        lg = np.log(np.maximum(base, 1e-30))
        for s_ in np.unique(st):
            j = np.flatnonzero(st == s_)
            j = j[np.argsort(od[j], kind="stable")]
            for sp in spaces:
                pk = _knn_stream(ctx, _knn_feats(ctx, f, sp)[j], base[j], sp, k, t0, floor)
                m = np.isfinite(pk).all(1)
                lg[j[m]] = lg[j[m]] + b_ * np.log(pk[m])
        return ctx.softmax(lg)
    return {"post": post}


def batch_d563():
    """T2-D563's pre-declared grid (the reviewer's), run with --base w20."""
    return [("w20_reimpl", fm, {"proto": 0.5}),
            ("knn_fm", knn, {"beta": 0.25}), ("knn_fm", knn, {"beta": 0.5}), ("knn_fm", knn, {"beta": 0.5, "k": 20}),
            ("knn_fm", knn, {"beta": 0.5, "floor": 0.2}),
            ("knn_side", knn, {"beta": 0.5, "space": "side"}), ("knn_side", knn, {"beta": 0.25, "space": "side"}),
            ("knn_both", knn, {"beta": 0.5, "space": "both"})]


def batch_d563c():
    """T2-D563's registered C read (run with --set C --base w20 --log d563_C.jsonl): the B passer only, the second-read rule."""
    return [("w20_reimpl", fm, {"proto": 0.5}), ("knn_both", knn, {"beta": 0.5, "space": "both"})]


# ---------------- T2-D566: the feedback-faithful teacher (the shipped confR: the FM rows = the arm's own adapted FM part) ----------------
TW = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10, "fm": 0.35}       # = t2_d512_teacher.py's W = the template's fam_w


def _prefix_decode(P, start, iters=10, n0=3, lam=0.95):
    """= scripts/local/t2_d512_teacher.py prefix_decode (T2-D511), verbatim."""
    K = P.shape[1]
    M = np.vstack([np.tile(np.asarray(start, np.float64), (int(n0), 1)), P.astype(np.float64)])
    w = float(lam) ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
    for _ in range(int(iters)):
        M = M * (w.sum() / K) / (M * w[:, None]).sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
    return M[int(n0):]


_TFC = {}


def _tfile(ctx, f, d):
    """the teacher file's raw routed CNN rows (R_fbc / R_sh / R_ifn) and their start vectors, re-ordered to IX[f]."""
    if (f, d) not in _TFC:
        z = np.load(ctx.TEACH / f"lc9_{f}__s{d}.npz")
        pos = {int(r): k for k, r in enumerate(z["rows"])}
        sel = np.array([pos[int(r)] for r in ctx.IX[f]])
        _TFC[(f, d)] = ({k: z[f"R_{k}"].astype(np.float64)[sel] for k in ("fbc", "sh", "ifn")},
                        {k: np.asarray(z[f"vec_{k}"], np.float64) for k in ("fbc", "sh", "ifn")}, z)
    return _TFC[(f, d)]


_D498 = {}


def _d498_fm(ctx, f):
    """the FIXED teacher's FM rows (D498's six members, prob_last mean) on IX[f] = what the shipped teacher_p1 files pooled (the check)."""
    if f not in _D498:
        import glob
        ps = []
        for a, seeds in (("stl", (760, 761, 762)), ("stlcar", (763, 764, 765))):
            for s in seeds:
                fs = glob.glob(f"{ctx.S_DIR}/ni26-t2p-d498-{a}_s{s}-[0-9]*/runs/lc9_{f}__fm-st_large-mean__sz__n00__*__s{s}.npz")
                assert len(fs) == 1, (a, s, f, fs)
                z = np.load(fs[0])
                pos = {int(r): k for k, r in enumerate(z["test_idx"])}
                p = z["prob_last"].astype(np.float64)
                ps.append((p / p.sum(1, keepdims=True))[np.array([pos[int(r)] for r in ctx.IX[f]])])
        _D498[f] = np.mean(ps, 0)
    return _D498[f]


_FMST_REC = {}


def fmstart_hook(kind, f, d, v):
    """T2-D642 (Track3 Q2): NI26_LAB_FMSTART = 'dump:<npz>' records, 'freeze:<npz>' REPLACES the FM start vector of (kind, fold, draw): kind
    '<lab mode>_fb' = fm_feedback's teacher-prefix start vfm (fbteach_fb in fbteach, mix_fb inside a mix's w20_fb idea), 'mix_final' = the saved
    mix's final-decode FM start (t2_lab.lab_mix, the final call only). Absent = unchanged (bitwise). A dump merges into the file and asserts
    equality with a value already recorded under the same key."""
    import os as _o
    spec = _o.environ.get("NI26_LAB_FMSTART", "")
    if not spec:
        return v
    mode, path = spec.split(":", 1)
    key = f"{kind}_{f}_{d}"
    if mode == "dump":
        if not _FMST_REC and _o.path.exists(path):
            _FMST_REC.update({k: np.asarray(x) for k, x in np.load(path).items()})
        vv = np.asarray(v, np.float64)
        assert key not in _FMST_REC or np.array_equal(_FMST_REC[key], vv), ("NI26_LAB_FMSTART dump: a different value for", key)
        _FMST_REC[key] = vv
        np.savez(path, **_FMST_REC)
        return v
    assert mode == "freeze", spec
    z = np.load(path)
    assert key in z.files, ("NI26_LAB_FMSTART freeze: no recorded value for", key)
    return np.asarray(z[key], np.float64)


def _fm_start(ctx, f, fmP):
    """the FM corrector's start: the OTHER folds' stranger mean of the FM rows (the teacher files' vec construction)."""
    v = np.mean([fmP[g][ctx.U[ctx.IX[g]] == ctx.TRU[g]].mean(0) for g in ctx.folds if g != f], 0)
    return v / v.sum()


def fm_feedback(ctx, proto=0.0, lam=1.0, frac=0.5, fm_rows="own", count=None, record=None, cks=None, fm_lp=None, soft_proto=False, wref=False,
                map_tau=None, tw=None, rescore=None):
    """T2-D566: the FM part with every refit (D553 side heads; + D560's prototypes / head-less refit when proto > 0) labelled by the SHIPPED
    confR teacher, built sequentially per stream: at refit c, Σ_k W_k prefix_decode(rows_k <= c) with the raw routed CNN rows and the FM rows =
    the arm's own CURRENT FM part (fm_rows 'own') or D498's fixed members (fm_rows 'd498' = the teacher_p1 files' construction: the check).
    T2-D572: fm_rows 'base' = the current members' UNADAPTED outputs (a stable current teacher). T2-D571: record[(f, d)] collects (c, rows of IX[f],
    the teacher posterior) at every refit, for the teacher-file export. T2-D570: ctx.STARTS (shipped starts) replaces the teacher's prefix starts.
    T2-D584 (wref): every refit on EVERY prefix window with ctx.d584_weights (side heads, the MI main-head refit, weighted prototype means).
    T2-D585 (map_tau): the prototypes mu_q ~ n_q m_q + tau w_q / |w_q| (w_q = the member's TRAINED main-head row), L2-normalised."""
    assert not (wref and (fm_lp is not None or soft_proto or map_tau is not None)), "T2-D584: one change per arm"
    TW_ = TW if tw is None else dict(tw)                 # T2-D591: the teacher's family weights (= the template's fam_w; None = W21's)
    assert set(TW_) == set(TW) and abs(sum(TW_.values()) - 1.0) < 1e-9, TW_
    assert map_tau is None or (proto and not soft_proto and fm_lp is None), "T2-D585: on W20's prototypes only"
    # T2-D607 (registered 0e7feec8): rescore = a dict -> at every refit, each member's NEW head also evaluated (a pure read, the same fitted
    # W_ / b_ / prototypes) on the stream's windows j[:e]; rescore[(f, d)] gets (stream, c, e, positions j[:e], the members' mean softmax there).
    # The default path (rescore None) is untouched. W21's form only (no wref / soft / map / fm_lp).
    assert rescore is None or not (wref or soft_proto or map_tau is not None or fm_lp is not None), "T2-D607: on W21's FM refits only"
    P = {}
    # the unadapted FM part per fold (the start vectors' basis) and the members' trained logits
    base_fm, ML = {}, {}
    for f in ctx.folds:
        mem = []
        for p in ctx.fm_paths[f]:
            dm = ctx.dumps(p)
            L = dm["feat"] @ dm["Wm"].T + dm["bm"]
            has_s = "Ws" in dm
            lg0 = L + (dm["z"] @ dm["Ws"].T + dm["bs"] if has_s else 0.0)
            mem.append(dict(dm=dm, L=L, lg0=lg0, has_s=has_s, fz=dm["feat"] / (np.linalg.norm(dm["feat"], axis=1, keepdims=True) + 1e-9)))
        ML[f] = mem
        base_fm[f] = np.mean([ctx.softmax(m["lg0"]) for m in mem], 0)
    for (f, d), parts in ctx.P.items():
        R, V, _ = _tfile(ctx, f, d)
        fmrows_fixed = _d498_fm(ctx, f) if fm_rows == "d498" else None
        vfm = _fm_start(ctx, f, {g: _d498_fm(ctx, g) for g in ctx.folds} if fm_rows == "d498" else base_fm)
        ST_ = getattr(ctx, "STARTS", None) or {}
        V = {k_: np.asarray(ST_[k_], np.float64) if k_ in ST_ else v_ for k_, v_ in V.items()}
        vfm = np.asarray(ST_["fm"], np.float64) if "fm" in ST_ else vfm
        vfm = fmstart_hook(f"{getattr(ctx, 'MODE', '')}_fb", f, d, vfm)   # T2-D642: recorded / frozen per NI26_LAB_FMSTART (absent = unchanged)
        cur = [m["lg0"].copy() for m in ML[f]]
        st, od = ctx.stream[ctx.IX[f]], ctx.order[ctx.IX[f]]
        for s_ in np.unique(st):
            j = np.flatnonzero(st == s_)
            j = j[np.argsort(od[j], kind="stable")]
            CK_ = tuple(cks) if cks else tuple(ctx.CKS)    # T2-D575: the FM heads' own refit points (the CNN-side refits keep ctx.CKS)
            for k, c in enumerate(CK_):
                if len(j) <= c:
                    break
                e = CK_[k + 1] if k + 1 < len(CK_) else len(j)
                a = j[:c]
                if fmrows_fixed is not None:
                    fmr = fmrows_fixed[a]
                elif fm_rows == "base":
                    fmr = base_fm[f][a]
                else:
                    assert fm_rows == "own", fm_rows
                    fmr = np.mean([ctx.softmax(x[a]) for x in cur], 0)
                q = sum(TW_[q_] * _prefix_decode(R[q_][a], V[q_]) for q_ in ("fbc", "sh", "ifn")) + TW_["fm"] * _prefix_decode(fmr, vfm)
                q = q / q.sum(1, keepdims=True)
                if record is not None:
                    record.setdefault((f, d), []).append((c, a.copy(), q.copy()))
                if fm_lp is not None:                    # T2-D579: the FM refits' labels = the confident half of LP(teacher) over a prefix graph
                    if fm_lp["space"] == "side":         # the 82-d window-mean side vector, z-scored over the prefix windows <= c only
                        Zp = _knn_feats(ctx, f, "side")[a]
                        Fg = (Zp - Zp.mean(0)) / (Zp.std(0) + 1e-6)
                    else:                                # 'fm': D569's graph (the members' L2-normalised pooled features concatenated)
                        Fg = _knn_feats(ctx, f, "fm")[a]
                    q = _lp_block(Fg, q, float(fm_lp.get("alpha", 0.5)), int(fm_lp.get("k", 10)))
                if wref:                                 # T2-D584: every prefix window, weight = confidence x capped class balance (mean 1)
                    ya, wv = ctx.d584_weights(q)
                    sel, ys = a, ya.astype(np.int64)
                else:
                    ya, keep = _conf(q, frac)
                    sel, ys, wv = a[keep], ya[keep].astype(np.int64), None
                if count is not None:
                    count["events"] = count.get("events", 0) + 1
                    count["absent"] = count.get("absent", 0) + int(len(np.unique(ys)) < 3)
                    count[f"events_{c}"] = count.get(f"events_{c}", 0) + 1          # T2-D575 (Track3 INFO): per refit point
                    count[f"absent_{c}"] = count.get(f"absent_{c}", 0) + int(len(np.unique(ys)) < 3)
                rs_ = [] if rescore is not None else None          # T2-D607: the members' new-head logits on j[:e]
                for i, m in enumerate(ML[f]):
                    dm = m["dm"]
                    if m["has_s"]:
                        W_, b_, _ = ctx.fit_head(m["L"][sel], dm["z"][sel], ys, dm["Ws"], dm["bs"], lam, sw=wv)
                        lg = m["L"][j[c:e]] + dm["z"][j[c:e]] @ W_.T + b_
                    elif proto:                          # the head-less member's main-head refit (D560 as confirmed)
                        W_, b_, _ = ctx.fit_head(np.zeros((len(sel), 3)), dm["feat"][sel], ys, dm["Wm"], dm["bm"], lam, sw=wv)
                        lg = dm["feat"][j[c:e]] @ W_.T + b_
                    else:
                        lg = m["lg0"][j[c:e]]
                    if proto and len(sel):
                        fz = m["fz"]
                        if soft_proto:                   # T2-D581: posterior-weighted class means over EVERY prefix window (no zero prototype)
                            mu = (q.T @ fz[a]) / np.maximum(q.sum(0), 1e-12)[:, None]
                        elif wv is not None:             # T2-D584: weighted class means (an absent class = a zero prototype, as in W20)
                            fs_ = fz[sel]
                            mu = np.stack([(wv[ys == q_] @ fs_[ys == q_]) / wv[ys == q_].sum() if (ys == q_).any() else np.zeros(fz.shape[1])
                                           for q_ in range(3)])
                        elif map_tau is not None:        # T2-D585: n_q m_q (= the class sum) + tau x the trained head row's direction
                            fs_ = fz[sel]
                            Wn = dm["Wm"] / (np.linalg.norm(dm["Wm"], axis=1, keepdims=True) + 1e-12)
                            mu = np.stack([fs_[ys == q_].sum(0) for q_ in range(3)]) + float(map_tau) * Wn
                        else:
                            mu = np.stack([fz[sel][ys == q_].mean(0) if (ys == q_).any() else np.zeros(fz.shape[1]) for q_ in range(3)])
                        mu = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
                        lg = lg + proto * 10.0 * (fz[j[c:e]] @ mu.T)
                    cur[i][j[c:e]] = lg
                    if rs_ is not None:                  # T2-D607: the SAME head on every delivered window up to e (the history + this segment)
                        jj = j[:e]
                        if m["has_s"]:
                            lp_ = m["L"][jj] + dm["z"][jj] @ W_.T + b_
                        elif proto:
                            lp_ = dm["feat"][jj] @ W_.T + b_
                        else:
                            lp_ = m["lg0"][jj].copy()
                        if proto and len(sel):
                            lp_ = lp_ + proto * 10.0 * (m["fz"][jj] @ mu.T)
                        rs_.append(lp_)
                if rs_ is not None:
                    rescore.setdefault((f, d), []).append((s_, c, e, j[:e].copy(), np.mean([ctx.softmax(x) for x in rs_], 0)))
        P[(f, d)] = dict(parts, fm=np.mean([ctx.softmax(x) for x in cur], 0))
    return {"P": P}


def _lp_block(Fr_a, T0, alpha_, kk_):
    """T2-D579: scripts/local/t2_lab.py MODE lpteach's per-stream block VERBATIM (= the template's _lp_smooth)."""
    c = len(T0)
    X = Fr_a - Fr_a.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Sm = X @ X.T
    np.fill_diagonal(Sm, -np.inf)
    nn = np.argsort(-Sm, 1, kind="stable")[:, :kk_]
    A_ = np.zeros((c, c))
    rr = np.repeat(np.arange(c), kk_)
    A_[rr, nn.ravel()] = np.maximum(Sm[rr, nn.ravel()], 0)
    A_ = np.maximum(A_, A_.T)
    dg = A_.sum(1) + 1e-9
    Sn = A_ / np.sqrt(dg[:, None] * dg[None, :])
    Fp = np.clip(np.linalg.solve(np.eye(c) - alpha_ * Sn, T0), 1e-9, None)
    return Fp / Fp.sum(1, keepdims=True)


def check_d566_teacher(ctx):
    """the reconstruction check: the D498-row teacher prefix posteriors == the shipped teacher_p1 files' post_r (every stream, every refit)."""
    dmax = 0.0
    for (f, d) in ctx.P:
        R, V, z = _tfile(ctx, f, d)
        fmr = _d498_fm(ctx, f)
        vfm = _fm_start(ctx, f, {g: _d498_fm(ctx, g) for g in ctx.folds})
        pos, post = ctx.teacher(f, d)
        st, od = ctx.stream[ctx.IX[f]], ctx.order[ctx.IX[f]]
        for s_ in np.unique(st):
            j = np.flatnonzero(st == s_)
            j = j[np.argsort(od[j], kind="stable")]
            for c in ctx.CKS:
                if len(j) <= c:
                    break
                a = j[:c]
                q = sum(TW[q_] * _prefix_decode(R[q_][a], V[q_]) for q_ in ("fbc", "sh", "ifn")) + TW["fm"] * _prefix_decode(fmr[a], vfm)
                q = q / q.sum(1, keepdims=True)
                ref = np.stack([post[c][pos[int(r)]] for r in ctx.IX[f][a]])
                dmax = max(dmax, float(np.abs(q - ref).max()))
    return dmax


def batch_d566():
    """T2-D566 (run with --base w19): the reconstruction check (D498 rows == the fixed teacher: must read 0.0), then W19 and W20 under the
    feedback teacher; W20_fb − W19_fb = the registered comparison (linear in the logged per-stranger / draw deltas)."""
    return [("w19_fixed", fm, {}), ("fb_check_d498", fm_feedback, {"fm_rows": "d498"}),
            ("w19_fb", fm_feedback, {}), ("w20_fb", fm_feedback, {"proto": 0.5})]


def fb_count(ctx, proto=0.5):
    """T2-D566 F3 (descriptive): W20 under the feedback teacher, counting the refit events whose confident rows lack a class."""
    c = {}
    r = fm_feedback(ctx, proto=proto, count=c)
    print(f"[lab] F3 absent-class refit events: {c.get('absent', 0)} of {c.get('events', 0)} (every fold x draw x stream x refit point)", flush=True)
    return r


def batch_d566c():
    """T2-D566 on C (--base w19) + the F3 count."""
    return batch_d566() + [("w20_fb_count", fb_count, {})]


def batch_d557f3():
    """R_W20D557 (the manager, 2026-11-18): R_W20 F3's absent-class count re-taken on C with the AUGMENTED members (run with NI26_LAB_D557=1,
    --set C --base w19, --log d557_F3_C.jsonl): the same counter as batch_d566c's w20_fb_count (W20 under the feedback teacher). Descriptive."""
    return [("w20d557_fb_count", fb_count, {})]


def batch_d572():
    """T2-D572 (registered 2026-11-18 before any number): the stable current teacher (W20's FM part, every refit labelled by the current members'
    UNADAPTED outputs) = the idea w20_st; read only through t2_lab_pair_read.py with its own P1 root (t2_d571_unw20d557_831)."""
    return [("w20_st", fm_feedback, {"proto": 0.5, "fm_rows": "base"})]


def batch_d575():
    """T2-D575 (registered 2026-11-18 before any number): an EARLY cheap HEAD refit with the CNN schedule kept = W20 under the feedback teacher
    with the FM heads (D553 side heads + D560's MI refit / prototypes) re-fitted at 40 / 60 / 100 / 150; P1 and D539b stay at 60 / 100 / 150."""
    return [("w20_fb_h40", fm_feedback, {"proto": 0.5, "cks": (40, 60, 100, 150)})]


def fb_count_h40(ctx, proto=0.5):
    """T2-D575 (Track3 INFO, descriptive): the absent-class refit events per refit point of w20_fb_h40 (40 / 60 / 100 / 150)."""
    c = {}
    r = fm_feedback(ctx, proto=proto, cks=(40, 60, 100, 150), count=c)
    print(f"[lab] D575 absent-class refit events per point: " + ", ".join(f"c {k}: {c.get(f'absent_{k}', 0)} / {c.get(f'events_{k}', 0)}"
                                                                        for k in (40, 60, 100, 150)), flush=True)
    return r


def batch_d575c():
    """T2-D575's descriptive absent-class count (run with --log d575_count_<set>.jsonl; never a confirm source)."""
    return [("w20_fb_h40_count", fb_count_h40, {})]


def batch_d579():
    """T2-D579 (registered 2026-11-18 before any number): LP for the FM refits' labels, z graph (arm a) / FM graph (arm b); read on W21's lp roots."""
    return [("w20_fb_zlp", fm_feedback, {"proto": 0.5, "fm_lp": {"space": "side", "alpha": 0.5, "k": 10}}),
            ("w20_fb_fmlp", fm_feedback, {"proto": 0.5, "fm_lp": {"space": "fm", "alpha": 0.5, "k": 10}})]


def batch_d584():
    """T2-D584 (registered 2026-11-18 16:26:28 before any number; cap 3 folded in 16:42:11): every refit on EVERY prefix window, weighted
    (confidence x capped class balance, mean 1); read with reader env NI26_LAB_D584=1 (D539b's side heads, same weights) on W21's lp roots."""
    return [("w20_fb_wref", fm_feedback, {"proto": 0.5, "wref": True})]


def fb_count_d585(ctx, proto=0.5, map_tau=10.0):
    """T2-D585: fm_feedback(proto 0.5, map_tau 10) + the absent-class refit-event count printed beside the result (Track3: reported with each
    delta; the count is a side effect, the probabilities are fm_feedback's)."""
    c = {}
    r = fm_feedback(ctx, proto=proto, map_tau=map_tau, count=c)
    print(f"[lab] D585 absent-class refit events: {c.get('absent', 0)} of {c.get('events', 0)} (every fold x draw x stream x refit point)", flush=True)
    return r


def batch_d585():
    """T2-D585 (registered 2026-11-18 16:26:28 before any number): D560's prototypes MAP-shrunk toward the trained head rows, tau 10 (pinned)."""
    return [("w20_fb_mapproto", fb_count_d585, {"proto": 0.5, "map_tau": 10.0})]


def tw_fm(x):
    """T2-D591: W21's family shares with the FM share set to x and the three CNN families scaled by (1 - x) / (1 - 0.35) (their ratios kept)."""
    return {k: (float(x) if k == "fm" else v * (1.0 - float(x)) / (1.0 - TW["fm"])) for k, v in TW.items()}


def fm_feedback_fm45(ctx, proto=0.5):
    """T2-D591 (registered 2026-11-18 before any number): the FM share 0.35 -> 0.45 in BOTH the output mix and the in-stream teacher (the template's
    fam_w feeds both); read only on the fm45 roots (whose CNN P1 / side-head teacher is the same 0.45 teacher)."""
    assert all(abs(ctx.W[k] - TW[k]) < 1e-4 for k in TW) and set(ctx.W) == set(TW), ("T2-D591: the lab's mix weights must be W21's", ctx.W)
    r = fm_feedback(ctx, proto=proto, tw=tw_fm(0.45))
    r.update(weights(ctx, fm_w=0.45))
    assert all(abs(r["W"][k] - tw_fm(0.45)[k]) < 1e-4 for k in TW), (r["W"], tw_fm(0.45))
    return r


def batch_d591():
    """T2-D591: the FM-family share step 0.35 -> 0.45 (teacher + mix), on its own fm45 roots."""
    return [("w20_fb_fm45", fm_feedback_fm45, {"proto": 0.5})]


def fm_feedback_fm40(ctx, proto=0.5):
    """T2-D655 (registered 2026-11-20 before any number): the FM share 0.35 -> 0.40 in BOTH the output mix and the in-stream teacher (= D591's form at
    half its step; the three CNN families keep their ratios, scaled by 0.60 / 0.65 = tw_fm(0.40)); read only on the fm40 roots (cell bw21fm40)."""
    assert all(abs(ctx.W[k] - TW[k]) < 1e-4 for k in TW) and set(ctx.W) == set(TW), ("T2-D655: the lab's mix weights must be W21's", ctx.W)
    r = fm_feedback(ctx, proto=proto, tw=tw_fm(0.40))
    r.update(weights(ctx, fm_w=0.40))
    assert all(abs(r["W"][k] - tw_fm(0.40)[k]) < 1e-12 for k in TW), (r["W"], tw_fm(0.40))
    return r


def batch_d655():
    """T2-D655: the FM-family share step 0.35 -> 0.40 (teacher + mix), on the fm40 roots of the cell bw21fm40."""
    return [("w20_fb_fm40", fm_feedback_fm40, {"proto": 0.5})]


RH_PACKAGE_SHARES = {"fm": 0.55, "cnn2": 0.1631868131868132, "cnn3": 0.06923076923076923}   # T2-D658: the RH package's config (launch_d658_rh.sh)


def fm_feedback_fm55(ctx, proto=0.5):
    """T2-D658 (registered 2026-11-21 18:50:58 before any number): RH's FM share 0.35 -> 0.55 in BOTH the output mix and the in-stream teacher
    (= D655's form at four times its step; the CNN families scaled by 0.45 / 0.65 = tw_fm(0.55)); read only on the fm55 roots (cell bw21fm55)."""
    assert all(abs(ctx.W[k] - TW[k]) < 1e-4 for k in TW) and set(ctx.W) == set(TW), ("T2-D658: the lab's mix weights must be W21's", ctx.W)
    r = fm_feedback(ctx, proto=proto, tw=tw_fm(0.55))
    r.update(weights(ctx, fm_w=0.55))
    assert all(abs(r["W"][k] - tw_fm(0.55)[k]) < 1e-12 for k in TW), (r["W"], tw_fm(0.55))
    return r


def batch_d658():
    """T2-D658: RH's FM-family share step 0.35 -> 0.55 (teacher + mix), on the fm55 roots of the cell bw21fm55."""
    return [("w20_fb_fm55", fm_feedback_fm55, {"proto": 0.5})]


def batch_d581():
    """T2-D581 (registered 2026-11-18 before any number): soft (posterior-weighted) prototypes; read on W21's lp roots."""
    return [("w20_fb_softproto", fm_feedback, {"proto": 0.5, "soft_proto": True})]
