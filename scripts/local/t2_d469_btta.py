#!/usr/bin/env python
"""T2-D469 (written before any number): BTTA-DG (Luo et al., ICLR 2026) mapped to T2, zero GPU. See the row.
Shape C (T2-D459: M7's weights, fresh CNN draws D419 / D429, D442 control FM). Per window, alpha = Dirichlet MLE (Minka fixed point)
over the 8 member probability vectors; feature log alpha (3-d). Per stream, causal, fresh state per stream:
  anchor a_t = causal Sinkhorn (10 / n0 3 / lambda 0.95) of the ensemble mix (the shipped decoder, unchanged);
  bank = earlier windows j < t with max a_j >= tau and normalised entropy(a_j) <= 0.8, labelled argmax a_j (anchor labels only:
         no feedback from the adapted output), FIFO <= 40 per class; one diagonal GMM per class (K components, own EM) once every
         class has >= max(6, 3K) entries; l_y = log p_GMM(log alpha_t | y);
  arm A (before the decoder): q_t ∝ exp(l_y) mix_t, then the decoder over the stream's q -> argmax;
  arm B (after the decoder):  r_t ∝ exp(l_y) a_t -> argmax.
Grid tau in {0.5, 0.7} x K in {1, 2}, picked per arm on the VAL cells (users P05, P08), judged on the TEST cells; base = the decoder alone.
    mem-guard python scripts/local/t2_d469_btta.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp, polygamma

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    S = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = S[src]
W = J["arms"]["M7f"]
idx = pd.read_csv(EXP / "index.csv")
U, ST, OD, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
GRID = [(t, k) for t in (0.5, 0.7) for k in (1, 2)]


def inv_digamma(y, it=5):
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1.0 / (y - digamma(1.0)))
    for _ in range(it):
        x = x - (digamma(x) - y) / polygamma(1, x)
    return x


def dirichlet_mle(P, it=50):
    P = np.clip(P, 1e-4, 1.0)
    P = P / P.sum(1, keepdims=True)
    lp = np.log(P).mean(0)
    m, v = P.mean(0), P.var(0) + 1e-6
    s = np.median(m * (1 - m) / v - 1)
    a = np.maximum(m * max(s, 0.5), 1e-3)
    for _ in range(it):
        a = inv_digamma(digamma(a.sum()) + lp)
    return a


def gmm_fit(Z, K, it=20):
    if K == 1:
        return np.ones(1), Z.mean(0, keepdims=True), Z.var(0, keepdims=True) + 1e-3
    mu = Z[[0, int(np.argmax(((Z - Z[0]) ** 2).sum(1)))]].copy()
    var = np.repeat(Z.var(0, keepdims=True) + 1e-3, K, 0)
    pi = np.full(K, 1.0 / K)
    for _ in range(it):
        ll = np.log(pi) - 0.5 * (((Z[:, None] - mu[None]) ** 2) / var[None] + np.log(2 * np.pi * var[None])).sum(-1)
        r = np.exp(ll - logsumexp(ll, 1, keepdims=True))
        nk = r.sum(0) + 1e-9
        pi, mu = nk / nk.sum(), (r.T @ Z) / nk[:, None]
        var = (r.T @ (Z ** 2)) / nk[:, None] - mu ** 2 + 1e-3
    return pi, mu, np.maximum(var, 1e-3)


def gmm_ll(g, z):
    pi, mu, var = g
    return float(logsumexp(np.log(pi) - 0.5 * (((z - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(-1)))


def adapt(base, F, tau, K):
    """base: (T, 3) probabilities in stream order; F: (T, 3) features; returns the (T, 3) log-likelihood terms (0 before the bank is ready)."""
    T = len(base)
    L = np.zeros((T, 3))
    bank = {c: [] for c in range(3)}
    g = {}
    nmin = max(6, 3 * K)
    for t in range(T):
        if all(len(bank[c]) >= nmin for c in range(3)):
            l = np.array([gmm_ll(g[c], F[t]) for c in range(3)])
            L[t] = l - logsumexp(l)
        a = base[t]
        h = -(a * np.log(np.maximum(a, 1e-12))).sum() / np.log(3)
        if a.max() >= tau and h <= 0.8:
            c = int(a.argmax())
            bank[c].append(F[t])
            bank[c] = bank[c][-40:]
            if len(bank[c]) >= nmin:
                g[c] = gmm_fit(np.array(bank[c]), K)
    return L


def task(job):
    proto, i, split = job
    parts, ix = [], None
    mix = 0.0
    for m, w in W.items():
        z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
        ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
        p = p.astype(np.float64)
        parts.append(p)
        mix = mix + w * p
    parts = np.stack(parts, 1)
    out = []
    for s_ in np.unique(ST[ix]):
        r = np.flatnonzero(ST[ix] == s_)
        r = r[np.argsort(OD[ix][r], kind="stable")]
        mx = mix[r]
        F = np.log(np.stack([dirichlet_mle(parts[j]) for j in r]))
        key = (np.zeros(len(r)), np.arange(len(r)))
        anc = causal_sinkhorn(mx, key, 10, 3, 0.95)
        preds = {"base": anc.argmax(1)}
        for tau, K in GRID:
            L = adapt(anc, F, tau, K)
            qa = np.exp(L + np.log(np.maximum(mx, 1e-12)))
            preds[f"A_t{tau}_K{K}"] = causal_sinkhorn(qa / qa.sum(1, keepdims=True), key, 10, 3, 0.95).argmax(1)
            preds[f"B_t{tau}_K{K}"] = (L + np.log(np.maximum(anc, 1e-12))).argmax(1)
        for arm, pr in preds.items():
            for c in np.unique(CELL[ix][r]):
                mk = CELL[ix][r] == c
                yy = Y[ix][r][mk]
                out.append((arm, proto, i, split, c, U[ix][r][mk][0], np.mean([np.mean(pr[mk][yy == v] == v) for v in np.unique(yy)])))
    return out


if __name__ == "__main__":
    jobs = [(p, i, s) for p in ("kit", "seen1") for i in range(12) for s in ("test", "val")]
    with Pool(12) as pool:
        rows = sum(pool.map(task, jobs), [])
    D = pd.DataFrame(rows, columns=["arm", "proto", "i", "split", "cell", "u", "ba"])
    D.to_csv(O / "lead/panel_2026-11-12_t2/d469_cells.csv", index=False)
    T = D[D.u != "P04"]
    key = ["proto", "i", "split", "cell"]
    base = T[T.arm == "base"].set_index(key).ba
    res = {}
    for arm in sorted(set(T.arm) - {"base"}):
        d = (T[T.arm == arm].set_index(key).ba - base).rename("d").reset_index()
        res[arm] = {sp: (d[d.split == sp].groupby("i").d.mean().mean(), int((d[d.split == sp].groupby("i").d.mean() > 0).sum())) for sp in ("val", "test")}
        print(f"{arm:12s} val {res[arm]['val'][0]:+.4f} ({res[arm]['val'][1]:2d}/12)   test {res[arm]['test'][0]:+.4f} ({res[arm]['test'][1]:2d}/12)", flush=True)
    for A in ("A", "B"):
        pick = max((a for a in res if a.startswith(A)), key=lambda a: res[a]["val"][0])
        d = (D[(D.arm == pick) & (D.split == "test")].set_index(key + ["u"]).ba - D[(D.arm == "base") & (D.split == "test")].set_index(key + ["u"]).ba).rename("d").reset_index()
        print(f"ARM {A}: picked on val = {pick}; TEST typical {res[pick]['test'][0]:+.4f} ({res[pick]['test'][1]}/12); per user "
              + " ".join(f"{u} {v:+.4f}" for u, v in d.groupby("u").d.mean().items()), flush=True)
