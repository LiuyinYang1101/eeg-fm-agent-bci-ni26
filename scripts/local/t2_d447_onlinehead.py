#!/usr/bin/env python
"""T2-D447 (pre-registered before any number): an ONLINE, ANCHORED linear head on FROZEN penultimate features, self-trained per stream on
the DECODED posterior of the stream's earlier windows (the manager's idea, 2026-11-11). Kill-first screen on the T2-D369 feature dumps
(kit protocol, one shipped seed per family; held-out users P04 P05 P08 P09; standardise + PCA 64 fitted on the six training users).
Head0 = multinomial logistic probe on the training users (static). At window t >= START of a stream the head is refitted on windows < t:
  min_W  sum_s CE(target_s, softmax(x_s W)) + LAM * ||W - W0||^2      (L-BFGS, warm start), then window t is predicted by W.
Targets: OR = true labels (ceiling, not shippable); PLm = M7's causal SK-decoded posterior (draw 0; the package as teacher);
PLs = the static probe's own causal SK-decoded posterior. Every arm is scored after the causal SK decode, per user.
    mem-guard python scripts/local/t2_d447_onlinehead.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
idx = pd.read_csv(EXP / "index.csv")
J = json.load(open(O / "t2_d408/spec_m7.json"))
START = 30
mix = np.zeros((len(idx), 3))
for m, w in J["arms"]["M7"].items():
    M = J["members"][m]
    z = np.load(M["kit"].format(s=M.get("offset", 0)))
    mix[z["test_idx"]] += w * z["prob_last"]
    mix[z["val_idx"]] += w * z["prob_val_last"]


def sk(p, rows):
    return causal_sinkhorn(p, (idx.stream.to_numpy()[rows], idx.order.to_numpy()[rows]), 10, 3, 0.95)


def typ(pred, rows):
    y, cells, sj = idx.y.to_numpy()[rows], idx.cell.to_numpy()[rows], idx.subject.str[-3:].to_numpy()[rows]
    out = {}
    for c in np.unique(cells):
        mk = cells == c
        out.setdefault(sj[mk][0], []).append(np.mean([np.mean(pred[mk][y[mk] == q] == q) for q in np.unique(y[mk])]))
    u = {k: float(np.mean(v)) for k, v in out.items()}
    return float(np.mean([v for k, v in u.items() if k != "P04"])), u


def refit(X, T, W0, lam, W):
    def f(w):
        Wm = w.reshape(W0.shape)
        lp = log_softmax(X @ Wm, 1)
        return -(T * lp).sum() + lam * ((Wm - W0) ** 2).sum(), (X.T @ (softmax(X @ Wm, 1) - T) + 2 * lam * (Wm - W0)).ravel()
    return minimize(f, W.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 30}).x.reshape(W0.shape)


res = {}
for fam in ("fbcnet_bboth", "shallow"):
    z = np.load(O / f"t2_d369/feat_{fam}.npz", allow_pickle=True)
    tr, ho = z["tr"], np.sort(z["ho"])
    sc = StandardScaler().fit(z["F"][tr])
    pc = PCA(64, random_state=0).fit(sc.transform(z["F"][tr]))
    f = lambda r: np.c_[pc.transform(sc.transform(z["F"][r])), np.ones(len(r))]   # noqa: E731
    Xtr, Xho = f(tr), f(ho)
    s2 = Xtr[:, :-1].std()
    Xtr[:, :-1] /= s2
    Xho[:, :-1] /= s2
    lr = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(Xtr, z["y"][tr])
    W0 = lr.coef_.T.copy()
    y = idx.y.to_numpy()[ho]
    p0 = softmax(Xho @ W0, 1)
    teach = {"OR": np.eye(3)[y], "PLm": sk(mix[ho], ho), "PLs": sk(p0, ho)}
    b0, u0 = typ(sk(p0, ho).argmax(1), ho)
    tm, _ = typ(teach["PLm"].argmax(1), ho)
    print(f"\n== {fam}: static probe typical {b0:.4f} " + " ".join(f"{k} {v:.3f}" for k, v in u0.items()) + f" | the M7 teacher on these windows {tm:.4f}", flush=True)
    st, od = idx.stream.to_numpy()[ho], idx.order.to_numpy()[ho]
    for arm, T in teach.items():
        for lam in (10.0, 30.0, 100.0):
            p = p0.copy()
            for s in np.unique(st):
                r = np.where(st == s)[0]
                r = r[np.argsort(od[r])]
                W = W0.copy()
                for t in range(START, len(r)):
                    W = refit(Xho[r[:t]], T[r[:t]], W0, lam, W)
                    p[r[t]] = softmax(Xho[r[t]] @ W)
            pred = sk(p, ho).argmax(1)
            b, u = typ(pred, ho)
            pos = idx.k.to_numpy()[ho] if "k" in idx else od
            late = np.array([od[i] >= START for i in range(len(ho))])
            res[(fam, arm, lam)] = b - b0
            print(f"  {arm:3s} lam {lam:5.0f}: typical {b:.4f} ({b - b0:+.4f})  " + " ".join(f"{k} {u[k] - u0[k]:+.3f}" for k in u)
                  + f" | window accuracy at order >= {START}: {np.mean(pred[late] == y[late]):.3f} vs static {np.mean(sk(p0, ho).argmax(1)[late] == y[late]):.3f}", flush=True)
print("\nprimary (lam 30):", {f"{k[0][:3]}-{k[1]}": round(v, 4) for k, v in res.items() if k[2] == 30.0})
