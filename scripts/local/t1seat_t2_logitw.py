#!/usr/bin/env python
"""Track1's outside seat on T2 (the manager's ask 2026-11-14 21:4x; lead/panel_2026-11-14_outliers/SEAT_T1_on_T2.md): a zero-GPU screen.
T1's one sealed lever was a per-batch WHITENING of the emission (second moment), where the mirror had only credited the mean. T2's
decode repairs the per-stream FIRST moment of the mix (causal Sinkhorn = class balance, worth +0.12). Not on T2's record by mechanism
(greps: whiten / temperature / logit / second moment): a causal per-stream SECOND-moment repair of the mix in log-ratio space before
the shipped decode. Nearest rows: D182 per-member temperature, D244 FM temperature, D362 class-conditional calibration, D375 feature
centring, D476 decode-then-pool, D478 train-as-decode.
Arms (M7's stored mix, t2_order_sim's loader; decode = the shipped causal SK 10 / n0 3 / lambda 0.95):
  A0     shipped.
  SHAPE  u = ilr(log mix) (2-d); around the stream's causal running mean m_t, u' = m_t + S_t (u - m_t), S_t = (C_t / gm(C_t))^(-1/2),
         C_t = the causal covariance of windows 1..t shrunk to isotropic with n0 = 20 pseudo-windows (shape only, scale kept).
  FULL   SHAPE plus the stream's scale set to a constant = the median over the OTHER split's streams of sqrt(gm(C_end)) (cross-fit).
  SCALE  scale only (no shape).
Read (fixed before any number): typical = non-P04 cells, 12 draws, kit + seen1, test + val. A lead iff an arm is >= +0.003 typical with
>= 9 / 12 draws up and no user below -0.010; else closed, no row asked. Prediction: all arms within +-0.002; P(lead) 0.08.
    mem-guard python scripts/local/t1seat_t2_logitw.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d408/spec_m7.json"))
idx = pd.read_csv(EXP / "index.csv")
B = np.array([[1, -1, 0], [1, 1, -2]], float).T
B /= np.linalg.norm(B, axis=0)          # 3 x 2 orthonormal basis of the centred log-ratio plane
N0 = 20.0


def gm(C):
    return np.sqrt(max(np.linalg.det(C), 1e-30))        # geometric mean of the two eigenvalues


def transform(mix, st, od, mode, const=None):
    """Causal per-stream second-moment repair. Returns the new probabilities and each stream's final scale sqrt(gm(C_end))."""
    L = np.log(np.clip(mix, 1e-9, 1))
    U = (L - L.mean(1, keepdims=True)) @ B
    out, scales = np.empty_like(mix), {}
    for s in np.unique(st):
        ii = np.where(st == s)[0]
        ii = ii[np.argsort(od[ii], kind="stable")]
        u = U[ii]
        s1, s2 = np.zeros(2), np.zeros((2, 2))
        v = np.empty_like(u)
        for t in range(len(u)):
            s1 += u[t]
            s2 += np.outer(u[t], u[t])
            n = t + 1
            m = s1 / n
            C = s2 / n - np.outer(m, m)
            iso = max(np.trace(C) / 2, 1e-6)
            C = (n * C + N0 * iso * np.eye(2)) / (n + N0)
            g = np.sqrt(gm(C))
            w, V = np.linalg.eigh(C / g ** 2)
            S = (V / np.sqrt(w)) @ V.T if mode in ("SHAPE", "FULL") else np.eye(2)
            k = (const / g) if (mode in ("FULL", "SCALE") and const is not None) else 1.0
            k = 1.0 + (k - 1.0) * n / (n + N0)
            v[t] = m + k * (S @ (u[t] - m))
        scales[s] = g
        z = v @ B.T
        p = np.exp(z - z.max(1, keepdims=True))
        out[ii] = p / p.sum(1, keepdims=True)
    return out, scales


def load(proto, i, split):
    mix, ix = 0.0, None
    for m, w in J["arms"]["M7"].items():
        M = J["members"][m]
        z = np.load(M[proto].format(s=M.get("offset", 0) + i))
        ix, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
        mix = mix + w * p.astype(np.float64)
    return mix, ix


res, per_user = {}, {}
for proto in ("kit", "seen1"):
    for i in range(12):
        D = {sp: load(proto, i, sp) for sp in ("test", "val")}
        const = {}
        for sp, (mix, ix) in D.items():
            _, sc = transform(mix, idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix], "NONE")
            const[sp] = float(np.median(list(sc.values())))
        for sp, (mix, ix) in D.items():
            other = const["val" if sp == "test" else "test"]
            st, od = idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix]
            y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
            for arm in ("A0", "SHAPE", "FULL", "SCALE"):
                p = mix if arm == "A0" else transform(mix, st, od, arm, other)[0]
                P = causal_sinkhorn(p, (st, od), 10, 3, 0.95).argmax(1)
                for c in np.unique(cells):
                    mk = cells == c
                    ba = np.mean([np.mean(P[mk][y[mk] == q] == q) for q in np.unique(y[mk])])
                    u = subj[mk][0]
                    per_user.setdefault(arm, {}).setdefault(u, {}).setdefault(i, []).append(ba)
                    if u != "P04":
                        res.setdefault(arm, {}).setdefault(i, []).append(ba)
base = np.array([np.mean(res["A0"][i]) for i in range(12)])
out = {}
for arm, r in res.items():
    v = np.array([np.mean(r[i]) for i in range(12)])
    us = {u: float(np.mean([np.mean(per_user[arm][u][i]) - np.mean(per_user["A0"][u][i]) for i in range(12)])) for u in sorted(per_user[arm])}
    out[arm] = dict(typical=float(v.mean()), delta=float(np.mean(v - base)), up=int((v > base).sum()), users=us)
    print(f"{arm:6s} typical {v.mean():.4f}  vs A0 {np.mean(v - base):+.4f} ({int((v > base).sum())} / 12 up)  "
          + "  ".join(f"{u} {d:+.4f}" for u, d in us.items()), flush=True)
(O / "lead/panel_2026-11-14_outliers").mkdir(parents=True, exist_ok=True)
json.dump(out, open(O / "lead/panel_2026-11-14_outliers/t1seat_logitw.json", "w"), indent=1)
