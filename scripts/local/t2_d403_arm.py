#!/usr/bin/env python
"""T2-D403 (backlog 95; pre-registered 2026-11-07 04:20, before any code or number): the bounded repair of T2-D395's ARM context head.
On outputs/t2_d369/feat_<family>.npz (kit protocol; t2_p3_screens.load_feats: standardise + PCA 64 on the training users):
  probe   = T2-D394's sklearn LogisticRegression on the training users (THE COMPARATOR; frozen inside every head, so context-free = it)
  arm     = probe logits + h(c) + Σ_r g_r(c) u_r v_rᵀ x (rank 4)          bias = probe logits + h(c)
  c       = the causal mean of the stream's earlier windows (prior = the training mean, n0 10), centred by that mean
Training: minibatches of 256 RANDOMIZED PREFIX EPISODES (a training stream ∝ its length, a start a, a window t >= a; the context from
windows a..t-1), Adam lr 3e-3; leave-one-training-user-out selection of steps {150, 500, 1500} x weight decay {1e-3, 1e-2} per kind (the
held user's mean cell BA under causal SK); final heads on all training users, fresh head seeds 100-104 averaged. Held-out kit users only in
the final read: genuine vs TRUE shuffled-user context (the position-matched context of a held-out stream of a DIFFERENT user), all under SK.
Bar: on EITHER family, arm genuine - probe >= +0.005 AND genuine - shuffled-user >= +0.010 typical, P04 >= -0.010 vs the probe.
  python t2_d403_arm.py --export E [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from t2_p3_screens import load_feats, probe, decode, cell_scores  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
N0, R, BS, LR = 10, 4, 256, 3e-3
GRID = [(s, w) for s in (150, 500, 1500) for w in (1e-3, 1e-2)]


def streams(rows, d):
    strm, ordr = d.stream.to_numpy(), d.order.to_numpy()
    out = {}
    for s in np.unique(strm[rows]):
        r = rows[strm[rows] == s]
        out[s] = r[np.argsort(ordr[r], kind="stable")]
    return out


def causal_ctx(Z, rows, d, mu0):
    """{row: the causal mean of its stream's EARLIER windows (prior mu0, weight N0)} as an array aligned with `rows`."""
    C = np.empty((len(rows), Z.shape[1])); pos = {r: k for k, r in enumerate(rows)}
    for s, r in streams(rows, d).items():
        cs = np.cumsum(Z[r], 0) - Z[r]
        for k, x in enumerate(r):
            C[pos[x]] = (cs[k] + N0 * mu0) / (k + N0)
    return C


class Head(torch.nn.Module):
    def __init__(self, D, kind, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind = kind
        self.H = torch.nn.Parameter(torch.zeros(3, D))
        if kind == "arm":
            self.G = torch.nn.Parameter(torch.randn(R, D, generator=g) / np.sqrt(D)); self.gb = torch.nn.Parameter(torch.zeros(R))
            self.U = torch.nn.Parameter(0.01 * torch.randn(R, 3, generator=g)); self.V = torch.nn.Parameter(0.01 * torch.randn(R, D, generator=g))

    def forward(self, base, x, c):
        z = base + c @ self.H.T
        if self.kind == "arm":
            gg = torch.tanh(c @ self.G.T + self.gb)
            z = z + torch.einsum("nr,rk,rd,nd->nk", gg, self.U, self.V, x)
        return z


def fit_head(Z, y, rows, d, clf, mu0, kind, steps, wd, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    st = list(streams(rows, d).values())
    lens = np.array([len(r) for r in st]); pst = lens / lens.sum()
    cums = [np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z[r], 0)]) for r in st]
    W, b = torch.tensor(clf.coef_, dtype=torch.float32), torch.tensor(clf.intercept_, dtype=torch.float32)
    h = Head(Z.shape[1], kind, seed)
    opt = torch.optim.Adam(h.parameters(), lr=LR, weight_decay=wd)
    for _ in range(steps):
        si = rng.choice(len(st), BS, p=pst)
        a = np.array([rng.integers(lens[k]) for k in si]); t = np.array([rng.integers(a_, lens[k]) for a_, k in zip(a, si)])
        rows_b = np.array([st[k][t_] for k, t_ in zip(si, t)])
        ctx = np.stack([(cums[k][t_] - cums[k][a_] + N0 * mu0) / (t_ - a_ + N0) for k, a_, t_ in zip(si, a, t)]) - mu0
        x = torch.tensor(Z[rows_b], dtype=torch.float32); c = torch.tensor(ctx, dtype=torch.float32)
        loss = torch.nn.functional.cross_entropy(h(x @ W.T + b, x, c), torch.tensor(y[rows_b]))
        opt.zero_grad(); loss.backward(); opt.step()
    return h, W, b


def head_proba(h, W, b, Z, rows, C, mu0):
    with torch.no_grad():
        x = torch.tensor(Z[rows], dtype=torch.float32)
        return torch.softmax(h(x @ W.T + b, x, torch.tensor(C - mu0, dtype=torch.float32)), 1).numpy()


def shuffled_ctx(Cho, ho, d, subj):
    """position-matched context of a held-out stream of a DIFFERENT user (the next such stream in sorted order)."""
    st = streams(ho, d); keys = sorted(st); pos = {r: k for k, r in enumerate(ho)}
    out = np.empty_like(Cho)
    for i, s in enumerate(keys):
        u = subj[st[s][0]]
        o = next(keys[(i + j) % len(keys)] for j in range(1, len(keys)) if subj[st[keys[(i + j) % len(keys)]][0]] != u)
        for k, x in enumerate(st[s]):
            out[pos[x]] = Cho[pos[st[o][min(k, len(st[o]) - 1)]]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(O / "t2_d403"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    subj = d.subject.str[-3:].to_numpy()
    out = {}
    for fam in ("fbcnet_bboth", "shallow"):
        Z, y, tr, ho = load_feats(fam, d)
        tr, ho = np.asarray(tr), np.asarray(ho)
        clf = probe(Z[tr], y[tr])
        assert np.allclose(clf.predict_proba(Z[ho]), torch.softmax(torch.tensor(Z[ho] @ clf.coef_.T + clf.intercept_), 1).numpy(), atol=1e-6)
        # ---- leave-one-training-user-out selection, per kind
        sel = {}
        for kind in ("arm", "bias"):
            tab = {}
            for (steps, wd) in GRID:
                sc = []
                for u in sorted(set(subj[tr])):
                    rf, re = tr[subj[tr] != u], tr[subj[tr] == u]
                    cf = probe(Z[rf], y[rf]); mu_f = Z[rf].mean(0)
                    h, W, b = fit_head(Z, y, rf, d, cf, mu_f, kind, steps, wd, seed=0)
                    P = head_proba(h, W, b, Z, re, causal_ctx(Z, re, d, mu_f), mu_f)
                    sc.append(cell_scores(decode(P, re, d, "causal"), re, d)[u])
                tab[f"{steps}/{wd:g}"] = float(np.mean(sc))
            best = max(tab, key=tab.get)
            sel[kind] = dict(table=tab, pick=best)
            print(f"[D403] {fam} {kind}: LOUO " + " · ".join(f"{k} {v:.4f}" for k, v in tab.items()) + f" → {best}", flush=True)
        # ---- final read on the held-out kit users
        mu0 = Z[tr].mean(0)
        Cho = causal_ctx(Z, ho, d, mu0); Csh = shuffled_ctx(Cho, ho, d, subj)
        res = {"probe": cell_scores(decode(clf.predict_proba(Z[ho]), ho, d, "causal"), ho, d)}
        for kind in ("arm", "bias"):
            steps, wd = sel[kind]["pick"].split("/")
            Pg = Ps = 0.0
            for seed in range(100, 105):
                h, W, b = fit_head(Z, y, tr, d, clf, mu0, kind, int(steps), float(wd), seed)
                Pg = Pg + head_proba(h, W, b, Z, ho, Cho, mu0) / 5
                Ps = Ps + head_proba(h, W, b, Z, ho, Csh, mu0) / 5
            res[f"{kind}_genuine"] = cell_scores(decode(Pg, ho, d, "causal"), ho, d)
            res[f"{kind}_shuffled"] = cell_scores(decode(Ps, ho, d, "causal"), ho, d)
        base = res["probe"]
        print(f"[D403] {fam}: " + " · ".join(f"{k} {v['typical']:.4f} ({v['typical'] - base['typical']:+.4f}; P04 {v['P04'] - base['P04']:+.4f})"
                                             for k, v in res.items()), flush=True)
        g, s_ = res["arm_genuine"], res["arm_shuffled"]
        met = bool(g["typical"] - base["typical"] >= 0.005 and g["typical"] - s_["typical"] >= 0.010 and g["P04"] - base["P04"] >= -0.010)
        out[fam] = dict(selection=sel, res=res, met=met)
        print(f"[D403] {fam}: arm genuine − probe {g['typical'] - base['typical']:+.4f}, genuine − shuffled-user {g['typical'] - s_['typical']:+.4f}, "
              f"P04 vs probe {g['P04'] - base['P04']:+.4f} → {'MET' if met else 'not met'}", flush=True)
    out["bar_met"] = any(out[f]["met"] for f in ("fbcnet_bboth", "shallow"))
    print(f"[D403] BAR (either family): {'MET' if out['bar_met'] else 'NOT MET'}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "arm.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
