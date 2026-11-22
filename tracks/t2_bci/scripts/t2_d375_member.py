#!/usr/bin/env python
"""T2-D375 step 1.5: the per-user-centred linear head as a deployable causal MEMBER, for the package read.
Per protocol (kit, seen1) and seed, the pooled CNN trained exactly as t2p_bench's task (folds, the training scale, aligned_all, the vp bank,
ni26.train.fit_eval with the shipped recipe), then on final_layer's input (standardised + PCA <= 64, fit on the TRAINING rows):
  *_cen: a logistic head fit on per-user-centred training features (each training user's TRAINING rows minus their mean), applied to
         the val / test windows with a CAUSAL per-stream running mean (prior = the training mean, weight N0 = 10; stream = subject |
         session in the kit 'order');
  *_lin: the control — the same head on raw features.
Writes <out>/<member>/<protocol>__s<seed>.npz with test_idx / prob_last / val_idx / prob_val_last (the keys t2_d320_rescreen reads).
  python t2_d375_member.py --export E --out DIR [--models fbcnet_bboth,shallow] [--protocols kit,seen1] [--seeds-fbc 192-203] [--seeds-sh 156-167]
"""
import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
import ni26.train as T  # noqa: E402

N0 = 10
DRES = False        # T2-D394 step 2: --dres also writes <tag>_dres (the per-window log-ratio adj, not probabilities)
SHIPPED = {"fbcnet_bboth": ("none", "r1_pulxx_vp2_e180_vr_rc75", "fbc"), "shallow": ("ea", "r1_pulx_vp2_e180_vr", "sh")}


def train_features(model, proto, seed, X, idx, meta, device, epochs=0):
    align, rname, _ = SHIPPED[model]
    (_, trm, vam, tem), = B.folds(idx, proto)
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    XA = B.aligned_all(X, idx, align, trm, 0.0, 0.05, scale)
    tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    tr_users = sorted(idx.iloc[tr]["subject"].unique())
    utr = idx.iloc[tr]["subject"].map({u: k for k, u in enumerate(tr_users)}).to_numpy()
    rec = T.RECIPES[rname] if not epochs else replace(T.RECIPES[rname], epochs=epochs)
    T.VP_BANK.clear()
    T.VP_BANK.update({k: v.to(device) for k, v in B.vp_bank(list(meta["ch_names"]), K=rec.vp_k, sigma=0.005 * rec.vp, gain=rec.vp_g, shift=rec.vp_s,
                                                           vref=rec.vp_r, cap=rec.vp_cap).items()})
    data = dict(Xtr=XA[tr], ytr=y[tr], Xva=XA[va], yva=y[va], cva=cells[va], Xte=XA[te], yte=y[te], cte=cells[te], ext=[], utr=utr,
                str=idx.iloc[tr]["session"].astype(str).to_numpy())
    r = T.fit_eval(model, data, rec, seed, device=device, sfreq=float(meta["sfreq"]), log_every=rec.epochs, return_state=True)
    net = T.build_model(model, X.shape[1], 3, XA.shape[2], float(meta["sfreq"]))
    net.load_state_dict(r["state_last"])
    net.eval()
    feat = []
    h = net.final_layer.register_forward_pre_hook(lambda m, inp: feat.append(inp[0].detach().flatten(1).numpy()))
    with torch.no_grad():
        for i in range(0, len(XA), 256):
            net(torch.as_tensor(XA[i:i + 256]))
    h.remove()
    return np.concatenate(feat), y, tr, va, te


def members(F, y, idx, tr, va, te):
    subj, stream, order = idx["subject"].to_numpy(), idx["stream"].to_numpy(), idx["order"].to_numpy()
    sc = StandardScaler().fit(F[tr])
    pca = PCA(min(64, F.shape[1], len(tr) - 1), random_state=0).fit(sc.transform(F[tr]))
    Z = pca.transform(sc.transform(F))
    Z = Z / Z[tr].std(0, keepdims=True)
    mu0 = Z[tr].mean(0)
    trs = np.zeros(len(Z), bool)
    trs[tr] = True
    Zc = Z.copy()
    for u in np.unique(subj[tr]):
        k = (subj == u) & trs
        Zc[k] = Z[k] - Z[k].mean(0)
    ev = np.concatenate([va, te])
    Zt = Z.copy()
    for s in np.unique(stream[ev]):
        r = np.flatnonzero(stream == s)
        r = r[np.argsort(order[r], kind="stable")]
        assert not trs[r].any(), f"an eval stream with training rows: {s}"
        Zt[r] = Z[r] - (np.cumsum(Z[r], 0) + N0 * mu0) / (np.arange(1, len(r) + 1)[:, None] + N0)
    cen = LogisticRegression(max_iter=3000).fit(Zc[tr], y[tr])
    lin = LogisticRegression(max_iter=3000).fit(Z[tr], y[tr])
    assert list(cen.classes_) == list(lin.classes_) == [0, 1, 2]
    out = {"cen": (cen.predict_proba(Zt[te]), cen.predict_proba(Zt[va])), "lin": (lin.predict_proba(Z[te]), lin.predict_proba(Z[va]))}
    if DRES:   # T2-D394 step 2 (G2 B1): the bounded target − source class log-likelihood ratio on the CENTRED features, causal per eval stream
        mus = np.stack([Zc[tr][y[tr] == c].mean(0) for c in range(3)])
        Rr = Zc[tr] - mus[y[tr]]
        Prec = np.linalg.pinv(np.cov(Rr.T) + 1e-3 * np.eye(Rr.shape[1]))
        ll = lambda x, m: -0.5 * float((x - m) @ Prec @ (x - m))  # noqa: E731
        anc = cen.predict_proba(Zt)                     # the anchor's pseudo-labels (the centred probe), never a label
        adj = np.zeros((len(Z), 3))
        for s_ in np.unique(stream[ev]):
            r = np.flatnonzero(stream == s_)
            r = r[np.argsort(order[r], kind="stable")]
            S, N = np.zeros((3, Z.shape[1])), np.zeros(3)
            for x in r:                                 # target means from EARLIER windows only (prior weight 10 on the source means)
                mt = (10 * mus + S) / (10 + N)[:, None]
                adj[x] = np.clip([ll(Zt[x], mt[c]) - ll(Zt[x], mus[c]) for c in range(3)], -2, 2)
                c_ = int(np.argmax(anc[x]))
                S[c_] += Zt[x]; N[c_] += 1
        out["dres"] = (adj[te], adj[va])
    return out


def seeds(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="fbcnet_bboth,shallow")
    ap.add_argument("--protocols", default="kit,seen1")
    ap.add_argument("--seeds-fbc", default="192-203")
    ap.add_argument("--seeds-sh", default="156-167")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=0, help="smoke tests only (0 = the shipped recipe's)")
    ap.add_argument("--dres", action="store_true", help="T2-D394 step 2: also write <tag>_dres = the bounded density-ratio adj per window")
    a = ap.parse_args()
    global DRES
    DRES = a.dres
    warnings.filterwarnings("ignore")
    torch.set_num_threads(8)
    out = Path(a.out)
    X, idx, meta = B.load_export(a.export)
    log = []
    for m in a.models.split(","):
        tag = SHIPPED[m][2]
        for proto in a.protocols.split(","):
            for s in seeds(a.seeds_fbc if tag == "fbc" else a.seeds_sh):
                F, y, tr, va, te = train_features(m, proto, s, X, idx, meta, a.device, a.epochs)
                res = members(F, y, idx, tr, va, te)
                for kind, (pte, pva) in res.items():
                    d = out / f"{tag}_{kind}"
                    d.mkdir(parents=True, exist_ok=True)
                    key = ("adj_te", "adj_va") if kind == "dres" else ("prob_last", "prob_val_last")    # T2-D394: an adj file is not a member
                    np.savez_compressed(d / f"{proto}__s{s}.npz", test_idx=te, val_idx=va, **{key[0]: pte.astype(np.float32), key[1]: pva.astype(np.float32)})
                acc = {k: float((v[0].argmax(1) == y[te]).mean()) for k, v in res.items() if k != "dres"}
                log.append(dict(model=m, proto=proto, seed=s, test_acc=acc))
                print(f"[d375m] {m} {proto} s{s}: test acc lin {acc['lin']:.4f} cen {acc['cen']:.4f}", flush=True)
    (out / "d375_member.json").write_text(json.dumps(dict(n0=N0, rows=log), indent=1))
    print("[d375m] done", flush=True)


if __name__ == "__main__":
    main()
