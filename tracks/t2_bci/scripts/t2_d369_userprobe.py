#!/usr/bin/env python
"""T2-D369 (zero GPU): is 'make the net not recognise the person' structurally capped on T2? On the shipped pooled CNN recipes, trained
exactly as t2p_bench's kit task (folds 'kit', the training scale, aligned_all, the vp bank, ni26.train.fit_eval), the input of final_layer
(T2-D293's adversary site) for every public window -> standardised + PCA (<= 64 dims, fit on the kit-TRAIN users) ->
  (a) a linear user-ID probe on the training users (5-fold CV; chance 1 / n_users);
  (b) principal angles between the class-LDA span (2 dims) and the user-LDA span (n_users - 1 dims), training users; null = random spans;
  (c) cross-user class accuracy (probe fit on the training users, scored on the val + test users) before / after projecting out the user
      span, vs projecting out random spans of the same size (20 draws); and user-ID CV accuracy after projecting out the class span.
  python t2_d369_userprobe.py --export E --out DIR [--models fbcnet_bboth,shallow] [--device cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
import ni26.train as T  # noqa: E402

SHIPPED = {"fbcnet_bboth": ("none", "r1_pulxx_vp2_e180_vr_rc75", 192), "shallow": ("ea", "r1_pulx_vp2_e180_vr", 156)}


def orth(M):
    q, _ = np.linalg.qr(M)
    return q


def cosines(A, Bm):
    return np.linalg.svd(orth(A).T @ orth(Bm), compute_uv=False)


def project_out(F, U):
    return F - (F @ U) @ U.T


def probe():
    return LogisticRegression(max_iter=3000, C=1.0)


def features(model, X, idx, meta, device, log, epochs=0):
    align, rname, seed = SHIPPED[model]
    (_, trm, vam, tem), = B.folds(idx, "kit")
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    XA = B.aligned_all(X, idx, align, trm, 0.0, 0.05, scale)
    tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    tr_users = sorted(idx.iloc[tr]["subject"].unique())
    utr = idx.iloc[tr]["subject"].map({u: k for k, u in enumerate(tr_users)}).to_numpy()
    rec = T.RECIPES[rname]
    if epochs:                                    # smoke tests only
        from dataclasses import replace
        rec = replace(rec, epochs=epochs)
    T.VP_BANK.clear()
    T.VP_BANK.update({k: v.to(device) for k, v in B.vp_bank(list(meta["ch_names"]), K=rec.vp_k, sigma=0.005 * rec.vp, gain=rec.vp_g, shift=rec.vp_s,
                                                           vref=rec.vp_r, cap=rec.vp_cap).items()})
    data = dict(Xtr=XA[tr], ytr=y[tr], Xva=XA[va], yva=y[va], cva=cells[va], Xte=XA[te], yte=y[te], cte=cells[te], ext=[], utr=utr,
                str=idx.iloc[tr]["session"].astype(str).to_numpy())
    t0 = time.time()
    r = T.fit_eval(model, data, rec, seed, device=device, sfreq=float(meta["sfreq"]), log_every=rec.epochs, return_state=True)
    log(f"{model}: trained {rname} seed {seed} in {time.time() - t0:.0f}s; kit test cellBA last {r.get('test_cell_bacc_last', r.get('test_cell_bacc', float('nan')))}")
    net = T.build_model(model, X.shape[1], 3, XA.shape[2], float(meta["sfreq"]))
    net.load_state_dict(r["state_last"])
    net.eval()
    feat = []
    h = net.final_layer.register_forward_pre_hook(lambda m, inp: feat.append(inp[0].detach().flatten(1).numpy()))
    with torch.no_grad():
        for i in range(0, len(XA), 256):
            net(torch.as_tensor(XA[i:i + 256]))
    h.remove()
    F = np.concatenate(feat)
    return F, y, idx["subject"].to_numpy(), tr, np.concatenate([va, te]), r


def analyse(F, y, subj, tr, ho, rng):
    sc = StandardScaler().fit(F[tr])
    d = min(64, F.shape[1], len(tr) - 1)
    pca = PCA(d, random_state=0).fit(sc.transform(F[tr]))
    Z = pca.transform(sc.transform(F))
    Z = Z / Z[tr].std(0, keepdims=True)
    users = sorted(set(subj[tr]))
    u = np.array([users.index(s) for s in subj[tr]])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out = dict(dim=int(d), n_train_users=len(users), chance_user=1 / len(users))
    out["user_probe_cv"] = float(cross_val_score(probe(), Z[tr], u, cv=cv).mean())
    out["class_probe_cv_train_users"] = float(cross_val_score(probe(), Z[tr], y[tr], cv=cv).mean())
    lc = LDA(n_components=2).fit(Z[tr], y[tr])
    lu = LDA(n_components=len(users) - 1).fit(Z[tr], u)
    Uc, Uu = orth(lc.scalings_[:, :2]), orth(lu.scalings_[:, :len(users) - 1])
    cs = cosines(Uc, Uu)
    null = [cosines(Uc, orth(rng.standard_normal((d, Uu.shape[1]))))[0] for _ in range(200)]
    out["class_user_cosines"] = [round(float(c), 3) for c in cs]
    out["null_max_cosine_mean"], out["null_max_cosine_p95"] = float(np.mean(null)), float(np.quantile(null, 0.95))

    def xuser(Zp):
        return float(probe().fit(Zp[tr], y[tr]).score(Zp[ho], y[ho]))
    out["xuser_class_acc"] = xuser(Z)
    out["xuser_class_acc_minus_user_span"] = xuser(project_out(Z, Uu))
    rnd = [xuser(project_out(Z, orth(rng.standard_normal((d, Uu.shape[1]))))) for _ in range(20)]
    out["xuser_class_acc_minus_random_span_mean"] = float(np.mean(rnd))
    out["removal_cost_user_span"] = out["xuser_class_acc"] - out["xuser_class_acc_minus_user_span"]
    out["removal_cost_random_span"] = out["xuser_class_acc"] - out["xuser_class_acc_minus_random_span_mean"]
    out["user_probe_cv_minus_class_span"] = float(cross_val_score(probe(), project_out(Z[tr], Uc), u, cv=cv).mean())
    out["strong"] = bool(cs[0] >= 0.5 and out["removal_cost_user_span"] >= 0.03 and out["removal_cost_user_span"] >= 3 * max(out["removal_cost_random_span"], 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="fbcnet_bboth,shallow")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=0, help="smoke tests only (0 = the shipped recipe's 180)")
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, idx, meta = B.load_export(a.export)
    res = {}
    log = lambda s: print(f"[d369] {s}", flush=True)  # noqa: E731
    for m in a.models.split(","):
        F, y, subj, tr, ho, r = features(m, X, idx, meta, a.device, log, a.epochs)
        np.savez_compressed(out / f"feat_{m}.npz", F=F.astype(np.float32), y=y, subj=subj, tr=tr, ho=ho)
        res[m] = analyse(F, y, subj, tr, ho, np.random.default_rng(0))
        log(f"{m}: " + json.dumps(res[m]))
    res["verdict"] = "STRONG (mechanism holds; record and stop)" if all(res[m]["strong"] for m in a.models.split(",")) else "WEAK (pre-register the tuned DANN re-run)"
    (out / "d369.json").write_text(json.dumps(res, indent=1))
    log("verdict: " + res["verdict"])


if __name__ == "__main__":
    main()
