#!/usr/bin/env python
"""T2-D404 (pre-registered 2026-11-07 04:27, before any code or number): T2-D403 completed faithfully on CROSS-FITTED features.
Two-way split of the six kit training users (sorted, alternated): A = {P01, P03, P07}, B = {P02, P06, P10}. Per family (T2-D369's shipped
recipe / seed / alignment, ni26.train.fit_eval as t2p_bench's kit task) and per split (net on A, heads on B; then net on B, heads on A):
  the net trains on its half only -> final_layer-input features of every window -> standardise + PCA 64 fit on the HEAD half -> the sklearn
  probe + T2-D403's heads (frozen probe inside, randomized prefix episodes, LOUO selection among the head half's users over the 3 x 2 grid,
  head seeds 100-104) trained on the head half (features out-of-fold for that net) -> held-out kit users (val P05 P08, test P04 P09) read in
  that net's basis, genuine vs TRUE shuffled-user context. Each arm's held-out probabilities averaged over the two splits, then causal SK.
Bar (= T2-D403's): on EITHER family, arm genuine - probe >= +0.005 AND genuine - shuffled-user >= +0.010 typical, P04 >= -0.010 vs probe.
  python t2_d404_xfit.py --export E [--out DIR] [--device cpu] [--epochs 0]
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402
import ni26.train as T  # noqa: E402
from t2_p3_screens import probe, decode, cell_scores  # noqa: E402
from t2_d403_arm import GRID, causal_ctx, fit_head, head_proba, shuffled_ctx  # noqa: E402
from t2_d369_userprobe import SHIPPED  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
HALVES = (("P01", "P03", "P07"), ("P02", "P06", "P10"))


def net_features(model, X, idx, meta, fit_rows, device, epochs):
    """the family's shipped recipe trained on fit_rows only (the kit task's scale / alignment / vp bank) -> final_layer inputs of every window."""
    align, rname, seed = SHIPPED[model]
    (_, trm, vam, tem), = B.folds(idx, "kit")
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    XA = B.aligned_all(X, idx, align, trm, 0.0, 0.05, scale)
    va, te = np.flatnonzero(vam), np.flatnonzero(tem)
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    users = sorted(idx.iloc[fit_rows]["subject"].unique())
    utr = idx.iloc[fit_rows]["subject"].map({u: k for k, u in enumerate(users)}).to_numpy()
    rec = T.RECIPES[rname] if not epochs else replace(T.RECIPES[rname], epochs=epochs)
    T.VP_BANK.clear()
    T.VP_BANK.update({k: v.to(device) for k, v in B.vp_bank(list(meta["ch_names"]), K=rec.vp_k, sigma=0.005 * rec.vp, gain=rec.vp_g, shift=rec.vp_s,
                                                           vref=rec.vp_r, cap=rec.vp_cap).items()})
    data = dict(Xtr=XA[fit_rows], ytr=y[fit_rows], Xva=XA[va], yva=y[va], cva=cells[va], Xte=XA[te], yte=y[te], cte=cells[te], ext=[], utr=utr,
                str=idx.iloc[fit_rows]["session"].astype(str).to_numpy())
    t0 = time.time()
    r = T.fit_eval(model, data, rec, seed, device=device, sfreq=float(meta["sfreq"]), log_every=rec.epochs, return_state=True)
    net = T.build_model(model, X.shape[1], 3, XA.shape[2], float(meta["sfreq"]))
    net.load_state_dict(r["state_last"]); net.eval()
    feat = []
    h = net.final_layer.register_forward_pre_hook(lambda m, inp: feat.append(inp[0].detach().flatten(1).cpu().numpy()))
    with torch.no_grad():
        for i in range(0, len(XA), 256):
            net(torch.as_tensor(XA[i:i + 256]))
    h.remove()
    print(f"[D404] {model}: net on {users} ({len(fit_rows)} windows) in {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(O / "t2_d404"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=0, help="smoke only (0 = the shipped 180)")
    a = ap.parse_args()
    torch.set_num_threads(8)
    X, idx, meta = B.load_export(a.export)
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    assert len(d) == len(idx) and np.array_equal(d.y.to_numpy(), idx["y"].to_numpy())
    subj = d.subject.str[-3:].to_numpy(); y = d.y.to_numpy().astype(int)
    (_, trm, vam, tem), = B.folds(idx, "kit")
    tr = np.flatnonzero(trm); ho = np.concatenate([np.flatnonzero(vam), np.flatnonzero(tem)])
    assert sorted(set(subj[tr])) == sorted(HALVES[0] + HALVES[1]), sorted(set(subj[tr]))
    out = {}
    for fam in ("fbcnet_bboth", "shallow"):
        acc = {k: [] for k in ("probe", "arm_genuine", "arm_shuffled", "bias_genuine", "bias_shuffled")}
        per_split = []
        for s_, (net_half, head_half) in enumerate(((HALVES[0], HALVES[1]), (HALVES[1], HALVES[0]))):
            fit_rows, H = tr[np.isin(subj[tr], net_half)], tr[np.isin(subj[tr], head_half)]
            F = net_features(fam, X, idx, meta, fit_rows, a.device, a.epochs)
            sc = StandardScaler().fit(F[H]); pca = PCA(min(64, F.shape[1], len(H) - 1), random_state=0).fit(sc.transform(F[H]))
            Z = pca.transform(sc.transform(F)); Z = Z / Z[H].std(0, keepdims=True)
            clf = probe(Z[H], y[H])
            sel = {}
            for kind in ("arm", "bias"):
                tab = {}
                for (steps, wd) in GRID:
                    scs = []
                    for u in head_half:
                        rf, re = H[subj[H] != u], H[subj[H] == u]
                        cf = probe(Z[rf], y[rf]); mu_f = Z[rf].mean(0)
                        h, W, b = fit_head(Z, y, rf, d, cf, mu_f, kind, steps, wd, seed=0)
                        P = head_proba(h, W, b, Z, re, causal_ctx(Z, re, d, mu_f), mu_f)
                        scs.append(cell_scores(decode(P, re, d, "causal"), re, d)[u])
                    tab[f"{steps}/{wd:g}"] = float(np.mean(scs))
                sel[kind] = dict(table=tab, pick=max(tab, key=tab.get))
                print(f"[D404] {fam} split {s_} {kind}: LOUO " + " · ".join(f"{k} {v:.4f}" for k, v in tab.items()) + f" → {sel[kind]['pick']}", flush=True)
            mu0 = Z[H].mean(0)
            Cho = causal_ctx(Z, ho, d, mu0); Csh = shuffled_ctx(Cho, ho, d, subj)
            Pp = clf.predict_proba(Z[ho]); P = {"probe": Pp}
            for kind in ("arm", "bias"):
                steps, wd = sel[kind]["pick"].split("/")
                Pg = Ps = 0.0
                for seed in range(100, 105):
                    h, W, b = fit_head(Z, y, H, d, clf, mu0, kind, int(steps), float(wd), seed)
                    Pg = Pg + head_proba(h, W, b, Z, ho, Cho, mu0) / 5
                    Ps = Ps + head_proba(h, W, b, Z, ho, Csh, mu0) / 5
                P[f"{kind}_genuine"], P[f"{kind}_shuffled"] = Pg, Ps
            sc_ = {k: cell_scores(decode(v, ho, d, "causal"), ho, d) for k, v in P.items()}
            eng = {k: float(np.abs(v - Pp).max()) for k, v in P.items() if k != "probe"}
            per_split.append(dict(net=net_half, heads=head_half, selection=sel, typical={k: v["typical"] for k, v in sc_.items()},
                                  P04={k: v["P04"] for k, v in sc_.items()}, engagement_max_abs_dp=eng))
            print(f"[D404] {fam} split {s_}: " + " · ".join(f"{k} {v['typical']:.4f}" for k, v in sc_.items()) +
                  " | max |Δp| vs probe " + ", ".join(f"{k} {v:.3f}" for k, v in eng.items()), flush=True)
            for k in acc:
                acc[k].append(P[k])
        res = {k: cell_scores(decode(np.mean(v, 0), ho, d, "causal"), ho, d) for k, v in acc.items()}
        base, g, s2 = res["probe"], res["arm_genuine"], res["arm_shuffled"]
        engaged = any(sp["engagement_max_abs_dp"]["arm_genuine"] > 0.01 for sp in per_split)
        met = bool(engaged and g["typical"] - base["typical"] >= 0.005 and g["typical"] - s2["typical"] >= 0.010 and g["P04"] - base["P04"] >= -0.010)
        out[fam] = dict(per_split=per_split, res=res, engaged=engaged, met=met)
        print(f"[D404] {fam} (2 splits averaged): " + " · ".join(f"{k} {v['typical']:.4f} ({v['typical'] - base['typical']:+.4f}; P04 "
                                                                f"{v['P04'] - base['P04']:+.4f})" for k, v in res.items()), flush=True)
        print(f"[D404] {fam}: arm genuine − probe {g['typical'] - base['typical']:+.4f}, genuine − shuffled-user {g['typical'] - s2['typical']:+.4f}, "
              f"P04 vs probe {g['P04'] - base['P04']:+.4f}; engaged {engaged} → {'MET' if met else 'not met'}", flush=True)
    out["bar_met"] = any(out[f]["met"] for f in ("fbcnet_bboth", "shallow"))
    print(f"[D404] BAR (either family): {'MET' if out['bar_met'] else 'NOT MET'}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "xfit.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
