#!/usr/bin/env python
"""T2-D413b PHASE A, DIAGNOSE (pre-registered 2026-11-09 11:23, before any D413b number): why did T2-D413's g0 generators fail Step 0?
Per protocol (kit, seen1), the SAME EMA weights (outputs/t2_d413/gen_<proto>_g0.pt), conditioned on every real training window's (user, class):
  samplers: mf1 = 1 step (as D413), mf2 = 2 MeanFlow steps (r 0.5), eu50 = 50 Euler steps of the instantaneous velocity u(z, t, t)
  (the multi-step flow-matching ceiling of the same net). For each: per-class mean log-PSD (Welch 1 s, 1-60 Hz) vs real (channel-averaged),
  the gate's 5-band Delta log power per class (channel-averaged), per-channel RMS ratio syn / real, per-window RMS percentiles, and the gate's
  (a) AUC (training windows only: a tuning metric). Training windows only: no val / test window is read.
Writes outputs/t2_d413b/diag/{diag.json, psd_<proto>.png, rms_<proto>.png}.
    gpu-submit ... python scripts/local/t2_d413b_diag.py --export E
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import welch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))
import t2p_meanflow as MF  # noqa: E402
from t2p_bench import folds, load_export  # noqa: E402

FS, BANDS = 120.0, ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))


def psd(X):
    f, P = welch(X, fs=FS, nperseg=120, axis=-1)
    return f, P                                                    # (n, C, F)


def logbp(f, P):
    return np.log(np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1))   # (n, C, 5)


def auc(Fr, Fs):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return float(cross_val_score(clf, np.r_[Fr, Fs], np.r_[np.zeros(len(Fr)), np.ones(len(Fs))], cv=cv, scoring="roc_auc").mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--gen-dir", default=str(REPO / "outputs" / "t2_d413"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "t2_d413b" / "diag"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, idx, meta = load_export(a.export)
    res = {}
    for proto in ("kit", "seen1"):
        ck = torch.load(Path(a.gen_dir) / f"gen_{proto}_g0.pt", map_location="cpu", weights_only=False)
        tr = ck["train_idx"]
        (_, trm, _, _), = folds(idx, proto)
        assert np.array_equal(np.flatnonzero(trm), tr)
        users = ck["users"]
        u = idx.iloc[tr]["subject"].map({s: k for k, s in enumerate(users)}).to_numpy()
        y = idx["y"].to_numpy()[tr]
        Xr = np.asarray(X[tr], dtype=np.float32)
        net = MF.UNet1D(Xr.shape[1], len(users)).to(dev); net.load_state_dict(ck["state"]); net.eval()
        mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
        Y, U = torch.as_tensor(y, device=dev), torch.as_tensor(u, device=dev)
        ue = net.uemb.weight.detach()[U]
        S = {}
        for name, k, eu in (("mf1", 1, False), ("mf2", 2, False), ("eu50", 50, True)):
            S[name] = MF.generate_k(net, Y, ue, mu, sd, torch.Generator().manual_seed(10_000), steps=k, euler=eu)
        f, Pr = psd(Xr)
        Br = logbp(f, Pr)
        R = dict(n_train=int(len(tr)), real_rms_pct=np.percentile(np.sqrt((Xr ** 2).mean((1, 2))), [5, 50, 95]).round(2).tolist())
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        rms_r = np.sqrt((Xr ** 2).mean((0, 2)))
        for name, Xs in S.items():
            _, Ps = psd(Xs)
            Bs = logbp(f, Ps)
            d = dict(rms_pct=np.percentile(np.sqrt((Xs ** 2).mean((1, 2))), [5, 50, 95]).round(2).tolist(),
                     ch_rms_ratio=dict(zip(("min", "median", "max"), np.percentile(np.sqrt((Xs ** 2).mean((0, 2))) / rms_r, [0, 50, 100]).round(3).tolist())),
                     auc=round(auc(Br.reshape(len(Br), -1), Bs.reshape(len(Bs), -1)), 4), band_dlog={})
            for c in range(3):
                kr, ks = y == c, y == c
                d["band_dlog"][c] = dict(zip([f"{lo}-{hi}" for lo, hi in BANDS], (Bs[ks].mean((0, 1)) - Br[kr].mean((0, 1))).round(3).tolist()))
                lr_, ls_ = np.log(Pr[kr].mean((0, 1))), np.log(Ps[ks].mean((0, 1)))
                axs[c].plot(f[1:61], ls_[1:61], lw=1.5, label=f"{name} (AUC {d['auc']:.2f})")
                if name == "mf1":
                    axs[c].plot(f[1:61], lr_[1:61], lw=2.5, color="k", label="real")
                d.setdefault("psd_dlog_by_hz", {})[c] = (ls_[1:61] - lr_[1:61]).round(3).tolist()
            ax2.plot(np.sqrt((Xs ** 2).mean((0, 2))) / rms_r, marker="o", ms=3, lw=1, label=name)
            R[name] = d
            print(f"== {proto} {name}: AUC {d['auc']:.3f}; window RMS p5/50/95 {d['rms_pct']} (real {R['real_rms_pct']}); channel RMS ratio {d['ch_rms_ratio']}; "
                  f"band dlog class0 {d['band_dlog'][0]}", flush=True)
        for c in range(3):
            axs[c].set_title(f"{proto}: class {c} mean log-PSD (channel-averaged)"); axs[c].set_xlabel("Hz")
        axs[0].set_ylabel("log power"); axs[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"psd_{proto}.png", dpi=110)
        ax2.axhline(1, color="k", lw=1); ax2.set_xlabel("channel"); ax2.set_ylabel("RMS syn / real"); ax2.set_title(f"{proto}: per-channel RMS ratio"); ax2.legend(fontsize=8)
        fig2.tight_layout(); fig2.savefig(out / f"rms_{proto}.png", dpi=110)
        plt.close("all")
        res[proto] = R
    json.dump(res, open(out / "diag.json", "w"), indent=1)
    print(f"wrote {out}/diag.json", flush=True)


if __name__ == "__main__":
    main()
