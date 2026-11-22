#!/usr/bin/env python
"""T2-D413b PHASE A (2): what still separates real from synthetic when D413's kit g0 net is sampled with 50 Euler steps (the diagnosis's
best sampler; AUC 1.000)? TRAINING WINDOWS ONLY. Per log-band-power feature (41 ch x 5 bands): Cohen's d (syn - real) / sd_real and the
spread ratio sd_syn / sd_real; per channel: the ratio of MEDIAN window RMS (robust to the artifact windows) and of mean RMS; the share of
the real windows' per-channel variance carried by the top 5 % windows (the heavy tail). Writes outputs/t2_d413b/diag/diag2.json.
    gpu-submit ... python scripts/local/t2_d413b_diag2.py --export E
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "local"))
import t2p_meanflow as MF  # noqa: E402
import t2_d413_gate as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--protocol", default="kit")
    a = ap.parse_args()
    dev = torch.device("cuda")
    X, idx, meta = G.load_export(a.export)
    ck = torch.load(REPO / "outputs" / "t2_d413" / f"gen_{a.protocol}_g0.pt", map_location="cpu", weights_only=False)
    tr, users = ck["train_idx"], ck["users"]
    u = idx.iloc[tr]["subject"].map({s: k for k, s in enumerate(users)}).to_numpy()
    y = idx["y"].to_numpy()[tr]
    Xr = np.asarray(X[tr], dtype=np.float32)
    net = MF.UNet1D(Xr.shape[1], len(users)).to(dev); net.load_state_dict(ck["state"]); net.eval()
    Y, U = torch.as_tensor(y, device=dev), torch.as_tensor(u, device=dev)
    Xs = MF.generate_k(net, Y, net.uemb.weight.detach()[U], ck["mu"].to(dev), ck["sd"].to(dev), torch.Generator().manual_seed(10_000), steps=50, euler=True)
    Fr, Fs = G.logbp(Xr), G.logbp(Xs)
    d = (Fs.mean(0) - Fr.mean(0)) / Fr.std(0)
    sr = Fs.std(0) / Fr.std(0)
    ch = list(meta["ch_names"])
    names = [f"{c}:{lo}-{hi}" for c in ch for lo, hi in G.BANDS]
    top = np.argsort(-np.abs(d))[:15]
    rr, rs = np.sqrt((Xr ** 2).mean(2)), np.sqrt((Xs ** 2).mean(2))       # (n, C) window RMS
    var_r = (Xr ** 2).mean(2)
    tail = [float(np.sort(var_r[:, c])[-int(0.05 * len(var_r)):].sum() / var_r[:, c].sum()) for c in range(len(ch))]
    out = dict(protocol=a.protocol, sampler="eu50",
               cohen_d=dict(median_abs=float(np.median(np.abs(d))), max_abs=float(np.abs(d).max()), share_abs_gt_0p5=float((np.abs(d) > 0.5).mean())),
               spread_ratio=dict(min=float(sr.min()), median=float(np.median(sr)), max=float(sr.max())),
               top_features=[(names[i], round(float(d[i]), 3), round(float(sr[i]), 3)) for i in top],
               band_mean_d={f"{lo}-{hi}": round(float(d.reshape(len(ch), 5)[:, k].mean()), 3) for k, (lo, hi) in enumerate(G.BANDS)},
               band_spread_ratio={f"{lo}-{hi}": round(float(np.median(sr.reshape(len(ch), 5)[:, k])), 3) for k, (lo, hi) in enumerate(G.BANDS)},
               ch_median_rms_ratio=dict(zip(ch, (np.median(rs, 0) / np.median(rr, 0)).round(3).tolist())),
               ch_mean_rms_ratio=dict(zip(ch, (rs.mean(0) / rr.mean(0)).round(3).tolist())),
               real_top5pct_var_share=dict(zip(ch, np.round(tail, 3).tolist())))
    json.dump(out, open(REPO / "outputs" / "t2_d413b" / "diag" / f"diag2_{a.protocol}.json", "w"), indent=1)
    for k in ("cohen_d", "spread_ratio", "band_mean_d", "band_spread_ratio"):
        print(k, out[k])
    print("top |d| features (name, d, sd ratio):", out["top_features"][:10])
    mr = np.median(rs, 0) / np.median(rr, 0)
    print("channel MEDIAN-RMS ratio min / median / max:", mr.min().round(3), np.median(mr).round(3), mr.max().round(3),
          "| lowest:", [(ch[i], round(float(mr[i]), 2)) for i in np.argsort(mr)[:5]])
    print("real top-5 % windows' share of channel variance: median", np.median(tail).round(3), "max", max(tail))


if __name__ == "__main__":
    main()
