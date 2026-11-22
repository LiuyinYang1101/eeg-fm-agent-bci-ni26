#!/usr/bin/env python
"""T2-D307 (zero GPU): the training-set cleaning drop lists for t2p_bench --drop-train, per protocol (kit / seen1), from each protocol's own
TRAINING windows only (no test / val window's data or statistics is used):
  artifact  per training window three EEG-only artifact proxies (the contract export has no EOG / EMG channels): max |x| (extreme amplitude),
            mean 0.5-4 Hz power over the frontal channels (a blink proxy), mean 30-55 Hz power over all channels (an EMG proxy); each ranked
            WITHIN its stream (subject | session) as a percentile; score = the max of the three; drop the top 10 % per stream;
  oofwrong  T2-D253's out-of-fold CNN blend (t2p_bench.oof_pcnn, full) on the training windows: among the windows whose OOF argmax != the
            label, margin = max other-class p - p(label); drop the top 10 % of each training user's windows by margin (confidently wrong).
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d307_flags.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
O = Path.home() / "workspace/outputs/neural_interface_challenge"
OOF = O / "sofia_s1/ni26-t2p-d253-oof-1535295/runs"
FRONTAL = ["AF3", "AF4", "F1", "F2", "F3", "F4", "F5", "F6", "Fz"]
FRAC = 0.10


def band_power(X, fs, lo, hi):
    f = np.fft.rfftfreq(X.shape[-1], 1.0 / fs)
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    return P[..., (f >= lo) & (f <= hi)].mean(-1)


def top_frac(score, groups, frac):
    out = []
    for g in np.unique(groups):
        r = np.flatnonzero(groups == g)
        k = int(round(frac * len(r)))
        if k:
            out.extend(r[np.argsort(-score[r], kind="stable")[:k]].tolist())
    return sorted(out)


def main():
    X, idx, meta = B.load_export(E)
    fs, names = float(meta["sfreq"]), list(meta["ch_names"])
    fr = [names.index(c) for c in FRONTAL]
    y = idx["y"].to_numpy().astype(int)
    out = {"artifact": {}, "oofwrong": {}}
    for proto in ("kit", "seen1"):
        trm = next(t for fname, t, v, te in B.folds(idx, proto) if fname == proto)
        rows = np.flatnonzero(trm)
        Xr = np.asarray(X[rows], dtype=np.float64)
        feats = [np.abs(Xr).max((1, 2)), band_power(Xr[:, fr], fs, 0.5, 4.0).mean(1), band_power(Xr, fs, 30.0, 55.0).mean(1)]
        strm = idx["stream"].to_numpy()[rows]
        pct = np.zeros((len(rows), 3))
        for g in np.unique(strm):
            r = np.flatnonzero(strm == g)
            for j, f in enumerate(feats):
                pct[r, j] = (np.argsort(np.argsort(f[r])) + 0.5) / len(r)
        art = rows[top_frac(pct.max(1), strm, FRAC)]
        P = B.oof_pcnn(str(OOF), proto, y, rows, full=True)
        yr = y[rows]
        p_y = P[np.arange(len(rows)), yr]
        other = P.copy(); other[np.arange(len(rows)), yr] = -1
        margin = np.where(P.argmax(1) != yr, other.max(1) - p_y, -np.inf)
        users = idx["subject"].to_numpy()[rows]
        sel = [i for i in top_frac(margin, users, FRAC) if np.isfinite(margin[i])]
        wrong = rows[sel]
        out["artifact"][proto], out["oofwrong"][proto] = art.tolist(), wrong.tolist()
        print(f"{proto}: {len(rows)} training windows; artifact drop {len(art)} ({len(art) / len(rows):.1%}); oofwrong drop {len(wrong)} "
              f"({len(wrong) / len(rows):.1%}; OOF acc {np.mean(P.argmax(1) == yr):.3f}); overlap {len(set(art) & set(wrong))}", flush=True)
    d = O / "t2_d307"; d.mkdir(exist_ok=True)
    for k, v in out.items():
        (d / f"drop_{k}.json").write_text(json.dumps(v))
    print("written:", [str(d / f"drop_{k}.json") for k in out])


if __name__ == "__main__":
    main()
