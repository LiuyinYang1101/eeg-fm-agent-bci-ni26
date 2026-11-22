#!/usr/bin/env python
"""T2-D516 (registered 2026-11-15 before any number; CPU): what is in the per-window DC shift the members' _demean removes?
  features: per-channel means of the contract window, delivered + CAR (82) -> outputs/t2_d515/features_dc.npz (read by t2_d515_read.py --features);
  descriptive: each window's mean minus its stream's causal running mean (windows < t), residualised on [1, order, onset] within the stream;
    per user the class contrasts MI-SUB / MI-WORD / SUB-WORD per channel (difference of means, Welch t); group mean over the ten users,
    cross-user sign agreement, region means, raw vs CAR size, per interface (Graz / BrainHero);
  time course: the long export (-3 .. +8 s re cue, row-aligned; the contract window = samples 480 .. 960), 0.5-s bin means minus the -3 .. 0 s
    baseline, per class, per user then averaged; frontal / central groups; class-difference curves.
    mem-guard python scripts/local/t2_d516_dc.py
"""
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import t2p_bench as B  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
LONG = O / "sofia_s1/ni26-t2p-export-long-1613891/export_proteus_long_m3p8"
X, idx, meta = B.load_export(str(EXP))
CH = list(meta["ch_names"])
U, STREAM, ORD, Y = idx.subject.to_numpy(), idx.stream.to_numpy(), idx.order.to_numpy(), idx.y.to_numpy().astype(int)
ONS, IFC = idx.onset_s.to_numpy().astype(float), idx.interface.to_numpy()
CLS = {0: "MI", 1: "SUB", 2: "WORD"}
PAIRS = [(0, 1), (0, 2), (1, 2)]
FRONT = [c for c in ("AF3", "AF4", "F1", "F2", "F3", "F4", "F5", "F6", "Fz") if c in CH]
CENT = [c for c in CH if c[:1] == "C" or c[:2] in ("FC", "CP")]
POST = [c for c in CH if c[:1] == "P" or c[:2] in ("PO", "TP")]
res = {}

m_raw = np.asarray(X, np.float64).mean(-1)                       # (N, C)
m_car = m_raw - m_raw.mean(1, keepdims=True)
D = O / "t2_d515"
np.savez_compressed(D / "features_dc.npz", F=np.hstack([m_raw, m_car]).astype(np.float32),
                    names=np.array([f"dc_raw_{c}" for c in CH] + [f"dc_car_{c}" for c in CH]), family=np.array(["dc_raw"] * len(CH) + ["dc_car"] * len(CH)))
print(f"[d516] features_dc.npz: {2 * len(CH)} window-mean features", flush=True)


def resid(M):
    """window mean minus the stream's causal running mean (windows < t; the first window dropped), residualised on [1, order, onset]."""
    out = np.full_like(M, np.nan)
    for st in np.unique(STREAM):
        r = np.flatnonzero(STREAM == st)
        r = r[np.argsort(ORD[r], kind="stable")]
        cm = np.cumsum(M[r], 0)
        dev = np.full((len(r), M.shape[1]), np.nan)
        dev[1:] = M[r][1:] - cm[:-1] / np.arange(1, len(r))[:, None]
        k = np.isfinite(dev[:, 0])
        A = np.column_stack([np.ones(k.sum()), ORD[r][k], ONS[r][k]])
        beta = np.linalg.lstsq(A, dev[k], rcond=None)[0]
        e = np.full_like(dev, np.nan)
        e[k] = dev[k] - A @ beta
        out[r] = e
    return out


def contrasts(R, mask=None):
    """-> {pair: (users x C) difference of class means, (users x C) Welch t}"""
    users = sorted(set(U))
    out = {}
    for a, b in PAIRS:
        dif, tt = [], []
        for u in users:
            m = (U == u) & np.isfinite(R[:, 0]) & (np.ones(len(U), bool) if mask is None else mask)
            xa, xb = R[m & (Y == a)], R[m & (Y == b)]
            d = xa.mean(0) - xb.mean(0)
            dif.append(d)
            tt.append(d / np.sqrt(xa.var(0, ddof=1) / len(xa) + xb.var(0, ddof=1) / len(xb)))
        out[(a, b)] = (np.array(dif), np.array(tt))
    return out, users


for view, M in (("raw", m_raw), ("car", m_car)):
    C_, users = contrasts(resid(M))
    for (a, b), (dif, tt) in C_.items():
        g = dif.mean(0)
        agree = (np.sign(dif) == np.sign(g)).mean(0)
        reg = {nm: float(g[[CH.index(c) for c in grp]].mean()) for nm, grp in (("frontal", FRONT), ("central", CENT), ("posterior", POST))}
        top = np.argsort(-np.abs(g))[:6]
        key = f"{view}_{CLS[a]}-{CLS[b]}"
        res[key] = dict(region_uV=reg, global_uV=float(g.mean()), top=[(CH[j], round(float(g[j]), 3), round(float(agree[j]), 2),
                        round(float(np.median(np.abs(tt[:, j]))), 2)) for j in top], mean_abs_t=round(float(np.median(np.abs(tt))), 2))
        print(f"[d516] {key}: regions uV {{{', '.join(f'{k} {v:+.3f}' for k, v in reg.items())}}}, global {g.mean():+.3f}; top (ch, uV, sign agreement, median |t|): "
              + "; ".join(f"{c} {v:+.3f} {s:.2f} {t:.1f}" for c, v, s, t in res[key]["top"]), flush=True)
# raw vs CAR: how much of the effect is common to every channel
for a, b in PAIRS:
    r_ = np.abs(contrasts(resid(m_raw))[0][(a, b)][0]).mean()
    c_ = np.abs(contrasts(resid(m_car))[0][(a, b)][0]).mean()
    res[f"car_over_raw_{CLS[a]}-{CLS[b]}"] = round(float(c_ / r_), 3)
    print(f"[d516] {CLS[a]}-{CLS[b]}: mean |contrast| raw {r_:.3f} uV, CAR {c_:.3f} uV -> CAR keeps {c_ / r_:.2f}", flush=True)
# per interface
for ifc in np.unique(IFC):
    C_, _ = contrasts(resid(m_car), IFC == ifc)
    for (a, b), (dif, tt) in C_.items():
        g = dif.mean(0)
        reg = {nm: round(float(g[[CH.index(c) for c in grp]].mean()), 3) for nm, grp in (("frontal", FRONT), ("central", CENT), ("posterior", POST))}
        res[f"car_{ifc}_{CLS[a]}-{CLS[b]}"] = reg
        print(f"[d516] CAR, interface {ifc}, {CLS[a]}-{CLS[b]}: regions uV {reg}; median |t| {np.median(np.abs(tt)):.2f}", flush=True)
# time course from the long export
XL = np.load(LONG / "X.npy", mmap_mode="r")
assert XL.shape[0] == len(X) and XL.shape[-1] == 1320
chk = max(float(np.abs(np.asarray(XL[i, :, 480:960], np.float64) - np.asarray(X[i], np.float64)).max()) for i in (0, 700, 2000))
print(f"[d516] long export row check: contract slice max |d| {chk:.2e}", flush=True)
bins = np.asarray(XL, np.float64).reshape(len(XL), len(CH), 22, 60).mean(-1)      # (N, C, 22): -3.0 .. +8.0 s in 0.5-s bins
bins = bins - bins[:, :, :6].mean(-1, keepdims=True)                              # baseline -3 .. 0 s
tc = {}
for nm, grp in (("frontal", FRONT), ("central", CENT), ("posterior", POST)):
    gi = [CH.index(c) for c in grp]
    per = np.array([[bins[(U == u) & (Y == c)][:, gi].mean((0, 1)) for c in (0, 1, 2)] for u in sorted(set(U))])   # (users, 3, 22)
    gm = per.mean(0)
    tc[nm] = {CLS[c]: [round(float(v), 2) for v in gm[c]] for c in (0, 1, 2)}
    for a, b in PAIRS:
        d = per[:, a] - per[:, b]
        agree = (np.sign(d) == np.sign(d.mean(0))).mean(0)
        tc[nm][f"{CLS[a]}-{CLS[b]}"] = [round(float(v), 2) for v in d.mean(0)]
        tc[nm][f"{CLS[a]}-{CLS[b]}_agree"] = [round(float(v), 2) for v in agree]
    print(f"[d516] time course {nm} (uV re -3..0 s; bins -3.0 .. +8.0 s, 0.5 s): " + " | ".join(
        f"{k}: " + " ".join(f"{v:+.1f}" for v in tc[nm][k]) for k in ("MI-SUB", "MI-WORD", "SUB-WORD")), flush=True)
res["time_course"] = tc
json.dump(res, open(O / "t2_d516.json", "w"), indent=1, default=str)
