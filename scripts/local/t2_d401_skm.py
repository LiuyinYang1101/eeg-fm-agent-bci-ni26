#!/usr/bin/env python
"""T2-D401 (pre-registered 2026-11-07 03:57, before any code or number): the MATCHED-STRENGTH Sinkhorn hindsight diagnostic that repairs the
confounded T2-D393. On M6's shape (outputs/t2_endphase/spec.json, arm M6), 12 draws x kit / seen1 x test / val, every stream decoded by
  C95    the shipped causal SKλ (10 it, n0 3, λ 0.95)                                   <- the anchor (t2p_report.causal_sinkhorn)
  T95    the SAME balance with hindsight: row t against every window of its stream, weights λ^|t-j| (the n0 pseudo-rows at their causal
         position before window 1), 10 it                                              <- THE DECISION ARM (differs from C95 only in the future)
  C95i50 causal, 50 it · C1 causal, λ 1 · F10 two-sided λ 1, 10 it, n0 3 · F50 = T2-D393's whole-stream arm (descriptive)
Bar: T95 - C95 typical >= +0.005 with >= 9 / 12. Also T95 - C95 on each stream's first 20 windows vs the rest (descriptive).
Self-check: the generic weighted balance with causal weights reproduces causal_sinkhorn (allclose) on the first block.
  python t2_d401_skm.py --export E [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(HERE.parent))
from t2p_report import causal_sinkhorn  # noqa: E402
from t2_p3_screens import sinkhorn_full, bal  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EARLY = 20


def weighted_sk(P, n0, lam, iters, two_sided):
    """one stream's rows in chronological order -> the balanced row t for every t: [n0 uniform pseudo-rows; p_1..p_n], row j weighted
    lam^|pos_t - j| (two-sided) or lam^(pos_t - j) for j <= pos_t and 0 after (causal); `iters` rounds of weighted column balance + row
    normalisation (= t2p_report.causal_sinkhorn's update with its weights)."""
    n, K = P.shape
    M0 = np.vstack([np.full((n0, K), 1.0 / K), P.astype(np.float64)])
    N = n0 + n
    pos = n0 + np.arange(n)[:, None]
    j = np.arange(N)[None, :]
    if two_sided:
        W = float(lam) ** np.abs(pos - j)
    else:
        W = np.where(j <= pos, float(lam) ** np.clip(pos - j, 0, None), 0.0)
    M = np.broadcast_to(M0, (n, N, K)).copy()
    ws = W.sum(1)
    for _ in range(int(iters)):
        M = M * (ws / K)[:, None, None] / (M * W[:, :, None]).sum(1, keepdims=True)
        M = M / M.sum(2, keepdims=True)
    return M[np.arange(n), n0 + np.arange(n)]


ARMS = {"C95": dict(n0=3, lam=0.95, iters=10, two_sided=False), "T95": dict(n0=3, lam=0.95, iters=10, two_sided=True),
        "C95i50": dict(n0=3, lam=0.95, iters=50, two_sided=False), "C1": dict(n0=3, lam=1.0, iters=10, two_sided=False),
        "F10": dict(n0=3, lam=1.0, iters=10, two_sided=True), "F50": None}


def decode(P, ix, d, arm):
    strm, ordr = d.stream.to_numpy()[ix], d.order.to_numpy()[ix]
    out, rank = np.empty(len(ix), int), np.empty(len(ix), int)
    for s in np.unique(strm):
        k = np.flatnonzero(strm == s)
        k = k[np.argsort(ordr[k], kind="stable")]
        rank[k] = np.arange(len(k))
        cfg = ARMS[arm]
        Q = sinkhorn_full(P[k]) if cfg is None else weighted_sk(P[k], **cfg)
        out[k] = Q.argmax(1)
    return out, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(O / "t2_d401"))
    a = ap.parse_args()
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    spec = json.load(open(O / "t2_endphase/spec.json"))
    M, w = spec["members"], spec["arms"]["M6"]
    y_all, cells_all, subj_all = d.y.to_numpy(), d.cell.to_numpy(), d.subject.str[-3:].to_numpy()
    res = {k: [] for k in ARMS}
    part = {k: {"early": [], "late": []} for k in ("C95", "T95")}
    checked = False
    for i in range(spec["n"]):
        rows = {k: [] for k in ARMS}
        prow = {k: {"early": [], "late": []} for k in part}
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                ix, mix = None, 0.0
                for m, wm in w.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or np.array_equal(ix, ix_)
                    ix, mix = ix_, mix + wm * p.astype(np.float64)
                if not checked:   # the generic causal balance = the shipped decoder
                    ref = causal_sinkhorn(mix, (d.stream.to_numpy()[ix], d.order.to_numpy()[ix]), 10, 3, 0.95)
                    strm, ordr = d.stream.to_numpy()[ix], d.order.to_numpy()[ix]
                    for s in np.unique(strm):
                        k = np.flatnonzero(strm == s); k = k[np.argsort(ordr[k], kind="stable")]
                        assert np.allclose(weighted_sk(mix[k], 3, 0.95, 10, False), ref[k], atol=1e-9), "generic causal != causal_sinkhorn"
                    checked = True
                    print("[D401] self-check: the generic causal balance reproduces causal_sinkhorn", flush=True)
                y, cells, subj = y_all[ix], cells_all[ix], subj_all[ix]
                for arm in ARMS:
                    pr, rank = decode(mix, ix, d, arm)
                    rows[arm] += [(subj[cells == c][0], bal(y[cells == c], pr[cells == c])) for c in np.unique(cells)]
                    if arm in part:
                        for nm, sel in (("early", rank < EARLY), ("late", rank >= EARLY)):
                            prow[arm][nm] += [(subj[(cells == c) & sel][0], bal(y[(cells == c) & sel], pr[(cells == c) & sel]))
                                              for c in np.unique(cells[sel])]
        for arm in ARMS:
            df = pd.DataFrame(rows[arm], columns=["user", "ba"])
            res[arm].append(dict(typical=df[df.user != "P04"].ba.mean(), P04=df[df.user == "P04"].ba.mean(),
                                 new=df[df.user.isin(["P05", "P08", "P09"])].ba.mean()))
        for arm in part:
            for nm in ("early", "late"):
                df = pd.DataFrame(prow[arm][nm], columns=["user", "ba"])
                part[arm][nm].append(df[df.user != "P04"].ba.mean())
        print(f"[D401] draw {i}: " + " · ".join(f"{k} {res[k][-1]['typical']:.4f}" for k in ARMS), flush=True)
    R = {k: pd.DataFrame(v) for k, v in res.items()}
    out = {"level": {k: v.mean().to_dict() for k, v in R.items()}, "vs_C95": {}}
    for k in ARMS:
        if k == "C95":
            continue
        D = R[k] - R["C95"]
        out["vs_C95"][k] = dict(D.mean().to_dict(), up=int((D.typical > 0).sum()))
        print(f"[D401] {k} − C95: typical {D.typical.mean():+.4f} ({int((D.typical > 0).sum())} / {len(D)}), P04 {D.P04.mean():+.4f}, "
              f"new {D.new.mean():+.4f}", flush=True)
    e = np.array(part["T95"]["early"]) - np.array(part["C95"]["early"])
    l_ = np.array(part["T95"]["late"]) - np.array(part["C95"]["late"])
    out["T95_vs_C95_by_position"] = dict(early=float(e.mean()), late=float(l_.mean()), early_up=int((e > 0).sum()), late_up=int((l_ > 0).sum()))
    print(f"[D401] T95 − C95 typical by position: first {EARLY} windows {e.mean():+.4f} ({int((e > 0).sum())} / 12) · later {l_.mean():+.4f} "
          f"({int((l_ > 0).sum())} / 12)", flush=True)
    dT = out["vs_C95"]["T95"]
    out["bar"] = dict(met=bool(dT["typical"] >= 0.005 and dT["up"] >= 9), rule="T95 − C95 typical >= +0.005 with >= 9 / 12")
    print(f"[D401] BAR (T95 − C95 ≥ +0.005, ≥ 9 / 12): {'MET' if out['bar']['met'] else 'NOT MET'}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "skm.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
